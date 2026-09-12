/*
 * Native Vulkan present path, written for the host GPU that pays for every
 * byte crossing PCIe.
 *
 * The SDL_GPU backend maps a transfer buffer and lets the GPU copy from it.
 * SDL prefers host-cached memory for that buffer, which on a discrete card is
 * system RAM: the copy is then the GPU pulling 14.7 MB per 1440p frame across
 * the bus, and on a card sitting in an x4 slot that alone ate half of its
 * graphics engine (measured: 50% of an RX 550 at 48 fps, before the compositor
 * even touched the frame).
 *
 * With a resizable BAR the whole of VRAM is host-visible, so the direction can
 * be reversed: the CPU writes the frame straight into VRAM (write-combined
 * stores, which the bus handles at full width) and the GPU's part of the
 * upload becomes a VRAM-to-VRAM copy that costs it next to nothing. This
 * backend asks for staging memory that is DEVICE_LOCAL and HOST_VISIBLE at the
 * same time and falls back to plain host memory - the SDL behaviour - when the
 * card has no such heap.
 *
 * Everything else is the minimum a swapchain needs: FIFO presentation (the
 * guest is the clock, tearing is not wanted), one image for the frame, a
 * clear-plus-blit into the swapchain image, two frames in flight.
 */
#include "present_internal.h"

#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FRAMES   2
#define MAX_IMGS 8

struct vk_ctx {
    int                              refs;
    VkInstance                       inst;
    VkPhysicalDevice                 phys;
    VkPhysicalDeviceMemoryProperties mem;
    VkDevice                         dev;
    uint32_t                         qfam;
    VkQueue                          queue;
    VkCommandPool                    pool;
    bool                             bar;      /* staging lives in VRAM */
    char                             label[64];
};

struct vk_state {
    struct vk_ctx *ctx;
    SDL_Window    *win;
    VkSurfaceKHR   surface;

    VkSwapchainKHR swap;
    VkFormat       swap_fmt;
    VkExtent2D     swap_ext;
    uint32_t       n_img;
    VkImage        img[MAX_IMGS];
    VkSemaphore    done[MAX_IMGS];   /* per swapchain image: blit finished */
    VkSemaphore    acquire[FRAMES];
    VkFence        fence[FRAMES];
    VkCommandBuffer cmd[FRAMES];
    uint32_t       frame;
    bool           swap_dirty;
    bool           acquire_failed;

    VkImage        tex;
    VkDeviceMemory tex_mem;
    uint32_t       tex_w, tex_h;
    bool           tex_ready;        /* already in TRANSFER_SRC layout */

    VkBuffer       stage;
    VkDeviceMemory stage_mem;
    void          *stage_map;
    VkDeviceSize   stage_bytes;
    VkCommandBuffer upcmd;
    VkFence        upfence;
    bool           up_pending;

    uint32_t src_w, src_h;
    bool     have_frame;
    uint64_t ns_upload, ns_present;
    uint64_t dbg_wait, dbg_acquire, dbg_submit, dbg_present, dbg_n, dbg_last;
    bool     dbg;
};

static int find_mem(const struct vk_ctx *c, uint32_t bits, VkMemoryPropertyFlags need,
                    VkMemoryPropertyFlags want, VkDeviceSize size)
{
    int best = -1;
    for (uint32_t i = 0; i < c->mem.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        const VkMemoryPropertyFlags f = c->mem.memoryTypes[i].propertyFlags;
        if ((f & need) != need) continue;
        if (c->mem.memoryHeaps[c->mem.memoryTypes[i].heapIndex].size < size) continue;
        if ((f & want) == want) return (int)i;
        if (best < 0) best = (int)i;
    }
    return best;
}

/* ------------------------------------------------------------ shared context */

static struct vk_ctx *ctx_create(SDL_Window *win)
{
    if (!SDL_Vulkan_LoadLibrary(NULL)) {
        fprintf(stderr, "vypr: vulkan: %s\n", SDL_GetError());
        return NULL;
    }
    struct vk_ctx *c = SDL_calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->refs = 1;

    Uint32 n_ext = 0;
    const char *const *ext = SDL_Vulkan_GetInstanceExtensions(&n_ext);
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vypr-window",
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .enabledExtensionCount = n_ext,
        .ppEnabledExtensionNames = ext,
    };
    if (vkCreateInstance(&ici, NULL, &c->inst) != VK_SUCCESS) {
        fprintf(stderr, "vypr: vulkan: vkCreateInstance failed\n");
        SDL_free(c);
        return NULL;
    }

    /* A throwaway surface, only to ask which device can present here. */
    VkSurfaceKHR probe = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(win, c->inst, NULL, &probe)) {
        fprintf(stderr, "vypr: vulkan: surface: %s\n", SDL_GetError());
        vkDestroyInstance(c->inst, NULL);
        SDL_free(c);
        return NULL;
    }

    uint32_t n_dev = 0;
    vkEnumeratePhysicalDevices(c->inst, &n_dev, NULL);
    VkPhysicalDevice devs[8];
    if (n_dev > 8) n_dev = 8;
    vkEnumeratePhysicalDevices(c->inst, &n_dev, devs);

    c->phys = VK_NULL_HANDLE;
    for (uint32_t d = 0; d < n_dev && !c->phys; d++) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &nq, NULL);
        VkQueueFamilyProperties q[16];
        if (nq > 16) nq = 16;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &nq, q);
        for (uint32_t i = 0; i < nq; i++) {
            VkBool32 can = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(devs[d], i, probe, &can);
            if (can && (q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                c->phys = devs[d];
                c->qfam = i;
                break;
            }
        }
    }
    SDL_Vulkan_DestroySurface(c->inst, probe, NULL);
    if (!c->phys) {
        fprintf(stderr, "vypr: vulkan: no device can present to this window\n");
        vkDestroyInstance(c->inst, NULL);
        SDL_free(c);
        return NULL;
    }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = c->qfam,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    const char *dev_ext[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = dev_ext,
    };
    if (vkCreateDevice(c->phys, &dci, NULL, &c->dev) != VK_SUCCESS) {
        fprintf(stderr, "vypr: vulkan: vkCreateDevice failed\n");
        vkDestroyInstance(c->inst, NULL);
        SDL_free(c);
        return NULL;
    }
    vkGetDeviceQueue(c->dev, c->qfam, 0, &c->queue);
    vkGetPhysicalDeviceMemoryProperties(c->phys, &c->mem);

    VkCommandPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = c->qfam,
    };
    vkCreateCommandPool(c->dev, &pci, NULL, &c->pool);

    /* Is there VRAM the CPU can write into? 64 MiB is plenty for a 4K frame. */
    c->bar = find_mem(c, ~0u,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      0, 64u << 20) >= 0;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(c->phys, &props);
    snprintf(c->label, sizeof(c->label), "vulkan/%s", c->bar ? "bar" : "ram");
    fprintf(stderr, "vypr: vulkan on %s, staging in %s\n", props.deviceName,
            c->bar ? "VRAM (resizable BAR)" : "system RAM (no host-visible VRAM)");
    return c;
}

static void ctx_release(struct vk_ctx *c)
{
    if (!c || --c->refs > 0) return;
    vkDeviceWaitIdle(c->dev);
    vkDestroyCommandPool(c->dev, c->pool, NULL);
    vkDestroyDevice(c->dev, NULL);
    vkDestroyInstance(c->inst, NULL);
    SDL_Vulkan_UnloadLibrary();
    SDL_free(c);
}

/* ----------------------------------------------------------------- swapchain */

static void swap_destroy(struct vk_state *p)
{
    struct vk_ctx *c = p->ctx;
    if (!p->swap) return;
    vkDeviceWaitIdle(c->dev);
    for (uint32_t i = 0; i < p->n_img; i++)
        if (p->done[i]) { vkDestroySemaphore(c->dev, p->done[i], NULL); p->done[i] = VK_NULL_HANDLE; }
    vkDestroySwapchainKHR(c->dev, p->swap, NULL);
    p->swap = VK_NULL_HANDLE;
    p->n_img = 0;
}

static bool swap_create(struct vk_state *p)
{
    struct vk_ctx *c = p->ctx;
    VkSurfaceCapabilitiesKHR caps;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c->phys, p->surface, &caps) != VK_SUCCESS)
        return false;

    uint32_t nf = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, p->surface, &nf, NULL);
    VkSurfaceFormatKHR fmts[32];
    if (nf > 32) nf = 32;
    vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, p->surface, &nf, fmts);
    VkSurfaceFormatKHR pick = fmts[0];
    for (uint32_t i = 0; i < nf; i++)
        if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM) { pick = fmts[i]; break; }

    int pw = 0, ph = 0;
    SDL_GetWindowSizeInPixels(p->win, &pw, &ph);
    VkExtent2D ext = caps.currentExtent;
    if (ext.width == 0xFFFFFFFFu) {
        ext.width  = (uint32_t)(pw > 0 ? pw : 1);
        ext.height = (uint32_t)(ph > 0 ? ph : 1);
    }
    if (ext.width  < caps.minImageExtent.width)  ext.width  = caps.minImageExtent.width;
    if (ext.height < caps.minImageExtent.height) ext.height = caps.minImageExtent.height;
    if (ext.width  > caps.maxImageExtent.width)  ext.width  = caps.maxImageExtent.width;
    if (ext.height > caps.maxImageExtent.height) ext.height = caps.maxImageExtent.height;
    if (ext.width == 0 || ext.height == 0) return false;   /* minimised */

    uint32_t count = caps.minImageCount + 1;
    if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;
    if (count > MAX_IMGS) count = MAX_IMGS;

    VkSwapchainKHR old = p->swap;
    VkSwapchainCreateInfoKHR sci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = p->surface,
        .minImageCount = count,
        .imageFormat = pick.format,
        .imageColorSpace = pick.colorSpace,
        .imageExtent = ext,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,
        .clipped = VK_TRUE,
        .oldSwapchain = old,
    };
    VkSwapchainKHR nw;
    VkResult r = vkCreateSwapchainKHR(c->dev, &sci, NULL, &nw);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vypr: vulkan: vkCreateSwapchainKHR: %d\n", (int)r);
        return false;
    }
    fprintf(stderr, "vypr: vulkan: swapchain %ux%u, %u images\n", ext.width, ext.height, count);
    if (old) swap_destroy(p);
    p->swap = nw;
    p->swap_fmt = pick.format;
    p->swap_ext = ext;

    p->n_img = MAX_IMGS;
    vkGetSwapchainImagesKHR(c->dev, p->swap, &p->n_img, p->img);
    VkSemaphoreCreateInfo si = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    for (uint32_t i = 0; i < p->n_img; i++) vkCreateSemaphore(c->dev, &si, NULL, &p->done[i]);
    p->swap_dirty = false;
    return true;
}

/* ------------------------------------------------------------ frame storage */

static void tex_destroy(struct vk_state *p)
{
    struct vk_ctx *c = p->ctx;
    if (p->up_pending) { vkWaitForFences(c->dev, 1, &p->upfence, VK_TRUE, UINT64_MAX); p->up_pending = false; }
    vkDeviceWaitIdle(c->dev);
    if (p->stage_map) { vkUnmapMemory(c->dev, p->stage_mem); p->stage_map = NULL; }
    if (p->stage)     { vkDestroyBuffer(c->dev, p->stage, NULL); p->stage = VK_NULL_HANDLE; }
    if (p->stage_mem) { vkFreeMemory(c->dev, p->stage_mem, NULL); p->stage_mem = VK_NULL_HANDLE; }
    if (p->tex)       { vkDestroyImage(c->dev, p->tex, NULL); p->tex = VK_NULL_HANDLE; }
    if (p->tex_mem)   { vkFreeMemory(c->dev, p->tex_mem, NULL); p->tex_mem = VK_NULL_HANDLE; }
    p->tex_ready = false;
    p->stage_bytes = 0;
}

static bool ensure_resources(struct vk_state *p, const struct vypr_frame_view *f)
{
    struct vk_ctx *c = p->ctx;
    const VkDeviceSize need = (VkDeviceSize)f->stride * f->height;
    if (p->tex && p->tex_w == f->width && p->tex_h == f->height && p->stage_bytes >= need)
        return true;
    tex_destroy(p);

    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .extent = { f->width, f->height, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if (vkCreateImage(c->dev, &ici, NULL, &p->tex) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(c->dev, p->tex, &mr);
    int mt = find_mem(c, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, mr.size);
    if (mt < 0) mt = find_mem(c, mr.memoryTypeBits, 0, 0, mr.size);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = (uint32_t)mt,
    };
    if (mt < 0 || vkAllocateMemory(c->dev, &mai, NULL, &p->tex_mem) != VK_SUCCESS) {
        fprintf(stderr, "vypr: vulkan: no memory for a %ux%u image\n", f->width, f->height);
        tex_destroy(p);
        return false;
    }
    vkBindImageMemory(c->dev, p->tex, p->tex_mem, 0);

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = need,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(c->dev, &bci, NULL, &p->stage) != VK_SUCCESS) { tex_destroy(p); return false; }
    vkGetBufferMemoryRequirements(c->dev, p->stage, &mr);
    /* The point of this file: prefer VRAM the CPU can write into. */
    mt = find_mem(c, mr.memoryTypeBits,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  c->bar ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                  mr.size);
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = (uint32_t)mt;
    if (mt < 0 || vkAllocateMemory(c->dev, &mai, NULL, &p->stage_mem) != VK_SUCCESS) {
        fprintf(stderr, "vypr: vulkan: no memory for a %llu byte staging buffer\n",
                (unsigned long long)need);
        tex_destroy(p);
        return false;
    }
    vkBindBufferMemory(c->dev, p->stage, p->stage_mem, 0);
    if (vkMapMemory(c->dev, p->stage_mem, 0, VK_WHOLE_SIZE, 0, &p->stage_map) != VK_SUCCESS) {
        tex_destroy(p);
        return false;
    }
    p->stage_bytes = need;
    p->tex_w = f->width;
    p->tex_h = f->height;
    p->tex_ready = false;
    return true;
}

static void image_barrier(VkCommandBuffer cmd, VkImage img,
                          VkImageLayout from, VkImageLayout to,
                          VkAccessFlags src_access, VkAccessFlags dst_access)
{
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, NULL, 0, NULL, 1, &b);
}

/* ------------------------------------------------------------------ backend */

static const char *vk_driver(void *impl)
{
    const struct vk_state *p = impl;
    return p && p->ctx ? p->ctx->label : "vulkan";
}

static void vk_destroy(void *impl)
{
    struct vk_state *p = impl;
    if (!p) return;
    struct vk_ctx *c = p->ctx;
    if (c) {
        vkDeviceWaitIdle(c->dev);
        tex_destroy(p);
        swap_destroy(p);
        for (int i = 0; i < FRAMES; i++) {
            if (p->acquire[i]) vkDestroySemaphore(c->dev, p->acquire[i], NULL);
            if (p->fence[i])   vkDestroyFence(c->dev, p->fence[i], NULL);
        }
        if (p->upfence) vkDestroyFence(c->dev, p->upfence, NULL);
        if (p->cmd[0])  vkFreeCommandBuffers(c->dev, c->pool, FRAMES, p->cmd);
        if (p->upcmd)   vkFreeCommandBuffers(c->dev, c->pool, 1, &p->upcmd);
        if (p->surface) SDL_Vulkan_DestroySurface(c->inst, p->surface, NULL);
        ctx_release(c);
    }
    SDL_free(p);
}

static void *vk_create(SDL_Window *win, void *share_impl)
{
    struct vk_state *p = SDL_calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->win = win;

    if (share_impl) {
        p->ctx = ((struct vk_state *)share_impl)->ctx;
        p->ctx->refs++;
    } else {
        p->ctx = ctx_create(win);
    }
    if (!p->ctx) { SDL_free(p); return NULL; }
    struct vk_ctx *c = p->ctx;

    if (!SDL_Vulkan_CreateSurface(win, c->inst, NULL, &p->surface)) {
        fprintf(stderr, "vypr: vulkan: surface: %s\n", SDL_GetError());
        vk_destroy(p);
        return NULL;
    }
    VkBool32 can = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(c->phys, c->qfam, p->surface, &can);
    if (!can) {
        fprintf(stderr, "vypr: vulkan: the shared device cannot present to this window\n");
        vk_destroy(p);
        return NULL;
    }

    VkCommandBufferAllocateInfo cai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = c->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = FRAMES,
    };
    vkAllocateCommandBuffers(c->dev, &cai, p->cmd);
    cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(c->dev, &cai, &p->upcmd);

    VkSemaphoreCreateInfo si = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                             .flags = VK_FENCE_CREATE_SIGNALED_BIT };
    for (int i = 0; i < FRAMES; i++) {
        vkCreateSemaphore(c->dev, &si, NULL, &p->acquire[i]);
        vkCreateFence(c->dev, &fi, NULL, &p->fence[i]);
    }
    vkCreateFence(c->dev, &fi, NULL, &p->upfence);

    p->dbg = getenv("VYPR_VKDEBUG") != NULL;
    if (!swap_create(p)) p->swap_dirty = true;   /* minimised at birth: retry on present */
    return p;
}

static bool vk_upload(void *impl, const struct vypr_frame_view *f)
{
    struct vk_state *p = impl;
    struct vk_ctx *c = p->ctx;
    const uint64_t t0 = SDL_GetTicksNS();

    if (f->stride % 4 != 0) return false;
    if (!ensure_resources(p, f)) return false;

    /* The previous copy has to be out of the staging buffer before it is
     * overwritten. It is VRAM to VRAM when the BAR is there, so this is short. */
    if (p->up_pending) {
        vkWaitForFences(c->dev, 1, &p->upfence, VK_TRUE, UINT64_MAX);
        p->up_pending = false;
    }
    vkResetFences(c->dev, 1, &p->upfence);

    memcpy(p->stage_map, f->pixels, (size_t)f->stride * f->height);

    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkResetCommandBuffer(p->upcmd, 0);
    vkBeginCommandBuffer(p->upcmd, &bi);
    image_barrier(p->upcmd, p->tex,
                  p->tex_ready ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region = {
        .bufferOffset = 0,
        .bufferRowLength = f->stride / 4,
        .bufferImageHeight = f->height,
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageExtent = { f->width, f->height, 1 },
    };
    vkCmdCopyBufferToImage(p->upcmd, p->stage, p->tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &region);
    image_barrier(p->upcmd, p->tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    vkEndCommandBuffer(p->upcmd);

    VkSubmitInfo sub = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &p->upcmd,
    };
    if (vkQueueSubmit(c->queue, 1, &sub, p->upfence) != VK_SUCCESS) return false;
    p->up_pending = true;
    p->tex_ready = true;

    p->src_w = f->width;
    p->src_h = f->height;
    p->have_frame = true;
    p->ns_upload += SDL_GetTicksNS() - t0;
    return true;
}

static void vk_present(void *impl)
{
    struct vk_state *p = impl;
    struct vk_ctx *c = p->ctx;
    const uint64_t t0 = SDL_GetTicksNS();

    if (p->swap_dirty || !p->swap) {
        if (!swap_create(p)) { p->ns_present += SDL_GetTicksNS() - t0; return; }
    }

    const uint32_t fr = p->frame % FRAMES;
    vkWaitForFences(c->dev, 1, &p->fence[fr], VK_TRUE, UINT64_MAX);
    const uint64_t t1 = SDL_GetTicksNS();

    uint32_t idx = 0;
    VkResult r = vkAcquireNextImageKHR(c->dev, p->swap, UINT64_MAX, p->acquire[fr],
                                       VK_NULL_HANDLE, &idx);
    const uint64_t t2 = SDL_GetTicksNS();
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        p->swap_dirty = true;
        p->ns_present += SDL_GetTicksNS() - t0;
        return;
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        if (!p->acquire_failed) fprintf(stderr, "vypr: vulkan: vkAcquireNextImageKHR: %d\n", (int)r);
        p->acquire_failed = true;
        p->ns_present += SDL_GetTicksNS() - t0;
        return;
    }
    p->acquire_failed = false;
    vkResetFences(c->dev, 1, &p->fence[fr]);

    VkCommandBuffer cmd = p->cmd[fr];
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkResetCommandBuffer(cmd, 0);
    vkBeginCommandBuffer(cmd, &bi);
    image_barrier(cmd, p->img[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  0, VK_ACCESS_TRANSFER_WRITE_BIT);

    int dx = 0, dy = 0, dw = (int)p->swap_ext.width, dh = (int)p->swap_ext.height;
    if (p->have_frame && p->tex)
        vypr_fit_rect((int)p->swap_ext.width, (int)p->swap_ext.height, p->src_w, p->src_h,
                      &dx, &dy, &dw, &dh);

    /* The margin of a fitted picture is never written by the blit; clear it
     * rather than show whatever the image held last. Skipped when the picture
     * covers the whole image - the common case in fullscreen. */
    if (!p->have_frame || dx > 0 || dy > 0 ||
        (uint32_t)dw < p->swap_ext.width || (uint32_t)dh < p->swap_ext.height) {
        VkClearColorValue black = { .float32 = { 0.f, 0.f, 0.f, 1.f } };
        VkImageSubresourceRange all = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, p->img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &all);
    }
    if (p->have_frame && p->tex) {
        VkImageBlit blit = {
            .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .srcOffsets = { { 0, 0, 0 }, { (int32_t)p->src_w, (int32_t)p->src_h, 1 } },
            .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .dstOffsets = { { dx, dy, 0 }, { dx + dw, dy + dh, 1 } },
        };
        const bool same = (uint32_t)dw == p->src_w && (uint32_t)dh == p->src_h;
        vkCmdBlitImage(cmd, p->tex, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       p->img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, same ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    }
    image_barrier(cmd, p->img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
    vkEndCommandBuffer(cmd);

    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo sub = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &p->acquire[fr],
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &p->done[idx],
    };
    vkQueueSubmit(c->queue, 1, &sub, p->fence[fr]);
    const uint64_t t3 = SDL_GetTicksNS();

    VkPresentInfoKHR pi = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &p->done[idx],
        .swapchainCount = 1,
        .pSwapchains = &p->swap,
        .pImageIndices = &idx,
    };
    r = vkQueuePresentKHR(c->queue, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) p->swap_dirty = true;
    else if (r != VK_SUCCESS) fprintf(stderr, "vypr: vulkan: vkQueuePresentKHR: %d\n", (int)r);
    const uint64_t t4 = SDL_GetTicksNS();

    if (p->dbg) {
        p->dbg_wait += t1 - t0; p->dbg_acquire += t2 - t1; p->dbg_submit += t3 - t2;
        p->dbg_present += t4 - t3; p->dbg_n++;
        if (t4 - p->dbg_last > 1000000000ull) {
            const double n = p->dbg_n ? (double)p->dbg_n : 1.0;
            fprintf(stderr, "vypr: vulkan: %llu frames | wait %.1f acquire %.1f submit %.1f present %.1f ms\n",
                    (unsigned long long)p->dbg_n, p->dbg_wait / n / 1e6, p->dbg_acquire / n / 1e6,
                    p->dbg_submit / n / 1e6, p->dbg_present / n / 1e6);
            p->dbg_wait = p->dbg_acquire = p->dbg_submit = p->dbg_present = p->dbg_n = 0;
            p->dbg_last = t4;
        }
    }

    p->frame++;
    p->ns_present += SDL_GetTicksNS() - t0;
}

static void vk_take_timings(void *impl, uint64_t *upload_ns, uint64_t *present_ns)
{
    struct vk_state *p = impl;
    if (upload_ns)  *upload_ns  = p->ns_upload;
    if (present_ns) *present_ns = p->ns_present;
    p->ns_upload = p->ns_present = 0;
}

const struct present_ops present_vk_ops = {
    .name         = "vulkan",
    .create       = vk_create,
    .destroy      = vk_destroy,
    .driver       = vk_driver,
    .upload       = vk_upload,
    .present      = vk_present,
    .take_timings = vk_take_timings,
};
