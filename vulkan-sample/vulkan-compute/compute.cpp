// Minimal Vulkan compute: doubles every float in a buffer on the GPU.
// Build: make     Run: make run
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define VK_CHECK(call)                                                        \
    do {                                                                      \
        VkResult r_ = (call);                                                 \
        if (r_ != VK_SUCCESS) {                                               \
            std::fprintf(stderr, "Vulkan error %d at %s:%d\n", (int)r_,       \
                         __FILE__, __LINE__);                                 \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

static std::vector<char> readFile(const char* path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f) { std::fprintf(stderr, "Cannot open %s (did you compile the shader?)\n", path); std::exit(1); }
    size_t size = (size_t)f.tellg();
    std::vector<char> data(size);
    f.seekg(0);
    f.read(data.data(), size);
    return data;
}

static uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    std::fprintf(stderr, "No suitable memory type\n");
    std::exit(1);
}

int main() {
    const uint32_t N = 1024;
    const VkDeviceSize bufferSize = N * sizeof(float);

    // 1. Instance (macOS/MoltenVK needs the portability extension)
    std::vector<const char*> instExts;
    VkInstanceCreateFlags instFlags = 0;
#ifdef __APPLE__
    instExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instFlags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "compute-demo";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.flags = instFlags;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t)instExts.size();
    ici.ppEnabledExtensionNames = instExts.data();
    VkInstance instance;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

    // 2. Physical device (first one found; prefer a discrete GPU)
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
    if (devCount == 0) { std::fprintf(stderr, "No Vulkan devices found\n"); return 1; }
    std::vector<VkPhysicalDevice> devices(devCount);
    vkEnumeratePhysicalDevices(instance, &devCount, devices.data());
    VkPhysicalDevice phys = devices[0];
    for (auto d : devices) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { phys = d; break; }
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    std::printf("Using device: %s\n", props.deviceName);

    // 3. Compute queue family
    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qCount, qprops.data());
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < qCount; i++) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }
    }
    if (family == UINT32_MAX) { std::fprintf(stderr, "No compute queue\n"); return 1; }

    // 4. Logical device + queue (enable portability_subset if the device lists it, e.g. MoltenVK)
    std::vector<const char*> devExts;
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> avail(extCount);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &extCount, avail.data());
    for (auto& e : avail) {
        if (std::strcmp(e.extensionName, "VK_KHR_portability_subset") == 0)
            devExts.push_back("VK_KHR_portability_subset");
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)devExts.size();
    dci.ppEnabledExtensionNames = devExts.data();
    VkDevice device;
    VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);

    // 5. Buffer (host-visible so we can read/write it directly)
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bufferSize;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer;
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(phys, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory memory;
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0));

    float* mapped = nullptr;
    VK_CHECK(vkMapMemory(device, memory, 0, bufferSize, 0, (void**)&mapped));
    for (uint32_t i = 0; i < N; i++) mapped[i] = (float)i;   // input: 0,1,2,...

    // 6. Descriptor set: binds our buffer to "binding = 0" in the shader
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 1;
    dslci.pBindings = &binding;
    VkDescriptorSetLayout setLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &setLayout));

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &poolSize;
    VkDescriptorPool pool;
    VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &pool));

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &setLayout;
    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &set));

    VkDescriptorBufferInfo dbi{buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    // 7. Pipeline
    auto code = readFile("shader.spv");
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size();
    smci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule shader;
    VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &shader));

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &setLayout;
    VkPipelineLayout pipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName = "main";
    cpci.layout = pipelineLayout;
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));

    // 8. Record commands: bind pipeline + descriptors, dispatch
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = family;
    VkCommandPool cmdPool;
    VK_CHECK(vkCreateCommandPool(device, &cpi, nullptr, &cmdPool));

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd));

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(cmd, (N + 255) / 256, 1, 1);   // number of workgroups, 256 threads each
    VK_CHECK(vkEndCommandBuffer(cmd));

    // 9. Submit and wait
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    // 10. Read back and verify
    int bad = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (mapped[i] != 2.0f * (float)i) bad++;
    }
    std::printf("First values: %g %g %g %g %g\n", mapped[0], mapped[1], mapped[2], mapped[3], mapped[4]);
    std::printf(bad == 0 ? "OK: all %u values doubled\n" : "FAILED: %d wrong values\n",
                bad == 0 ? N : (uint32_t)bad);

    // Cleanup
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    vkUnmapMemory(device, memory);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return bad == 0 ? 0 : 1;
}
