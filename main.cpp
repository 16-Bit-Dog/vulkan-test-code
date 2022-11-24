// code here for refrence when on the go; realized I sometimes need sample code when coding vulkan and DX12 [hopefull] stuff

//https://vulkan-tutorial.com/en/Drawing_a_triangle/Graphics_pipeline_basics/Introduction
// code mostly gotten from above ^ --> for learning purposes, this was an amazing resource

// NOTE: most of my notes miss optional members for structs --> these may be usful, so always look at them if I need to do something spesific

// I cant say this enough; https://vulkan-tutorial.com/en/ <-- this is an amazing guy for doing this this website turtorial


// just so I remember: indices[] is the array for loaded model verticies


// for testing purposes on a laptop, vsync was turned on --> if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)    <-- that controls it, uncomment to turn off vsync to immediately feed frames

#define GLM_ENABLE_EXPERIMENTAL
#include <../glm/gtx/hash.hpp>
#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <../glm/glm.hpp>
#include <../glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define NOMINMAX
#include <Windows.h>

#include <chrono>
#include <glfw3.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <set>
#include <fstream>
#include <array>
#include <algorithm>
#include <unordered_map>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <shaderc/shaderc.h>



const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int modelCount = 6;
const int textureCount = 3;

std::string MODEL_PATH = "model/1.obj";
std::string TEXTURE_PATH = "textures/1.png";



const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {  //dunno what this is
    "VK_LAYER_KHRONOS_validation" //standart validation layers
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME // declare a default list of extensions
};


#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

//////////////////make a debug messager
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}
//////////////////

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily; // second set of familys that are alone
    std::optional<uint32_t> presentFamily; // same as above

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities; // surface capabilitys of device
    std::vector<VkSurfaceFormatKHR> formats; //formats supported by device
    std::vector<VkPresentModeKHR> presentModes; //present modes in device
};


//////////////////////////////////////////////// custom vertex data
struct Vertex { // vertex struct
    glm::vec3 pos; // vector 3 for 3 dimensions ;)
    glm::vec3 color;
    glm::vec2 texCoord; //texture cordinates

    static VkVertexInputBindingDescription getBindingDescription() {  // tell gpu how to pass this information around
        VkVertexInputBindingDescription bindingDescription{}; // rate to load data from memory through vertexes; nuber of bytes between entry
        bindingDescription.binding = 0; // index of binding in array  --> we have 1 vertex array, so only look at array one
        bindingDescription.stride = sizeof(Vertex); //number of bytes from entry A to entry B (vertex A to B)
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // read size of vertex, moving to next data entry after vertex is read
        //Input rate vars:
  //    VK_VERTEX_INPUT_RATE_VERTEX: Move to the next data entry after each vertex
  //    VK_VERTEX_INPUT_RATE_INSTANCE : Move to the next data entry after each instance


        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() { // function to fill in vertex struct


        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        attributeDescriptions[0].binding = 0; // which binding [index] the data comes from 
        attributeDescriptions[0].location = 0; // location in bytes (?) -- just start at beginning
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT; // color enumeration format is red and green
        /* this implicitly implys byte size (its for the format member)
        VarType: varFormat
        float: VK_FORMAT_R32_SFLOAT
        vec2: VK_FORMAT_R32G32_SFLOAT
        vec3: VK_FORMAT_R32G32B32_SFLOAT
        vec4: VK_FORMAT_R32G32B32A32_SFLOAT
        */
        attributeDescriptions[0].offset = offsetof(Vertex, pos); // number of bytes since the start of the per-vertex data to read


        attributeDescriptions[1].binding = 0; // index for where data comes from
        attributeDescriptions[1].location = 1; //location of bytes
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT; // rgb 
        attributeDescriptions[1].offset = offsetof(Vertex, color); // number of bytes 

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }

    bool operator==(const Vertex& other) const { // a custom operator to compare verticies
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }

};

namespace std { // hash functions are the idea of returning a hash code to see if values are equal...
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}



struct UniformBufferObject {
    alignas(16) glm::mat4 model; //4x4 matrix --> alligned to 16 byte boundary
    alignas(16) glm::mat4 view; //4x4  --> alligned to 16 byte boundary
    alignas(16) glm::mat4 proj; //4x4  --> alligned to 16 byte boundary


};

const std::vector<Vertex> vertices = { // array format:  {vertex pos, -1.0f to 1.0f (max to minimum)}, {r,g,b} (-1.0f is min, 1.0f is max)
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}}, // if your wondering on the format; first is the position, second is color, third is tex coordinates (this is 4 rows, so 4 verticies with these properties)
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},


    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}


};

// use uint16 or uint32 based on how many verticies we need --> INDEX BUFFERS ARE JUST FOR REUSING VERTEXES
const std::vector<uint16_t> indices = { 0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4 }; // make vector with index of vertex's on the verticie buffer --> we call the vertecie 0, then 1, then 2, then 2, then 3, then 0
////////////////////////////////////////////////


class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:

    GLFWwindow* window; // make window var

    POINT XYspeed1;

    POINT XYspeed2;

    float projectionVarFUN = -1;

    int radianPOSx;
    int radianPOSy;


    int FPScount;
    int score;
    std::chrono::system_clock::time_point timeStart;
    std::chrono::system_clock::time_point timeEnd;

    std::chrono::steady_clock::time_point startTime = std::chrono::high_resolution_clock::now();

    VkInstance instance; // make instance for vulkan
    VkDebugUtilsMessengerEXT debugMessenger; // debugger var
    VkSurfaceKHR surface; // surface interface object to interact in windows with vk


    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; //physical GPU device handle
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT; // get sample count device supports for MSAA - since by default we use 1 pixel, the default is 1 sample per pixel
    VkDevice device; // the GPU device (software controlled)

    VkQueue presentQueue; // control logical device creation
    VkQueue graphicsQueue; // the graphics adapters avaible are stored here

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat; // format  images in
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews; // make an image view vector to store image views in when creating image views for swapchain
    std::vector<VkFramebuffer> swapChainFramebuffers; // the frame buffer vector to hold them

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout; // layout of descriptor var
    VkPipelineLayout pipelineLayout; // pipeline layout for viewport stuff
    VkPipeline graphicsPipeline1;
    VkPipeline graphicsPipeline2;

    VkCommandPool commandPool;

    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    uint32_t mipLevels; // mip maps are levels of images, farther away means smaller image, and that means less lines of code to parse through --> we divide by 2 each time to evenly get smaller images
    std::vector <VkImage> textureImage{};
    std::vector <VkDeviceMemory> textureImageMemory{};
    std::vector <VkImageView> textureImageView;

    std::vector <VkImage> MiniTextureImage{}; // one off creation for testing <-- not used in public
    std::vector <VkDeviceMemory> MiniTextureImageMemory{};
    std::vector <VkImageView> MiniTextureImageView{};
    int MiniTextureCount;



    VkSampler textureSampler;



    std::vector < std::vector<Vertex> > vertices;
    std::vector < std::vector<uint32_t> > indices; // loading indices var; is for vertexes to be stored --> this is a vector of vector to allow storage of many models dynamically... yes I'm inconsistant... lucky this is my personal project...


    std::vector <VkBuffer> vertexBuffer; // store the vertex buffer
    std::vector <VkDeviceMemory> vertexBufferMemory; // store buffer memory

    std::vector <VkBuffer> indexBuffer; // store index data
    std::vector <VkDeviceMemory> indexBufferMemory; // index ram storage

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;

    // these 2 below are waiting for a signal that says process is done 
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences; // stall until cpu/gpu is done operations for frame --> deals with in use/flight frame
    std::vector<VkFence> imagesInFlight; // stall until cpu/gpu is done operations for the images showing

    size_t currentFrame = 0; // so we use the correct semaphore pair each time

    bool framebufferResized = false; // check if frame buffer (window) has its size changed

    void initWindow() {
        glfwInit(); // start glfw api

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // dont know what this definition is
              //  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // make client window not resizable

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr); // make window to width and height specified; also name it vulkan (and make no childs and stuff)
        glfwSetWindowUserPointer(window, this); // get pointer of current window; arbitrary value
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback); //set size of frame buffer to the new size call back

    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) { // Make sure 
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window)); // force the GLFW window pointer (gotten from the user pointer) to be casted into hello triangle application "application window" type
        app->framebufferResized = true; // make fence to be true
    }


    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        createTextureImage();
        createTextureImageView();
        createTextureSampler(); // gotta get to do that AA16x ;)
        loadModel();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }





    void mainLoop() {
        while (!glfwWindowShouldClose(window)) { // if close flag for "window" is true 
            glfwPollEvents(); // call all processes to finish processing (I assume to close it)
            glfwPollEvents(); // call all processes to finish processing (I assume to close it)
            drawFrame();
        }

        vkDeviceWaitIdle(device); // wait applies to "device"
    }


    void cleanupSwapChain() { // destroy frame buffers, command buffers, pipeline, pipeline layout, renderpass, vk image, KHR swap chain device

        vkDestroyImageView(device, colorImageView, nullptr);
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorImageMemory, nullptr);

        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

        vkDestroyPipeline(device, graphicsPipeline1, nullptr);
        vkDestroyPipeline(device, graphicsPipeline2, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }



    void cleanup() {
        //most of this is self explanitory...
        cleanupSwapChain();

        vkDestroySampler(device, textureSampler, nullptr);

        for (int i = 0; i < textureImage.size(); i++) {

            vkDestroyImageView(device, textureImageView[i], nullptr);


            vkDestroyImage(device, textureImage[i], nullptr); // clean image
            vkFreeMemory(device, textureImageMemory[i], nullptr); // clean image
        }
        for (int i = 0; i < MiniTextureImage.size(); i++) {


            vkDestroyImageView(device, MiniTextureImageView[i], nullptr);


            vkDestroyImage(device, MiniTextureImage[i], nullptr); // clean image
            vkFreeMemory(device, MiniTextureImageMemory[i], nullptr); // clean image

        }

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        for (int i = 0; i < indexBuffer.size(); i++) {
            vkDestroyBuffer(device, indexBuffer[i], nullptr);
            vkFreeMemory(device, indexBufferMemory[i], nullptr);

            vkDestroyBuffer(device, vertexBuffer[i], nullptr);
            vkFreeMemory(device, vertexBufferMemory[i], nullptr);
        }


        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr); // destory command pool

        vkDestroyDevice(device, nullptr); // destroy device instance

        if (enableValidationLayers) { // if debug mode yall gotta destroy em
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }



    void recreateSwapChain() { // remake swapchain to re-render images; bet frame biffer size for the temp width and height var, then
        // wait for devices and then rerender
        int width = 0, height = 0;
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device); //pause program when device is not done commands
        // flip the model used
        //


        cleanupSwapChain(); // clean up the swap chain to make it again --> remake what needs to be updated for the next frame
        createSwapChain();
        createImageViews();
        createRenderPass();
        /*

        from https://marcelbraghetto.github.io/a-simple-triangle/2019/10/06/part-28/ <-- another amazing guy for helping me troubleshoot features with his written turtorial

        : loop through each mesh instance in the meshes list to render

  - get the mesh from the asset manager
  - populate the push constants with the transformation matrix of the mesh
  - bind the pipeline to the command buffer
  - bind the vertex buffer for the mesh to the command buffer
  - bind the index buffer for the mesh to the command buffer
  - get the texture instance to apply to the mesh from the asset manager
  - get the descriptor set for the texture that defines how it maps to the pipeline's descriptor set layout
  - bind the descriptor set for the texture into the pipeline's descriptor set layout
  - tell the command buffer to draw the mesh

: end loop

        */
        createGraphicsPipeline();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();

    }


    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{}; // add some app info that can be used by drivers to speed up
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // says this is a application info struct
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; //says this is a vk instance info struct being made
        createInfo.pApplicationInfo = &appInfo; // set application info

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size()); // add info for vk program ?
        createInfo.ppEnabledExtensionNames = extensions.data(); // name of extensions (containor of sorts)

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo; // enable debugger stuff
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo; // convert debug messanger into vk debug* type
        }
        else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }











    ////////////////////////////////
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) { // chose which memory type to use
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties); // get all memory properties of device - now we know what we have avaible like amount of ram

        // memHeap is distince resources like vram or ram for when Vram runs out

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {// 
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        // above checks for memory type listed as propeerties in the function var


        throw std::runtime_error("failed to find suitable memory type!");
    }
    ////////////////////////////////





    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) { // debug messanger info 
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT; //says this is a debug info struct being made
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() { // if validation layers are on, setup debuger
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {

        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) { // make vk instance and gflw window surface linked to vulkan &surface for interfacing with windows
            throw std::runtime_error("failed to create window surface!");
        }

    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr); // pre allocate for device count

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount); // convert all devices into a containor of vk devices
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());  // setup devices 

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) { // check if device is good
                physicalDevice = device; // if good stop looking and make the chosen device the enumerated device
                msaaSamples = getMaxUsableSampleCount();
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;


        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }



    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        //store capabilitys of device and a bit more handling of capabilitys
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        //store capabilitys of device and a bit more handling of capabilitys


        uint32_t imageCount = textureCount; // minimum images; we do 1 more to prevent the need for the driver to stall the program until another image can be added
        if (swapChainSupport.capabilities.maxImageCount < imageCount) { // 0 means no maximum
            imageCount = swapChainSupport.capabilities.maxImageCount; // 
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        ///DETAILS OF SWAP CHAIN
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1; // layers of image are 1 unless using steroscopic 3d for anaglyph 3d or somthing like it
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // in case of using modified images with post
        ///^DETAILS OF SWAP CHAIN^

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT; // allows multipul familys to use file
            createInfo.queueFamilyIndexCount = 2; //say 2 familys are needed to be linked for later
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // faster, but only use if family of graphics and present is the same
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // says we do not want transformations
        createInfo.presentMode = presentMode; // sets the present mode
        createInfo.clipped = VK_TRUE; //means we do not care about color of pixels covered by another window - do not rerender covered pixels

        createInfo.oldSwapchain = VK_NULL_HANDLE; // if swap must be remade, just dont remake it

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) { // make the swap chain
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

    }

    void createImageViews() {

        swapChainImageViews.resize(swapChainImages.size()); // resize the vector of images into the acctual size of the image containor; remember how vetctors have extra space (we want to remove that)

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1); // make image views...
        }
    }






    void createRenderPass() {

        //////////////////////////////////
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat; // format of image to deal with
        colorAttachment.samples = msaaSamples; // sample rate - used for multisampling which I am doing

        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clear constants at the start of render pass each and every render pass after
        /*
        for loadOp

        VK_ATTACHMENT_LOAD_OP_LOAD: Preserve the existing contents of the attachment
        VK_ATTACHMENT_LOAD_OP_CLEAR: Clear the values to a constant at the start
        VK_ATTACHMENT_LOAD_OP_DONT_CARE: Existing contents are undefined; we don't care about them


        */


        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        /*
        for storeOp

        VK_ATTACHMENT_STORE_OP_STORE: Rendered contents will be stored in memory and can be read later
        VK_ATTACHMENT_STORE_OP_DONT_CARE: Contents of the framebuffer will be undefined after the rendering operation - bassically discard old frames


        */


        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // these are stencil buffers... but we dont use them... so who cares, same for store
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;// ^^^^^
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // 
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        /*
        * // will be talked about more in the textureing chapter of the guide I am looking at
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: Images used as color attachment
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: Images to be presented in the swap chain
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: Images to be used as destination for a memory copy operation
        */

        //////////////////////////////////


        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat(); // image format is the same as shown
        depthAttachment.samples = msaaSamples; // MSAA is on depth as well
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription colorAttachmentResolve{}; // SINCE WE CANT SHOW MULTISAMPLED IMAGES RIGHT AWAY WE NEED TO HAVE A BASE IMAGE - made from this struct
        colorAttachmentResolve.format = swapChainImageFormat;
        colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;


        VkAttachmentReference colorAttachmentRef{}; //dunno
        colorAttachmentRef.attachment = 0; // refrences a index of the color affect
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // best proformance color post processing effect

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentResolveRef{}; // WE ALSO NEED A struct for color refrence for MSAA images
        colorAttachmentResolveRef.attachment = 2;
        colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


        VkSubpassDescription subpass{}; // post processing stuff
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; // says this is a subpass for graphics and will have more in the future
        subpass.colorAttachmentCount = 1; // color attachment count; length of color attachments
        subpass.pColorAttachments = &colorAttachmentRef; // use info from colorattachmentRef - the data is from the fragment shader - data is gotten from the renderPassInfo pAttachments member?
        subpass.pDepthStencilAttachment = &depthAttachmentRef;
        subpass.pResolveAttachments = &colorAttachmentResolveRef; //
       /*

        Other subpass refrences for getting color (for above)


        pColorAttachments: get color from fragment shader layout()
        pInputAttachments: Attachments that are read from a shader

        pDepthStencilAttachment: Attachment for depth and stencil data
        pPreserveAttachments: Attachments that are not used by this subpass, but for which the data must be preserved

        */

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL; // external subpass is used
        dependency.dstSubpass = 0; // refers to the subpass of this index
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // wait on this stage for src; wait for color data to be read from the output stage
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // wait on this stage for dst; allow writing color data when we want
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };
        VkRenderPassCreateInfo renderPassInfo{}; //renderpass data
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size()); // number of attachment
        renderPassInfo.pAttachments = attachments.data(); // the p...attachment data
        renderPassInfo.subpassCount = 1; // subpass count - length of subpass (struct) array
        renderPassInfo.pSubpasses = &subpass; //subpass
        renderPassInfo.dependencyCount = 1; // total dependancy's
        renderPassInfo.pDependencies = &dependency; // array of dependency's


        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) { // make renderpass and store in renderPass var
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0; // have no binding in shader
        uboLayoutBinding.descriptorCount = swapChainImages.size(); //how many descriptors are used; could be used like an array to control bones of model
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; //type of resource descriptor  
        uboLayoutBinding.pImmutableSamplers = nullptr; // for image sampling
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // which stage will we refrence the descriptor; here it says vertex bit

        VkDescriptorSetLayoutBinding samplerLayoutBinding{}; // making a binding
        samplerLayoutBinding.binding = 1; // 1 binding for the sampler, and 1 description, ect...
        samplerLayoutBinding.descriptorCount = swapChainImages.size();
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // combined image sampler descriptor in the fragment shader (remmeber *that* file)
        samplerLayoutBinding.pImmutableSamplers = nullptr; // turn off changing of sampler during run time...  could allow filters to be hot swapable on
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // also can use vertex shader file if this is on to make deformations with height map

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };


        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; // layout struct type of telling layout info
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size()); // bindings count, by getting int version of array size
        layoutInfo.pBindings = bindings.data(); // 1 layout object refrenced 

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) { /// make descriptor on device, with layout info, without an allocator for memory,
            //^ final value is where to put the compiled descriptor layout [put inside a var]

            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }




    void createGraphicsPipeline() {


        std::string tempName = "compile.bat";
        system(tempName.c_str());

        auto vertShaderCode = readFile("shaders/vert.spv"); // byte file is loaded
        auto fragShaderCode = readFile("shaders/frag.spv"); // ^

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode); //store shader module
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode); //^

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{}; // vertex shader info
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; // assign to shader pipeline
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; // assign to vertex stage
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main"; // shader controlled

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{}; // fragment of shader info
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; // assign to shader pipeline
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT; // assign to fragment stage
        fragShaderStageInfo.module = fragShaderModule;  // fragment controlled
        fragShaderStageInfo.pName = "main";
        // if we dont include something like pSpecializationInfo it is automatically nullptr

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo }; // vertex and frag data 


        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO; // format of vertex

        auto bindingDescription = Vertex::getBindingDescription(); // binding description
        auto attributeDescriptions = Vertex::getAttributeDescriptions(); // attribute description --> shader controls offset of texture on 3d model; also use this for offsets; so make this offset value changed to shift in unison

        vertexInputInfo.vertexBindingDescriptionCount = 1; // 1 vertex descriptors
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()); // cast descriptor array (number of items) to an int
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; //refrence to binding description
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data(); // data of attribute descriptions

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{}; //type of geometry to be drawn is stated here

        /*
        TOPOLOGY members for how to draw geometry:

        VK_PRIMITIVE_TOPOLOGY_POINT_LIST: points from vertices
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST : line from every 2 vertices without reuse
        VK_PRIMITIVE_TOPOLOGY_LINE_STRIP : the end vertex of every line is used as start vertex for the next line
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : triangle from every 3 vertices without reuse
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : the second and third vertex of every triangle are used as first two vertices of the next triangle
        */

        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; // type of struct
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // we will draw triangles from 3 vertexes
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        std::array<VkViewport, 2> viewport{}; // where we will render
        viewport[0].x = 0.0f; //width to start rendering  --> will fit image into the border made through stretching it and such
        viewport[0].y = 0.0f; //height to start rendering
        viewport[0].width = (float)swapChainExtent.width / 2; // width to render (right how much) --> will fit image into the border made
        viewport[0].height = (float)swapChainExtent.height; // height to render (down how much)
        viewport[0].minDepth = 0.0f; // view distance min
        viewport[0].maxDepth = 1.0f; // view distance max 

        viewport[1].x = (float)swapChainExtent.width / 2; //width to start rendering  --> will fit image into the border made through stretching it and such
        viewport[1].y = 0.0f; //height to start rendering
        viewport[1].width = (float)swapChainExtent.width / 2; // width to render (right how much) --> will fit image into the border made
        viewport[1].height = (float)swapChainExtent.height; // height to render (down how much)
        viewport[1].minDepth = 0.0f; // view distance min
        viewport[1].maxDepth = 1.0f; // view distance max 



        VkRect2D scissor1{}; //-- > how much to cut image.. yes acctually just cut parts out
        scissor1.offset = { int(0.0f), int(0.0f) }; //where to start cut off in x and y
        scissor1.extent = { swapChainExtent.width / 2, swapChainExtent.height }; // where to end cut off 

        VkRect2D scissor2{}; //-- > how much to cut image.. yes acctually just cut parts out
        scissor2.offset = { int(swapChainExtent.width / 2), int(0.0f) }; //where to start cut off in x and y
        scissor2.extent = { swapChainExtent.width, swapChainExtent.height }; // where to end cut off 


        VkPipelineViewportStateCreateInfo viewportState1{};
        viewportState1.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO; // type of struct
        viewportState1.viewportCount = 1; // // use selected view:  you can use multipul views on some gpus, but make sure when qurrying in the gpu for features that it supports this feature
        viewportState1.pViewports = &viewport[0]; // use selected view port
        viewportState1.scissorCount = 1; // use selected sissor:  you can use multipul sissors on some gpus, but make sure when qurrying in the gpu that it supports this feature
        viewportState1.pScissors = &scissor1; // this could be an array if multipul sissors are used

        VkPipelineViewportStateCreateInfo viewportState2{};
        viewportState2.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO; // type of struct
        viewportState2.viewportCount = 1; // // use selected view:  you can use multipul views on some gpus, but make sure when qurrying in the gpu for features that it supports this feature
        viewportState2.pViewports = &viewport[1]; // use selected view port
        viewportState2.scissorCount = 1; // use selected sissor:  you can use multipul sissors on some gpus, but make sure when qurrying in the gpu that it supports this feature
        viewportState2.pScissors = &scissor2; // this could be an array if multipul sissors are used



        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO; //type of struct
        rasterizer.depthClampEnable = VK_FALSE; // clamp fragments that would be thrown away
        rasterizer.rasterizerDiscardEnable = VK_FALSE; // disable rasterizer



        /*
        VK_POLYGON_MODE_FILL: fill the area of the polygon with fragments
        VK_POLYGON_MODE_LINE: polygon edges are drawn as lines
        VK_POLYGON_MODE_POINT: polygon vertices are drawn as points
        */

        rasterizer.polygonMode = VK_POLYGON_MODE_FILL; //fill polygon with fragments (more above ^) --> like everything, other than fill is a gpu feature
        rasterizer.lineWidth = 1.0f; //thickness of line when making image; I assume this can also allow the black border lines in borderland games to happen with little prof impact
        rasterizer.cullMode = VK_CULL_MODE_NONE; //cull the front faces, back faces, or both.. invisible obj vibes in unity  --> none means none are discarded
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; //vertex order, to be front clockwise, or counter, or back
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{}; // AA for sharp pixels --> adds fragment shaders that rasterize in the same pixel
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = FALSE; // https://www.khronos.org/registry/vulkan/specs/1.0/html/vkspec.html#primsrast-sampleshading : disabled or enabled --> false is disabled, this is sample shading to add sample shading to textures, so jagged edges on inside colors are a non important thing
        multisampling.rasterizationSamples = msaaSamples;
        multisampling.minSampleShading = 0.2f; // minimum sample shading
        //multisampling.pSampleMask = nullptr; // Optional
        //multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
        //multisampling.alphaToOneEnable = VK_FALSE; // Optional

        // VkPipelineDepthStencilStateCreateInfo this is nullpts; but it is fairly important for using depth and stencil tests

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE; // get depth of new fragments compared to the depth buffer (see if they are discarded)
        depthStencil.depthWriteEnable = VK_TRUE; // check if new fragments that pass depth test should be written to depth buffer
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS; // comparison technique used to see if the stuff are discarded or not
        depthStencil.depthBoundsTestEnable = VK_FALSE; // allow the bottom 2 optionals if true to act as a "render distance"
        depthStencil.minDepthBounds = 0.0f; // Optional --> write stuff that are only in these depth bounds (render distance)
        depthStencil.maxDepthBounds = 1.0f; // Optional --> write stuff that are only in these depth bounds (render distance)

        depthStencil.stencilTestEnable = VK_FALSE; //stencil buffer operations
        //depthStencil.front{}; // Optional
       // depthStencil.back{}; // Optional

        VkPipelineColorBlendAttachmentState colorBlendAttachment{}; // this struct blends and changes color data as it is sent, so yeah, you need to use this for every frame if changing colors
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT; // include color R,G,B, and Alpha
        colorBlendAttachment.blendEnable = VK_FALSE; // turned off blend
        //colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional these all are blend ing factors and such
        //colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        //colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        //colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        //colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        //colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
        // to add colors we normally add a color with a lower alpha value 


        VkPipelineColorBlendStateCreateInfo colorBlending{}; // like above, but this actually sets blend constants only; these are used for the calculations above
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO; // type of struct
        colorBlending.logicOpEnable = VK_FALSE;  // off
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // 
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;


        //
        //VkPipelineDynamicStateCreateInfo dynamicState{};
        //dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        //dynamicState.dynamicStateCount = 1;
        //dynamicState.pDynamicStates = &dynamicStateVar;
        ////if you want to change pipeline adjustable values during runtime without making a new pipeline you can use VkDynamicState:


        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; // pipeline info struct type
        pipelineLayoutInfo.setLayoutCount = 1; // descriptor array length if array
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout; // descriptors
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = 0; // changed my approach to use something else



        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }



        VkGraphicsPipelineCreateInfo pipelineInfo1{}; // compiles all info into 1 struct
        pipelineInfo1.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo1.stageCount = 2;
        pipelineInfo1.pStages = shaderStages;
        pipelineInfo1.pVertexInputState = &vertexInputInfo;
        pipelineInfo1.pInputAssemblyState = &inputAssembly;
        pipelineInfo1.pViewportState = &viewportState1;
        pipelineInfo1.pRasterizationState = &rasterizer;
        pipelineInfo1.pMultisampleState = &multisampling;
        pipelineInfo1.pDepthStencilState = &depthStencil;
        pipelineInfo1.pColorBlendState = &colorBlending;
        pipelineInfo1.layout = pipelineLayout;
        pipelineInfo1.renderPass = renderPass;
        pipelineInfo1.subpass = 0;
        pipelineInfo1.basePipelineHandle = VK_NULL_HANDLE;

        VkGraphicsPipelineCreateInfo pipelineInfo2{}; // compiles all info into 1 struct
        pipelineInfo2.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo2.stageCount = 2;
        pipelineInfo2.pStages = shaderStages;
        pipelineInfo2.pVertexInputState = &vertexInputInfo;
        pipelineInfo2.pInputAssemblyState = &inputAssembly;
        pipelineInfo2.pViewportState = &viewportState2;
        pipelineInfo2.pRasterizationState = &rasterizer;
        pipelineInfo2.pMultisampleState = &multisampling;
        pipelineInfo2.pDepthStencilState = &depthStencil;
        pipelineInfo2.pColorBlendState = &colorBlending;
        pipelineInfo2.layout = pipelineLayout;
        pipelineInfo2.renderPass = renderPass;
        pipelineInfo2.subpass = 0;
        pipelineInfo2.basePipelineHandle = VK_NULL_HANDLE;


        // seperate pipelines since I felt like I wanted to do this; could have put them all in 1 for a more efficent process... but meh
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo1, nullptr, &graphicsPipeline1) != VK_SUCCESS) { // pipeline 1 for view port 1    -add pipeline cache/second thingy

            throw std::runtime_error("failed to create graphics pipeline!");
        }

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo2, nullptr, &graphicsPipeline2) != VK_SUCCESS) { // pipeline 2 for view port 1
            throw std::runtime_error("failed to create graphics pipeline!");
        }


        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);

    }



    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size()); // resize containor to hold all framebuffers

        for (size_t i = 0; i < swapChainImageViews.size(); i++) { // iterate through all frame buffers and make a 
            std::array<VkImageView, 3> attachments = { // make array of attachments, 3 per for depth and color and swap chain image?
                colorImageView,
                depthImageView,
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass; // state which render pass is going to be used for the rendering spesifications
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size()); // number of attachments to use
            framebufferInfo.pAttachments = attachments.data(); // link attachments array to the attachments of the render pass
            framebufferInfo.width = swapChainExtent.width; // width of frame buffer
            framebufferInfo.height = swapChainExtent.height; // height of frame buffer
            framebufferInfo.layers = 1; // layer count 

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) { // make a frame buffer
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }



    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice); // get queue family

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; // struct type
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(); // value of graphics family; so we can submit buffers of graphics to them
        poolInfo.flags = 0; // Optional
/*
     flags value

     VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: Hint that command buffers are rerecorded with new commands very often (may change memory allocation behavior)

     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: Allow command buffers to be rerecorded individually, without this flag they all have to be reset together
*/



        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) { // make the buffer obj
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createColorResources() {
        VkFormat colorFormat = swapChainImageFormat;

        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
        colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }


    void createDepthResources() {
        VkFormat depthFormat = findDepthFormat(); // get the depth format

        /*
        depth formats:

        VK_FORMAT_D32_SFLOAT: 32-bit float for depth
        VK_FORMAT_D32_SFLOAT_S8_UINT: 32-bit signed float for depth and 8 bit stencil component
        VK_FORMAT_D24_UNORM_S8_UINT: 24-bit float for depth and 8 bit stencil component
        */

        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1); // third param for our function sats we have a depth bit on
    }


    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
        // point of function is to get supported depth format
        for (VkFormat format : candidates) {
            VkFormatProperties props;

            /* properties for VkFormatProperties

            linearTilingFeatures: Use cases that are supported with linear tiling
            optimalTilingFeatures: Use cases that are supported with optimal tiling
            bufferFeatures: Use cases that are supported for buffers

            */

            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props); // look for the depth format supported...

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");

    }

    VkFormat findDepthFormat() {
        return findSupportedFormat(
            { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT; // top 3 are for getting format of deapth...
    }


    void createMiniTextureImage(std::string TexPath, int CurrentMiniImage) { // get and make texture...

        MiniTextureCount += 1;

        stbi_uc* pixels;

        int texWidth, texHeight, texChannels; // var for sizes


        pixels = stbi_load(TexPath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha); // return from image chosen, the sizes, then colors, according to the color type of RGB and alpha

        TexPath = "textures/1.png";


        //  TEXTURE_PATH.replace(1)

          // NOT IMPLEMENTED YET ^ get second texture - I could make this dynamic, but I dont want to right now... plus its fairly simple, just automate the way I added a second texture

        VkDeviceSize imageSize = texWidth * texHeight * 4; // make var the size of image for the RGBA values --> since we use PNG's we need Alpha
        // made it so I only can use 1024x1024 images... This is not dynamic fully
        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1; // get number of mip maps that can be made from dividing by 2 --> the number of times the width or height (MAX means chose the larger) can be divided by 2 evenly. This is the mip map levels we can make
        //^ number of mipmap levels we can make        stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha); // return from image chosen, the sizes, then colors, according to the color type of RGB and alpha


        if (!pixels) { // if no pixels were found, throw a error
            throw std::runtime_error("failed to load texture image!");


            MiniTextureImage.push_back(NULL); // pre make vector
            MiniTextureImageMemory.push_back(NULL); // pre fill stuff

            VkBuffer stagingBuffer; // buffer for visible memory to handdle the texture
            VkDeviceMemory stagingBufferMemory;
            createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory); //make buffer the size of image size, with certain properties

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data); // map memoory in the quantity of pixel values to be added tp data
            memcpy(data, pixels, static_cast<size_t>(imageSize)); // copy pixel values to data
            vkUnmapMemory(device, stagingBufferMemory);

            stbi_image_free(pixels); // empty array of pixels since data is storing it for now

            createImage(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, MiniTextureImage[CurrentMiniImage], MiniTextureImageMemory[CurrentMiniImage]); // we use SRC and DST to say this image may be a source and destination  - since mipmaps may be used or taken from here

            transitionImageLayout(MiniTextureImage[CurrentMiniImage], VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels); // move the image, though we use transfer SRC to DST optimal since vulkan allows us to then move the mipmaps independent of an image
            copyBufferToImage(stagingBuffer, MiniTextureImage[CurrentMiniImage], static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
            //transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps <-- yes this is me using notes someone else made; the point of this is to learn, not be lazy

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);

            generateMipmaps(MiniTextureImage[CurrentMiniImage], VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);

        }
    }

    void createTextureImage() { // get and make texture...


        std::vector <stbi_uc*> pixels{};

        int texWidth, texHeight, texChannels; // var for sizes

        for (int i = 0; i < textureCount; i++) {


            pixels.push_back(stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha)); // return from image chosen, the sizes, then colors, according to the color type of RGB and alpha

            if (i < textureCount) {


                TEXTURE_PATH = "textures/" + std::to_string(i + 2) + ".png";
            }
        }

        TEXTURE_PATH = "textures/1.png";


        //  TEXTURE_PATH.replace(1)

          // NOT IMPLEMENTED YET ^ get second texture - I could make this dynamic, but I dont want to right now... plus its fairly simple, just automate the way I added a second texture

        VkDeviceSize imageSize = texWidth * texHeight * 4; // make var the size of image for the RGBA values --> since we use PNG's we need Alpha
        // made it so I only can use 1024x1024 images... This is not dynamic fully
        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1; // get number of mip maps that can be made from dividing by 2 --> the number of times the width or height (MAX means chose the larger) can be divided by 2 evenly. This is the mip map levels we can make
        //^ number of mipmap levels we can make        stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha); // return from image chosen, the sizes, then colors, according to the color type of RGB and alpha

        for (int i = 0; i < pixels.size(); i++) { // test pixels... 
            if (!pixels[i]) { // if no pixels were found, throw a error
                throw std::runtime_error("failed to load texture image!");
            }

            textureImage.push_back(NULL); // pre make vector
            textureImageMemory.push_back(NULL); // pre fill stuff

            VkBuffer stagingBuffer; // buffer for visible memory to handdle the texture
            VkDeviceMemory stagingBufferMemory;
            createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory); //make buffer the size of image size, with certain properties

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data); // map memoory in the quantity of pixel values to be added tp data
            memcpy(data, pixels[i], static_cast<size_t>(imageSize)); // copy pixel values to data
            vkUnmapMemory(device, stagingBufferMemory);

            stbi_image_free(pixels[i]); // empty array of pixels since data is storing it for now

            createImage(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage[i], textureImageMemory[i]); // we use SRC and DST to say this image may be a source and destination  - since mipmaps may be used or taken from here

            transitionImageLayout(textureImage[i], VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels); // move the image, though we use transfer SRC to DST optimal since vulkan allows us to then move the mipmaps independent of an image
            copyBufferToImage(stagingBuffer, textureImage[i], static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
            //transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps <-- yes this is me using notes someone else made; the point of this is to learn, not be lazy

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);

            generateMipmaps(textureImage[i], VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);

        }
    }

    // THIS IS NORMALLY DONE BEFORE RUNTIME; THIS IS JUST BECAUSE IT IS EXTRA PRACTICE CODE
    void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) { // make mip maps
        // Check if image format supports linear blitting
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties); // get properties of image format to be sure it supports linear filtering
        //

        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) { // make sure properties have tilling features
         /*
         (https://vulkan-tutorial.com/en/Generating_Mipmaps)

         linearTilingFeatures, optimalTilingFeatures and bufferFeatures that each describe how the format
         can be used depending on the way it is used. We create a texture image with the optimal tiling format,
         so we need to check optimalTilingFeatures. Support for the linear filtering feature can be checked with
         the VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT


         */

            throw std::runtime_error("texture image format does not support linear blitting!");
        }

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{}; // since we have many image transitions we will reuse this barrier
        // subresourceRange.miplevel, oldLayout, newLayout, srcAccessMask, and dstAccessMask are changed later, so we can reuse this var as a result
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;


        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;

        for (uint32_t i = 1; i < mipLevels; i++) { // starts iterating through mipmaps at 1 since we say mipmaps have 1 as a minimum - since otherwise you dont have a image
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, nullptr,
                0, nullptr,
                1, &barrier); // wait until stuff was transfered from previous blit command - as the barrier repisents the mipmap looked at -1

            VkImageBlit blit{}; // we are gonna mess with mip map images... --> goal is to take previous image and divide by 2
            blit.srcOffsets[0] = { 0, 0, 0 }; // where the 3d region will be blitted from
            blit.srcOffsets[1] = { mipWidth, mipHeight, 1 }; // where the 3d region will be blitted from; we get the width and height of the mipmap - z is 1 since all 2d iamges are 1
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = { 0, 0, 0 }; //where we will blit to
            blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 }; //where we will blit to. x and y are divided by 2, and z is 1 since a 2d image has a depth of 1 --> if we cannot divide by 2, we make it 1
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;

            vkCmdBlitImage(commandBuffer, // record the blit command  --> we use 1 image for the Source (SRC) image and DST (destination) image since we are making this mipmap from 1 original image
                image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit, // store blit command in command buffer to process mip map when needed
                VK_FILTER_LINEAR); // add a filter of linear to interpolate pixels

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer, // I think this is just for a barrier to make sure the bit transfer is done
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

            if (mipWidth > 1) mipWidth /= 2; // prevents image from being 0 (or less than 1), helps if image is not a square 
            if (mipHeight > 1) mipHeight /= 2; // prevents image from being 0, helps if image is not a square
        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer, // add barrier to final mip level --> we otherwise dont achive this since we work the buffer based on i-1
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        endSingleTimeCommands(commandBuffer);
    }


    VkSampleCountFlagBits getMaxUsableSampleCount() { // get MSAA support of device 
        VkPhysicalDeviceProperties physicalDeviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

        VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts; // get depth buffer sample count and color buffer
        if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
        if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
        if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
        if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
        if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
        if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

        return VK_SAMPLE_COUNT_1_BIT; // return 1 if others are not supported
    }

    void createMiniTextureImageView(int miniImage) { // all mini versions are for 1 time texture creations

        MiniTextureImageView[miniImage] = createImageView(MiniTextureImage[miniImage], VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels); // makes the view for the acctual renderer... 

    }

    void createTextureImageView() {
        for (int i = 0; i < textureImage.size(); i++) {
            textureImageView.push_back(createImageView(textureImage[i], VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels)); // makes the view for the acctual renderer... 
        }
    }

    void createTextureSampler() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO; // sampler info struct type
        samplerInfo.magFilter = VK_FILTER_LINEAR; // how to interpoloate texels
        samplerInfo.minFilter = VK_FILTER_LINEAR; // how to interpoloate texels

        //
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        /*
        vars for above:

        VK_SAMPLER_ADDRESS_MODE_REPEAT: Repeat the texture when going beyond the image dimensions.
        VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: Like repeat, but inverts the coordinates to mirror the image when going beyond the dimensions.
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: Take the color of the edge closest to the coordinate beyond the image dimensions.
        VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: Like clamp to edge, but instead uses the edge opposite to the closest edge.
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: Return a solid color when sampling beyond the dimensions of the image.



        */

        //

        samplerInfo.anisotropyEnable = VK_TRUE; // turn on AA --> yes you can turn this off...
        samplerInfo.maxAnisotropy = 16.0f; // AA16x for 16 samples in an image
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK; // returned color whne sampling is past border
        samplerInfo.unnormalizedCoordinates = VK_FALSE; // use the format [0,1) for textles cordnates if false, use [0,width) and for height other wise, allows you to break ratio is true
        samplerInfo.compareEnable = VK_FALSE; // use comparison of image for filtering
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = static_cast<float>(mipLevels); // load mip maps max lod to be equal to the mip map files we can make
        samplerInfo.mipLodBias = 0.0f;


        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) { // make the sampler for better image sharpness
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; // view info struct
        viewInfo.image = image; // image controlled
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; // type of view
        viewInfo.format = format; // format of image
        viewInfo.subresourceRange.aspectMask = aspectFlags; // colors and how to control?
        viewInfo.subresourceRange.baseMipLevel = 0; // rest I already talk about
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0; // array level to start at
        viewInfo.subresourceRange.layerCount = 1; // layers number...
        //viewInfo.components   // this is defaulted at  VK_COMPONENT_SWIZZLE_IDENTITY for 0

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) { // test if view of image can be made and make it
            throw std::runtime_error("failed to create texture image view!");
        }

        return imageView;
    }


    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO; // image info struct
        imageInfo.imageType = VK_IMAGE_TYPE_2D; // 2d image...
        imageInfo.extent.width = width; // width of image
        imageInfo.extent.height = height; // height of image
        imageInfo.extent.depth = 1; // depth of 1 or 0, I chose 1 since its not in a mipmap layer
        imageInfo.mipLevels = mipLevels; // mip map layers are equal to the mipmaps that can be made from dividing image by 2 (log2)
        imageInfo.arrayLayers = 1; // 1 image, not a layered texture
        imageInfo.format = format; // format of image --> tells vulkan the pixel layout
        imageInfo.tiling = tiling; // tilling/offset
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        /*

        VK_IMAGE_LAYOUT_UNDEFINED: Not usable by the GPU and the very first transition will discard the texels.
        VK_IMAGE_LAYOUT_PREINITIALIZED: Not usable by the GPU, but the first transition will preserve the texels.

        */
        imageInfo.usage = usage; // VK_IMAGE_USAGE_SAMPLED_BIT means it can be used as a texture, Transfer_DST_BIT means it is just a transfer destination
        imageInfo.samples = numSamples; // multisamlping, 1 sample
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements); // size of ram needed for image

        VkMemoryAllocateInfo allocInfo{}; // requirments and info 
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size; //size of ram needed
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties); // memory type that should be used is found

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) { // allocate memory for the image
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0); // bind the objects together
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{}; // protect memory from being messed with
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; // the struct to protect ram
        barrier.oldLayout = oldLayout;// layout of image, undefined means you dont care
        barrier.newLayout = newLayout; // give it properties it needs
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; // ignore family --> use this if you want to transfer ownership of queue familys
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image; // associated image
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels; // mipmap levels to render lower quality image
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        /*

        Undefined -> transfer destination: transfer writes that don't need to wait on anything
        Transfer destination -> shader reading: shader reads should wait on transfer writes, specifically the shader reads in the fragment shader, because that's where we're going to use the texture

        the following handdles these transitions of an image

        */

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0; // make the source access mask not exist since it has no inheritance from it having no mipmaps to follow
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; //allow the transfer of layouts from the destination access mask

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; // refers to pre barrier operations
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT; // this is not a pipeline state; its a stage of transfering something - its a psudo stage that we can refrence
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; // allow writing to this layout of the texture var from the source
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; // read from the var rather than write to inherit

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT; // this is not a pipeline state; its a stage of transfering something - its a psudo stage that we can refrence
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; // to deal with shaders
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier( // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPipelineBarrier.html
            commandBuffer, // sync the command buffer pipeline stage
            sourceStage, destinationStage, // 
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier // 
        );

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{}; // copy struct info; only really copy color... and width/height with 0 depth
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1; // 1 layer to the image
        region.imageOffset = { 0, 0, 0 }; // offset pixel values, really self explanitory
        region.imageExtent = {
            width,
            height,
            1
        };

        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region); // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdCopyBufferToImage.html
        //command buffer to record to, then the buffer to use, then the image to use, then the use case to transfer bits, then the number of regieons to copy (I have 1 layer I need to copy and therefore 1 region?), region var refrence to use as info struct for copying style and modifications to do while copying

        endSingleTimeCommands(commandBuffer);
    }

    void loadModel() {

        size_t positionOfNum;

        for (int loop = 0; loop <= modelCount; loop++) {


            std::vector <uint32_t> tempIndice; // to add to the main vector of indices
            std::vector <Vertex> tempVertices; // to add to the main vector of indices

            tinyobj::attrib_t attrib;// clear these values each read by reinitializing because I don't know if free and such works/how they work for these - rather do this for now since dead memory is a non-issue if it happens to exist 
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string warn, err;


            if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
                throw std::runtime_error(warn + err);
            }

            if (loop <= modelCount - 2) {
                MODEL_PATH = "model\\" + std::to_string(loop + 2) + ".obj";
            }

            std::unordered_map<Vertex, uint32_t> uniqueVertices{};




            for (const auto& shape : shapes) { // combine all faces into 1 model --> they are normally "seperate"
                for (const auto& index : shape.mesh.indices) { //iterate through each indice in mesh



                    Vertex vertex{};

                    vertex.pos = {
                        attrib.vertices[3 * index.vertex_index + 0] + 1 * loop, //since these are floats I multiply by 3?
                        attrib.vertices[3 * index.vertex_index + 1] ,
                        attrib.vertices[3 * index.vertex_index + 2] + 1 * loop // move obj pos
                    };

                    vertex.texCoord = { // flip image vertically to fix model texture
                        attrib.texcoords[2 * index.texcoord_index + 0], // this vector does not work if I do not set texcoods when making the obj
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                    };

                    vertex.color = { 1.0f, 1.0f, 1.0f };


                    if (uniqueVertices.count(vertex) == 0) { // count number of times a value appears in verticies array to make sure that it does not appear twice in the end result
                        uniqueVertices[vertex] = static_cast<uint32_t>(tempVertices.size());
                        tempVertices.push_back(vertex); //dump all vertex's in a vector for verticies
                    }

                    tempIndice.push_back(uniqueVertices[vertex]); //add to unique vertex unordered map
                }





                indices.push_back(tempIndice);
                vertices.push_back(tempVertices);
            }
        }
    }

    void createVertexBuffer() {
        for (int i = 0; i < vertices.size(); i++) {

            vertexBuffer.push_back(NULL);
            vertexBufferMemory.push_back(NULL);

            VkDeviceSize bufferSize = sizeof(vertices[i][0]) * vertices[i].size();

            VkBuffer stagingBuffer; //buffer for mapped vertex data
            VkDeviceMemory stagingBufferMemory; // memory buffer for memory allocation
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);//make buffer using a user made function
            /*
            meaning of flags:
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT: Buffer can be used as source in a memory transfer operation.
            VK_BUFFER_USAGE_TRANSFER_DST_BIT: Buffer can be used as destination in a memory transfer operation.

            */


            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data); // maps access a region of memory specified by an offset and size after delcaring device and memory looked at(offset is 0, size is bufferinfo.size). fourth var is flags (not using any, so made 0), final is were we dump the output data 
            memcpy(data, vertices[i].data(), (size_t)bufferSize); // map data to vertice.data, with size of bufferSize
            vkUnmapMemory(device, stagingBufferMemory); // unmap device and vertex buffer memory - waisted VRAM/RAM otherwise

        // ways to map data (used way 1):
        // 1. Use a memory heap that is host coherent, indicated with VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        // 2.  Call vkFlushMappedMemoryRanges after writing to the mapped memory, and call vkInvalidateMappedMemoryRanges before reading from the mapped memory


            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer[i], vertexBufferMemory[i]); //make the buffer again using vertex buffer values versus buffer memory


            copyBuffer(stagingBuffer, vertexBuffer[i], bufferSize); //user made function to copy buffer in a valid way

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }
    }

    // same as vertex buffer, but uses indices of vertex for refrence rather than vertex's

    void createIndexBuffer() { // make buffer for index of shape verticies --> look at create vertex buffer function for better explinations
        for (int i = 0; i < indices.size(); i++) {

            indexBuffer.push_back(NULL);
            indexBufferMemory.push_back(NULL);

            VkDeviceSize bufferSize = sizeof(indices[i][i]) * indices[i].size(); // make a buffer size for the index buffer
            // the size of the buffer is the number of indices multiplied by the var type (prevent out of bounds memory errors)

            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;

            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, indices[i].data(), (size_t)bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer[i], indexBufferMemory[i]); // only other diffrence is the use of index_Buffer_bit rather than vertex_buffer_bit; this flag says we use a index buffer -- I think this is more for drivers than anything? 

            copyBuffer(stagingBuffer, indexBuffer[i], bufferSize);


            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);


        }
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(swapChainImages.size()); //resize to size of image number --> array to handle images supplied
        uniformBuffersMemory.resize(swapChainImages.size()); // resize to swap chain count --> array to handle images supplied

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
        }
    }




    void createDescriptorPool() {

        std::array<VkDescriptorPoolSize, textureCount * 2> poolSizes{}; // pool for frame and a discriptor - 2 of em

        for (int i = 0; i < (textureCount) * 2; i += 2) {
            poolSizes[i].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // uniform buffer descriptor var
            poolSizes[i].descriptorCount = static_cast<uint32_t>(swapChainImages.size()); // number of descriptors is images in swap chain
             // struct array part 2 
            poolSizes[i + 1].descriptorCount = static_cast<uint32_t>(textureCount); // seocond this is used but for the second poolSize this time
            poolSizes[i + 1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // sampler and fragment shader are working together

        }


        VkDescriptorPoolCreateInfo poolInfo{}; // info of pool 
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; // info of pool type
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size()); // number of pPoolSizes
        poolInfo.pPoolSizes = poolSizes.data(); // poolsize var
        poolInfo.maxSets = static_cast<uint32_t>(textureCount); // max sets of discriptors (one for each discriptor
        poolInfo.flags = 0; // can be VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT too allow changing it during runtime


        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) { // make descriptor pool
            throw std::runtime_error("failed to create descriptor pool!");
        }


    }


    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout); // make discriptor vector size of swap chain images count, and using descriptor set layout as a base layout

        VkDescriptorSetAllocateInfo allocInfo{}; // add info to the descriptors
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; // type of struct for descriptors when adding info to them
        allocInfo.descriptorPool = descriptorPool; //descriptor pool made in CreateDescriptorPool
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size()); // 
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(textureCount); // origran expects vector size of descriptorSets equal to swap chain image count
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) { // make descriptors, with allocInfo, and device, putting it inside the member "data" of DescriptorSets
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < swapChainImages.size(); i++) { // for every swap chain image
            VkDescriptorBufferInfo bufferInfo{}; // make a buffer
            bufferInfo.buffer = uniformBuffers[i]; // use the uniform buffer corrisponding with the swap chain image
            bufferInfo.offset = 0; // no offset
            bufferInfo.range = sizeof(UniformBufferObject); // descriptor updates bytes (we want to update the amount of a Uniform Buffer Object... 


            VkDescriptorImageInfo imageInfo{}; // image info var
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // read only
            imageInfo.imageView = textureImageView[i]; // image view setup
            imageInfo.sampler = textureSampler; // sampler for image
             // other members for image info struct can go here

            std::array<VkWriteDescriptorSet, 2> descriptorWrites{}; // configuration of descriptors need this struct
            // descriptor for views made previously
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;  // using image info rather than buffer info since all image related vars are inside the imageInfo struct



          //  descriptorWrite.pImageInfo = nullptr; // Optional - mess with image data
          //  descriptorWrite.pTexelBufferView = nullptr; // Optional - buffer to mess with images...

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr); // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkUpdateDescriptorSets.html
        }
    }

    void swapTextureImage(int ImageToReplace, int MiniToReplaceWith) { // not used yet

        VkImageView tempMini = MiniTextureImageView[MiniToReplaceWith];

        VkImageView tempReg = textureImageView[ImageToReplace];

        memcpy(textureImageView[ImageToReplace], tempMini, sizeof(VkImageView)); // copy memory so I don't need to keep track of tempMini image   --> move images used

        memcpy(MiniTextureImageView[MiniToReplaceWith], tempReg, sizeof(VkImageView)); // copy memory so I don't need to keep track of tempRegular image --> move image mini over to regular views

    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) { // we can make many diffrent buffers with this function parameter

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;// buffer struct
        bufferInfo.size = size;// number of vertexe's multiplied by size of a vertex (bytes)
        bufferInfo.usage = usage; // purpose of data use
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;// controlled and used by spesific buffer

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements; // struct of mem requirments
/*   members:

    size: The size of the required amount of memory in bytes, may differ from bufferInfo.size.
    alignment: The offset in bytes where the buffer begins in the allocated region of memory, depends on bufferInfo.usage and bufferInfo.flags.
    memoryTypeBits: Bit field of the memory types that are suitable for the buffer.
*/
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);  // auto calculate memory needed, and put inside memRequirments struct

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties); // memory types we check for in our GPU

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) { // allocate memory needed using allocInfo stuff inside vertex buffer memory ---------->   NOT SUPPOSED TO ALLOCATE EVERY FRAME, use a allocator or something like: "https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator" for a open source allocator 
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0); // bind memory with device, vertex buffer (memory binded is second last) --> the fourth perameter is the offset
    }



    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{}; // temp command buffer to copy stuff that should not be copyable
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;//all of this stuff is explained in a diffrent part of the document
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer; // the temp command buffer var
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer); // allocate to the buffer

        VkCommandBufferBeginInfo beginInfo{};  // telling driver how we use the buffer to be efficent
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo); // start the command buffer

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{}; // to submit the info we have a info struct with command buffer(s) [we have 1 only]--> we have no semaphores or fences since we want this data to process —As soon As Possible—
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1; // we have 1 command buffer...
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE); // submit command buffer without a fence or semaphore since it does not matter if we do it right away --> if you needed many simmilar completions a fence could be used to sync all operations to improve efficency of the program

        vkQueueWaitIdle(graphicsQueue);  // wait for the graphics queue

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);  // clean up the 1 time buffer
    }





    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }




    void createCommandBuffers() {

        commandBuffers.resize(swapChainFramebuffers.size()); // resize command buffers array to number of frame buffers because we should have graphics family queue filled with same number

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; // allocation info struct
        allocInfo.commandPool = commandPool; // fill command pool var to give settings and values to command buffer to use
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        /*
        .level values:

        VK_COMMAND_BUFFER_LEVEL_PRIMARY: Can be submitted to a queue for execution, but cannot be called from other command buffers.
        VK_COMMAND_BUFFER_LEVEL_SECONDARY: Cannot be submitted directly, but can be called from primary command buffers.

        */


        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) { // fill command buffers
            throw std::runtime_error("failed to allocate command buffers!");
        }

        for (size_t i = 0; i < commandBuffers.size(); i++) { // for all command buffers do the following:
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; // command buffer info struct
            beginInfo.flags = 0; // default is 
            beginInfo.pInheritanceInfo = nullptr; // Optional // for secondary command buffers, it says which state to inherit from the primary
            /*
            The flags parameter specifies how we're going to use the command buffer. The following values are available:

                VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: The command buffer will be rerecorded right after executing it once.
                VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT : This is a secondary command buffer that will be entirely within a single render pass.
                VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : The command buffer can be resubmitted while it is also already pending execution.
                */

            if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) { // reset the buffer --> // I can use my loops to sync or un sync textures (also need to change bind descriptor index  ;)
                throw std::runtime_error("failed to begin recording command buffer!");
            }


            VkRenderPassBeginInfo renderPassInfo{}; // make render pass info to render
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO; // type of struct
            renderPassInfo.renderPass = renderPass; // render pass its self
            renderPassInfo.framebuffer = swapChainFramebuffers[i]; //attachment/buffer
            renderPassInfo.renderArea.offset = { 0, 0 };  // to push render area size x or y values
            renderPassInfo.renderArea.extent = swapChainExtent; //contributes to size of render area <-- I really should make 2 of these, one for each view... but it really is not important currently



            std::array<VkClearValue, 2> clearValues{}; // clear vales are made
            clearValues[0].color = { 0.0f, 1.0f, 0.0f, 1.0f }; // max alpha when nothing is present -> its how to get that RGB pain background





            clearValues[1].depthStencil = { 1.0f, 0 }; // range of depth for 1 is far view plane and close is 0

            renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());  // add to renderpassinfo the clear values number
            renderPassInfo.pClearValues = clearValues.data(); // add what are clear values




                                                                                               /*

            VK_SUBPASS_CONTENTS_INLINE: The render pass commands will be embedded in the primary command buffer itself and no secondary command buffers will be executed.
            VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS: The render pass commands will be executed from secondary command buffers.

            */


            std::vector <VkBuffer> vertexBuffers; // vertex buffer array preallocated with vertexBuffer made already   -  vertexBuffer[0] 

            VkDeviceSize offsets[] = { 0 };

            for (int i = 0; i < vertexBuffer.size(); i++) {

                vertexBuffers.push_back(vertexBuffer[i]);

            }


            vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); //reset render pass related to command buffers with renderpass info now - imbed inside primary buffer


            for (int y = 0; y < vertexBuffer.size(); y++) {

                // tried each of these: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkShaderStageFlagBits.html
                vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline1); // second parameter says the pipeline is a graphics or compute pipeline; we said it is a graphics pipeline

                vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, &vertexBuffers[y], offsets); // bind to command buffer

                vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer[y], 0, VK_INDEX_TYPE_UINT32); // if any atribute is diffrent, we need to rebind data to the command buffer--> for the final flag, UINT32 can be used if we need it due to vertex count --> almost always need 32^32
                    //^^ --> we bind this by: commandBuffer being added too, index buffer being looked at, offset, type of parameters 

                // THIS ONE JUST MAKES 2 IDENTICAL but diffrent textured VIEWS - I have a funny idea that I may show someone... want this here    vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[0], 0, nullptr); // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindDescriptorSets.html

                //offset moves texture...
                vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[std::clamp(y - 2, 0, textureCount - 1)], 0, nullptr); // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindDescriptorSets.html

                vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices[y].size()), 1, 0, 0, 0); //draw the index buffer, using: command buffer, number of indices, number of instances, first index, vertex offset, and first instance (if more than 1 exists, otherwise make it 0)



            }
            vkCmdEndRenderPass(commandBuffers[i]); // end render pass recording to the command buffer

            // add             vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline1); // second parameter says the pipeline is a graphics or compute pipeline; we said it is a graphics pipeline
 // but I need to redo the pipeline, raise offset and reduce extent

            clearValues[0].color = { 0.0f, 0.0f, 1.0f, 1.0f }; // max alpha when nothing is present -> its how to get that RGB pain background
            renderPassInfo.renderArea.offset = { int(swapChainExtent.width / 2), int(0) };  // to push render area size x or y values
            renderPassInfo.renderArea.extent = swapChainExtent; //contrivutes to size of render area

            vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); //reset render pass related to command buffers with renderpass info now - imbed inside primary buffer

            for (int y = 0; y < vertexBuffer.size(); y++) {


                vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline2); // second parameter says the pipeline is a graphics or compute pipeline; we said it is a graphics pipeline

                vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, &vertexBuffers[y], offsets); // bind to command buffer

                vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer[y], 0, VK_INDEX_TYPE_UINT32); // if any atribute is diffrent, we need to rebind data to the command buffer--> for the final flag, UINT32 can be used if we need it due to vertex count --> almost always need 32^32
                //^^ --> we bind this by: commandBuffer being added too, index buffer being looked at, offset, type of parameters 

               // THIS ONE JUST MAKES 2 IDENTICAL but diffrent textured VIEWS - I have a funny idea that I may show someone... want this here vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[1], 0, nullptr); // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindDescriptorSets.html

                vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[std::clamp(y - 1, 0, textureCount - 1)], 0, nullptr); // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindDescriptorSets.html

                vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices[y].size()), 1, 0, 0, 0); //draw the index buffer, using: command buffer, number of indices, number of instances, first index, vertex offset, and first instance (if more than 1 exists, otherwise make it 0)
            }

            vkCmdEndRenderPass(commandBuffers[i]); // end render pass recording to the command buffer

            if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) { // // huzah, check if the render pass failed...z
                throw std::runtime_error("failed to record command buffer!");
            }

        }
    }


    //   void moveObjects(std::vector <VkBuffer> vertices) {

     //  }


    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT); // like this we will always have enough space for semaphores to render the correct image at the right time, same for below; otherwise we will be out of sync for rendering frames
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO; // semaphone/signal thingy struct

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; // fence struct type for cpu/gpu sync
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // fence starts as a signaled bit, so then you dont have a stuck fence

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) { //make signals to wait for signals to pass 
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }


    //int projectionVarFUN; // im'a put this here for ctrl f surfing

    void updateUniformBuffer(uint32_t currentImage) { // change model stuff here and views here...  --> each image handles model

        if (XYspeed1.x != NULL) {
            XYspeed2 = XYspeed1;
        }
        GetCursorPos(&XYspeed1);

        //spin
        int netspeedx = XYspeed1.x - XYspeed2.x;
        int netspeedy = XYspeed1.y - XYspeed2.y;

        radianPOSx += netspeedx;

        radianPOSy += netspeedy;
        //spin



        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();


        UniformBufferObject ubo{};

        //ubo.model = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f+ radianPOSy), glm::vec3(0.0f, 0.0f, 1.0f)); //model power to rotate, angle to rotate, angle to rotate (x,y,z) rotate 
        //^ mouse controls model movment
        ubo.model = glm::rotate(glm::mat4(1.0f), glm::radians(1.0f * 5 + time * currentImage * time * 20), glm::vec3(0.0f, 0.0f, 1.0f)); //model power to rotate, angle to rotate, angle to rotate (x,y,z) rotate 
        ubo.model *= glm::translate(glm::mat4(1.0), glm::vec3(1, 1, 1 * sin(time * 7.5)));
        ubo.model *= glm::scale(glm::mat4(1.0), glm::vec3(1, 1 * sin(time * 7.5), 1));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(75.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
        ubo.proj[1][1] *= projectionVarFUN;

        //  projectionVarFUN += (projectionVarFUN > -0.8) ? -0.001 : 0.1; // bouncy bois'



        void* data;

        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);


    }



    void drawFrame() {

        /*
        1. get image from swap chain
        2. excute command buffer with the image as attachemnt in frame buffer
        3. return image to swap chain for showing off
        */
        timeStart = std::chrono::system_clock::now();

        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);  // wait for 1 or more fences to be signaled to force the cpu/gpu to be in sync with one another

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex); // get next image from swap chain; parameters: logical device, swapchain to get image,  64bit int removes timeout, next 2 are syncorinisation object to control it, net is image index of Vk image, also returns:
        //suboptimal or out of date vk render pass;  

        if (result == VK_ERROR_OUT_OF_DATE_KHR) { // out of date means we need a new swap chain
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) { // 
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        updateUniformBuffer(imageIndex); // update buffers

        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) { // make sure we are assigning new frames
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }

        imagesInFlight[imageIndex] = inFlightFences[currentFrame];



        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] }; // which semaphore to wait for before excution --> and which stage on the pipeline
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; // state waiting on
        submitInfo.waitSemaphoreCount = 1; // can be more if more semaphores are needed
        submitInfo.pWaitSemaphores = waitSemaphores; // the wait semaphore to signal when excution is done basics
        submitInfo.pWaitDstStageMask = waitStages; // the stage wating on

        submitInfo.commandBufferCount = 1; // 1 command buffer
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex]; // image result from the command buffer

        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] }; // which semaphore says execution is done
        submitInfo.signalSemaphoreCount = 1; // number of em
        submitInfo.pSignalSemaphores = signalSemaphores; // which semaphore to signal once the command buffer is done

        vkResetFences(device, 1, &inFlightFences[currentFrame]); // reset fence value...

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) { // if the fence is done for the current frame, the GPU and CPU are in sync, meaning you can continue
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1; // how many wait signals/semaphores
        presentInfo.pWaitSemaphores = signalSemaphores; // wait for this until info is presented

        VkSwapchainKHR swapChains[] = { swapChain };

        presentInfo.swapchainCount = 1; // number of swap chains

        presentInfo.pSwapchains = swapChains; // the chain array/var

        presentInfo.pImageIndices = &imageIndex; // index of image to read

        presentInfo.pResults = nullptr; // Optional // see if results of sawp worked; we use 1 chain so we dont need this

        vkQueuePresentKHR(presentQueue, &presentInfo); // presents an image to the swap chain

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) { // if out of date render, or not optimal, or frame buffer was resized from window changing size, remake sawp chain
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS) { // if error is none of the above we now have less possibilitys for the swap image
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT; // move 1 frame forward; the modulos means that the frame index loops, preventing unexpected visuals

        timeEnd = std::chrono::system_clock::now();

        score = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();

        FPScount = (1 / 1e-6) / score; // 1*by decimal places of micro second divided by passed microseconds)

        std::cout << "to pause click the console, to continue click 'enter' after clicking the console \n";
        std::cout << "time to elapse (microseconds) is: " << score << "\n";
        std::cout << "estimated FPS: " << FPScount << "\n\n";
    }




    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{}; // made new shader struct to store data 
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); // byte code stored in here from what was read

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) { // make shader module...
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }









    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) { //used to set formats needed
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { //  check if we have [certain things needed for program]: VK_FORMAT_B8G8R8A8_SRGB means that we store the B, G, R and alpha channels in that order with an 8 bit unsigned integer for a total of 32 bits per pixel. The colorSpace member indicates if the SRGB color space is supported or not using the VK_COLOR_SPACE_SRGB_NONLINEAR_KHR flag ; gotten from the vulkan-tutorial.com website
                return availableFormat; // if above is true we return formats to recognize all formats avaible is good
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        /*
VK_PRESENT_MODE_IMMEDIATE_KHR: Images submitted by your application are transferred to the screen right away, which may result in tearing.
VK_PRESENT_MODE_FIFO_KHR: The swap chain is a queue where the display takes an image from the front of the queue when the display is refreshed and the program inserts rendered images at the back of the queue. If the queue is full then the program has to wait. This is most similar to vertical sync as found in modern games. The moment that the display is refreshed is known as "vertical blank".
VK_PRESENT_MODE_FIFO_RELAXED_KHR: This mode only differs from the previous one if the application is late and the queue was empty at the last vertical blank. Instead of waiting for the next vertical blank, the image is transferred right away when it finally arrives. This may result in visible tearing.
VK_PRESENT_MODE_MAILBOX_KHR: This is another variation of the second mode. Instead of blocking the application when the queue is full, the images that are already queued are simply replaced with the newer ones. This mode can be used to implement triple buffering, which allows you to avoid tearing with significantly less latency issues than standard vertical sync that uses double buffering.
        from https://vulkan-tutorial.com/en/Drawing_a_triangle/Presentation/Swap_chain
        (like most of the code)
        */


        for (const auto& availablePresentMode : availablePresentModes) { //for everything inside availablePresentModes do the following
         //   if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) { // check for mode support for just forcing hardware to run without vsync
         //       return availablePresentMode;
         //   }
        }

        return VK_PRESENT_MODE_FIFO_KHR; // this mode is always avaible, so if push comes to shove just use this
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

            return actualExtent;
        }
    }


    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities); // get basic capabilitys of device and return in details.capabilities

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr); // get number of format surface capabilitys

        if (formatCount != 0) {
            details.formats.resize(formatCount); //resize vector of format surface capabilities to size of format's avaible
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data()); //// get format capabilitys of device
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);  // get number of present mode capabilitys

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data()); //// get present capabilitys of device
        }

        return details; // return capabilities, format, and present mode's
    }











    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device); // return true or false based on extensions avaible in device

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures); // gotta get some features

        return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy; // return true or false if extensions are supported, AA is not supported by all gpu's
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr); // return number of extensions (to pre allocate in a vector later)

        std::vector<VkExtensionProperties> availableExtensions(extensionCount); // premake a vector to store extension data in
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data()); // get extentions and put in "availableExtensions"

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end()); // check for required extensions from begining of containor to end

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName); // clean memory from the extension searcher
        }

        return requiredExtensions.empty();
    }


    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport); // check if the device supports the queue family in "i"

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());


        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary); // for every binary number inside file name; ate means start reading at the end of file, binary means you are reading binary

        if (!file.is_open()) {  // check if open is false
            throw std::runtime_error("failed to open file!");
        }


        size_t fileSize = (size_t)file.tellg(); // get file size
        std::vector<char> buffer(fileSize); // make buffer the size of file size to store binary

        file.seekg(0); // move the seeker that reads data to first char
        file.read(buffer.data(), fileSize); //read data and add to the char buffer; read a total number of file size

        file.close();

        return buffer;
    }


    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
};



int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
