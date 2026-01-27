#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        return 1; \
    }
    // 简单的CUDA内核：将内存全部设置为特定值
    __global__ void set_memory(char* ptr, char value, size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            ptr[idx] = value;
        }
    }

int main() {
    cudaError_t err;
    const size_t SIZE = 4096;  // 4KB

    std::cout << "==========================================" << std::endl;
    std::cout << "Testing: cudaHostRegister + cudaHostGetDevicePointer" << std::endl;
    std::cout << "         then try cudaIpcGetMemHandle" << std::endl;
    std::cout << "==========================================" << std::endl;

    // 1. 分配普通主机内存
    std::cout << "\n[1] Allocating host memory with malloc..." << std::endl;
    void* host_mem = malloc(SIZE);
    if (!host_mem) {
        std::cout << "malloc failed!" << std::endl;
        return 1;
    }
    std::cout << "    Host memory address: " << host_mem << std::endl;

    // 初始化一些数据
    memset(host_mem, 0xAA, SIZE);

    // 2. 注册为CUDA可访问内存
    std::cout << "\n[2] Registering host memory with CUDA..." << std::endl;
    err = cudaHostRegister(host_mem, SIZE, cudaHostRegisterDefault);
    CHECK_CUDA(err);
    std::cout << "    cudaHostRegister: SUCCESS" << std::endl;

    // 3. 获取设备指针
    std::cout << "\n[3] Getting device pointer..." << std::endl;
    void* dev_ptr = nullptr;
    err = cudaHostGetDevicePointer(&dev_ptr, host_mem, 0);
    CHECK_CUDA(err);
    std::cout << "    cudaHostGetDevicePointer: SUCCESS" << std::endl;
    std::cout << "    Device pointer: " << dev_ptr << std::endl;

    // 4. 检查指针属性
    std::cout << "\n[4] Checking pointer attributes..." << std::endl;
    cudaPointerAttributes attr;
    err = cudaPointerGetAttributes(&attr, dev_ptr);
    CHECK_CUDA(err);

    std::cout << "    Memory type: ";
    switch (attr.type) {
        case cudaMemoryTypeUnregistered:
            std::cout << "Unregistered"; break;
        case cudaMemoryTypeHost:
            std::cout << "Host"; break;
        case cudaMemoryTypeDevice:
            std::cout << "Device"; break;
        case cudaMemoryTypeManaged:
            std::cout << "Managed"; break;
        default:
            std::cout << "Unknown"; break;
    }
    std::cout << std::endl;

    std::cout << "    Device: " << attr.device << std::endl;
    std::cout << "    Host pointer: " << attr.hostPointer << std::endl;
    std::cout << "    Device pointer: " << attr.devicePointer << std::endl;

    // 5. 尝试获取IPC内存句柄
    std::cout << "\n[5] Trying to get IPC memory handle..." << std::endl;
    cudaIpcMemHandle_t ipc_handle;

    std::cout << "    Attempt 1: Using host pointer (" << host_mem << ")..." << std::endl;
    err = cudaIpcGetMemHandle(&ipc_handle, host_mem);
    if (err == cudaSuccess) {
        std::cout << "    ❌ UNEXPECTED: cudaIpcGetMemHandle with host pointer SUCCEEDED!" << std::endl;
    } else {
        std::cout << "    ✅ Expected: cudaIpcGetMemHandle failed with: "
                  << cudaGetErrorString(err) << " (error code: " << err << ")" << std::endl;
    }

    std::cout << "\n    Attempt 2: Using device pointer (" << dev_ptr << ")..." << std::endl;
    err = cudaIpcGetMemHandle(&ipc_handle, dev_ptr);
    if (err == cudaSuccess) {
        std::cout << "    ❌ UNEXPECTED: cudaIpcGetMemHandle with device pointer SUCCEEDED!" << std::endl;
    } else {
        std::cout << "    ✅ Expected: cudaIpcGetMemHandle failed with: "
                  << cudaGetErrorString(err) << " (error code: " << err << ")" << std::endl;
    }

    // 6. 对比测试：真正的设备内存
    std::cout << "\n[6] Comparison test: Real device memory..." << std::endl;
    void* real_dev_mem = nullptr;
    err = cudaMalloc(&real_dev_mem, SIZE);
    CHECK_CUDA(err);
    std::cout << "    Real device memory allocated at: " << real_dev_mem << std::endl;

    // 检查真实设备内存的属性
    err = cudaPointerGetAttributes(&attr, real_dev_mem);
    CHECK_CUDA(err);

    std::cout << "    Real device memory type: ";
    switch (attr.type) {
        case cudaMemoryTypeDevice:
            std::cout << "Device"; break;
        default:
            std::cout << "Not Device (" << attr.type << ")"; break;
    }
    std::cout << std::endl;

    // 尝试获取真实设备内存的IPC句柄
    std::cout << "\n    Attempt with real device memory..." << std::endl;
    err = cudaIpcGetMemHandle(&ipc_handle, real_dev_mem);
    if (err == cudaSuccess) {
        std::cout << "    ✅ cudaIpcGetMemHandle with real device memory SUCCEEDED!" << std::endl;
    } else {
        std::cout << "    ❌ Unexpected failure: " << cudaGetErrorString(err) << std::endl;
    }

    // 7. 测试：通过设备指针进行内核访问
    std::cout << "\n[7] Testing kernel access through device pointer..." << std::endl;


    // 通过dev_ptr启动内核
    std::cout << "    Launching kernel through dev_ptr..." << std::endl;
    dim3 blocks(8, 1, 1);
    dim3 threads(256, 1, 1);
    set_memory<<<blocks, threads>>>((char*)dev_ptr, 0x55, SIZE);

    err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        std::cout << "    ✅ Kernel executed successfully through dev_ptr" << std::endl;

        // 验证主机内存被修改
        char* host_check = (char*)host_mem;
        std::cout << "    Checking first 4 bytes in host memory: ";
        for (int i = 0; i < 4; i++) {
            printf("%02X ", (unsigned char)host_check[i]);
        }
        std::cout << std::endl;
    } else {
        std::cout << "    ❌ Kernel failed: " << cudaGetErrorString(err) << std::endl;
    }

    // 8. 测试：通过真实设备内存启动内核
    std::cout << "\n[8] Testing kernel access through real device memory..." << std::endl;
    set_memory<<<blocks, threads>>>((char*)real_dev_mem, 0x77, SIZE);
    err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        std::cout << "    ✅ Kernel executed successfully through real device memory" << std::endl;
    }

    // 9. 清理
    std::cout << "\n[9] Cleaning up..." << std::endl;
    cudaHostUnregister(host_mem);
    free(host_mem);
    cudaFree(real_dev_mem);

    std::cout << "\n==========================================" << std::endl;
    std::cout << "CONCLUSION:" << std::endl;
    std::cout << "1. cudaHostGetDevicePointer returns a MAPPED pointer," << std::endl;
    std::cout << "   not true device memory." << std::endl;
    std::cout << "2. cudaIpcGetMemHandle ONLY works with true device" << std::endl;
    std::cout << "   memory allocated by cudaMalloc." << std::endl;
    std::cout << "3. Mapped pointers CAN be used in kernels, but" << std::endl;
    std::cout << "   CANNOT be used with CUDA IPC." << std::endl;
    std::cout << "==========================================" << std::endl;

    return 0;
}
