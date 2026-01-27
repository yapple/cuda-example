#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
#include <cstring>

// 错误检查宏（与发送端一致）
#define CHECK_CUDA_ERROR(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (file: %s, line: %d)\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 从文件读取IPC句柄信息
bool loadIpcHandleFromFile(cudaIpcMemHandle_t& ipcHandle, const char* filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for reading IPC handle\n");
        return false;
    }
    // 读取cudaIpcMemHandle_t
    file.read(reinterpret_cast<char*>(&ipcHandle), sizeof(cudaIpcMemHandle_t));
    file.close();
    return true;
}

int main() {
    // 1. 定义参数
    const int DATA_SIZE = 1024;  // 与发送端保持一致
    int* d_sharedMem = nullptr;  // 共享设备内存指针（接收端通过IPC打开）
    cudaIpcMemHandle_t ipcHandle; // 从文件读取的IPC句柄

    // 2. 初始化CUDA（与发送端使用同一块显卡）
    CHECK_CUDA_ERROR(cudaSetDevice(0));

    // 3. 从文件读取IPC句柄
    if (!loadIpcHandleFromFile(ipcHandle, "ipc_handle.dat")) {
        exit(EXIT_FAILURE);
    }
    printf("Receiver: Loaded IPC handle from file 'ipc_handle.dat'\n");

    // 4. 核心调用：通过IPC句柄打开共享设备内存（cudaIpcOpenMemHandle）
    // 第二个参数：访问权限（cudaIpcMemLazyEnablePeerAccess = 懒加载对等访问，常用）
    CHECK_CUDA_ERROR(cudaIpcOpenMemHandle((void**)&d_sharedMem, ipcHandle, 
                                          cudaIpcMemLazyEnablePeerAccess));
    printf("Receiver: Opened shared device memory at pointer: %p\n", d_sharedMem);

    // 5. 从共享设备内存读取数据到主机内存
    int* h_hostData = new int[DATA_SIZE];
    CHECK_CUDA_ERROR(cudaMemcpy(h_hostData, d_sharedMem, DATA_SIZE * sizeof(int), 
                                cudaMemcpyDeviceToHost));
    printf("Receiver: Read data from shared device memory\n");

    // 6. 验证数据（检查前10个和最后10个数据，确保传输正确）
    bool dataValid = true;
    for (int i = 0; i < 10; ++i) {
        if (h_hostData[i] != i) {
            dataValid = false;
            break;
        }
    }
    for (int i = DATA_SIZE - 10; i < DATA_SIZE; ++i) {
        if (h_hostData[i] != i) {
            dataValid = false;
            break;
        }
    }
    if (dataValid) {
        printf("Receiver: Data verification passed! (前10个和后10个数据正确)\n");
    } else {
        printf("Receiver: Data verification failed!\n");
    }

    // 7. 释放资源（先关闭IPC共享内存，再释放主机内存）
    delete[] h_hostData;
    CHECK_CUDA_ERROR(cudaIpcCloseMemHandle(d_sharedMem));
    printf("Receiver: Closed IPC mem handle and exited\n");

    return 0;
}
