#include <cuda_runtime.h>  // 仅保留这个头文件即可，已包含IPC相关声明
#include <stdio.h>
#include <fstream>
#include <cstring>

// 错误检查宏（保持不变，CUDA编程必备）
#define CHECK_CUDA_ERROR(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (file: %s, line: %d)\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 保存IPC句柄信息到文件（供接收端读取）
bool saveIpcHandleToFile(const cudaIpcMemHandle_t& ipcHandle, const char* filename) {
    std::ofstream file(filename, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for writing IPC handle\n");
        return false;
    }
    // 写入cudaIpcMemHandle_t（本质是固定大小的字节数组）
    file.write(reinterpret_cast<const char*>(&ipcHandle), sizeof(cudaIpcMemHandle_t));
    file.close();
    return true;
}

int main() {
    // 1. 定义参数
    const int DATA_SIZE = 1024;  // 数据大小（1024个int）
    int* d_devMem = nullptr;     // 设备内存指针（发送端分配）
    cudaIpcMemHandle_t ipcHandle; // IPC句柄

    // 2. 初始化CUDA（选择第0块显卡，多卡环境需与接收端一致）
    CHECK_CUDA_ERROR(cudaSetDevice(0));

    // 3. 分配设备内存（必须是cudaMalloc分配的设备内存，不可用页锁定主机内存）
    CHECK_CUDA_ERROR(cudaMalloc(&d_devMem, DATA_SIZE * sizeof(int)));
    printf("Sender: Allocated device memory at pointer: %p\n", d_devMem);

    // 4. 向设备内存写入测试数据
    int* h_hostData = new int[DATA_SIZE];
    for (int i = 0; i < DATA_SIZE; ++i) {
        h_hostData[i] = i; // 测试数据：0,1,2,...,1023
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_devMem, h_hostData, DATA_SIZE * sizeof(int), 
                                cudaMemcpyHostToDevice));
    printf("Sender: Written test data to device memory\n");

    // 5. 核心调用：获取设备内存的IPC句柄（cudaIpcMemGetMemHandle）
    CHECK_CUDA_ERROR(cudaIpcGetMemHandle(&ipcHandle, d_devMem));
    printf("Sender: Got IPC handle for device memory\n");

    // 6. 将IPC句柄保存到文件（传递给接收端）
    if (!saveIpcHandleToFile(ipcHandle, "ipc_handle.dat")) {
        goto clean_up;
    }
    printf("Sender: Saved IPC handle to file 'ipc_handle.dat'\n");

    // 7. 等待接收端处理（防止发送端提前释放内存）
    printf("Sender: Waiting for receiver to process... (Press Enter to exit)\n");
    getchar();

clean_up:
    // 8. 释放资源（接收端先关闭共享内存，发送端再释放cudaMalloc内存）
    delete[] h_hostData;
    CHECK_CUDA_ERROR(cudaFree(d_devMem));
    printf("Sender: Freed device memory and exited\n");

    return 0;
}
