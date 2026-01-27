#include "kvcache_ipc_common.h"
#include <fstream>
#include <iostream>
#include <cstring>

int main() {
    // 步骤 1：初始化CUDA，绑定显卡1（与进程C的显卡区分）
    CHECK_CUDA_ERROR(cudaSetDevice(1));
    std::cout << "\n【进程B（显卡1）】已初始化 CUDA 设备 1" << std::endl;

    // 步骤 2：从文件读取进程A生成的KVCache IPC句柄
    cudaIpcMemHandle_t kvcache_ipc_handle;
    std::ifstream handle_file(KVCACHE_IPC_HANDLE_FILE, std::ios::binary | std::ios::in);
    if (!handle_file.is_open()) {
        fprintf(stderr, "【进程B（显卡1）】无法打开句柄文件（请先启动进程A）\n");
        exit(EXIT_FAILURE);
    }
    handle_file.read((char*)&kvcache_ipc_handle, sizeof(cudaIpcMemHandle_t));
    handle_file.close();
    std::cout << "【进程B（显卡1）】已读取 KVCache IPC 句柄" << std::endl;

    // 步骤 3：打开IPC句柄，获取进程A的KVCache共享内存地址
    KVCacheData* d_kvcache = nullptr;
    unsigned int flags = cudaIpcMemLazyEnablePeerAccess; // 懒加载对等访问，提升性能
    CHECK_CUDA_ERROR(cudaIpcOpenMemHandle((void**)&d_kvcache, kvcache_ipc_handle, flags));
    std::cout << "【进程B（显卡1）】已打开 KVCache 共享内存，地址：" << d_kvcache << std::endl;

    // 步骤 4：读写KVCache数据（验证共享功能）
    // 4.1 读取KVCache数据
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // 等待数据迁移到显卡1
    std::cout << "【进程B（显卡1）】读取 KVCache 信息：" << std::endl;
    std::cout << "  cache_id: " << d_kvcache->cache_id << std::endl;
    std::cout << "  valid_length: " << d_kvcache->valid_length << std::endl;
    std::cout << "  第0个Key: " << d_kvcache->key_cache[0] << "，第0个Value: " << d_kvcache->value_cache[0] << std::endl;

    // 4.2 修改KVCache数据（写入新数据，进程C可读取到该修改）
    d_kvcache->valid_length = 600;
    for (int i = 512; i < 600; ++i) {
        d_kvcache->key_cache[i] = (float)i / 100.0f + 100.0f;
        d_kvcache->value_cache[i] = (float)i * 2.0f + 100.0f;
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "【进程B（显卡1）】已修改 KVCache 数据（更新valid_length为600）" << std::endl;

    // 步骤 5：关闭IPC句柄（使用完成后必须关闭，否则进程A无法释放内存）
    CHECK_CUDA_ERROR(cudaIpcCloseMemHandle(d_kvcache));
    std::cout << "【进程B（显卡1）】已关闭 KVCache IPC 句柄" << std::endl;

    return 0;
}
