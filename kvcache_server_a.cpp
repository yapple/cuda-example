#include "kvcache_ipc_common.h"
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <cstring>

int main() {
    // 步骤 1：初始化CUDA（进程A可绑定任意显卡，此处绑定显卡0）
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    std::cout << "【进程A（中心KVCache）】已初始化 CUDA 设备 0" << std::endl;

    // 步骤 2：分配【托管内存】（关键：支持跨显卡+跨进程共享，用于KVCache）
    KVCacheData* d_kvcache = nullptr;
    size_t kvcache_size = sizeof(KVCacheData);
    // cudaMemAttachGlobal：允许所有进程/显卡附加访问该托管内存
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_kvcache, kvcache_size, cudaMemAttachGlobal));
    std::cout << "【进程A（中心KVCache）】已分配 KVCache 托管内存，地址：" << d_kvcache << std::endl;

    // 步骤 3：初始化KVCache数据（托管内存可直接主机端读写，无需手动拷贝）
    d_kvcache->cache_id = 0x12345678;
    d_kvcache->valid_length = 512;
    // 初始化Key/Value缓存（填充测试数据）
    for (int i = 0; i < 512; ++i) {
        d_kvcache->key_cache[i] = (float)i / 100.0f;
        d_kvcache->value_cache[i] = (float)i * 2.0f;
    }
    // 同步确保数据写入完成（托管内存懒加载，手动同步避免数据不一致）
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "【进程A（中心KVCache）】已初始化 KVCache 数据（cache_id: " << d_kvcache->cache_id << "）" << std::endl;

    // 步骤 4：生成CUDA IPC句柄（核心：给KVCache内存生成跨进程共享标识）
    cudaIpcMemHandle_t kvcache_ipc_handle;
    CHECK_CUDA_ERROR(cudaIpcGetMemHandle(&kvcache_ipc_handle, d_kvcache));
    std::cout << "【进程A（中心KVCache）】已生成 KVCache IPC 句柄" << std::endl;

    // 步骤 5：将IPC句柄写入文件（传递给进程B、C，实际场景可改用套接字/管道）
    std::ofstream handle_file(KVCACHE_IPC_HANDLE_FILE, std::ios::binary | std::ios::out);
    if (!handle_file.is_open()) {
        fprintf(stderr, "【进程A（中心KVCache）】无法打开句柄文件\n");
        exit(EXIT_FAILURE);
    }
    handle_file.write((const char*)&kvcache_ipc_handle, sizeof(cudaIpcMemHandle_t));
    handle_file.close();
    std::cout << "【进程A（中心KVCache）】已将 IPC 句柄写入文件：" << KVCACHE_IPC_HANDLE_FILE << std::endl;

    // 步骤 6：等待进程B、C完成KVCache读写（阻塞等待，按回车释放资源）
    std::cout << "【进程A（中心KVCache）】等待进程B、C处理 KVCache...（按回车释放内存）" << std::endl;
    std::cin.get();

    // 步骤 7：释放KVCache托管内存（需等B、C都关闭IPC句柄后再释放）
    CHECK_CUDA_ERROR(cudaFree(d_kvcache));
    // 清理句柄文件
    remove(KVCACHE_IPC_HANDLE_FILE.c_str());
    std::cout << "【进程A（中心KVCache）】已释放 KVCache 内存并清理句柄文件" << std::endl;

    return 0;
}
