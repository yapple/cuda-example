#ifndef KVCACHE_IPC_COMMON_H
#define KVCACHE_IPC_COMMON_H

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

// KVCache 数据结构（简化版，贴合大模型场景）
struct KVCacheData {
    uint64_t cache_id;          // KVCache 唯一标识
    float key_cache[1024];      // Key 缓存（简化为固定大小数组）
    float value_cache[1024];    // Value 缓存（简化为固定大小数组）
    uint32_t valid_length;      // 有效数据长度
};

// 3个进程间传递IPC句柄的文件路径
const std::string KVCACHE_IPC_HANDLE_FILE = "./kvcache_ipc_handle.bin";

// CUDA 错误检查宏
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t _err = (err); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(_err), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#endif // KVCACHE_IPC_COMMON_H
