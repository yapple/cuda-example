
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
// mmap() 所需头文件
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

int main() {
  const size_t N = 1024;
  size_t size = N * sizeof(float);

  // -------------------------- 替换 malloc() 为 mmap() 匿名映射 --------------------------
  float* h_data = (float*)mmap(
      nullptr,                // 让内核自动选择映射内存的起始地址
      size,                   // 映射内存的大小（字节）
      PROT_READ | PROT_WRITE, // 映射内存的权限：可读、可写
      MAP_ANONYMOUS | MAP_PRIVATE, // 标志：匿名映射（无关联文件）、私有映射（仅当前进程可见）
      -1,                     // 匿名映射时，文件描述符设为 -1
      0                       // 偏移量，匿名映射时设为 0
  );

  // 检查 mmap() 是否成功
  if (h_data == MAP_FAILED) {
      perror("mmap failed"); // 打印系统错误信息
      return 1;
  }

  // -------------------------- 原有 CUDA 逻辑不变 --------------------------
  // device memory
  float* d_data = nullptr;
  cudaError_t cuda_err = cudaMalloc(&d_data, size);
  if (cuda_err != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cuda_err) << std::endl;
      munmap(h_data, size); // 先释放 mmap 内存，避免泄漏
      return 1;
  }

  // 注意：此处 d_data 未初始化，拷贝出来的 h_data 是随机垃圾值，不影响功能验证
  cudaError_t err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

  if(err != cudaSuccess) {
      std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
  } else {
      std::cout << "Copy succeeded!" << std::endl;
  }

  // -------------------------- 释放资源（替换 free() 为 munmap()） --------------------------
  // 释放 mmap 映射的内存
  if (munmap(h_data, size) == -1) {
      perror("munmap failed");
  }
  // 释放 CUDA 设备内存
  cudaFree(d_data);

  return 0;
}
