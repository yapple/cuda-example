
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
// mmap() 所需头文件
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring> // for memset()

int main() {
  const size_t N = 1024;
  size_t size = N * sizeof(float);
  const char* shared_file_path = "./cuda_shared_mem.dat"; // 关联的磁盘文件路径

  // -------------------------- 步骤 1：创建/打开一个磁盘文件 --------------------------
  int fd = open(
      shared_file_path,
      O_RDWR | O_CREAT | O_TRUNC, // 读写模式 | 不存在则创建 | 存在则清空内容
      S_IRUSR | S_IWUSR // 文件权限：所有者可读、可写
  );
  if (fd == -1) {
      perror("open failed");
      return 1;
  }

  // -------------------------- 步骤 2：扩展文件大小（与映射内存大小一致） --------------------------
  // mmap() 不会自动扩展文件大小，需先通过 ftruncate() 调整文件尺寸
  if (ftruncate(fd, size) == -1) {
      perror("ftruncate failed");
      close(fd);
      return 1;
  }

  // -------------------------- 步骤 3：将文件映射到进程内存空间（替换 malloc()） --------------------------
  float* h_data = (float*)mmap(
      nullptr,                // 让内核自动选择映射内存的起始地址
      size,                   // 映射内存的大小（字节）
      PROT_READ | PROT_WRITE, // 映射内存的权限：可读、可写
      MAP_SHARED,             // 标志：共享映射（修改会同步到文件，其他进程可见）
      fd,                     // 关联的文件描述符
      0                       // 偏移量，从文件开头开始映射
  );

  // 检查 mmap() 是否成功
  if (h_data == MAP_FAILED) {
      perror("mmap failed");
      close(fd);
      return 1;
  }

  // 可选：初始化映射内存（验证写入功能）
  memset(h_data, 0, size); // 全部置 0

  // -------------------------- 原有 CUDA 逻辑不变 --------------------------
  float* d_data = nullptr;
  cudaError_t cuda_err = cudaMalloc(&d_data, size);
  if (cuda_err != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cuda_err) << std::endl;
      munmap(h_data, size);
      close(fd);
      return 1;
  }

  // 注意：d_data 未初始化，拷贝结果为随机值
  cudaError_t err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

  if(err != cudaSuccess) {
      std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
  } else {
      std::cout << "Copy succeeded!" << std::endl;
  }

  // -------------------------- 步骤 4：释放资源（顺序不可乱） --------------------------
  // 1. 解除内存映射
  if (munmap(h_data, size) == -1) {
      perror("munmap failed");
  }
  // 2. 关闭文件描述符
  close(fd);
  // 3. （可选）删除临时文件（若不需要保留共享数据）
  // remove(shared_file_path);

  // 4. 释放 CUDA 设备内存
  cudaFree(d_data);

  return 0;
}
