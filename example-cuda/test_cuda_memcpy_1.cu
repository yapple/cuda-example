#include <cuda_runtime.h>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>

// 共享内存配置（与 demo2 保持一致）
const char* SHARED_FILE_PATH = "./cuda_shared_data.dat";
const size_t N = 1024;  // 数据元素个数
const size_t DATA_SIZE = N * sizeof(float);  // 数据总字节数

int main() {
    // 步骤 1：创建/打开共享文件
    int fd = open(
        SHARED_FILE_PATH,
        O_RDWR | O_CREAT | O_TRUNC,  // 读写 | 不存在则创建 | 存在则清空
        S_IRUSR | S_IWUSR  // 所有者可读可写
    );
    if (fd == -1) {
        perror("demo1: open shared file failed");
        return 1;
    }

    // 步骤 2：扩展文件大小至数据总字节数（mmap 不会自动扩展文件）
    if (ftruncate(fd, DATA_SIZE) == -1) {
        perror("demo1: ftruncate file failed");
        close(fd);
        return 1;
    }

    // 步骤 3：将文件映射到进程内存（共享映射，供 demo2 读取）
    float* shared_mem = (float*)mmap(
        nullptr,
        DATA_SIZE,
        PROT_READ | PROT_WRITE,  // 可读可写
        MAP_SHARED,              // 共享映射：修改同步到文件和其他进程
        fd,
        0
    );
    if (shared_mem == MAP_FAILED) {
        perror("demo1: mmap failed");
        close(fd);
        return 1;
    }

    // 步骤 4：CUDA 锁页映射内存（可选但推荐，提升 cudaMemcpy 效率，避免额外拷贝）
    cudaError_t cuda_err = cudaHostRegister(
        shared_mem,
        DATA_SIZE,
        cudaHostRegisterDefault  // 默认属性：可读写、供 CUDA 高效访问
    );
    if (cuda_err != cudaSuccess) {
        std::cerr << "demo1: cudaHostRegister failed: " << cudaGetErrorString(cuda_err) << std::endl;
        munmap(shared_mem, DATA_SIZE);
        close(fd);
        return 1;
    }

    // 步骤 5：往共享内存写入数据（自定义数据：0, 1, 2, ..., N-1）
    std::cout << "demo1: 开始往共享内存写入数据..." << std::endl;
    for (size_t i = 0; i < N; ++i) {
        shared_mem[i] = 100 + static_cast<float>(i);  // 写入连续浮点值
    }
    std::cout << "demo1: 数据写入完成！前 10 个数据：";
    for (size_t i = 0; i < 10; ++i) {  // 打印前 10 个数据验证
        std::cout << shared_mem[i] << " ";
    }
    std::cout << std::endl;

    // 步骤 6：释放资源（顺序不可乱）
    // 6.1 解除 CUDA 锁页
    cuda_err = cudaHostUnregister(shared_mem);
    if (cuda_err != cudaSuccess) {
        std::cerr << "demo1: cudaHostUnregister failed: " << cudaGetErrorString(cuda_err) << std::endl;
    }

    // 6.2 解除内存映射
    if (munmap(shared_mem, DATA_SIZE) == -1) {
        perror("demo1: munmap failed");
    }

    // 6.3 关闭文件描述符（保留共享文件，供 demo2 读取，不删除）
    close(fd);

    std::cout << "demo1: 操作完成，共享文件已保留，等待 demo2 读取..." << std::endl;
    return 0;
}
