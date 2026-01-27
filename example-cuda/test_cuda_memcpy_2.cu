#include <cuda_runtime.h>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>

// 共享内存配置（与 demo1 完全一致，必须同步修改）
const char* SHARED_FILE_PATH = "./cuda_shared_data.dat";
const size_t N = 1024;  // 数据元素个数
const size_t DATA_SIZE = N * sizeof(float);  // 数据总字节数

int main() {
    // 步骤 1：打开 demo1 创建的共享文件（仅打开，不创建/清空）
    int fd = open(
        SHARED_FILE_PATH,
        O_RDWR,  // 仅读写（文件已由 demo1 创建）
        S_IRUSR | S_IWUSR
    );
    if (fd == -1) {
        perror("demo2: open shared file failed (请先运行 demo1 创建共享文件)");
        return 1;
    }

    // 步骤 2：将文件映射到进程内存（共享映射，读取 demo1 写入的数据）
    float* shared_mem = (float*)mmap(
        nullptr,
        DATA_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        0
    );
    if (shared_mem == MAP_FAILED) {
        perror("demo2: mmap failed");
        close(fd);
        return 1;
    }
/*
    // 步骤 3：CUDA 锁页映射内存（提升拷贝效率）
    cudaError_t cuda_err = cudaHostRegister(
        shared_mem,
        DATA_SIZE,
        cudaHostRegisterDefault
    );
    if (cuda_err != cudaSuccess) {
        std::cerr << "demo2: cudaHostRegister failed: " << cudaGetErrorString(cuda_err) << std::endl;
        munmap(shared_mem, DATA_SIZE);
        close(fd);
        return 1;
    }
*/
    // 步骤 4：分配 GPU 显存（用于存储从共享内存拷贝的数据）
    float* d_data = nullptr;
    cudaError_t cuda_err = cudaMalloc(&d_data, DATA_SIZE);
    if (cuda_err != cudaSuccess) {
        std::cerr << "demo2: cudaMalloc failed: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaHostUnregister(shared_mem);
        munmap(shared_mem, DATA_SIZE);
        close(fd);
        return 1;
    }
    std::cout << "demo2: GPU 显存分配成功！" << std::endl;

    // 步骤 5：将共享内存数据拷贝到 GPU 显存（Host → Device）
    std::cout << "demo2: 开始将共享内存数据拷贝到 GPU 显存..." << std::endl;
    cuda_err = cudaMemcpy(
        d_data,               // 目标：GPU 显存地址
        shared_mem,           // 源：共享内存地址（已锁页的主机内存）
        DATA_SIZE,
        cudaMemcpyHostToDevice  // 拷贝方向：主机 → 设备
    );
    if (cuda_err != cudaSuccess) {
        std::cerr << "demo2: cudaMemcpy failed: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaFree(d_data);
        cudaHostUnregister(shared_mem);
        munmap(shared_mem, DATA_SIZE);
        close(fd);
        return 1;
    }
    std::cout << "demo2: 数据拷贝到 GPU 显存完成！" << std::endl;

    // 步骤 6：验证显存数据（可选：从显存拷贝回主机内存，打印验证）
    float* h_verify = (float*)malloc(DATA_SIZE);
    if (h_verify == nullptr) {
        perror("demo2: malloc verify buffer failed");
        cudaFree(d_data);
        cudaHostUnregister(shared_mem);
        munmap(shared_mem, DATA_SIZE);
        close(fd);
        return 1;
    }

    // 从显存拷贝回主机验证
    cuda_err = cudaMemcpy(h_verify, d_data, DATA_SIZE, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        std::cerr << "demo2: cudaMemcpy verify failed: " << cudaGetErrorString(cuda_err) << std::endl;
        free(h_verify);
        cudaFree(d_data);
        cudaHostUnregister(shared_mem);
        munmap(shared_mem, DATA_SIZE);
        close(fd);
        return 1;
    }

    // 打印前 10 个数据验证（应与 demo1 写入的数据一致）
    std::cout << "demo2: 从 GPU 显存验证数据（前 10 个）：";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << h_verify[i] << " ";
    }
    std::cout << std::endl;

    // 步骤 7：释放所有资源（顺序不可乱）
    // 7.1 释放验证用主机内存
    free(h_verify);

    // 7.2 释放 GPU 显存
    cudaFree(d_data);
/*
    // 7.3 解除 CUDA 锁页
    cuda_err = cudaHostUnregister(shared_mem);
    if (cuda_err != cudaSuccess) {
        std::cerr << "demo2: cudaHostUnregister failed: " << cudaGetErrorString(cuda_err) << std::endl;
    }
*/
    // 7.4 解除内存映射
    if (munmap(shared_mem, DATA_SIZE) == -1) {
        perror("demo2: munmap failed");
    }

    // 7.5 关闭文件描述符
    close(fd);

    // 7.6 （可选）删除共享文件（验证完成后清理）
    if (remove(SHARED_FILE_PATH) == -1) {
        perror("demo2: remove shared file failed (可选，不影响功能)");
    }

    std::cout << "demo2: 所有操作完成！" << std::endl;
    return 0;
}
