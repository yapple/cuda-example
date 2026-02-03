#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>

#define SHM_SIZE 1024
// 拼接 /proc 路径的缓冲区大小
#define PATH_SIZE 256

int main(int argc, char *argv[]) {
    // 校验命令行参数
    if (argc != 3) {
        fprintf(stderr, "用法: %s <服务端PID> <服务端memfd数值>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // 解析命令行输入的 PID 和 fd
    pid_t server_pid = atoi(argv[1]);
    int server_fd = atoi(argv[2]);
    char proc_path[PATH_SIZE];

    // 拼接 /proc/[pid]/fd/[fdnum] 路径
    snprintf(proc_path, PATH_SIZE, "/proc/%d/fd/%d", server_pid, server_fd);
    printf("尝试打开共享内存路径: %s\n", proc_path);

    // 1. 打开 proc 路径，获取客户端本地的 fd
    int client_mfd = open(proc_path, O_RDWR);
    if (client_mfd == -1) {
        perror("open 失败");
        fprintf(stderr, "失败原因：权限不足/服务端已退出/fd无效\n");
        exit(EXIT_FAILURE);
    }

    // 2. 映射共享内存，与服务端使用相同参数
    char *shm_ptr = mmap(
        NULL,
        SHM_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        client_mfd,
        0
    );
    if (shm_ptr == MAP_FAILED) {
        perror("mmap 失败");
        close(client_mfd);
        exit(EXIT_FAILURE);
    }

    // 3. 读取并打印共享内存数据
    printf("===== 客户端读取结果 =====\n");
    printf("共享内存数据: %s\n", shm_ptr);
    printf("==========================\n");

    // 4. 释放资源
    munmap(shm_ptr, SHM_SIZE);
    close(client_mfd);

    return 0;
}
