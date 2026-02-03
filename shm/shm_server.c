#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/memfd.h>
#include <errno.h>

// 共享内存配置
#define MFD_NAME "demo_shm"
#define SHM_SIZE 1024
// 服务端写入的测试数据
#define SERVER_MSG "Server: Hello, Shared Memory! pid=%d, fd=%d"

int main() {
    // 1. 创建 memfd，flags=0 默认行为
    int mfd = syscall(SYS_memfd_create, MFD_NAME, 0);
    if (mfd == -1) {
        perror("memfd_create 失败");
        exit(EXIT_FAILURE);
    }

    // 2. 必须设置共享内存大小（默认0字节无法映射）
    if (ftruncate(mfd, SHM_SIZE) == -1) {
        perror("ftruncate 失败");
        close(mfd);
        exit(EXIT_FAILURE);
    }

    // 3. mmap 映射共享内存，MAP_SHARED 是跨进程共享关键
    char *shm_ptr = mmap(
        NULL,           // 由内核分配映射地址
        SHM_SIZE,       // 映射大小
        PROT_READ | PROT_WRITE,  // 可读可写权限
        MAP_SHARED,     // 共享映射，修改对所有进程可见
        mfd,            // memfd 文件描述符
        0               // 偏移量为0
    );
    if (shm_ptr == MAP_FAILED) {
        perror("mmap 失败");
        close(mfd);
        exit(EXIT_FAILURE);
    }

    // 4. 清空内存并写入数据
    memset(shm_ptr, 0, SHM_SIZE);
    snprintf(shm_ptr, SHM_SIZE, SERVER_MSG, getpid(), mfd);

    // 5. 打印关键信息，供客户端使用
    printf("===== 服务端信息 =====\n");
    printf("进程 PID: %d\n", getpid());
    printf("memfd 数值: %d\n", mfd);
    printf("共享内存路径: /proc/%d/fd/%d\n", getpid(), mfd);
    printf("已写入数据: %s\n", shm_ptr);
    printf("======================\n");
    printf("服务端持续运行中，按 Ctrl+C 退出...\n");

    // 6. 保持进程运行，防止 memfd 被关闭、共享内存释放
    while (1) {
        sleep(3600);
    }

    // 理论上不会执行到此处，用于规范收尾
    munmap(shm_ptr, SHM_SIZE);
    close(mfd);
    return 0;
}
