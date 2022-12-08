/**************************************************************************
*Copyright:FZU
*Author: GTJ
*Date:2022-11-22
*Description: Client Program
**************************************************************************/
#include <iostream>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define MAXBUFF 1024

int main(int argc, char **argv)
{
    int serverFd;
    char buffer[MAXBUFF] = {0};
    struct sockaddr_in servaddr;
    FILE *fq = nullptr;

    if (argc != 4) 
    {
        /*判断输入参数个数是否正确*/
        printf("usage: ./client <file> <ipaddress> <port>\n");
        return -1;
    }

    if ((serverFd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        /*创建套接字并判断是否成功*/
        printf("create socket error\n");
        return -1;
    }

    memset(&servaddr, 0, sizeof(servaddr));                         //初始化结构体
    servaddr.sin_family = AF_INET;                                  //设置地址族
    servaddr.sin_port = htons(atoi(argv[3]));                       //根据输入设置端口

    if (inet_pton(AF_INET, argv[2], &servaddr.sin_addr) <= 0)
    {
        /*将IP转换为网络地址并判断是否成功*/
        printf("inet_pton error for %s\n", argv[2]);
        return -1;
    }

    if (connect(serverFd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
    {
        /*发出连接请求判断是否连接成功*/
        printf("connect error\n");
        return -1;
    }

    if ((fq = fopen(argv[1], "rb")) == NULL)
    {
        /*判断文件是否打开*/
        close(serverFd);
        return -1;
    }

    while (!feof(fq)) 
    {
        /*循环读取文件并发送*/
        size_t readLen = fread(buffer, 1, sizeof(buffer), fq);
        if (readLen != write(serverFd, buffer, readLen))
        {
            printf("write error.\n");
            break;
        }
    }

    close(serverFd);
    fclose(fq);
    return 0;
}
