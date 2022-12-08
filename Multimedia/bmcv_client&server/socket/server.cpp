/**************************************************************************
*Copyright:FZU
*Author: GTJ
*Date:2022-11-22
*Description: Server Program
**************************************************************************/

#include <stdio.h>
#include <string.h>
#include <netinet/in.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <queue>

#define MAXBUFF 1024

std::queue<std::string> g_dataQue;                              //全局队列
std::mutex g_mx;                                                //互斥锁

void writeThread()
{
    /*写线程*/
    FILE *out_put = fopen("recv_data.mp4", "w+");
    sleep(1);                                                  //休眠一秒，确保队列中有数据

    while (true)
    {
        /*从队列中读取数据并存储*/
        if (g_dataQue.size() == 0)
            break;
        g_mx.lock();
        std::string data = g_dataQue.front();
        g_dataQue.pop();
        g_mx.unlock();
        fwrite((void *)data.data(), 1, data.size(), out_put);
    }

    fclose(out_put);
}

int main(int argc, char **argv)
{
    int listenFd, clientFd;
    struct sockaddr_in servaddr;

    if ((listenFd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        /*创建套接字*/
        printf("create socket error\n");
        return -1;
    }

    memset(&servaddr, 0, sizeof(servaddr));                     //初始化结构体
    servaddr.sin_family = AF_INET;                              //设置地址族协议
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);               //设置地址
    servaddr.sin_port = htons(6666);                            //设置默认端口

    if (bind(listenFd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
    {
        /*绑定套接字地址和端口*/
        printf("bind socket error\n");
        return -1;
    }

    if (listen(listenFd, 10) < 0)
    {
        /*开启监听*/
        printf("listen socket error\n");
        return -1;
    }

    struct sockaddr_in client_addr;
    socklen_t size = sizeof(client_addr);

    if ((clientFd = accept(listenFd, (struct sockaddr *)&client_addr, &size)) < 0)
    {
        /*建立连接*/
        printf("accept socket error\n");
        return -1;
    }

    std::thread write_thread(writeThread);
    size_t readLen = 0;

    while (true)
    {
        /*循环读取客户端消息*/
        char buff[MAXBUFF] = {0};
        readLen = read(clientFd, buff, MAXBUFF);

        if (readLen <= 0)
            break;
        std::string data(buff, readLen);
        g_mx.lock();                                        //上锁
        g_dataQue.push(data);
        g_mx.unlock();                                      //解锁
    }

    write_thread.join();
    close(clientFd);
    close(listenFd);
    return 0;
}
