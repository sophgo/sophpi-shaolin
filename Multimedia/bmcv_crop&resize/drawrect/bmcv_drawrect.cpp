/*
*	kk 2022/11/2 
*	bmcv_drawrect.cpp //该文件主要实现了在图片上绘制矩形
*/ 
#include <iostream>
#include <vector>
#include "bmcv_api.h"
#include "common.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


static void usage(char *program_name)
{
    av_log(NULL, AV_LOG_ERROR, "Usage: \n\t%s [input_filename]  [bmcv or opencv] \n", program_name);
    av_log(NULL, AV_LOG_ERROR, "\t%s exampel.jpeg bmcv.\n", program_name);
}


//此部分演示无读取图片操作，只在纯黑图片绘制矩形操作 
int main(int argc, char *argv[]) {

    if (argc != 3) {
        usage(argv[0]);
        return -1;
    }
    //采用BMCV进行处理
    if(0 == strcmp("bmcv",argv[2])) {
        cv::Mat Input,Out;
        Input = cv::imread(argv[1], 1);	
        bm_handle_t handle;
        bm_dev_request(&handle, 0);

        //定义图片数据 
        int image_h = 600;
        int image_w = 600;
        int channel = 3; 
        bm_image src;
        
        std::unique_ptr<unsigned char[]> src_data(
            new unsigned char[image_w * image_h * channel]);
	memset(src_data.get(), 0, channel*image_w*image_h);
        
		//创建bmcv对象 
        bm_image_create(handle, image_h, image_w, FORMAT_RGB_PLANAR,
        DATA_TYPE_EXT_1N_BYTE, &src);
			
        
        bm_image_alloc_contiguous_mem(1, &src, 1);
		
        //unsigned char * input_img_data = src_data.get();
         //bm_image_copy_host_to_device(src, (void **)&input_img_data);
        //绘制矩形数据（起始点和绘制长度） 	
        bmcv_rect_t rect;
        rect.start_x = 100;
        rect.start_y = 100;
        rect.crop_w = 200;
        rect.crop_h = 300;
        
	cv::bmcv::toBMI(Input,&src);
        
	bmcv_image_draw_rectangle(handle, src, 1, &rect, 3, 255, 0, 0);
		
        //unsigned char *res_img_data = src_data.get();
        //bm_image_copy_device_to_host(src, (void **)&res_img_data);
        cv::bmcv::toMAT(&src, Out);
        cv::imwrite("out.jpg", Out);
		
        bm_image_free_contiguous_mem(1, &src);
        bm_image_destroy(src);
        bm_dev_free(handle);

        return 0; 
 
    } 
    
    //采用OPENCV函数处理
    if(0 == strcmp("opencv",argv[2])) {
        Mat src = imread(argv[1], 1);
        if(!src.data) {
            cout << "can't not found image" << endl;
            return -1;
        }
        //绘制矩形
	Rect rect = Rect(200,100,300,300);
	Scalar color = Scalar(255,0,0);
	rectangle(src,rect,color,2,LINE_8);
	cv::imwrite("out.jpg", src);

        return 0;
    }	
    
    usage(argv[0]);
    return -1;
}

