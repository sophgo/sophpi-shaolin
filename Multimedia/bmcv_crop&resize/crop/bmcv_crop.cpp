/*
*	kk 2022/11/2 
*	bmcv_crop.cpp //该文件主要实现了对图片进行crop操作，
*                   将原图片分为小块 
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
    av_log(NULL, AV_LOG_ERROR, "Usage: \n\t%s [input_filename]  ...\n", program_name);
    av_log(NULL, AV_LOG_ERROR, "\t%s exampel.jpeg.\n", program_name);
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        usage(argv[0]);
        return -1;
    }
    
    bm_handle_t handle;					  //获取句柄 
    bm_dev_request(&handle, 0);
    int width =  600;					  //定义图片数据 
    int height = 600;
        
    bmcv_rect_t crop_attr;
    crop_attr.start_x = 0;
    crop_attr.start_y = 0;
    crop_attr.crop_w = 600;
    crop_attr.crop_h = 600;

    cv::Mat Input,Out,Test; 				  // opencv读取图片 
    Input = cv::imread(argv[1], 0);	
		
    // 智能指针获取分配内存数据 
    std::unique_ptr<unsigned char[]> src_data(
        new unsigned char[width * height]);
    std::unique_ptr<unsigned char[]> res_data(
	new unsigned char[width * height]);
    memset(src_data.get(), 0x11, width * height);
    memset(res_data.get(), 0, width * height);
    
    // bmcv处理 
    bm_image input, output;
	
    //v创建输入的BMCV IMAGE
    bm_image_create(handle, height, width, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE, &input);

    // bm_image_alloc_dev_mem（input）;                      //利用这个函数也可以申请设备内存
    bm_image_alloc_contiguous_mem(1, &input, 1); 	        // 分配device memory 
	
    unsigned char * input_img_data = src_data.get();
    bm_image_copy_host_to_device(input, (void **)&input_img_data);	
	
    // 创建输出的BMCV IMAGE
    bm_image_create(handle, height, width, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE, &output);
		
    bm_image_alloc_contiguous_mem(1, &output, 1);	
		
    cv::bmcv::toBMI(Input,&input);                         //自动进行内存同步
    // cv::bmcv::toBMI(Input,&input,true);                 //与上面等同
    // bm_image_copy_host_to_device                        //也可以使用这个函数
		
    // bmcv图像处理 
    if (BM_SUCCESS != bmcv_image_crop(handle, 1, &crop_attr, input,&output)) {
        std::cout << "bmcv sobel error !!!" << std::endl;
	bm_image_destroy(input);
	bm_image_destroy(output);
	bm_dev_free(handle);
	return -1;
    }
    unsigned char *res_img_data = res_data.get();
    bm_image_copy_device_to_host(output, (void **)&res_img_data);
    // 将输出结果转成mat数据并保存 
    cv::bmcv::toMAT(&output, Out);
    cv::imwrite("out.jpg", Out);
	
    bm_image_free_contiguous_mem(1, &input);
    bm_image_free_contiguous_mem(1, &output);
    bm_image_destroy(input);
    bm_image_destroy(output);
    bm_dev_free(handle);

    return 0; 
}

