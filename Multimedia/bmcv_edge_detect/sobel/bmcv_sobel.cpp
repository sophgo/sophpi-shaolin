/*
*	kk 2022/11/2 
*	bmcv_sobel.cpp //该文件主要实现了对图片的进行sobel边缘检测，
*                        并将边缘检测结果输出 
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


int main(int argc, char *argv[]) {
  
    if (argc != 3) {
        usage(argv[0]);
        return -1;
    }
    //采用BMCV进行处理
    if(0 == strcmp("bmcv", argv[2])) {
        bm_handle_t handle;					  //获取句柄 
        bm_dev_request(&handle, 0);
        int width =  600;					  //定义图片数据 
        int height = 600;
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
	if (BM_SUCCESS != bmcv_image_sobel(handle, input, output, 0, 1)) {
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
    //采用OPENCV进行处理
    if(0 == strcmp("opencv", argv[2])) {
	Mat src1,src,dst;
	src1 = cv::imread(argv[1], 1);
	resize(src1, src, Size(src1.cols/2, src1.rows/2));
	if (!src.data) {
	    printf("cannot load image ...");
	    return -1;
	} 

	GaussianBlur(src, dst, Size(3,3), 0);
	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	    
        //sobel算子
	Mat dst_x, dst_y;
	Sobel(src_gray, dst_x, -1, 1, 0); 		    // x方向导数运算参数，可取0,1,2
	Sobel(src_gray, dst_y, -1, 0, 1); 		    // y方向导数运算参数
	addWeighted(dst_x, 0.5, dst_y, 0.5, 0, dst);
	cv::imwrite("out.jpg", dst);

	return 0;
    }
    
    usage(argv[0]);
    return -1;
}

