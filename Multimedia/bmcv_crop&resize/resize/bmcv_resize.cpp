/*
*	kk 2022/11/2 
*	bmcv_resize.cpp //该文件主要实现了对图片的进行resize操作，
*                     如放大、缩小、抠图等 
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

    if(0 == strcmp("bmcv",argv[2])) {
        // 获取句柄 
        bm_handle_t handle;
        bm_dev_request(&handle, 0);
        
        // 设置图片格式，设置剪切大小 
        int image_num = 1;
        int crop_w = 400, crop_h = 400, resize_w = 400, resize_h = 400;
        int image_w = 1000, image_h = 1000;
        int img_size_i = image_w * image_h * 3;
        int img_size_o = resize_w * resize_h * 3;
        
        // 智能指针 
        std::unique_ptr<unsigned char[]> img_data(
            new unsigned char[img_size_i * image_num]);
        std::unique_ptr<unsigned char[]> res_data(
            new unsigned char[img_size_o * image_num]);
		
        bmcv_resize_image resize_attr[image_num];
        bmcv_resize_t resize_img_attr[image_num];
	
	// 设置resize配置 
        for (int img_idx = 0; img_idx < image_num; img_idx++) {
            resize_img_attr[img_idx].start_x = 0;
            resize_img_attr[img_idx].start_y = 0;
            resize_img_attr[img_idx].in_width = crop_w;
            resize_img_attr[img_idx].in_height = crop_h;
            resize_img_attr[img_idx].out_width = resize_w;
            resize_img_attr[img_idx].out_height = resize_h;
        }
		
        for (int img_idx = 0; img_idx < image_num; img_idx++) {
            resize_attr[img_idx].resize_img_attr = &resize_img_attr[img_idx];
            resize_attr[img_idx].roi_num = 1;
            resize_attr[img_idx].stretch_fit = 1;
            resize_attr[img_idx].interpolation = BMCV_INTER_NEAREST;
        }
        bm_image input[image_num];
        bm_image output[image_num];
        // opencv读取数据 
        cv::Mat Input,Out;
        Input = cv::imread(argv[1], 1);
		
        for (int img_idx = 0; img_idx < image_num; img_idx++) {
            int input_data_type = DATA_TYPE_EXT_1N_BYTE;
            bm_image_create(handle, image_h, image_w, FORMAT_BGR_PLANAR,
                (bm_image_data_format_ext)input_data_type, &input[img_idx]);
	}
        bm_image_alloc_contiguous_mem(image_num, input, 1);
		
		
        for (int img_idx = 0; img_idx < image_num; img_idx++) {
             int output_data_type = DATA_TYPE_EXT_1N_BYTE;
             bm_image_create(handle,resize_h,resize_w,FORMAT_BGR_PLANAR,
                 (bm_image_data_format_ext)output_data_type,&output[img_idx]);
        }
        bm_image_alloc_contiguous_mem(image_num, output, 1);
        // 转数据格式 
        cv::bmcv::toBMI(Input,&input[0]);
        // resize处理 
        bmcv_image_resize(handle, image_num, resize_attr, input, output);
		
		
        cv::bmcv::toMAT(&output[0], Out);
        cv::imwrite("out.jpg", Out);
		
        bm_image_free_contiguous_mem(image_num, input);
        bm_image_free_contiguous_mem(image_num, output);
        for (int i = 0; i < image_num; i++) {
            bm_image_destroy(input[i]);
            bm_image_destroy(output[i]);
        }

        return 0; 

    } 
    
    if (0==strcmp("opencv",argv[2])) {

        Mat img = imread(argv[1], 1);
        if (img.empty()) {
            cout <<"无法读取图像" <<endl;
            return 0;
        }
	 
        int height = img.rows;
        int width = img.cols;
        
        // 缩小图像，比例为(0.3, 0.5)
        Size dsize = Size(round(0.3 * width), round(0.5 * height));
        Mat shrink;
        resize(img, shrink, dsize, 0, 0, INTER_AREA);
	 	cv::imwrite("small.jpg", shrink);
        // 放大图像，比例为(1.6, 1.2)
        float fx = 1.6;
        float fy = 1.2;
        Mat enlarge;
        resize(img, enlarge, Size(), fx, fy, INTER_CUBIC);
        cv::imwrite("large.jpg", enlarge);
        return 0;
    } 
    
    usage(argv[0]);
    return -1;
}

