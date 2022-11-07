#ifndef FALSR_HPP
#define FALSR_HPP
#include <iostream>
#include <vector>
#include <assert.h>
#include <time.h>
#include "opencv2/opencv.hpp"
#include "bmruntime_interface.h"

#define USE_OPENCV
#include "bmnn_utils.h"

#define INPUT_H 270
#define INPUT_W 480
#define RATIO 2
#define TIME_UNIT 1000
 
 struct input_shape
 {
    int b;
    int c;
    int h;
    int w;
 };
 
class Falsr{
   
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    int max_batch;
    int ret;
    bm_device_mem_t input_dev_mem_1;
    bm_device_mem_t input_dev_mem_2;
    input_shape input_shape_1;
    input_shape input_shape_2;
    float input_scale_1;
    float input_scale_2;
    bool int8_flag_;

    private:
    /* handle of low level device */
    bm_handle_t bm_handle_;
    float *input_y;
    float *input_pbpr;
    std::shared_ptr<BMNNTensor> tensor_1;
    std::shared_ptr<BMNNTensor> tensor_2;

    //Preprocess
    int preprocess(cv::Mat& input_frame, cv::Mat& input_frame_sec);
    //Forward
    void forward();
    //Postprocess
    int postprocess();
    //Input Frame Resize To Model Shape
    void resize(cv::Mat& input_frame);

    public:
    Falsr(std::shared_ptr<BMNNContext> context);
    virtual ~Falsr();
    std::vector<cv::Mat> out_images;
    std::vector<cv::Mat> input_images;
    //YPbPr transpose matrix
    std::vector<std::vector<float>> ypbpr_from_rgb = {{0.299   , -0.168736  , 0.5   },
                                                    {0.587     , -0.331264  , -0.418688},
                                                    {0.114     , 0.5        ,-0.081312}};
   //Convert 2D-vector to opencv mat                              
    cv::Mat vector2mat(std::vector<std::vector<float>> vector_in);
    //Convert RGB to YPbPr
    void rgb2ypbpr(cv::Mat &rgb, float *input_ptr, std::vector<int> chanel_idx);
    void rgb2ypbpr_int8(cv::Mat &rgb, int8_t *input_ptr, std::vector<int> chanel_idx, float input_scale);
    //Init bmodel info and memory
    int init();
    //Input Frame Do Super Resolution
    void do_sr(std::vector<cv::Mat> &input_images);

};

#endif