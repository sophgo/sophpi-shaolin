#include "falsr.hpp"

Falsr::Falsr(std::shared_ptr<BMNNContext> context):m_bmContext(context) {
  std::cout << "Falsr ..." << std::endl;
}
Falsr::~Falsr(){}

// Init bmodel info and memory
int Falsr::init(){

    bm_status_t status = bm_dev_request(&bm_handle_, 0);
    assert(BM_SUCCESS == status);

    //Get Network
    m_bmNetwork = m_bmContext->network(0);

    //Get model info
    max_batch = m_bmNetwork->maxBatch();
    tensor_1 = m_bmNetwork->inputTensor(0);
    input_shape_1.h = tensor_1->get_shape()->dims[1];
    input_shape_1.w = tensor_1->get_shape()->dims[2];
    input_shape_1.c = tensor_1->get_shape()->dims[3];

    tensor_2 = m_bmNetwork->inputTensor(1);
    input_shape_2.h = tensor_2->get_shape()->dims[1];
    input_shape_2.w = tensor_2->get_shape()->dims[2];
    input_shape_2.c = tensor_2->get_shape()->dims[3];

    unsigned int input_size_1 = input_shape_1.h * input_shape_1.w * input_shape_1.c;
    unsigned int input_size_2 = input_shape_2.h * input_shape_2.w * input_shape_2.c;

    input_scale_1 = tensor_1->get_scale();   
    input_scale_2 = tensor_2->get_scale(); 

    //Request input tensor memory and relate to device memory
    input_y = new float[input_size_1];
    input_pbpr = new float[input_size_2];

    bm_malloc_device_byte(bm_handle_, &input_dev_mem_1, input_size_1*sizeof(float));
    bm_malloc_device_byte(bm_handle_, &input_dev_mem_2, input_size_2*sizeof(float));

    tensor_1->set_device_mem(&input_dev_mem_1);
    tensor_2->set_device_mem(&input_dev_mem_2);

    tensor_1->set_shape_by_dim(0, 1);
    tensor_2->set_shape_by_dim(0, 1);

    return 0;
}

//Do super-resolution for input frames
void Falsr::do_sr(std::vector<cv::Mat>& input_frames){
    float pre_time, forward_time, post_time;
    for (size_t i = 0; i < input_frames.size(); ++i){
        clock_t start_t = clock();
        cv::Mat input_frame = input_frames[i];

        //Preprocess
        resize(input_frame);
        cv::Mat input_frame_sec;
        input_frame_sec = input_frame.clone();
        ret = preprocess(input_frame, input_frame_sec);
        CV_Assert(ret == 0);
        clock_t pre_t = clock();
        pre_time += pre_t - start_t;

        //Forward
        forward();
        clock_t forward_t = clock();
        forward_time += forward_t - pre_t;

        //Postprocess
        ret = postprocess();
        CV_Assert(ret == 0);
        clock_t post_t = clock();
        post_time += post_t - forward_t;
        std::cout << "preprocess frame " << i << " time: " << (post_t - start_t)/TIME_UNIT << "ms" << std::endl;
    }
    std::cout << "average preprocess time: " << (pre_time / float(input_frames.size()))/TIME_UNIT << "ms" << std::endl;
    std::cout << "average forward time: " << (forward_time / float(input_frames.size()))/TIME_UNIT << "ms" << std::endl;
    std::cout << "average postprocess time: " << (post_time / float(input_frames.size()))/TIME_UNIT << "ms" << std::endl;
}

//Preprocess: 1.Generate two inputs.  2.Copy inputs to device memory
int Falsr::preprocess(cv::Mat& input_frame, cv::Mat& input_frame_sec){
    cv::Mat input_frame_2;
    cv::resize(input_frame_sec, input_frame_2, cv::Size(input_frame.cols * 2, input_frame.rows * 2), 0, 0, cv::INTER_CUBIC);
    std::vector<int> channel_idx_1 = {0};
    std::vector<int> channel_idx_2 = {1, 2};
    rgb2ypbpr(input_frame, input_y, channel_idx_1);
    rgb2ypbpr(input_frame_2, input_pbpr, channel_idx_2);

    bm_memcpy_s2d(bm_handle_, input_dev_mem_1, input_y);
    bm_memcpy_s2d(bm_handle_, input_dev_mem_2, input_pbpr);

    return 0;

}

//Forward
void Falsr::forward(){
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
}

//Resize input frame
void Falsr::resize(cv::Mat& input_frame){
    int h = input_frame.rows;
    int w = input_frame.cols;
    if (h > INPUT_H || w > INPUT_W){
        cv::resize(input_frame, input_frame, cv::Size(INPUT_H, INPUT_W));
    }
    else if (h < INPUT_H || w < INPUT_W){
        std::cout << "Input shape must bigger than model shape" << std::endl;
        assert(false);
    }
}

//postprocess
int Falsr::postprocess(){
    std::shared_ptr<BMNNTensor> outputTensor = m_bmNetwork->outputTensor(0);
    auto output_shape = m_bmNetwork->outputTensor(0)->get_shape();
    auto output_scale = m_bmNetwork->outputTensor(0)->get_scale();
    auto output_dims = output_shape->num_dims;
    float* out_ptr = (float*)outputTensor->get_cpu_data();
    float out_num;
    cv::Mat out_image = cv::Mat::zeros(output_shape->dims[1], output_shape->dims[2], CV_8UC3);
    for (int h = 0; h < out_image.rows; ++h){
        for (int w = 0; w < out_image.cols; ++w){
            for (int c = 0; c < output_shape->dims[3]; ++c){
                out_num = (out_ptr[h*out_image.cols*output_shape->dims[3] + w*output_shape->dims[3] + c]) * 255.0;
                out_num = std::min((int)out_num, 255);
                out_image.at<cv::Vec3b>(h, w)[c] = out_num;
            }
        }
    }
    out_images.push_back(out_image);
    return 0;
}

#ifdef False
cv::Mat Falsr::vector2mat(std::vector<std::vector<float>> vector_in){
    cv::Mat out_mat(3, 3, CV_64FC1);
    for (int i = 0; i < vector_in.size(); ++i){
        for (int j = 0; j < vector_in[0].size(); ++j){
            out_mat.at<float>(i, j) = vector_in[i][j];
        }
    }
    return out_mat;
}
#endif

//Convert RGB to YPbPr
void Falsr::rgb2ypbpr(cv::Mat &rgb, float *input_ptr, std::vector<int> chanel_idx){
    int idx_offset = chanel_idx[0];
    std::vector<int>::iterator it;
    for (int h = 0; h < rgb.rows; ++h){
        for (int w = 0; w < rgb.cols; ++w){
            for (it = chanel_idx.begin(); it != chanel_idx.end(); ++it){
                int idx = *it;
                float temp = 0;
                for (int c = 0 ; c < 3; ++c){
                    float rgb_num = rgb.at<cv::Vec3b>(h, w)[c];
                    temp += (rgb_num / 255.0) * ypbpr_from_rgb[c][idx];
                }
                *(input_ptr + (h*chanel_idx.size()*rgb.cols + w*chanel_idx.size() + (idx - idx_offset))) = temp;
            }
        }
    }
}

#ifdef False
void Falsr::rgb2ypbpr_int8(cv::Mat &rgb, int8_t *input_ptr, std::vector<int> chanel_idx, float input_scale){
    int idx_offset = chanel_idx[0];
    std::vector<int>::iterator it;
    for (int h = 0; h < rgb.rows; ++h){
        for (int w = 0; w < rgb.cols; ++w){
            for (it = chanel_idx.begin(); it != chanel_idx.end(); ++it){
                int idx = *it;
                float temp = 0;
                for (int c = 0 ; c < 3; ++c){
                    float rgb_num = rgb.at<cv::Vec3b>(h, w)[c];
                    temp += (rgb_num / 255.0) * ypbpr_from_rgb[c][idx];
                }
                temp *= input_scale;
                temp = (int8_t) temp;
                *(input_ptr + (h*chanel_idx.size()*rgb.cols + w*chanel_idx.size() + (idx - idx_offset))) = temp;
            }
        }
    }
}
#endif