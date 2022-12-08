#include <iostream>
#include <vector>
#include "bmcv_api.h"
#include "common.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <memory>
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
	int H = 600;
	int W = 600;
	int C = 1;
	int dim = 1;
	int channels[1] = {0};
	int histSizes[] = {256};
	float ranges[] = {0, 256};
	int totalHists = 1;
	for (int i = 0; i < dim; ++i)
		totalHists *= histSizes[i];
	bm_handle_t handle = nullptr;
	bm_status_t ret = bm_dev_request(&handle, 0);
	float *inputHost = new float[C * W * H ];
	float *outputHost = new float[totalHists];

	if (argc != 2)
	{
		cout << "Please input a **grey** picture."
			 << endl
			 << "Usage: " << argv[0] << " [grey_picture]" << endl;
		bm_dev_free(handle);
		exit(1);
	}

	cv::Mat Input;
	Input = cv::imread(argv[1], 0);
	uint8_t pix;
	Input.at<uint8_t>(598,576) = 1;
	for (int i = 0; i < C; ++i)
	{
		for (int j = 0; j < H *W; ++j)
		{
			pix = Input.at<uint8_t>(j/W,j%W);
			inputHost[i * H * W + j] = (float)pix;
		}
	}

	if (ret != BM_SUCCESS)
	{
		printf("bm_dev_request failed. ret = %d\n", ret);
		exit(-1);
	}
	bm_device_mem_t input, output;
	ret = bm_malloc_device_byte(handle, &input, C * H * W * 4);
	if (ret != BM_SUCCESS)
	{
		printf("bm_malloc_device_byte failed. ret = %d\n", ret);
		exit(-1);
	}
	ret = bm_memcpy_s2d(handle, input, inputHost);

	if (ret != BM_SUCCESS)
	{
		printf("bm_memcpy_s2d failed. ret = %d\n", ret);
		exit(-1);
	}
	ret = bm_malloc_device_byte(handle, &output, totalHists * 4);
	if (ret != BM_SUCCESS)
	{
		printf("bm_malloc_device_byte failed. ret = %d\n", ret);
		exit(-1);
	}
	ret = bmcv_calc_hist(handle,
						 input,
						 output,
						 C,
						 H,
						 W,
						 channels,
						 dim,
						 histSizes,
						 ranges,
						 0);
	if (ret != BM_SUCCESS)
	{
		printf("bmcv_calc_hist failed. ret = %d\n", ret);
		exit(-1);
	}
	ret = bm_memcpy_d2s(handle, outputHost, output);
	if (ret != BM_SUCCESS)
	{
		printf("bm_memcpy_d2s failed. ret = %d\n", ret);
		exit(-1);
	}

	int scale = 2;
	int hist_height = 256;
	Mat hist_img = Mat::zeros(hist_height, 256 * scale, CV_8UC3);
	double max_val;
	Mat hist = Mat::zeros(1, 256 , CV_32FC3);;
	for (int i = 0; i < 250; i++)
	{
		hist.at<float>(i) = outputHost[i];
	}
	minMaxLoc(hist, 0, &max_val, 0, 0);
	for (int i = 0; i < 250; i++)
	{
		float bin_val = hist.at<float>(i);
		int intensity = cvRound(bin_val * hist_height / max_val);
		rectangle(hist_img, Point(i * scale, hist_height - 1), Point((i + 1) * scale - 1, hist_height - intensity), Scalar(255, 255, 255));
	}
	imwrite("bmcv_calc_hist_out.jpg", hist_img);

	bm_free_device(handle, input);
	bm_free_device(handle, output);
	bm_dev_free(handle);
	delete[] inputHost;
	delete[] outputHost;
	return 0;


}
