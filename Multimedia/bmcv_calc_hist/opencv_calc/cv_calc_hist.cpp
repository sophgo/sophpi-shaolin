#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Please input a grey picture."
             << endl
             << "Usage: " << argv[0] << " [grey_picture]" << endl;
        exit(1);
    }
    Mat src;
    src = imread(argv[1], 0);

    const int channels[] = {0};
    Mat hist;
    int dims = 1;
    const int histsize[] = {256};
    float pranges[] = {0, 255};
    const float *ranges[] = {pranges};

    calcHist(&src, 1, channels, Mat(), hist, dims, histsize, ranges, true, false);

    int scale = 2;
    int hist_height = 256;
    Mat hist_img = Mat::zeros(hist_height, 256 * scale, CV_8UC3);
    double max_val;
    minMaxLoc(hist, 0, &max_val, 0, 0);
    for (int i = 0; i < 256; i++)
    {
        float bin_val = hist.at<float>(i);
        int intensity = cvRound(bin_val * hist_height / max_val);
        rectangle(hist_img, Point(i * scale, hist_height - 1), Point((i + 1) * scale - 1, hist_height - intensity), Scalar(255, 255, 255));
    }
    imwrite("cv_calc_hist_out.jpg", hist_img);

    return 0;
}
