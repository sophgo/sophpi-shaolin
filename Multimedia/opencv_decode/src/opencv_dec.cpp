#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

const char *progressbar_icon = "||||||//////------\\\\\\\\\\\\";

int main(int argc, char *argv[])
{
    string input_file, output_file;

    if (argc < 2)
    {
        cout << "Please input input_file name and output_file name."
             << endl
             << "Usage: " << argv[0] << " [input file] [output file]" << endl
             << "\t"
             << "[input file] - input file name." << endl
             << "\t"
             << "[output file] - output file name(default = opencv_decode_out.yuv)." << endl;
        exit(1);
    }
    if (argc == 3)
    {
        input_file = argv[1];
        output_file = argv[2];
    }
    else
    {
        input_file = argv[1];
        output_file = "opencv_decode_out.yuv";
    }

    VideoCapture cap;
    FILE *output_stream;

    cap.open(input_file, CAP_FFMPEG, 0);
    if (!cap.isOpened())
    {
        cout << "Could not open the input video for write: " << input_file << endl;
        return -1;
    }
    int fps = cap.get(CAP_PROP_FPS);
    cout << "FPS : " << fps << endl;
    long long frames = cap.get(CAP_PROP_FRAME_COUNT);
    cout << "Total Frame : " << frames << endl;

    output_stream = fopen(output_file.data(), "w+");
    if (output_stream == NULL)
    {
        cout << "open output file faild." << endl;
        exit(1);
    }

    Mat image;
    long long framecount = 0;
    cap.set(CAP_PROP_OUTPUT_YUV, PROP_TRUE);

    while (true)
    {
        if (!cap.read(image))
        {
            break;
        }

        bmcv::downloadMat(image);

        for (int i = 0; i < image.avRows(); i++)
        {
            fwrite((char *)image.avAddr(0) + i * image.avStep(0), 1, image.avCols(), output_stream);
        }
        for (int i = 0; i < image.avRows() / 2; i++)
        {
            fwrite((char *)image.avAddr(1) + i * image.avStep(1), 1, image.avCols() / 2, output_stream);
        }
        for (int i = 0; i < image.avRows() / 2; i++)
        {
            fwrite((char *)image.avAddr(2) + i * image.avStep(2), 1, image.avCols() / 2, output_stream);
        }
        printf("\rfinish %ld / %ld [%c].", ++framecount, frames, progressbar_icon[framecount % 24]);
        fflush(stdout);
    }

    cout << "Total Decode " << framecount << " frames" << endl;
    fclose(output_stream);
    return 0;
}
