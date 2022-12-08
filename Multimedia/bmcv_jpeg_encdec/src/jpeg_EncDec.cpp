#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
extern "C"
{
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
#include "libavformat/avformat.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
}

using namespace std;
using namespace cv;
void FnEncode(const char *filenpath, int output_en);
void FnDecode(const char *filenpath, int output_en);
int main(int argc, char *argv[])
{
    string input_file;
    int codec_type;
    int outputEnable;
    if (argc < 2)
    {
        cout << "Please input input_file name , codec type and select if output."
             << endl
             << "Usage: " << argv[0] << " [input file] [codec-type] ([output enable] default = 1)" << endl
             << "\t"
             << " [input file] - the picture you want to encode or decode." << endl
             << "\t"
             << " [codec-type] - the codec type you want to use . 1 -> encode ,2 -> decode" << endl
             << "\t"
             << " [output enable] - if enable output .1 -> enable , 0 -> disable." << endl;
        exit(1);
    }
    if (argc == 3)
    {
        input_file = argv[1];
        codec_type = atoi(argv[2]);
        outputEnable = atoi(argv[3]);
    }
    else
    {
        input_file = argv[1];
        codec_type = atoi(argv[2]);
        outputEnable = 1;
    }

    switch (codec_type)
    {
    case 1:
        /* code */

        FnEncode(input_file.data(), outputEnable);
        cout << "encode finish ." << endl;
        break;
    case 2:

        FnDecode(input_file.data(), outputEnable);
        cout << "decode finish ." << endl;
        break;
    default:
        cout << "please input correct codec type number." << endl
             << "   "
             << " [codec-type] - the codec type you want to use . 1 -> encode ,2 -> decode" << endl;
        break;
    }
    return 0;
}

void FnEncode(const char *filenpath, int output_en)
{
    Mat save = imread(filenpath, IMREAD_UNCHANGED);
    vector<uint8_t> encoded;
    imencode(".jpg", save, encoded);

    if (output_en)
    {
        char *str = "encodeImage.jpg";
        int bufLen = encoded.size();
        if (bufLen)
        {
            uint8_t *pYuvBuf = encoded.data();
            FILE *fclr = fopen(str, "wb");
            fwrite(pYuvBuf, 1, bufLen, fclr);
            fclose(fclr);
        }
    }
}

void FnDecode(const char *filenpath, int output_en)
{
    ifstream in(filenpath, ios::binary);
    string s((istreambuf_iterator<char>(in)), (istreambuf_iterator<char>()));
    in.close();
    vector<char> pic(s.c_str(), s.c_str() + s.length());
    Mat image;
    imdecode(pic, IMREAD_UNCHANGED, &image);

    if (output_en)
    {
        char *str = "decodeImage.bmp";
        imwrite(str, image);
    }
}
