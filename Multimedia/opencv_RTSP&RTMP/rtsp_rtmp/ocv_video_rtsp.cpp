/**
  @file videocapture_basic.cpp
  @brief A very basic sample for using VideoCapture and VideoWriter
  @author PkLab.net
  @date Aug 24, 2016
*/
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdio.h>
#include <queue>

#include <sys/time.h>
#include <unistd.h>
#include <signal.h>

#define IMAGE_MATQUEUE_NUM 25


#define BM_ALIGN16(_x)             (((_x)+0x0f)&~0x0f)
#define BM_ALIGN32(_x)             (((_x)+0x1f)&~0x1f)
#define BM_ALIGN64(_x)             (((_x)+0x3f)&~0x3f)


using namespace cv;
using namespace std;



typedef struct  threadArg{
    const char  *inputUrl;
    const char  *codecType;
    int         frameNum;
    const char  *outputName;
    int         yuvEnable;
    int         roiEnable;
    int         deviceId;
    const char  *encodeParams;
    int         startWrite;
    int         fps;
    int         imageCols;
    int         imageRows;
    queue<Mat*> *imageQueue;
}THREAD_ARG;


//std::queue<Mat> g_image_queue;
std::mutex g_video_lock;

int exit_flag = 0;



void *videoWriteThread(void *arg){

    THREAD_ARG *threadPara = (THREAD_ARG *)(arg);
    FILE *fp_out           = NULL;
    char *out_buf          = NULL;
    int is_stream          = 0;
    int quit_times         = 0;
    VideoWriter            writer;
    Mat                    image;
    string outfile         = "";
    string encodeparms     = "";
    int64_t curframe_start = 0;
    int64_t curframe_end   = 0;
    Mat *toEncImage;

    struct timeval   tv;

    if(threadPara->encodeParams)
        encodeparms = threadPara->encodeParams;

    if((strcmp(threadPara->outputName,"NULL") != 0) && (strcmp(threadPara->outputName,"null") != 0))
        outfile = threadPara->outputName;

    if(strstr(threadPara->outputName,"rtmp://") || strstr(threadPara->outputName,"rtsp://"))
        is_stream = 1;

    if(strcmp(threadPara->codecType,"H264enc") ==0)
    {
        writer.open(outfile, VideoWriter::fourcc('H', '2', '6', '4'),
        threadPara->fps,
        Size(threadPara->imageCols, threadPara->imageRows),
        encodeparms,
        true,
        threadPara->deviceId);
    }
    else if(strcmp(threadPara->codecType,"H265enc") ==0)
    {
       writer.open(outfile, VideoWriter::fourcc('h', 'v', 'c', '1'),
        threadPara->fps,
        Size(threadPara->imageCols, threadPara->imageRows),
        encodeparms,
        true,
        threadPara->deviceId);
    }
    else if(strcmp(threadPara->codecType,"MPEG2enc") ==0)
    {
       writer.open(outfile, VideoWriter::fourcc('M', 'P', 'G', '2'),
        threadPara->fps,
        Size(threadPara->imageCols, threadPara->imageRows),
        true,
        threadPara->deviceId);
    }

    if(!writer.isOpened())
    {

        return (void *)-1;

    }

    while(1){
        if(is_stream){
            gettimeofday(&tv, NULL);
            curframe_start= (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
        }

        //if(threadPara->startWrite && !g_image_queue.empty()) {
        if(threadPara->startWrite && !threadPara->imageQueue->empty()) {
 
             g_video_lock.lock();
             //writer.write(g_image_queue.front());
             toEncImage = threadPara->imageQueue->front();
             g_video_lock.unlock();
             writer.write(*toEncImage);

        }else{

            usleep(2000);

            quit_times++;
        }
        //only Push video stream
        if(is_stream){

            gettimeofday(&tv, NULL);
            curframe_end = (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;

            if(curframe_end - curframe_start > 1000000 / threadPara->fps)
                continue;

            usleep((1000000 / threadPara->fps) - (curframe_end - curframe_start));

        }
        //if((exit_flag && g_image_queue.size() == 0) || quit_times >= 1000){//No bitstream exits after a delay of three seconds
        if((exit_flag && threadPara->imageQueue->size() == 0) || quit_times >= 300){//No bitstream exits after a delay of three seconds
            break;
        }
    }
    writer.release();
    if(fp_out != NULL){
        fclose(fp_out);
        fp_out = NULL;
    }

    if(out_buf != NULL){
        free(out_buf);
        out_buf = NULL;
    }

    return (void *)0;

}



int main(int argc, char* argv[])
{
    string input_file;
    string output_file;
    if (argc < 2)
    {
        cout << "Please input input_file name and output_file name."
             << endl
             << "Usage: " << argv[0] << " [input file] [output file]" << endl
             << "\t"
             << "[input file] - input file name(example = rtsp://admin:admin123@192.168.1.124:554/id=0). "  << endl
             << "\t"
             << "[output file] - output file name(example = rtmp://192.168.1.125:1935/live/home)." << endl;
        exit(1);
    }
    if (argc == 3)
    {
        input_file = argv[1];
        output_file = argv[2];
    }
    else
    {
        input_file = "rtsp://admin:admin123@192.168.1.124:554/id=0";
        output_file = "rtmp://192.168.1.125:1935/live/home";
    }

    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;

    struct timeval tv1, tv2;
    pthread_t threadId;


    THREAD_ARG *threadPara = (THREAD_ARG *)malloc(sizeof(THREAD_ARG));
    memset(threadPara,0,sizeof(THREAD_ARG));

    if(!threadPara){
        return -1;
    }
    threadPara->imageQueue   = new queue<Mat*>;
    threadPara->outputName = output_file.data();
    threadPara->codecType  = "H265enc";
    threadPara->frameNum   = 10000;    //±àÂëµÄÖ¡Êý
    threadPara->deviceId = 0;
    threadPara->startWrite = 1;
    threadPara->yuvEnable = 1;
    threadPara->roiEnable = 1;
    threadPara->inputUrl = input_file.data();

    cap.open(threadPara->inputUrl, CAP_FFMPEG, threadPara->deviceId);
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open input Url\n";
        return -1;
    }

    // Set Resamper
    cap.set(CAP_PROP_OUTPUT_SRC, 1.0);
    double out_sar = cap.get(CAP_PROP_OUTPUT_SRC);
    cout << "CAP_PROP_OUTPUT_SAR: " << out_sar << endl;

    if(threadPara->yuvEnable == 1){
        cap.set(cv::CAP_PROP_OUTPUT_YUV, PROP_TRUE);
    }

    // Set scalar size
    //int height = (int) cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    //int width  = (int) cap.get(cv::CAP_PROP_FRAME_WIDTH);
    cout << "orig CAP_PROP_FRAME_HEIGHT: " << (int) cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "orig CAP_PROP_FRAME_WIDTH: " << (int) cap.get(CAP_PROP_FRAME_WIDTH) << endl;

    if(threadPara->startWrite)
    {
        Mat image;
        cap.read(image);
        //threadPara->imageQueue->push(image);
        threadPara->fps = cap.get(CAP_PROP_FPS);
        threadPara->imageCols = image.cols;
        threadPara->imageRows = image.rows;
        pthread_create(&threadId, NULL, videoWriteThread, threadPara);

    }

    //--- GRAB AND WRITE LOOP

    gettimeofday(&tv1, NULL);

    for (int i=0; i < threadPara->frameNum; i++)
    {
        if(exit_flag){
            break;
        }
        //Mat image;
        Mat *image = new Mat;
        // wait for a new frame from camera and store it into 'frame'
        cap.read(*image);

        // check if we succeeded
        //if (image.empty()) {
        if (image->empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            if ((int)cap.get(CAP_PROP_STATUS) == 2) {     // eof
                cout << "file ends!" << endl;
                cap.release();
                cap.open(threadPara->inputUrl, CAP_FFMPEG, threadPara->deviceId);
                if(threadPara->yuvEnable == 1){
                    cap.set(cv::CAP_PROP_OUTPUT_YUV, PROP_TRUE);
                }
                cout << "loop again " << endl;
            }
            continue;
        }
        g_video_lock.lock();
        //g_image_queue.push(image);
        threadPara->imageQueue->push(image);
        g_video_lock.unlock();
        if(threadPara->startWrite){
            //while(g_image_queue.size() >= IMAGE_MATQUEUE_NUM){
            while(threadPara->imageQueue->size() >= IMAGE_MATQUEUE_NUM){
                usleep(2000);
                if(exit_flag){
                    break;
                }
            }
        }
        if ((i+1) % 300 == 0)
        {
            unsigned int time;

            gettimeofday(&tv2, NULL);
            time = (tv2.tv_sec - tv1.tv_sec)*1000 + (tv2.tv_usec - tv1.tv_usec)/1000;

            printf("current process is %f fps!\n", (i * 1000.0) / (float)time);
        }
    }


    pthread_join(threadId, NULL);

    cap.release();

    if(threadPara->imageQueue){
        delete threadPara->imageQueue;
        threadPara->imageQueue = NULL;
    }

    if(threadPara){
        free(threadPara);
        threadPara = NULL;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
