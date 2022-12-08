
#include <iostream>
#include "videoDec.h"

#ifdef __cplusplus
extern "C"
{
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#ifdef __cplusplus
};
#endif

using namespace std;

const char *progressbar_icon = "|||///---\\\\\\";

int main(int argc, char *argv[])
{
    AVFormatContext *pFormatCtx;
    int i, videoindex;
    AVCodecContext *pCodecCtx;
    AVCodec *pCodec;
    AVFrame *pFrame, *pFrameYUV;
    uint8_t *out_buffer;
    AVPacket *packet;
    int y_size;
    int ret, got_picture;
    struct SwsContext *img_convert_ctx;
    VideoDec_FFMPEG reader;
    string input_file, output_file;

    if (argc < 2)
    {
        cout << "Please input input_file name and output_file name."
             << endl
             << "Usage: " << argv[0] << " [input file] [output file]" << endl
             << "\t"
             << "[input file] - input file name." << endl
             << "\t"
             << "[output file] - output file name(default = ffmpeg_decode_out.yuv)." << endl;

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
        output_file = "ffmpeg_decode_out.yuv";
    }

    FILE *fp_yuv = fopen(output_file.data(), "wb+");
    av_log_set_level(AV_LOG_DEBUG); // set debug level
    reader.openDec(input_file.data(), 1, "h264_bm", 100, 60, 0, 0);

    pFormatCtx = reader.ifmt_ctx;

    videoindex = -1;
    for (i = 0; i < pFormatCtx->nb_streams; i++)
        if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            videoindex = i;
            break;
        }

    if (videoindex == -1)
    {
        printf("Didn't find a video stream.\n");
        return -1;
    }
    cout << "video index " << videoindex << endl;

    pCodecCtx = reader.video_dec_ctx;
    pCodec = reader.decoder;

    pFrame = av_frame_alloc();
    pFrameYUV = av_frame_alloc();
    out_buffer = (uint8_t *)av_malloc(avpicture_get_size(AV_PIX_FMT_YUV420P, pCodecCtx->width, pCodecCtx->height));
    avpicture_fill((AVPicture *)pFrameYUV, out_buffer, AV_PIX_FMT_YUV420P, pCodecCtx->width, pCodecCtx->height);
    packet = (AVPacket *)av_malloc(sizeof(AVPacket));

    av_dump_format(pFormatCtx, 0, input_file.data(), 0);

    img_convert_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt,
                                     pCodecCtx->width, pCodecCtx->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL);
    long long framecount = 0;
    while (av_read_frame(pFormatCtx, packet) >= 0)
    { //读取一帧压缩数据
        if (packet->stream_index == videoindex)
        {

            // fwrite(packet->data,1,packet->size,fp_h264); //把H264数据写入fp_h264文件
            ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, packet); //解码一帧压缩数据
            if (ret < 0)
            {
                printf("Decode Error.\n");
                return -1;
            }

            if (got_picture)
            {
                sws_scale(img_convert_ctx, (const uint8_t *const *)pFrame->data, pFrame->linesize, 0, pCodecCtx->height,
                          pFrameYUV->data, pFrameYUV->linesize);

                y_size = pCodecCtx->width * pCodecCtx->height;
                fwrite(pFrameYUV->data[0], 1, y_size, fp_yuv);     // Y
                fwrite(pFrameYUV->data[1], 1, y_size / 4, fp_yuv); // U
                fwrite(pFrameYUV->data[2], 1, y_size / 4, fp_yuv); // V

                printf("\rfinish %lld [%c].", ++framecount, progressbar_icon[framecount % 12]);
                fflush(stdout);
            }
        }
        av_free_packet(packet);
    }
    // flush decoder
    /*当av_read_frame()循环退出的时候，实际上解码器中可能还包含剩余的几帧数据。
    因此需要通过“flush_decoder”将这几帧数据输出。
   “flush_decoder”功能简而言之即直接调用avcodec_decode_video2()获得AVFrame，而不再向解码器传递AVPacket。*/
    while (1)
    {
        ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, packet);
        if (ret < 0)
            break;
        if (!got_picture)
            break;
        sws_scale(img_convert_ctx, (const uint8_t *const *)pFrame->data, pFrame->linesize, 0, pCodecCtx->height,
                  pFrameYUV->data, pFrameYUV->linesize);

        int y_size = pCodecCtx->width * pCodecCtx->height;
        fwrite(pFrameYUV->data[0], 1, y_size, fp_yuv);     // Y
        fwrite(pFrameYUV->data[1], 1, y_size / 4, fp_yuv); // U
        fwrite(pFrameYUV->data[2], 1, y_size / 4, fp_yuv); // V

        printf("\rfinish %lld [%c].", ++framecount, progressbar_icon[framecount % 12]);
        fflush(stdout);
    }

    sws_freeContext(img_convert_ctx);

    //关闭文件以及释放内存
    fclose(fp_yuv);

    cout << "Total Decode " << framecount << " frames" << endl;
    av_frame_free(&pFrameYUV);
    av_frame_free(&pFrame);

    return 0;
}
