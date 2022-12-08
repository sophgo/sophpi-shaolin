#ifndef __FF_VIDEO_DECODE_H
#define __FF_VIDEO_DECODE_H

#include <iostream>

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

class VideoDec_FFMPEG
{
public:
    VideoDec_FFMPEG();
    ~VideoDec_FFMPEG();

    AVFormatContext *ifmt_ctx;
    AVCodec *decoder;
    AVCodecContext *video_dec_ctx;
    AVCodecParameters *video_dec_par;

    int width;
    int height;
    int pix_fmt;

    int video_stream_idx;
    AVFrame *frame;
    AVPacket pkt;
    int refcount;

    int openDec(const char *filename, int codec_name_flag,
                const char *coder_name, int output_format_mode = 100,
                int extra_frame_buffer_num = 5,
                int sophon_idx = 0, int pcie_no_copyback = 0);

    void closeDec();

    AVCodecParameters *getCodecPar();
    AVFrame *grabFrame();
    AVCodecContext *get_video_dec_ctx()
    {
        return video_dec_ctx;
    }
    AVFormatContext *get_avFormatCtx()
    {
        return ifmt_ctx;
    }
    AVCodec *get_decoder()
    {
        return decoder;
    }

private:
    AVCodec *findBmDecoder(AVCodecID dec_id, const char *name = "h264_bm",
                           int codec_name_flag = 0,
                           enum AVMediaType type = AVMEDIA_TYPE_VIDEO);

    int openCodecContext(int *stream_idx, AVCodecContext **dec_ctx,
                         AVFormatContext *fmt_ctx, enum AVMediaType type,
                         int codec_name_flag, const char *coder_name,
                         int output_format_mode = 100,
                         int extra_frame_buffer_num = 5,
                         int sophon_idx = 0, int pcie_no_copyback = 0);
};

#endif /*__FF_VIDEO_DECODE_H*/