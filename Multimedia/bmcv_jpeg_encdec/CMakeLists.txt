cmake_minimum_required(VERSION 3.10)

project(lab_12)

set(TARGET_1 jpeg_EncDec)

set(SRC_1 ./src/jpeg_EncDec.cpp)

# 以下无需修改

set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ./bin)

include_directories(./src/ /workspace//include/decode /workspace/include/ffmpeg /workspace//include/opencv/opencv4 /workspace/include/bmlib)
link_directories(/workspace/lib/decode/soc /workspace/lib/ffmpeg/soc /workspace/lib/opencv/soc /workspace/lib/bmnn/soc)
set(LIBRARIES bmion
    bmjpulite
    bmjpuapi
    bmvpulite
    bmvpuapi
    bmvideo
    bmvppapi
    yuv
    avcodec
    avformat
    avutil
    swresample
    swscale
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_videoio
    bmcv
    bmlib
    pthread
    stdc++)

add_compile_options(-Wall -O2 -std=gnu++11 -fPIC)

add_executable(${TARGET_1} ${SRC_1})

target_link_libraries(${TARGET_1}  ${LIBRARIES})

