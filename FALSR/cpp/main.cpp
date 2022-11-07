#include "falsr.hpp"

int main(int argc, char *argv[]){

    // Profiling
    const char *keys="{bmodel |xxx.bmodel | bmodel file path}"
    "{video_path | xxx.mp4 | Input mp4 path}"
    "{result_path | xxx.mp4 | Output mp4 path}"
    "{tpuid | 0 | TPU device id}"
    "{out_imgs_path | path | Output pridict frame path}"
    "{help | 0 | Print help information}";
    
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    std::string bmodel_file = parser.get<std::string>("bmodel");
    size_t dev_id = parser.get<int>("tpuid");
    BMNNHandlePtr handle = std::make_shared<BMNNHandle>(dev_id);
    std::string video_file = parser.get<std::string>("video_path");
    std::string result_path = parser.get<std::string>("result_path");
    std::string out_imgs_path = parser.get<std::string>("out_imgs_path");
    
    // Load bmodel
    clock_t start_t = clock();
    std::shared_ptr<BMNNContext> bm_ctx = std::make_shared<BMNNContext>(handle, bmodel_file.c_str());
    Falsr falsr(bm_ctx);
    CV_Assert(0 == falsr.init());

    //Get Video Frames
    cv::VideoCapture capture(video_file);
	if (!capture.isOpened()){
        std::cout << "open video " << video_file << " failed!" << std::endl;
		return -1;
	}
    int w = int(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = int(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "resolution of input stream: " << h << "," << w << std::endl;
    cv::Mat frame(w, h, CV_32FC3);
    std::vector<cv::Mat> input_images;
    while (1)
    {   
        cv::Mat frame(w, h, CV_32FC3);
        if (!capture.read(frame)) {
            std::cout << "Read frame failed!" << std::endl;
            break;
        }
        input_images.push_back(frame);

    }
    clock_t init_t = clock();
    std::cout << "prepare bmodel and get video frames time: " << (init_t - start_t)/TIME_UNIT << "ms" << std::endl;

    //Pridict super-resolution frames
    falsr.do_sr(input_images);

    //Write pridict frames into video
    clock_t before_writer_t = clock();
    cv::VideoWriter video_writer;
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    std::cout << result_path << std::endl;
    video_writer.open(result_path, codec, 25, cv::Size(INPUT_W*2, INPUT_H*2), true);
    if (!video_writer.isOpened()) {
        std::cout << "Could not open the output video file for write\n";
        return -1;
    }
    for (size_t i = 0; i < falsr.out_images.size(); ++i){
        video_writer.write(falsr.out_images[i]);
        std::string img_idx = std::to_string(i);
        cv::imwrite(out_imgs_path + img_idx +".png", falsr.out_images[i]);
    }
    clock_t after_writer_t = clock();
    std::cout << "!!! super-resolution video done !!!" << std::endl;
    std::cout << "write video time: " << (after_writer_t - before_writer_t)/TIME_UNIT << "ms" << std::endl;
    std::cout << "------total video super resolution time: " << (after_writer_t - start_t)/TIME_UNIT << "ms" << "------" << std::endl;
    return 0;
}
