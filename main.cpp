#define CPU_ONLY
#define USE_OPENCV

#include "caffe/net.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "vector"
#include "fstream"

int main(int argc, char** argv) {
    // initialize logging
    google::InitGoogleLogging(argv[0]);
    google::SetCommandLineOption("GLOG_minloglevel", "2");
    
    // load network
    caffe::Net<float> net("res/deploy.prototxt", caffe::TEST);
    net.CopyTrainedLayersFrom("res/squeezenet_v1.1.caffemodel");
    
    // input layer
    boost::shared_ptr<caffe::MemoryDataLayer<float> > inputLayer;
    inputLayer = boost::static_pointer_cast<caffe::MemoryDataLayer<float> >(net.layer_by_name("data"));
    
    // load image
    cv::Mat img = cv::imread("res/cat.jpg");
    cv::resize(img, img, cv::Size(inputLayer->height(), inputLayer->width()));

    // classify
    std::vector<cv::Mat> inputData(1, img);
    inputLayer->AddMatVector(inputData, std::vector<int>(1));
    const float* probs = net.Forward()[1]->cpu_data();
    
    // print top-1 prediction
    std::string className;
    std::ifstream ilsvrcClasses("res/imagenet-classes.txt");
    int class_id = 0;
    while (std::getline(ilsvrcClasses, className)) {
        if(class_id++ == probs[0]) {
            printf("Class: '%s'\tScore: %.2f", className.c_str(), probs[1]);
            break;
        }
    }
    ilsvrcClasses.close();
    
    return 0;
}