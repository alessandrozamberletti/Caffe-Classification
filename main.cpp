#define CPU_ONLY
#define USE_OPENCV

#include <opencv2/imgproc.hpp>

#include "caffe/net.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/blob.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "vector"
#include "fstream"

int main(int argc, char** argv) {
    // log errors
    google::InitGoogleLogging(argv[0]);
    google::SetCommandLineOption("GLOG_minloglevel", "2");
    
    // load network
    caffe::Net<float> net("res/deploy.prototxt", caffe::TEST);
    net.CopyTrainedLayersFrom("res/squeezenet_v1.1.caffemodel");
    
    // load image
    cv::Mat img = cv::imread("res/grumpy.jpg");
    cv::resize(img, img, cv::Size(227, 227));
    
    // prepare input layer
    boost::shared_ptr<caffe::MemoryDataLayer<float> > memoryDataLayer;
    memoryDataLayer = boost::static_pointer_cast<caffe::MemoryDataLayer<float> >(net.layer_by_name("data"));

    // classify
    std::vector<cv::Mat> input_img(1, img);
    memoryDataLayer->AddMatVector(input_img, std::vector<int>(1));
    caffe::Blob<float> *preds = net.ForwardPrefilled()[1];
    const float* probabilities = preds->cpu_data();
    
    // print top-1 class
    std::string ilsvrc_class;
    std::ifstream ilsvrc_classes("res/imagenet-classes.txt");
    int idx = 0;
    while (std::getline(ilsvrc_classes, ilsvrc_class)) {
        if(idx++ == probabilities[0]) {
            printf("Class: '%s'\tScore: %.2f", 
                    ilsvrc_class.c_str(), 
                    probabilities[1]);
            break;
        }
    }
    
    return 0;
}

