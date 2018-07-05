#define CPU_ONLY
#define USE_OPENCV

#include <opencv2/imgproc.hpp>

#include "caffe/net.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/blob.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "vector"

int main(int argc, char** argv) {
    
    caffe::Net<float> net("res/deploy.prototxt", caffe::TEST);
    net.CopyTrainedLayersFrom("res/squeezenet_v1.1.caffemodel");
    
    cv::Mat img = cv::imread("res/grumpy.jpg");
    cv::resize(img, img, cv::Size(227, 227));
    
    boost::shared_ptr<caffe::MemoryDataLayer<float> > memoryDataLayer;
    memoryDataLayer = boost::static_pointer_cast<caffe::MemoryDataLayer<float> >(net.layer_by_name("data"));

    std::vector<cv::Mat> input_img;
    input_img.push_back(img);
    memoryDataLayer->AddMatVector(input_img, std::vector<int>(1));
    caffe::Blob<float> *out = net.ForwardPrefilled()[1];

    return 0;
}

