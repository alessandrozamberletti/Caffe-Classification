#define CPU_ONLY
#define USE_OPENCV

#include "caffe/net.hpp"

using namespace std;

int main(int argc, char** argv) {
    
    caffe::Net<float> cnn("res/lenet.prototxt", caffe::TEST);

    return 0;
}

