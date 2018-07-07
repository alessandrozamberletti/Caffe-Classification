# caffe-opencv
Image classification using Caffe and OpenCV in 50 lines of code

## Instructions [Ubuntu 16.04](http://releases.ubuntu.com/16.04/)
* Install dependencies
```console
sudo apt-get update &&
sudo apt-get -y upgrade &&
sudo apt-get -y install build-essential cmake git libgtk2.0-dev pkg-config python-dev python-numpy libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libboost-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev libopenblas-dev wget libcurl4-openssl-dev libatlas-base-dev
```
* Install [OpenCV v3.0.0](https://github.com/opencv/opencv/archive/3.0.0.zip)
```console
```
* Install [Caffe v1.0RC5](https://github.com/BVLC/caffe/archive/rc5.zip)
```console
wget https://github.com/BVLC/caffe/archive/rc5.zip &&
unzip rc5.zip &&
cd caffe-rc5 &&
cp Makefile.config.example Makefile.config &&
sed -i 's\# CPU_ONLY := 1\CPU_ONLY := 1\' Makefile.config &&
sed -i 's\# USE_OPENCV := 0\USE_OPENCV := 1\' Makefile.config &&
sed -i 's\# OPENCV_VERSION := 3\OPENCV_VERSION := 3\' Makefile.config &&
make all -j`nproc --all` &&
sudo make distribute
```
