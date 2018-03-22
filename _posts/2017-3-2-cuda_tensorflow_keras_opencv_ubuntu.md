---

layout: post

title: Ubuntu 17.10에서 CUDA, cuDNN, Tensorflow, Keras, OpenCV 개발환경 구축

---

# 1.gtx 1080ti 그래픽 드라이버 설치

> 기존 드라이버를 업그레이드 할 경우 CUDA와 호환되지 않아 새로 설치 해야 한다.

## 1)설치전 사전 작업

### Ubuntu 저장소 변경

기본 우분투 저장소(Repository)가 속도가 느려서 ftp.daumkakao.com로 변경

```
# 백업
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak 
# streamlined editor를 이용해서 kr.archive.ubuntu.com 를 ftp.daumkakao.com 로 바꿔준다
sudo sed 's/kr.archive.ubuntu.com/ftp.daumkakao.com/g' /etc/apt/sources.list
```

### Nouveau 모드 NVIDIA 드라이버 비활성화

- Disable Nouveau  

/etc/modprobe.d/blacklist-nouveau.conf 에 추가 

```
sudo nano etc/modprobe.d/blacklist-nouveau.conf 
```

```
blacklist nouveau
options nouveau modeset=0
```

- initramfs 디스크 업데이트

```
sudo update-initramfs -u
```

### Text 모드로 전환 

- Kill the Display Manager

```
sudo service gdm stop 
```

- Go to Runlevel 3

```bash
sudo init 3
```

## 2)NVIDIA 드라이버 설치 

[NVIDIA 드라이버 다운로드]: http://www.nvidia.co.kr/Download/index.aspx

> 설치전 사전 작업을 진행하지 않을 경우 오류가 발생함.

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
# NVIDIA-Linux-x86_64-390.25.run 버전이 390 임을 확인
sudo apt install nvidia-390 nvidia-390-dev
```

## 3)Reboot 후 드라이버 설치 확인

```
nvidia - smi
```

![nvidia-smi](https://yannishin.github.com/images/nvidia-smi.png)

설치 오류가 있을 경우 빈화면이 나온다.



# 2.CUDA 9.1, cuDNN 7.1설치

## 1)설치전 사전 작업

- 종속 라이브러리 설치 :

```
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
```

- GCC 6 설치

> CUDA 9.X는 Ubuntu 17.10에 설치된 GCC 7.2 대신 GCC 6을 사용해야 하기 때문에 GCC 6을 설치

```
sudo apt install gcc-6
sudo apt install g++-6
```



## 2)CUDA 9.1

[CUDA 9.1 다운로드]: https://developer.nvidia.com/cuda-downloads

### Text 모드로 전환 

- Kill the Display Manager

```
sudo service gdm stop 
```

- Go to Runlevel 3

```bash
sudo init 3
```

### 설치

[설치문서]: https://docs.nvidia.com/cuda/



```
sudo bash cuda_9.1.85_387.26_linux.run --override
```

> 아래와 같이 설치하고 드라이버를 설치하지 않도록 설정

![cudatoolkit](https://yannishin.github.com/images/cudatoolkit.jpg)



- Patch 설치

```bash
sudo bash cuda_9.1.85.1_linux.run
sudo bash cuda_9.1.85.1_linux.run
```

- CUDA 9.x가 gcc v6 을 시용하도록 설정

```
sudo ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc
sudo ln -s /usr/bin/g++-6 /usr/local/cuda/bin/g++
```

- LD_LIBRARY_PATH에 CUDA 9 라이브러리 파일 추가 :

```
sudo nano /etc/ld.so.conf.d/cudalibs.conf
```

```
LD_LIBRARY_PATH=/usr/local/cuda/lib64
```

환경변수 추가

```
sudo nano ~/.bashrc
```

파일 끝에 추가

```
export PATH=${PATH}:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib64
```



GUI 모드로 Reboot

```
sudo service gdm start
sudo reboot
```



## 3)cuDNN

### 다운로드

https://developer.nvidia.com/cudnn 으로 접속하여 개발자 계정으로 로그인후 관련 파일을 다운로드

[Library 다운로드]: https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.1.1/prod/9.1_20180214/cudnn-9.1-linux-x64-v7.1
[Runtime Library 다운로드]: https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.1.1/prod/9.1_20180214/Ubuntu16_04-x64/libcudnn7_7.1.1.5-1+cuda9.1_amd64
[Developer Library 다운로드]: https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.1.1/prod/9.1_20180214/Ubuntu16_04-x64/libcudnn7-dev_7.1.1.5-1+cuda9.1_amd64	"다운로드"
[Code Sample 다운로드]: https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.1.1/prod/9.1_20180214/Ubuntu16_04-x64/libcudnn7-doc_7.1.1.5-1+cuda9.1_amd64	"다운로드"

### cuDNN 7.11 & libraries 설치

[설치문서]: http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html

```
tar -xvzf cudnn-9.1-linux-x64-v7.1.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

sudo dpkg -i libcudnn7_7.1.1.5-1+cuda9.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.1.1.5-1+cuda9.1_amd64.deb
sudo dpkg -i libcudnn7-doc_7.1.1.5-1+cuda9.1_amd64.deb
```

### 설치확인

- cuDNN을 테스트하기 위해 mnistCUDNN을 빌드

```
cp -r /usr/src/cudnn_samples_v7/ $HOME
cd $HOME/cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN
```

- cuDNN이 올바르게 설치되면 다음과 같이 표시

> Test passed!



# 3.Tensorflow,Keras, Theano 설치

## 1)Bazel 설치

[설치문서]: https://docs.bazel.build/versions/master/install-ubuntu.html#install-on-ubuntu

- JDK 8 다운로드, 설치

```
sudo apt-get install openjdk-8-jdk
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update && sudo apt-get install oracle-java8-installer
```

- Bazel 다운로드, 설치

```
#Download
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

#install
sudo apt-get update && sudo apt-get install --fix-missing bazel
#upgrade
sudo apt-get upgrade bazel
```

## 2)Python 종속성 설치

```
#딥러닝 프레임 워크의 의존성 설치
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libopencv-dev

#파이썬 2와 3을 boost, lmdb, glog, blas 등과 같은 다른 중요한 패키지와 함께 설치
sudo apt-get install -y --no-install-recommends libboost-all-dev doxygen
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev libblas-dev 
sudo apt-get install -y libatlas-base-dev libopenblas-dev libgphoto2-dev libeigen3-dev libhdf5-dev 
sudo apt-get install -y python-dev python-pip python-nose python-numpy python-scipy python-wheel
sudo apt-get install -y python3-dev python3-pip python3-nose python3-numpy python3-scipy python3-wheel

```





## 3)tensorflow 1.5 설치

[설치문서]: https://www.tensorflow.org/install/install_sources

### Tensorflow 소스 컴파일

- 컴파일을 위한 설정

```
git clone https://github.com/tensorflow/tensorflow 
cd tensorflow
./configure
```

- 빌드

```
 #GPU를 지원하는 TensorFlow 용 pip 패키지를 작성
 #--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" (TensorFlow는 구형 ABI를 사용하는 gcc 4로 제작되어 빌드시이전 ABI와 호환가능하도록 추가)
 bazel build --config=opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package 
 #build wheel
 bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

- pip 패키지 설치

```
cd /tmp/tensorflow_pkg
# /tmp/tensorflow_pkg 폴더에 생성된 패키지로 설치
sudo pip install tensorflow-1.6.0-py2-none-any.whl 
```

- 설치확인

```
# Python
import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))
```

### Keras, Theano 설치

```
sudo pip install keras
sudo pip install Theano	
```

```
import numpy
numpy.__version__
import tensorflow
tensorflow.__version__
import keras
Using TensorFlow backend.
keras.__version__
import theano
theano.__version__
```



# 4.OPEN CV 3.4 치치

## 1)install the dependencies

```
#C/C++ 컴파일러와 관련 라이브러리, make 같은 도구 설치
sudo apt-get install build-essential 
#cmake 설치
sudo apt-get install cmake

sudo apt-get remove -y x264 libx264-dev
sudo apt-get install -y checkinstall yasm
sudo apt-get install -y libjpeg8-dev libjasper-dev libpng12-dev

sudo apt-get install -y libtiff5-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev

sudo apt-get install -y libxine2-dev libv4l-dev
sudo apt-get install -y libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt-get install -y libqt4-dev libgtk2.0-dev libtbb-dev
sudo apt-get install -y libfaac-dev libmp3lame-dev libtheora-dev
sudo apt-get install -y libvorbis-dev libxvidcore-dev
sudo apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install -y x264 v4l-utils
```

## 2)컴파일 

### OpenCV 3.4.1 다운로드

```
# OpenCV 3.4.1 다운로드
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.4.1
cd ..
```

### OpenCV-contrib 3.4.1 다운로드

```
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 3.4.1
cd ..
```

> 나중에 OpenCV를 업그레이드하거나 제거하는 데 필요할 수있는 빌드 폴더를 같은 위치에 유지



## 3)빌드

```
cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D CMAKE_CXX_FLAGS=-std=c++11 \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D WITH_CUDA=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON .. \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++-5 \
      -DCMAKE_C_COMPILER=/usr/bin/gcc-5
      
nproc
# nproc CPU Core 수 12
make -j12
sudo make install
sudo ldconfig
```

참조 URL

[ubuntu 17.10 + CUDA 9.0 + cuDNN 7 + tensorflow源码编译]: https://zhuanlan.zhihu.com/p/30781460
[Install CUDA 9.1 and cuDNN 7 for TensorFlow 1.5.0]: https://medium.com/@xinh3ng/install-cuda-9-1-and-cudnn-7-for-tensorflow-1-5-0-cda36239bc68
[Ubuntu 17.10 - Cuda 9/Tensorflow 1.4]: https://tweakmind.com/ubuntu-17-10-cuda-9-tensorflow-1-4/
[https://github.com/heethesh/Install-TensorFlow-OpenCV-GPU-Ubuntu-17.10]: https://github.com/heethesh/Install-TensorFlow-OpenCV-GPU-Ubuntu-17.10
[ubuntu 16.04에 opencv_contrib 포함하여 OpenCV 3.4 설치]: http://webnautes.tistory.com/1030
[How to install OpenCV 3.4.0 on Ubuntu 16.04]: http://www.python36.com/how-to-install-opencv340-on-ubuntu1604/
[https://www.learnopencv.com/install-opencv3-on-ubuntu/]: https://www.learnopencv.com/install-opencv3-on-ubuntu/

