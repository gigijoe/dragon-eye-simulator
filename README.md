# DRAGON EYE is a F3F automatic base

The intention of this project is to testing algorithm of F3F automatic base on PC. The input source is H.264 video file (.mp4 or .mkv). The output can be display on screen and save to H.264 video file (.mp4 or .mkv). Further more it can also display intermediate processing frame on screen. So, it's a good tool for development.

![Alt text](Screenshot.png?raw=true "Screenshot")

## Hardware requirement

Nvidia graphic card which supports cuda (For example GTX1050)

## Software environment

Ubuntu 20.04 / GCC-9 / CMake  
OpenCV 4.4.0  
cuda 11.0  

## Build & run

```
$ mkdir build
$ cd build
$ cmake ../
$ make
$ ./dragon-eye-cuda
```

## Video source

All video are in directory video/  
Change this line on dragon-eye-cuda.cpp  

```
  #define VIDEO_INPUT_FILE "../video/baseB001.mkv"
```

## Result

[Video](https://www.youtube.com/watch?v=g1BrMynNwn8)  

[Video](https://youtu.be/D6D2nifsbDQ)




