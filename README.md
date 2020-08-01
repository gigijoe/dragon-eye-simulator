# DRAGON EYE is a F3F automatic base

The intention of this project is to testing algorithm of F3F automatic base on PC. The input source is H.264 video file (.mp4 or .mkv). The output can be display on screen and save to H.264 video file (.mp4 or .mkv). Further more it can also display intermediate processing frame on screen. So, it's a good tool for development.

## Hardware requirement

Nvidia graphic card which supports cuda

## Software requirement

OpenCV 4.1.0

cuda 10.2

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

  #define VIDEO_INPUT_FILE "../video/baseB001.mkv"





