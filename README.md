# dragon-eye-simulator - F3F automatic base simulator

The intention of this project is to testing algorithm of F3F automatic base on PC. The input source is H.264 / H.265 video file (.mp4 or .mkv). The output can be display on screen and save to H.264 video file (.mp4 or .mkv). Further more it can also display intermediate processing frame on screen. So, it's a good tool for development.

![Alt text](Screenshot.png?raw=true "Screenshot")

## Hardware environment

* Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz
* Nvidia graphic card which supports cuda (For example GTX1050)

## Software environment

* Ubuntu 20.04 / GCC-9 / CMake  
* OpenCV 4.4.0  
* cuda 11.0  

## Technology

Base on image processing technology morphology operators, background subtraction and image smoothing, etc ...

## Processing flow

<pre>
Camera -> BGR to GRAY -> MOG2 Background Subtraction -> Erode -> Dilate -> Find Contour -> Anti cloud -> Moving Targets  
       
Moving Targets -> Targets Tracker -> Find if cross the line
</pre>

## Build & run

```
$ mkdir build
$ cd build
$ cmake ../
$ make
$ ./dragon-eye-simulator ../video/baseB001.mkv
```

* Press key 'P' to pause / continue
* Press key 'ESC' to quit

## Video options

If output result to screen

```
  #define VIDEO_OUTPUT_SCREEN
```

If output result to file

```
  #define VIDEO_OUTPUT_FILE "base"
```

## Result

[Video 1](https://www.youtube.com/watch?v=g1BrMynNwn8)  

[Video 2](https://youtu.be/D6D2nifsbDQ)

[Video 3](https://youtu.be/fjrS-nNDypw)

## Reference

[OpenCV3学习（10.3）基于高斯混合模型的背景建模BackgroundSubtractorMOG/MOG2](https://blog.csdn.net/qq_30815237/article/details/87120195)
