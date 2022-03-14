# dragon-eye-simulator - F3F electronic judging system simulator

The intention of this project is to testing algorithm of F3F electronic judging system on PC. The input source is H.264 / H.265 video file (.mp4 or .mkv). The output can be display on screen and save to H.264 video file (.mp4 or .mkv). Further more it can also display intermediate processing frame on screen. So, it's a good tool for development.

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

## MOG2 threshold

MOG2 threshold default value is 8. 
The lower the value the more sensitive the camera.
The suggested threshold value is between 8 to 32.
Setup MOG2 threshold from environment variable

```
$ export MOG2_THRESHOLD=16
```

## Video options

If output result to screen

```
  #define VIDEO_OUTPUT_SCREEN
```

If output result to file

```
  #define VIDEO_OUTPUT_FILE "base"
```

## Image Inference

Download model engine
```
wget https://drive.google.com/file/d/1-gd8gdyl4l8k4vXObf6GFGeXYoIptAsc/view?usp=sharing -O model.engine
```

Enable inference
```
  #define IMAGE_INFERENCE
```

## Result

[Demo Video](https://youtu.be/XU2usfSNTS0)

## Reference

[OpenCV3学习（10.3）基于高斯混合模型的背景建模BackgroundSubtractorMOG/MOG2](https://blog.csdn.net/qq_30815237/article/details/87120195)

## Donate

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://paypal.me/stevegigijoe)

