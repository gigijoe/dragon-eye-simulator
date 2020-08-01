#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

//#include <opencv2/cudacodec.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaobjdetect.hpp>

#include <errno.h>
#include <fcntl.h> 
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

using namespace cv;
using namespace std;

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

#define CAMERA_WIDTH 1280
#define CAMERA_HEIGHT 720
#define CAMERA_FPS 60
#define MIN_TARGET_WIDTH 8
#define MIN_TARGET_HEIGHT 8
#define MAX_TARGET_WIDTH 320
#define MAX_TARGET_HEIGHT 320

#define MAX_NUM_TARGET 3
#define MAX_NUM_TRIGGER 4
#define MAX_NUM_FRAME_MISSING_TARGET 10

#define MIN_COURSE_LENGTH            120    /* Minimum course length of RF trigger after detection of cross line */
#define MIN_TARGET_TRACKED_COUNT     3      /* Minimum target tracked count of RF trigger after detection of cross line */

//#define VIDEO_INPUT_FILE "../video/baseA011.mkv"
#define VIDEO_INPUT_FILE "../video/baseB001.mkv"

#ifdef VIDEO_INPUT_FILE
static int videoWidth, videoHeight;
#endif

#define VIDEO_OUTPUT_SCREEN
#define VIDEO_OUTPUT_DIR "."
//#define VIDEO_OUTPUT_FILE "base"

#ifdef VIDEO_OUTPUT_FILE
#ifndef VIDEO_OUTPUT_SCREEN
#define VIDEO_OUTPUT_SCREEN
#endif 
#endif

#define VIDEO_FRAME_DROP 30

static bool bShutdown = false;

void sig_handler(int signo)
{
    if(signo == SIGINT) {
        printf("SIGINT\n");
        bShutdown = true;
    }
}

/*
*
*/

static inline bool ContoursSort(vector<cv::Point> contour1, vector<cv::Point> contour2)  
{  
    //return (contour1.size() > contour2.size()); /* Outline length */
    return (cv::contourArea(contour1) > cv::contourArea(contour2)); /* Area */
}  

inline void writeText( Mat & mat, const string text )
{
   int fontFace = FONT_HERSHEY_SIMPLEX;
   double fontScale = 1;
   int thickness = 1;  
   Point textOrg( 10, 40 );
   putText( mat, text, textOrg, fontFace, fontScale, Scalar(0, 0, 0), thickness, cv::LINE_8 );
}

class Target
{
protected:
    double m_arcLength;
    unsigned long m_frameTick;
    uint8_t m_triggerCount;

public:
    Target(Rect & roi, unsigned long frameTick) : m_arcLength(0), m_frameTick(frameTick), m_triggerCount(0) {
        m_rects.push_back(roi);
        m_points.push_back(roi.tl());
        m_frameTick = frameTick;
    }

    vector< Rect > m_rects;
    vector< Point > m_points;
    Point m_velocity;

    void Update(Rect & roi, unsigned long frameTick) {
        if(m_rects.size() > 0)
            m_arcLength += norm(roi.tl() - m_rects.back().tl());

        if(m_points.size() == 1) {
            m_velocity.x = (roi.tl().x - m_rects.back().tl().x);
            m_velocity.y = (roi.tl().y - m_rects.back().tl().y);
        } else if(m_points.size() > 1) {
            m_velocity.x = (m_velocity.x + (roi.tl().x - m_rects.back().tl().x)) / 2;
            m_velocity.y = (m_velocity.y + (roi.tl().y - m_rects.back().tl().y)) / 2;
        }
        m_rects.push_back(roi);
        m_points.push_back(roi.tl());
        m_frameTick = frameTick;
    }
#if 0
    void Reset() {
        Rect r = m_rects.back();
        Point p = r.tl();
        m_rects.clear();
        m_points.clear();
        m_rects.push_back(r);
        m_points.push_back(p);
        m_triggerCount = 0;
        //m_frameTick =
        //m_arcLength = 0; /* TODO : Do clear this ? */
    }
#endif
    inline double ArcLength() { return m_arcLength; }
    inline unsigned long FrameTick() { return m_frameTick; }
    inline Rect & LatestRect() { return m_rects.back(); }
    inline Point & LatestPoint() { return m_points.back(); }
    inline void Trigger() { m_triggerCount++; }
    inline uint8_t TriggerCount() { return m_triggerCount; }
    //inline Point & Velocity() { return m_velocity; }
    inline size_t RectCount() { return m_rects.size(); }
};

static inline bool TargetSort(Target & a, Target & b)
{
    return a.LatestRect().area() > b.LatestRect().area();
}

class Tracker
{
private:
    unsigned long m_frameTick;
    list< Target > m_targets;
    Target *m_primaryTarget;

public:
    Tracker() : m_frameTick(0), m_primaryTarget(0) {}

    void Update(vector< Rect > & roiRect) {
        const int euclidean_distance = 240;

        if(m_primaryTarget) {
            Target *t = m_primaryTarget;
            int i;
            for(i=0; i<roiRect.size(); i++) {
                Rect r = t->m_rects.back();
                if((r & roiRect[i]).area() > 0) /* Target tracked ... */
                    break;                

                unsigned long f = m_frameTick - t->FrameTick();
                r.x += t->m_velocity.x * f;
                r.y += t->m_velocity.y * f;
                if((r & roiRect[i]).area() > 0) /* Target tracked with velocity ... */
                    break;
#if 0
                if(cv::norm(r.tl()-roiRect[i].tl()) < euclidean_distance) /* Target tracked with velocity and Euclidean distance ... */
                    break;
#endif
                r = t->m_rects.back();
                if(cv::norm(r.tl()-roiRect[i].tl()) < euclidean_distance) /* Target tracked with Euclidean distance ... */
                    break;
            }
            if(i != roiRect.size()) { /* Primary Target tracked */
                t->Update(roiRect[i], m_frameTick);
                return;
            }
        }

        for(list< Target >::iterator t=m_targets.begin();t!=m_targets.end();) { /* Try to find lost targets */
            int i;
            for(i=0; i<roiRect.size(); i++) {
                Rect r = t->m_rects.back();
                if((r & roiRect[i]).area() > 0) /* Target tracked ... */
                    break;                

                unsigned long f = m_frameTick - t->FrameTick();
                r.x += t->m_velocity.x * f;
                r.y += t->m_velocity.y * f;
                if((r & roiRect[i]).area() > 0) /* Target tracked with velocity ... */
                    break;
#if 0
                if(cv::norm(r.tl()-roiRect[i].tl()) < euclidean_distance) /* Target tracked with velocity and Euclidean distance ... */
                    break;
#endif
                r = t->m_rects.back();
                if(cv::norm(r.tl()-roiRect[i].tl()) < euclidean_distance) /* Target tracked with Euclidean distance ... */
                    break;
            }
            if(i == roiRect.size()) { /* Target missing ... */
                if(m_frameTick - t->FrameTick() > MAX_NUM_FRAME_MISSING_TARGET) { /* Target still missing for over X frames */
#if 1            
                    Point p = t->m_points.back();
                    printf("lost target : %d, %d\n", p.x, p.y);
#endif
                    if(&(*t) == m_primaryTarget)
                        m_primaryTarget = 0;

                    t = m_targets.erase(t); /* Remove tracing target */
                    continue;
                }
            }
            t++;
        }

        for(int i=0; i<roiRect.size(); i++) { /* Try to find NEW target for tracking ... */
            list< Target >::iterator t;
            for(t=m_targets.begin();t!=m_targets.end();t++) {
                Rect r = t->m_rects.back();
                if((r & roiRect[i]).area() > 0) /* Next step tracked ... */
                    break;

                unsigned long f = m_frameTick - t->FrameTick();
                r.x += t->m_velocity.x * f;
                r.y += t->m_velocity.y * f;
                if((r & roiRect[i]).area() > 0) /* Next step tracked with velocity ... */
                    break;
#if 0
                if(cv::norm(r.tl()-roiRect[i].tl()) < euclidean_distance) /* Target tracked with velocity and Euclidean distance ... */
                    break;
#endif
                r = t->m_rects.back();
                if(cv::norm(r.tl()-roiRect[i].tl()) < euclidean_distance) /* Target tracked with Euclidean distance ... */                   
                    break;
            }
            if(t == m_targets.end()) { /* New target */
#if 0
                if(roiRect[i].y > 960)
                    continue;
#endif
                m_targets.push_back(Target(roiRect[i], m_frameTick));
#if 1            
                printf("new target : %d, %d\n", roiRect[i].tl().x, roiRect[i].tl().y);
#endif
            } else
                t->Update(roiRect[i], m_frameTick);
        }
        m_frameTick++;

        if(m_targets.size() > 1)
            m_targets.sort(TargetSort);

        if(m_targets.size() > 0) {
            if(m_primaryTarget == 0)
                m_primaryTarget = &m_targets.front();
        }
    }

    Target *PrimaryTarget() {
//        if(m_targets.size() == 0)
//            return 0;

//        m_targets.sort(TargetSort);
#if 0
        list< Target >::iterator t;
        for(t=m_targets.begin();t!=m_targets.end();t++) {
            if(t->ArcLength() > 320)
                return &(*t);
        }
        return 0;
#else
        return m_primaryTarget;
#endif
    }
};

class FrameQueue
{
public:
    struct cancelled {};

public:
    FrameQueue() : cancelled_(false) {}

    void push(Mat const & image);
    Mat pop();

    void cancel();
    bool isCancelled() { return cancelled_; }

private:
    std::queue<cv::Mat> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    bool cancelled_;
};

void FrameQueue::cancel()
{
    std::unique_lock<std::mutex> mlock(mutex_);
    cancelled_ = true;
    cond_.notify_all();
}

void FrameQueue::push(cv::Mat const & image)
{
    while(queue_.size() > 30) { /* Prevent memory overflow ... */
        usleep(1000); /* Wait for 1 ms */
    }

    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(image);
    cond_.notify_one();
}

Mat FrameQueue::pop()
{
    std::unique_lock<std::mutex> mlock(mutex_);

    while (queue_.empty()) {
        if (cancelled_) {
            throw cancelled();
        }
        cond_.wait(mlock);
        if (cancelled_) {
            throw cancelled();
        }
    }

    Mat image(queue_.front());
    queue_.pop();
    return image;
}

#if defined(VIDEO_OUTPUT_FILE)
FrameQueue videoWriterQueue;

void VideoWriterThread(int width, int height)
{    
    Size videoSize = Size((int)width,(int)height);
    VideoWriter writer;
    char filePath[64];
    int videoOutoutIndex = 0;
    while(videoOutoutIndex < 1000) {
        snprintf(filePath, 64, "%s/%s%03d.mp4", VIDEO_OUTPUT_DIR, VIDEO_OUTPUT_FILE, videoOutoutIndex);
        FILE *fp = fopen(filePath, "rb");
        if(fp) { /* file exist ... */
            fclose(fp);
            videoOutoutIndex++;
        } else
            break; /* File doesn't exist. OK */
    }
    writer.open(filePath, VideoWriter::fourcc('X', '2', '6', '4'), 30, videoSize);
    cout << "Vodeo output " << filePath << endl;
    try {
        while(1) {
            Mat frame = videoWriterQueue.pop();
            writer.write(frame);
        }
    } catch (FrameQueue::cancelled & /*e*/) {
        // Nothing more to process, we're done
        std::cout << "FrameQueue " << " cancelled, worker finished." << std::endl;
    }    
}

#endif

int main(int argc, char**argv)
{
    double fps = 0;

    if(signal(SIGINT, sig_handler) == SIG_ERR)
        printf("\ncan't catch SIGINT\n");

    Mat frame, capFrame;
    cuda::GpuMat gpuFrame;

    cuda::printShortCudaDeviceInfo(cuda::getDevice());
    std::cout << cv::getBuildInformation() << std::endl;

#ifdef VIDEO_INPUT_FILE
    VideoCapture cap(VIDEO_INPUT_FILE, cv::CAP_FFMPEG);
#else
    int index = 0;    
    if(argc > 1)
        index = atoi(argv[1]);
    VideoCapture cap(index);

    cout << "Video input (" << static_cast<int32_t>(cap.get(CAP_PROP_FRAME_WIDTH)) << "x" << static_cast<int32_t>(cap.get(CAP_PROP_FRAME_HEIGHT))
            << ") at " << cap.get(CAP_PROP_FPS) << " FPS." << endl;
#endif
    if(!cap.isOpened()) {
        cout << "Could not open video" << endl;
        return 1;
    }

#ifdef VIDEO_INPUT_FILE
    cap.read(capFrame);

    videoWidth = capFrame.cols;
    videoHeight = capFrame.rows;
#else
    cap.set(CAP_PROP_FOURCC ,VideoWriter::fourcc('M', 'J', 'P', 'G') );
    cap.set(CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
    cap.set(CAP_PROP_FPS, 30.0);

    cout << "Video input (" << static_cast<int32_t>(cap.get(CAP_PROP_FRAME_WIDTH)) << "x" << static_cast<int32_t>(cap.get(CAP_PROP_FRAME_HEIGHT))
        << ") at " << cap.get(CAP_PROP_FPS) << " FPS." << endl;
    cout << "Drop first " << VIDEO_FRAME_DROP << " for camera stable ..." << endl;
    for(int i=0;i<VIDEO_FRAME_DROP;i++) {
        if(!cap.read(frame))
            printf("Error read camera frame ...\n");
    }
#endif

#if defined(VIDEO_OUTPUT_FILE)
    thread outThread(&VideoWriterThread, capFrame.cols, capFrame.rows);
#endif
    //Ptr<BackgroundSubtractor> bsModel = createBackgroundSubtractorKNN();
    //Ptr<BackgroundSubtractor> bsModel = createBackgroundSubtractorMOG2();
    /* 30 : history, 16 : threshold */
    Ptr<cuda::BackgroundSubtractorMOG2> bsModel = cuda::createBackgroundSubtractorMOG2(90, 16, false);

    bool doUpdateModel = true;
    bool doSmoothMask = true;

    Mat foregroundMask, background;
#ifdef VIDEO_OUTPUT_SCREEN
    Mat outFrame;
#endif
    cuda::GpuMat gpuForegroundMask;
    Ptr<cuda::Filter> gaussianFilter;
    Ptr<cuda::Filter> erodeFilter;
    Ptr<cuda::Filter> erodeFilter2;

    Tracker tracker;
    Target *primaryTarget = 0;

    int cx, cy;
    while(1) {
        if(cap.read(capFrame))
            break;
        if(bShutdown)
            return 0;
    }

    cx = (capFrame.cols / 2) - 1;
    cy = capFrame.rows-1;

    high_resolution_clock::time_point t1(high_resolution_clock::now());

    while(cap.read(capFrame)) {
#if 0
        /* (contrast) alpha = 3.2, (brightness) beta = 50 */   
        capFrame.convertTo(capFrame, -1, 3.2, 50);
#endif
        cvtColor(capFrame, frame, COLOR_BGR2GRAY);
        //frame = capFrame;
#ifdef VIDEO_OUTPUT_SCREEN
        capFrame.copyTo(outFrame);
#ifdef VIDEO_INPUT_FILE
        line(outFrame, Point(cx, 0), Point(cx, cy), Scalar(0, 255, 0), 1);
#else
        line(outFrame, Point(0, cy), Point(cx, cy), Scalar(0, 255, 0), 1);
#endif //VIDEO_INPUT_FILE
#endif //VIDEO_OUTPUT_SCREEN
        int erosion_size = 6;   
        Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                          cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), 
                          cv::Point(erosion_size, erosion_size) );
#if 0 /* Very poor performance ... Running by CPU is 10 times quick */
        gpuFrame.upload(frame);
/*
        if(gpuFrame.channels() == 3) {
            cuda::GpuMat destMat;
            cuda::cvtColor(gpuFrame, destMat, COLOR_BGR2BGRA);
            gpuFrame = destMat;
        } 
*/
        if(erodeFilter.empty()) 
            erodeFilter = cuda::createMorphologyFilter(MORPH_ERODE, gpuFrame.type(), element);
        erodeFilter->apply(gpuFrame, gpuFrame);
#else
        erode(frame, frame, element);
        gpuFrame.upload(frame);	
#endif    
        // pass the frame to background bsModel
        bsModel->apply(gpuFrame, gpuForegroundMask, doUpdateModel ? -1 : 0);

        if(gaussianFilter.empty())
            gaussianFilter = cuda::createGaussianFilter(gpuForegroundMask.type(), gpuForegroundMask.type(), Size(5, 5), 3.5);

        // show foreground image and mask (with optional smoothing)
        if(doSmoothMask) {
            gaussianFilter->apply(gpuForegroundMask, gpuForegroundMask);
            //cuda::threshold(gpuForegroundMask, gpuForegroundMask, 10.0, 255.0, THRESH_BINARY);
            
			/* Erode & Dilate */
            int erosion_size = 6;   
            Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                          cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), 
                          cv::Point(erosion_size, erosion_size) );
#if 0
        if(erodeFilter2.empty()) 
            erodeFilter2 = cuda::createMorphologyFilter(MORPH_ERODE, gpuForegroundMask.type(), element);
        erodeFilter2->apply(gpuForegroundMask, gpuForegroundMask);
        gpuForegroundMask.download(foregroundMask);
#else
            gpuForegroundMask.download(foregroundMask);
            erode(foregroundMask, foregroundMask, element);
#endif
        } else
            gpuForegroundMask.download(foregroundMask);

		vector< vector<Point> > contours;
    	vector< Vec4i > hierarchy;
//    	findContours(foregroundMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
        findContours(foregroundMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        sort(contours.begin(), contours.end(), ContoursSort); /* Contours sort by area, controus[0] is largest */

        vector<Rect> boundRect( contours.size() );
        vector<Rect> roiRect;

        RNG rng(12345);

    	for(int i=0; i<contours.size(); i++) {
    		approxPolyDP( Mat(contours[i]), contours[i], 3, true );
       		boundRect[i] = boundingRect( Mat(contours[i]) );
       		//drawContours(contoursImg, contours, i, color, 2, 8, hierarchy);
    		if(boundRect[i].width > MIN_TARGET_WIDTH && 
                boundRect[i].height > MIN_TARGET_HEIGHT &&
    			boundRect[i].width <= MAX_TARGET_WIDTH && 
                boundRect[i].height <= MAX_TARGET_HEIGHT) {
                    roiRect.push_back(boundRect[i]);
#ifdef VIDEO_OUTPUT_SCREEN
                    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                    rectangle( outFrame, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
#endif
    		}
            if(roiRect.size() >= MAX_NUM_TARGET) /* Deal ROI with largest area only */
                break;
    	}

        tracker.Update(roiRect);

        primaryTarget = tracker.PrimaryTarget();
        if(primaryTarget) {
#ifdef VIDEO_OUTPUT_SCREEN
            Rect r = primaryTarget->m_rects.back();
            rectangle( outFrame, r.tl(), r.br(), Scalar( 255, 0, 0 ), 2, 8, 0 );

            if(primaryTarget->m_points.size() > 1) { /* Minimum 2 points ... */
                for(int i=0;i<primaryTarget->m_points.size()-1;i++) {
                    line(outFrame, primaryTarget->m_points[i], primaryTarget->m_points[i+1], Scalar(0, 255, 255), 1);
                }
            }
#endif
            if(primaryTarget->ArcLength() > MIN_COURSE_LENGTH && 
                primaryTarget->RectCount() > MIN_TARGET_TRACKED_COUNT) {
#ifdef VIDEO_INPUT_FILE
                if((primaryTarget->m_points[0].x > cx && primaryTarget->LatestPoint().x <= cx) ||
                    (primaryTarget->m_points[0].x < cx && primaryTarget->LatestPoint().x >= cx)) {
#else
                if((primaryTarget->m_points[0].y > cy && primaryTarget->LatestPoint().y <= cy) ||
                    (primaryTarget->m_points[0].y < cy && primaryTarget->LatestPoint().y >= cy)) {
#endif //VIDEO_INPUT_FILE
                    if(primaryTarget->TriggerCount() < MAX_NUM_TRIGGER) { /* Triggle 4 times maximum  */
#ifdef VIDEO_OUTPUT_SCREEN
#ifdef VIDEO_INPUT_FILE
                        line(outFrame, Point(cx, 0), Point(cx, cy), Scalar(0, 0, 255), 3);
#else
                        line(outFrame, Point(0, cy), Point(cx, cy), Scalar(0, 0, 255), 3);
#endif //VIDEO_INPUT_FILE
#endif //VIDEO_OUTPUT_FRAME
                        printf("T R I G G E R - %d\n", primaryTarget->TriggerCount());
                        primaryTarget->Trigger();
                    }
                }
            }
        }
#if 1
        resize(frame, frame, Size(frame.cols * 3 / 4, frame.rows * 3 / 4));
        imshow("erode", frame);
#endif
#if 1
        resize(foregroundMask, foregroundMask, Size(foregroundMask.cols * 3 / 4, foregroundMask.rows * 3 / 4));
        imshow("foreground mask", foregroundMask);
#endif
#if 0
        bsModel->getBackgroundImage(gpuForegroundMask);

        gpuForegroundMask.download(background);

        if(!background.empty()) {
            resize(background, background, Size(background.cols * 3 / 4, background.rows * 3 / 4));
            imshow("mean background image", background );
        }
#endif
#ifdef VIDEO_OUTPUT_SCREEN
        char str[32];
        snprintf(str, 32, "FPS : %.2lf", fps);
        writeText(outFrame, string(str));
#if defined(VIDEO_OUTPUT_FILE)
        videoWriterQueue.push(outFrame.clone());
#endif        
        resize(outFrame, outFrame, Size(outFrame.cols * 3 / 4, outFrame.rows * 3 / 4));
        namedWindow("Out Frame", WINDOW_AUTOSIZE);
        imshow("Out Frame", outFrame);
#endif
#ifdef VIDEO_OUTPUT_SCREEN
        int k = waitKey(1);
        if(k == 27) {
            break;
        } else if(k == 'p') {
            while(waitKey(1) != 'p') {
                if(bShutdown)
                    break;
            }
        }
#endif
        if(bShutdown)
            break;

        high_resolution_clock::time_point t2(high_resolution_clock::now());
        double dt_us(static_cast<double>(duration_cast<microseconds>(t2 - t1).count()));
        //std::cout << (dt_us / 1000.0) << " ms" << std::endl;
        fps = (1000000.0 / dt_us);
        std::cout << "FPS : " << fixed  << setprecision(2) <<  fps << std::endl;

        t1 = high_resolution_clock::now();
    }

    cap.release();

#if defined(VIDEO_OUTPUT_FILE)
    videoWriterQueue.cancel();
    outThread.join();
#endif

    std::cout << "Finished ..." << endl;

    return 0;     
}
