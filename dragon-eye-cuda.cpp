#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

//#include <opencv2/cudacodec.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaarithm.hpp>

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

//#define VIDEO_INPUT_FILE "../video/baseA000.mkv"
//#define VIDEO_INPUT_FILE "../video/baseA011.mkv"
//#define VIDEO_INPUT_FILE "../video/baseA017.mkv"
//#define VIDEO_INPUT_FILE "../video/baseB000.mkv"
#define VIDEO_INPUT_FILE "../video/baseB001.mkv"
//#define VIDEO_INPUT_FILE "../video/baseB002.mkv"
//#define VIDEO_INPUT_FILE "../video/baseB004.mkv"
//#define VIDEO_INPUT_FILE "../video/baseB030.mkv"
//#define VIDEO_INPUT_FILE "../video/baseB036.mkv"
//#define VIDEO_INPUT_FILE "../video/baseB042.mkv"
//#define VIDEO_INPUT_FILE "../video/baseB044.mkv"
//#define VIDEO_INPUT_FILE "../video/baseB048.mkv"

#define VIDEO_OUTPUT_SCREEN
//#define VIDEO_OUTPUT_FILE "base"
#define VIDEO_OUTPUT_DIR "."

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
#if 0
        if(m_triggerCount >= MAX_NUM_TRIGGER)
            Reset();
#endif
    }

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

void contour_moving_object(Mat & foregroundMask, vector<Rect> & roiRect, int y_offset = 0)
{
    uint32_t num_target = 0;

    vector< vector<Point> > contours;
    vector< Vec4i > hierarchy;
//  findContours(foregroundMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    findContours(foregroundMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), ContoursSort); /* Contours sort by area, controus[0] is largest */

    vector<Rect> boundRect( contours.size() );
    for(int i=0; i<contours.size(); i++) {
        approxPolyDP( Mat(contours[i]), contours[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours[i]) );
        //drawContours(contoursImg, contours, i, color, 2, 8, hierarchy);
        if(boundRect[i].width > MIN_TARGET_WIDTH && 
            boundRect[i].height > MIN_TARGET_HEIGHT &&
            boundRect[i].width <= MAX_TARGET_WIDTH && 
            boundRect[i].height <= MAX_TARGET_HEIGHT) {
                boundRect[i].y += y_offset;
                roiRect.push_back(boundRect[i]);
                if(++num_target >= MAX_NUM_TARGET)
                    break;
        }
    }    
}

void extract_moving_object(Mat & frame, Mat & element, Ptr<cuda::Filter> & erodeFilter, Ptr<cuda::Filter> & gaussianFilter, 
    Ptr<cuda::BackgroundSubtractorMOG2> & bsModel, vector<Rect> & roiRect, int y_offset = 0)
{
    Mat foregroundMask;
    cuda::GpuMat gpuFrame;
    cuda::GpuMat gpuForegroundMask;
#if 0 /* Very poor performance ... Running by CPU is 10 times quick */
    gpuFrame.upload(frame);
    erodeFilter->apply(gpuFrame, gpuFrame);
#else
    erode(frame, frame, element);
    gpuFrame.upload(frame);
#endif    
    // pass the frame to background bsGrayModel
    bsModel->apply(gpuFrame, gpuForegroundMask, -1);
    gaussianFilter->apply(gpuForegroundMask, gpuForegroundMask);
    //cuda::threshold(gpuForegroundMask, gpuForegroundMask, 10.0, 255.0, THRESH_BINARY);
#if 0 /* Very poor performance ... Running by CPU is 10 times quick */
    erodeFilter->apply(gpuForegroundMask, gpuForegroundMask);
    gpuForegroundMask.download(foregroundMask);
#else
    gpuForegroundMask.download(foregroundMask);
    erode(foregroundMask, foregroundMask, element);
#endif

    contour_moving_object(foregroundMask, roiRect, y_offset);
}

int main(int argc, char**argv)
{
    double fps = 0;

    if(signal(SIGINT, sig_handler) == SIG_ERR)
        printf("\ncan't catch SIGINT\n");

    Mat capFrame;
    //cuda::GpuMat gpuFrame;
#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
    Mat outFrame;
#endif

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
#else
    cap.set(CAP_PROP_FOURCC ,VideoWriter::fourcc('M', 'J', 'P', 'G') );
    cap.set(CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
    cap.set(CAP_PROP_FPS, 30.0);

    cout << "Video input (" << static_cast<int32_t>(cap.get(CAP_PROP_FRAME_WIDTH)) << "x" << static_cast<int32_t>(cap.get(CAP_PROP_FRAME_HEIGHT))
        << ") at " << cap.get(CAP_PROP_FPS) << " FPS." << endl;
    cout << "Drop first " << VIDEO_FRAME_DROP << " for camera stable ..." << endl;
    for(int i=0;i<VIDEO_FRAME_DROP;i++) {
        if(!cap.read(capFrame))
            printf("Error read camera frame ...\n");
    }
#endif

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

#if defined(VIDEO_OUTPUT_FILE)
    thread outThread(&VideoWriterThread, capFrame.cols, capFrame.rows);
#endif

    int erosion_size = 6;   
    Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                    cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), 
                    cv::Point(-1, -1) ); /* Default anchor point */

    Ptr<cuda::Filter> erodeFilter1 = cuda::createMorphologyFilter(MORPH_ERODE, CV_8UC1, element);
    Ptr<cuda::Filter> erodeFilter2 = cuda::createMorphologyFilter(MORPH_ERODE, CV_8UC1, element);
    Ptr<cuda::Filter> gaussianFilter1 = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 5.0);
    Ptr<cuda::Filter> gaussianFilter2 = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 5.0);
    Ptr<cuda::BackgroundSubtractorMOG2> bsModel1 = cuda::createBackgroundSubtractorMOG2(90, 48, false);
    Ptr<cuda::BackgroundSubtractorMOG2> bsModel2 = cuda::createBackgroundSubtractorMOG2(90, 48, false);

    high_resolution_clock::time_point t1(high_resolution_clock::now());

    while(cap.read(capFrame)) {
#if 0
        /* (contrast) alpha = 2.2, (brightness) beta = 50 */   
        capFrame.convertTo(capFrame, -1, 2.2, 50);
#endif

#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
        capFrame.copyTo(outFrame);
#ifdef VIDEO_INPUT_FILE
        line(outFrame, Point(cx, 0), Point(cx, cy), Scalar(0, 255, 0), 1);
#else
        line(outFrame, Point(0, cy), Point(cx, cy), Scalar(0, 255, 0), 1);
#endif //VIDEO_INPUT_FILE
#endif //VIDEO_OUTPUT_SCREEN

        vector<Rect> roiRect;
#if 0
        Mat grayFrame, roiFrame; 
        cvtColor(capFrame, grayFrame, COLOR_BGR2GRAY);
        thread th1(extract_moving_object, std::ref(grayFrame), std::ref(element), std::ref(erodeFilter1), std::ref(gaussianFilter1), std::ref(bsModel1), std::ref(roiRect), 0);
        th1.detach();

        Mat hsvFrame;
        int y_offset = capFrame.rows * 2 / 3;
        capFrame(Rect(0, y_offset, capFrame.cols, capFrame.rows - y_offset)).copyTo(roiFrame);
        cvtColor(roiFrame, hsvFrame, COLOR_BGR2HSV);
        Mat hsvCh[3];
        split(hsvFrame, hsvCh);
        extract_moving_object(hsvCh[0], element, erodeFilter2, gaussianFilter2, bsModel2, roiRect, y_offset);

        if(th1.joinable())
            th1.join();
#else
        /* Gray color space for whole region */

        Mat grayFrame, roiFrame;
        cvtColor(capFrame, grayFrame, COLOR_BGR2GRAY);
//imshow("GRAY frame", grayFrame);
        extract_moving_object(grayFrame, element, erodeFilter1, gaussianFilter1, bsModel1, roiRect);

        /* HSV color space Hue channel for bottom 1/3 region */

        Mat hsvFrame;
        int y_offset = capFrame.rows * 2 / 3;
        capFrame(Rect(0, y_offset, capFrame.cols, capFrame.rows - y_offset)).copyTo(roiFrame);
        cvtColor(roiFrame, hsvFrame, COLOR_BGR2HSV);
        Mat hsvCh[3];
        split(hsvFrame, hsvCh);
        extract_moving_object(hsvCh[0], element, erodeFilter2, gaussianFilter2, bsModel2, roiRect, y_offset);
#endif

#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
        RNG rng(12345);
        for(int i=0;i<roiRect.size();i++) {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            rectangle( outFrame, roiRect[i].tl(), roiRect[i].br(), color, 2, 8, 0 );
        }
#endif

        tracker.Update(roiRect);

        primaryTarget = tracker.PrimaryTarget();
        if(primaryTarget) {
#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
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
#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
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
#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
        char str[32];
        snprintf(str, 32, "FPS : %.2lf", fps);
        writeText(outFrame, string(str));
#if defined(VIDEO_OUTPUT_FILE)
        videoWriterQueue.push(outFrame.clone());
#endif
#ifdef VIDEO_OUTPUT_SCREEN
        resize(outFrame, outFrame, Size(outFrame.cols * 3 / 4, outFrame.rows * 3 / 4));
        namedWindow("Out Frame", WINDOW_AUTOSIZE);
        imshow("Out Frame", outFrame);
#endif
#endif
#ifdef VIDEO_OUTPUT_SCREEN
        int k = waitKey(1);
        if(k == 27) { /* Press key 'ESC' to quit */
            break;
        } else if(k == 'p') { /* Press key 'p' to pause or resume */
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
