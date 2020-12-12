#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

//#include <opencv2/cudacodec.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

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
#include <list>

//using std::chrono::high_resolution_clock;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

#define CAMERA_WIDTH 1280
#define CAMERA_HEIGHT 720
#define CAMERA_FPS 60
#define MIN_TARGET_WIDTH 8
#define MIN_TARGET_HEIGHT 8
#define MAX_TARGET_WIDTH 320
#define MAX_TARGET_HEIGHT 320

#define MAX_NUM_TARGET 6
#define MAX_NUM_TRIGGER 4
#define MAX_NUM_FRAME_MISSING_TARGET 6

#define MIN_COURSE_LENGTH            120    /* Minimum course length of RF trigger after detection of cross line */
#define MIN_TARGET_TRACKED_COUNT     3      /* Minimum target tracked count of RF trigger after detection of cross line */

#define VIDEO_INPUT_FILE

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

static inline bool ContoursSortByArea(vector<cv::Point> contour1, vector<cv::Point> contour2)  
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

    vector< Rect > m_rects;
    vector< Point > m_vectors;
    Point m_velocity;

    Point center(Rect & r) {
        return Point(r.tl().x + (r.width / 2), r.tl().y + (r.height / 2));
    }

public:
    Target(Rect & roi, unsigned long frameTick) : m_arcLength(0), m_frameTick(frameTick), m_triggerCount(0) {
        m_rects.push_back(roi);
        m_frameTick = frameTick;
    }

    void Reset() {
        Rect r = m_rects.back();
        m_rects.clear();
        m_rects.push_back(r);
        m_triggerCount = 0;
        //m_frameTick =
        //m_arcLength = 0; /* TODO : Do clear this ? */
    }

    void Update(Rect & roi, unsigned long frameTick) {
        if(m_rects.size() > 0)
            m_arcLength += norm(roi.tl() - m_rects.back().tl());
#if 1
        if(m_rects.size() == 1) {
            m_velocity.x = (roi.tl().x - m_rects.back().tl().x);
            m_velocity.y = (roi.tl().y - m_rects.back().tl().y);
        } else if(m_rects.size() > 1) {
            m_velocity.x = (m_velocity.x + (roi.tl().x - m_rects.back().tl().x)) / 2;
            m_velocity.y = (m_velocity.y + (roi.tl().y - m_rects.back().tl().y)) / 2;
        }
#else
        if(m_rects.size() >= 1) {
            m_velocity.x = (roi.tl().x - m_rects.back().tl().x);
            m_velocity.y = (roi.tl().y - m_rects.back().tl().y);
        }
#endif
        if(m_rects.size() >= 1) {
            Point p = roi.tl() - m_rects.back().tl();
            m_vectors.push_back(p);
        }

        m_rects.push_back(roi);
        m_frameTick = frameTick;
#if 0
        if(m_triggerCount >= MAX_NUM_TRIGGER)
            Reset();
#endif
    }

    int DotProduct(Point p) {
        size_t i = m_rects.size();
        if(i < 2)
            return 0;
        i--;
        Point v[2];
        v[0].x = p.x - m_rects[i].tl().x;
        v[0].y = p.y - m_rects[i].tl().y;
        v[1].x = m_rects[i].tl().x - m_rects[i-1].tl().x;
        v[1].y = m_rects[i].tl().y - m_rects[i-1].tl().y;
        int dp = v[0].x * v[1].x + v[0].y * v[1].y;
        return dp;
    }

    double CosineAngle(Point p) {
        size_t i = m_rects.size();
        if(i < 2)
            return 0;

        i--;
        Point v1, v2;
        v1.x = p.x - m_rects[i].tl().x;
        v1.y = p.y - m_rects[i].tl().y;
        v2.x = m_rects[i].tl().x - m_rects[i-1].tl().x;
        v2.y = m_rects[i].tl().y - m_rects[i-1].tl().y;

        /* A.B = |A||B|cos() */
        /* cos() = A.B / |A||B| */

        return v1.dot(v2) / (norm(v1) * norm(v1));
    }

    void Draw(Mat & outFrame) {
        Rect r = LastRect();
        rectangle( outFrame, r.tl(), r.br(), Scalar( 255, 0, 0 ), 2, 8, 0 );

        if(m_rects.size() > 1) { /* Minimum 2 points ... */
            for(int i=0;i<m_rects.size()-1;i++) {
                Point p0 = m_rects[i].tl();
                Point p1 = m_rects[i+1].tl();
                line(outFrame, p0, p1, Scalar(0, 0, 255), 1);
                //Point v = p1 - p0;
                //printf("[%d,%d]\n", v.x, v.y);
            }
        }        
    }

    inline double ArcLength() { return m_arcLength; }
    inline unsigned long FrameTick() { return m_frameTick; }
    inline Rect & LastRect() { return m_rects.back(); }
#if 0    
    inline Point BeginPoint() { return m_rects[0].tl(); }
    inline Point EndPoint() { return m_rects.back().tl(); }
#else
    inline Point BeginPoint() { return center(m_rects[0]); }
    inline Point EndPoint() { return center(m_rects.back()); }
#endif    
    inline void Trigger() { m_triggerCount++; }
    inline uint8_t TriggerCount() { return m_triggerCount; }
    //inline Point & Velocity() { return m_velocity; }
    inline size_t RectCount() { return m_rects.size(); }

    friend class Tracker;
};

static inline bool TargetSortByArea(Target & a, Target & b)
{
    return a.LastRect().area() > b.LastRect().area();
}

class Tracker
{
private:
    unsigned long m_frameTick;
    list< Target > m_targets;

public:
    Tracker() : m_frameTick(0) {}

    void Update(list< Rect > & roiRect) {
        const int euclidean_distance = 120;

        for(list< Target >::iterator t=m_targets.begin();t!=m_targets.end();) { /* Try to find lost targets */
            list<Rect>::iterator rr;
            for(rr=roiRect.begin();rr!=roiRect.end();rr++) {
                Rect r = t->m_rects.back();
                if((r & *rr).area() > 0) { /* Target tracked ... */
                    //if(t->DotProduct(rr->tl()) >= 0) /* Two vector less than 90 degree */
                        break;                
                }

                unsigned long f = m_frameTick - t->FrameTick();
                r.x += t->m_velocity.x * f;
                r.y += t->m_velocity.y * f;
                if((r & *rr).area() > 0) { /* Target tracked with velocity ... */
                    //if(t->DotProduct(rr->tl()) >= 0) /* Two vector less than 90 degree */
                        break;
                }

                if(cv::norm(r.tl()-rr->tl()) < euclidean_distance) { /* Target tracked with velocity and Euclidean distance ... */
                    if(t->DotProduct(rr->tl()) >= 0) /* Two vector less than 90 degree */
                        break;
                }
#if 0
                r = t->m_rects.back();
                if(cv::norm(r.tl()-rr->tl()) < euclidean_distance) { /* Target tracked with Euclidean distance ... */
                    if(t->DotProduct(rr->tl()) >= 0) /* Two vector less than 90 degree */
                        break;
                }
#endif
            }
            if(rr == roiRect.end()) { /* Target missing ... */
                bool ignoreMissingTarget = false;
                if(t->m_rects.size() <= MAX_NUM_FRAME_MISSING_TARGET) { 
                    for(int j=0;j<t->m_vectors.size();j++) {
                        Point p = t->m_vectors[j];
                        if(p.x == 0 && p.y == 0) { /* With none moving behavior. Maybe fake signal ... */
                            ignoreMissingTarget = true;
                            break;
                        }
                    }
                }
                if(m_frameTick - t->FrameTick() > MAX_NUM_FRAME_MISSING_TARGET || /* Target still missing for over X frames */
                    ignoreMissingTarget) { 
#if 1            
                    Point p = t->m_rects.back().tl();
                    //printf("lost target : %d, %d\n", p.x, p.y);
                    printf("lost target : %d, %d\n", t->m_velocity.x, t->m_velocity.y);
#endif          
                    t = m_targets.erase(t); /* Remove tracing target */
                    continue;
                }
            } else { /* Target tracked ... */
                t->Update(*rr, m_frameTick);
                roiRect.erase(rr);
            }
            t++;
        }

        for(list<Rect>::iterator rr=roiRect.begin();rr!=roiRect.end();rr++) {    
            m_targets.push_back(Target(*rr, m_frameTick));
#if 1            
            printf("new target : %d, %d\n", rr->tl().x, rr->tl().y);
#endif
        }

        m_frameTick++;

        if(m_targets.size() > 1)
            m_targets.sort(TargetSortByArea);
    }

    inline list< Target > & TargetList() { return m_targets; }
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

void contour_moving_object(Mat & frame, Mat & foregroundMask, list<Rect> & roiRect, int y_offset = 0)
{
    uint32_t num_target = 0;

    vector< vector<Point> > contours;
    vector< Vec4i > hierarchy;
//  findContours(foregroundMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    findContours(foregroundMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), ContoursSortByArea); /* Contours sort by area, controus[0] is largest */

    vector<Rect> boundRect( contours.size() );
    for(int i=0; i<contours.size(); i++) {
        //approxPolyDP( Mat(contours[i]), contours[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours[i]) );
        //drawContours(contoursImg, contours, i, color, 2, 8, hierarchy);
        if(boundRect[i].width > MAX_TARGET_WIDTH &&
            boundRect[i].height > MAX_TARGET_HEIGHT)
            continue; /* Extremely large object */

        if(boundRect[i].width < MIN_TARGET_WIDTH && 
            boundRect[i].height < MIN_TARGET_HEIGHT)
            break; /* Rest are small objects, ignore them */

#if 1 /* Anti cloud ... */
        double minVal; 
        double maxVal; 
        Point minLoc; 
        Point maxLoc;

        Mat roiFrame = frame(boundRect[i]);
        minMaxLoc(roiFrame, &minVal, &maxVal, &minLoc, &maxLoc ); 
            /* If difference of max and min value of ROI rect is too small then it could be noise such as cloud or sea */
        if((maxVal - minVal) < 24)
            continue; /* Too small, drop it. */
#endif
#if 1
        if(roiFrame.cols > roiFrame.rows && (roiFrame.cols >> 4) > roiFrame.rows)
            continue; /* Ignore thin object */
#endif                        
        boundRect[i].y += y_offset;
        roiRect.push_back(boundRect[i]);
        if(++num_target >= MAX_NUM_TARGET)
            break;
    }    

}

void extract_moving_object(Mat & frame, 
    Mat & elementErode, Mat & elementDilate, 
    Ptr<cuda::Filter> & erodeFilter, Ptr<cuda::Filter> & dilateFilter, Ptr<cuda::Filter> & gaussianFilter, 
    Ptr<cuda::BackgroundSubtractorMOG2> & bsModel, 
    list<Rect> & roiRect, int y_offset = 0)
{
    Mat foregroundMask;
    cuda::GpuMat gpuFrame;
    cuda::GpuMat gpuForegroundMask;

    gpuFrame.upload(frame); 
    // pass the frame to background bsGrayModel
    //gaussianFilter->apply(gpuFrame, gpuFrame);
    bsModel->apply(gpuFrame, gpuForegroundMask, 0.05);
    //cuda::threshold(gpuForegroundMask, gpuForegroundMask, 10.0, 255.0, THRESH_BINARY);
#if 1 /* Run with GPU */
    erodeFilter->apply(gpuForegroundMask, gpuForegroundMask);
    dilateFilter->apply(gpuForegroundMask, gpuForegroundMask);
    gpuForegroundMask.download(foregroundMask);
#else /* Run with CPU */
    gpuForegroundMask.download(foregroundMask);
    morphologyEx(foregroundMask, foregroundMask, MORPH_ERODE, elementErode);
    morphologyEx(foregroundMask, foregroundMask, MORPH_DILATE, elementDilate);
#endif

#ifdef VIDEO_OUTPUT_SCREEN
        Mat tFrame;
        foregroundMask.copyTo(tFrame);
        resize(tFrame, tFrame, Size(tFrame.cols * 3 / 4, tFrame.rows * 3 / 4));
        namedWindow("FG Frame", WINDOW_AUTOSIZE);
        imshow("FG Frame", tFrame);
#endif

    contour_moving_object(frame, foregroundMask, roiRect, y_offset);
}

int main(int argc, char**argv)
{
    double fps = 0;

    if(signal(SIGINT, sig_handler) == SIG_ERR)
        printf("\ncan't catch SIGINT\n");

    Mat capFrame, bgFrame;
#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
    Mat outFrame;
#endif

    cuda::printShortCudaDeviceInfo(cuda::getDevice());
    std::cout << cv::getBuildInformation() << std::endl;

#ifdef VIDEO_INPUT_FILE
    VideoCapture cap;
    if(argc > 1)
        cap.open(argv[1], cv::CAP_FFMPEG);
    else {
        cout << "Usage : dragon-eye-cuda <vidoe file>" << endl;
        return 0;
    }
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

    int erosion_size = 1;   
    Mat elementErode = cv::getStructuringElement(cv::MORPH_RECT,
                    cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), 
                    cv::Point(-1, -1) ); /* Default anchor point */

    int dilate_size = 2;   
    Mat elementDilate = cv::getStructuringElement(cv::MORPH_RECT,
                    cv::Size(2 * dilate_size + 1, 2 * dilate_size + 1), 
                    cv::Point(-1, -1) ); /* Default anchor point */

    Ptr<cuda::Filter> erodeFilter = cuda::createMorphologyFilter(MORPH_ERODE, CV_8UC1, elementErode);
    Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, CV_8UC1, elementDilate);

    /* background history count, varThreshold, shadow detection */
    Ptr<cuda::BackgroundSubtractorMOG2> bsModel = cuda::createBackgroundSubtractorMOG2(30, 64, false);
    //cout << bsModel->getVarInit() << " / " << bsModel->getVarMax() << " / " << bsModel->getVarMax() << endl;
    /* Default variance of each gaussian component 15 / 75 / 75 */ 
    bsModel->setVarInit(15);
    bsModel->setVarMax(20);
    bsModel->setVarMin(4);
    Ptr<cuda::Filter> gaussianFilter = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(5, 5), 0);

    steady_clock::time_point t1(steady_clock::now());

    int dropCount = 30;
    while(dropCount-- > 0)
        cap.read(capFrame);

    cvtColor(capFrame, bgFrame, COLOR_BGR2GRAY);

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

        list<Rect> roiRect;
        /* Gray color space for whole region */
        Mat grayFrame, roiFrame;
        cvtColor(capFrame, grayFrame, COLOR_BGR2GRAY);

        extract_moving_object(grayFrame, elementErode, elementDilate, erodeFilter, dilateFilter, gaussianFilter, bsModel, roiRect);

#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
        RNG rng(12345);
        for(list<Rect>::iterator rr=roiRect.begin();rr!=roiRect.end();rr++) {
            //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            rectangle( outFrame, rr->tl(), rr->br(), Scalar(0, 255, 0), 2, 8, 0 );
        }
#endif

        tracker.Update(roiRect);

        list< Target > & targets = tracker.TargetList();

        for(list< Target >::iterator t=targets.begin();t!=targets.end();t++) {
#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
            t->Draw(outFrame);
#endif
            if(t->ArcLength() > MIN_COURSE_LENGTH && 
                t->RectCount() > MIN_TARGET_TRACKED_COUNT) {
#ifdef VIDEO_INPUT_FILE
                if((t->BeginPoint().x > cx && t->EndPoint().x <= cx) ||
                    (t->BeginPoint().x < cx && t->EndPoint().x >= cx)) {
#else
                if((t->BeginPoint().y > cy && t->EndPoint().y <= cy) ||
                    (t->BeginPoint().y < cy && t->EndPoint().y >= cy)) {
#endif //VIDEO_INPUT_FILE
                    if(t->TriggerCount() < MAX_NUM_TRIGGER) { /* Triggle 4 times maximum  */
#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
#ifdef VIDEO_INPUT_FILE
                        line(outFrame, Point(cx, 0), Point(cx, cy), Scalar(0, 0, 255), 3);
#else
                        line(outFrame, Point(0, cy), Point(cx, cy), Scalar(0, 0, 255), 3);
#endif //VIDEO_INPUT_FILE
#endif //VIDEO_OUTPUT_FRAME
                        printf("T R I G G E R - %d\n", t->TriggerCount());
                        t->Trigger();
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

        steady_clock::time_point t2(steady_clock::now());
        double dt_us(static_cast<double>(duration_cast<microseconds>(t2 - t1).count()));
#if 1
        while(dt_us < 33000) {
            usleep(1000);
            t2 = steady_clock::now();
            dt_us = static_cast<double>(duration_cast<microseconds>(t2 - t1).count());
        }
#endif
        //std::cout << (dt_us / 1000.0) << " ms" << std::endl;
        fps = (1000000.0 / dt_us);
        std::cout << "FPS : " << fixed  << setprecision(2) <<  fps << std::endl;

        t1 = steady_clock::now();
    }

    cap.release();

#if defined(VIDEO_OUTPUT_FILE)
    videoWriterQueue.cancel();
    outThread.join();
#endif

    std::cout << "Finished ..." << endl;

    return 0;     
}
