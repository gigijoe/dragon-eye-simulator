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

#define MAX_NUM_TARGET                  6
#define MAX_NUM_TRIGGER                 1
#define MAX_NUM_FRAME_MISSING_TARGET    6

#define MIN_COURSE_LENGTH               120    /* Minimum course length of RF trigger after detection of cross line */
#define MIN_TARGET_TRACKED_COUNT        4      /* Minimum target tracked count of RF trigger after detection of cross line (4 * 33ms = 132ms) */

#undef NEW_TARGET_RESTRICTION

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

const std::string currentDateTime() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%H:%M:%S %m/%d", &tstruct);
    return buf;
}

/*
*
*/

inline void writeText( Mat & mat, const string text, const Point textOrg)
{
   int fontFace = FONT_HERSHEY_SIMPLEX;
   double fontScale = 1;
   int thickness = 1;  
   //Point textOrg( 10, 40 );
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
    double m_maxVector, m_minVector;
    Point m_velocity;

    Point Center(Rect & r) {
        return Point(r.tl().x + (r.width / 2), r.tl().y + (r.height / 2));
    }

    double CosineAngle(const Point & p1, const Point & p2, const Point & p3) {
        Point v1, v2;
        v1.x = p1.x - p2.x;
        v1.y = p1.y - p2.y;
        v2.x = p2.x - p3.x;
        v2.y = p2.y - p3.y;

        /* A.B = |A||B|cos() */
        /* cos() = A.B / |A||B| */
        return v1.dot(v2) / (norm(v1) * norm(v2));
    }

public:
    Target(Rect & roi, unsigned long frameTick) : m_arcLength(0), m_frameTick(frameTick), m_triggerCount(0), m_maxVector(0), m_minVector(0) {
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
        if(m_rects.size() > 0) {
            Point p = roi.tl() - m_rects.back().tl();
            double v = norm(p);

            m_arcLength += v;
            m_vectors.push_back(p);

            if(m_rects.size() == 1) {
                m_maxVector = v;
                m_minVector = v;
            } else if(v > m_maxVector)
                m_maxVector = v;
            else if(v < m_minVector)
                m_minVector = v;
        }
#if 0
        if(m_rects.size() == 1) {
            m_velocity.x = (roi.tl().x - m_rects.back().tl().x);
            m_velocity.y = (roi.tl().y - m_rects.back().tl().y);
        } else if(m_rects.size() > 1) {
            m_velocity.x = (m_velocity.x + (roi.tl().x - m_rects.back().tl().x)) / 2;
            m_velocity.y = (m_velocity.y + (roi.tl().y - m_rects.back().tl().y)) / 2;
        }
#else
        if(m_rects.size() >= 1) {
            int histroy_count = 3;
            if(m_rects.size() < histroy_count) {
                m_velocity.x = (roi.tl().x - m_rects.back().tl().x);
                m_velocity.y = (roi.tl().y - m_rects.back().tl().y);
            } else {            
                int c = 0;
                for(vector< Point >::reverse_iterator rit=m_vectors.rbegin();rit!=m_vectors.rend();++rit) {
                    m_velocity.x = (m_velocity.x + rit->x) / 2;
                    m_velocity.y = (m_velocity.y + rit->y) / 2;
                    if(++c >= histroy_count)
                        break;
                }
            }
        }
#endif
        m_rects.push_back(roi);
        m_frameTick = frameTick;
#if 0
        if(m_triggerCount >= MAX_NUM_TRIGGER)
            Reset();
#endif
    }

    int DotProduct(const Point & p) {
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

    double CosineAngle(const Point & p) {
        size_t i = m_rects.size();
        if(i < 2)
            return 0;
        --i;
        Point v1, v2;
        v1.x = p.x - m_rects[i].tl().x;
        v1.y = p.y - m_rects[i].tl().y;
        v2.x = m_rects[i].tl().x - m_rects[i-1].tl().x;
        v2.y = m_rects[i].tl().y - m_rects[i-1].tl().y;

        /* A.B = |A||B|cos() */
        /* cos() = A.B / |A||B| */
        return v1.dot(v2) / (norm(v1) * norm(v2));
    }

    void Draw(Mat & outFrame, bool drawAll = false) {
        Rect r = LastRect();
        rectangle( outFrame, r.tl(), r.br(), Scalar( 255, 0, 0 ), 1, 8, 0 );

        //RNG rng(12345);
        if(m_rects.size() > 1) { /* Minimum 2 points ... */
            for(int i=0;i<m_rects.size()-1;i++) {
                Point p0 = m_rects[i].tl();
                Point p1 = m_rects[i+1].tl();
                line(outFrame, p0, p1, Scalar(0, 0, 255), 1);
                //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                //line(outFrame, p0, p1, color, 1);
                //Point v = p1 - p0;
                //printf("[%d,%d]\n", v.x, v.y);
                if(drawAll)
                    rectangle( outFrame, m_rects[i].tl(), m_rects[i].br(), Scalar( 196, 0, 0 ), 1, 8, 0 );
            }
        }        
    }

    void Info() {
        printf("### samples = %lu, area = %d, arc length = %f, abs length = %f, velocity = %f\n", 
            m_rects.size(), AverageArea(), m_arcLength, AbsLength(), norm(m_velocity));
        printf("### max vector = %f, min vector = %f\n", m_maxVector, m_minVector);      
        for(auto v : m_vectors) {
            printf("%f\t", norm(v));
        }
        printf("\n");
        printf("### history\n");
        for(auto p : m_rects) {
            printf("(%4d,%4d)<%5d>\t", p.tl().x, p.tl().y, p.area());
        }
        printf("\n");
    }

    inline size_t VectorCount() { return m_vectors.size(); }
    inline double VectorDistortion() {
        return m_minVector > 0 ? (m_maxVector / m_minVector) : 0;
    }

    inline double ArcLength() { return m_arcLength; }
    inline double AbsLength() {         
        if(m_rects.size() > 1)
            return norm(m_rects.back().tl() - m_rects[0].tl());
        else
            return 0;
    }    
    inline unsigned long FrameTick() { return m_frameTick; }
    inline Rect & LastRect() { return m_rects.back(); }
#if 0    
    inline Point BeginPoint() { return m_rects[0].tl(); }
    inline Point EndPoint() { return m_rects.back().tl(); }
#else
    inline Point BeginPoint() { return Center(m_rects[0]); }
    inline Point EndPoint() { return Center(m_rects.back()); }
#endif    
    inline void Trigger() { m_triggerCount++; }
    inline uint8_t TriggerCount() { return m_triggerCount; }
    inline Point & Velocity() { return m_velocity; }
    inline double NormVelocity() { return norm(m_velocity); }
    int AverageArea() {
        uint32_t a = 0;
        for(auto r : m_rects) {
            a += r.area();
        }
        return a / m_rects.size();
    }
    inline size_t RectCount() { return m_rects.size(); }

    friend class Tracker;
};

static inline bool TargetSortByArea(Target & a, Target & b)
{
    return a.LastRect().area() > b.LastRect().area();
}

static Rect MergeRect(Rect & r1, Rect & r2) {
    Rect r;
    r.x = min(r1.x, r2.x);
    r.y = min(r1.y, r2.y);
    r.width = r2.x + r2.width - r1.x;
    r.height = max(r1.y + r1.height, r2.y + r2.height) - r.y;
    return r;
}

class Tracker
{
private:
    unsigned long m_frameTick;
    list< Target > m_targets;
    Rect m_newTargetRestrictionRect;
    list< list< Rect > > m_newTargetsHistory;

public:
    Tracker() : m_frameTick(0) {}

    void NewTargetRestriction(const Rect & r) {
        m_newTargetRestrictionRect = r;
    }

    Rect NewTargetRestriction() const {   return m_newTargetRestrictionRect; }

    void Update(list< Rect > & roiRect) {
        ++m_frameTick;
        for(list< Target >::iterator t=m_targets.begin();t!=m_targets.end();) { /* Try to find lost targets */
            list<Rect>::iterator rr;
            Rect r1 = t->m_rects.back();
            Rect r2 = r1;
            unsigned long f = m_frameTick - t->FrameTick();
            r2.x += t->m_velocity.x * f;
            r2.y += t->m_velocity.y * f;

            double n0 = 0;
            if(t->m_vectors.size() > 0)
                n0 = cv::norm(t->m_velocity);
            
            for(rr=roiRect.begin();rr!=roiRect.end();++rr) {

                if(r1.area() > (rr->area() * 100) ||
                    rr->area() > (r1.area() * 100))
                    continue;

                double n1 = cv::norm(rr->tl() - r1.tl());
/*
                if(rr->area() < 800 && n1 > 300)
                    continue;
*/
#if 1
                if(n1 > (CAMERA_HEIGHT / 2))
                    continue; /* Too far */

                if(t->m_vectors.size() > 1) {
                    double n = cv::norm(t->m_vectors.back());
                    if((n1 > (n * 10) || n > (n1 * 10)))
                        continue; /* Too far */
                }
#endif
                if((r1 & *rr).area() > 0) { /* Target tracked ... */
                    //if(t->DotProduct(rr->tl()) >= 0) /* Two vector less than 90 degree */
                        break;                
                }

                if(t->m_vectors.size() > 0) {
                    if((r2 & *rr).area() > 0) { /* Target tracked with velocity ... */
                        //if(t->DotProduct(rr->tl()) >= 0) /* Two vector less than 90 degree */
                            break;
                    }
                }

                if(t->m_vectors.size() == 0) { /* new target with zero velocity */
                    if(n1 < (rr->width + rr->height) * 4) { /* Target tracked with Euclidean distance ... */                    
                        break;
                    }
                } else if(n1 < (n0 * f)) { /* Target tracked with velocity and Euclidean distance ... */
#ifdef VIDEO_INPUT_FILE
                    if(rr->y >= (CAMERA_WIDTH * 4 / 5)) {
#else
                    if(rr->x >= (CAMERA_WIDTH * 4 / 5)) {
#endif
                        double a = t->CosineAngle(rr->tl());
                        if(a > 0.9659) /* cos(PI/12) */
                            break;
                    } else {
                        double a = t->CosineAngle(rr->tl());
                        if(a > 0.8587) /* cos(PI/6) */
                            break;
                    }
                }
            }
#if 1
            if(rr == roiRect.end() && 
                t->m_vectors.size() > 0) { /* Target missing ... */
                for(rr=roiRect.begin();rr!=roiRect.end();++rr) { /* */

                    if(r1.area() > (rr->area() * 100) ||
                        rr->area() > (r1.area() * 100))
                        continue;

                    double n1 = cv::norm(rr->tl() - r1.tl());
#if 1
                    if(n1 > (CAMERA_HEIGHT / 2))
                        continue; /* Too far */

                    double n = cv::norm(t->m_vectors.back());
                    if((n1 > (n * 10) || n > (n1 * 10)))
                        continue; /* Too far */
#endif
                    double a = t->CosineAngle(rr->tl());
                    double n2 = cv::norm(rr->tl() - r2.tl());
#if 0
                    if(a > 0.5 && 
                        n2 < (n0 * 1)) { /* cos(PI/3) */
                        break;
                    }               

                    if(a > 0.8587 && 
                        n2 < (n0 * 2)) { /* cos(PI/6) */
                        break;
                    }
#endif
                    /* This number has been tested by various video. Don't touch it !!! */
                    /* Tracking longer distance if target samples is over 20 */
                    if(a > 0.9659 && 
                        n2 < (n0 * (2 + (t->RectCount() > 20 ? 1 : 0)))) { /* cos(PI/12) */
                        break;
                    }
                }
            }
#else
            if(rr == roiRect.end()) { /* Target missing ... */
                for(rr=roiRect.begin();rr!=roiRect.end();++rr) { /* */
                    if(cv::norm(r2.tl()-rr->tl()) < ((rr->width + rr->height) * 2)) { /* Target tracked with velocity and Euclidean distance ... */
                        if(t->DotProduct(rr->tl()) >= 0) /* Two vector less than 90 degree */
                            break;
                    }
                }
            }
#endif
            if(rr == roiRect.end()) { /* Target missing ... */
                uint32_t compensation = (t->RectCount() / 10); /* Tracking more frames with more sample */
                if(compensation > 4)
                    compensation = 4;
                if(m_frameTick - t->FrameTick() > MAX_NUM_FRAME_MISSING_TARGET + compensation) { /* Target still missing for over X frames */
                    Point p = t->m_rects.back().tl();
                    printf("lost target : %d, %d\n", p.x, p.y);
                    //printf("lost target : %d, %d\n", t->m_velocity.x, t->m_velocity.y);
                    t = m_targets.erase(t); /* Remove tracing target */
                    continue;
                }
            } else { /* Target tracked ... */
                t->Update(*rr, m_frameTick);
                roiRect.erase(rr);
            }
            ++t;
        }

        list< Rect > newTargetList;

        for(list<Rect>::iterator rr=roiRect.begin();rr!=roiRect.end();++rr) {
            if((m_newTargetRestrictionRect & *rr).area() > 0) {
                continue;
            }

            uint32_t overlap_count = 0;
            for(auto & l : m_newTargetsHistory) {
                for(auto & r : l) {
#ifdef VIDEO_INPUT_FILE
                    if(rr->y < (CAMERA_WIDTH * 4 / 5))
#else
                    if(rr->x < (CAMERA_WIDTH * 4 / 5))
#endif
                        continue;
                    if((r & *rr).area() > 0) { /* new target overlap previous new target */
                        ++overlap_count;
                    }
                }
            }
#ifdef VIDEO_INPUT_FILE
            if(rr->y >= (CAMERA_WIDTH * 4 / 5)) {
#else
            if(rr->x >= (CAMERA_WIDTH * 4 / 5)) {
#endif
                if(overlap_count > 0) {
                    rr->x = rr->x - rr->width;
                    rr->y = rr->y - rr->height;
                    rr->width = rr->width << 1;
                    rr->height = rr->height << 1;
                }
                newTargetList.push_back(*rr);
            }

            if(overlap_count < 2) {
                m_targets.push_back(Target(*rr, m_frameTick));
                printf("new target : %d, %d\n", rr->tl().x, rr->tl().y);
            } else
                printf("fake target (%u)\n", overlap_count);
        }

        m_newTargetsHistory.push_back(newTargetList);
        if(m_newTargetsHistory.size() >= 90) {
            m_newTargetsHistory.pop_front();
        }

        if(m_targets.size() > 1)
            m_targets.sort(TargetSortByArea);
    }

    inline list< Target > & TargetList() { return m_targets; }
    inline list< list< Rect > > & NewTargetHistory() { return m_newTargetsHistory; }
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
    sort(contours.begin(), contours.end(), [](vector<cv::Point> contour1, vector<cv::Point> contour2) {  
            //return (contour1.size() > contour2.size()); /* Outline length */
            return (cv::contourArea(contour1) > cv::contourArea(contour2)); /* Area */
        }); /* Contours sort by area, controus[0] is largest */

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

        Mat roiFrame = frame(boundRect[i]);
#if 1 /* Anti cloud ... */
        double minVal; 
        double maxVal; 
        Point minLoc; 
        Point maxLoc;

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
    Ptr<cuda::Filter> & erodeFilter, Ptr<cuda::Filter> & dilateFilter, 
    Ptr<cuda::BackgroundSubtractorMOG2> & bsModel, 
    list<Rect> & roiRect, int y_offset = 0)
{
    Mat foregroundMask;
    cuda::GpuMat gpuFrame;
    cuda::GpuMat gpuForegroundMask;

    gpuFrame.upload(frame); 
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

    int cx, cy;
    while(1) {
        if(cap.read(capFrame))
            break;
        if(bShutdown)
            return 0;
    }

    cx = (capFrame.cols / 2) - 1;
    cy = capFrame.rows-1;

    Tracker tracker;

#ifdef NEW_TARGET_RESTRICTION    
#ifdef VIDEO_INPUT_FILE
    //tracker.NewTargetRestriction(Rect(160, 1080, 400, 200));
    tracker.NewTargetRestriction(Rect(cx - 200, cy - 200, 400, 200));
    //tracker.NewTargetRestriction(Rect(cx - 360, cy - 200, 720, 200));
#else
    tracker.NewTargetRestriction(Rect(cy - 200, cx - 200, 400, 200));
#endif
#endif

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
    Ptr<cuda::BackgroundSubtractorMOG2> bsModel = cuda::createBackgroundSubtractorMOG2(30, 10, false);
    /* https://blog.csdn.net/m0_37901643/article/details/72841289 */
    /* Default variance of each gaussian component 15 / 75 / 75 */ 
    //cout << bsModel->getVarInit() << " / " << bsModel->getVarMax() << " / " << bsModel->getVarMax() << endl;
    bsModel->setVarInit(15);
    bsModel->setVarMax(20);
    bsModel->setVarMin(4);

    steady_clock::time_point t1(steady_clock::now());

    int dropCount = 30;
    while(dropCount-- > 0)
        cap.read(capFrame);

    cvtColor(capFrame, bgFrame, COLOR_BGR2GRAY);

    uint64_t loopCount = 0;

    while(cap.read(capFrame)) {
        ++loopCount;
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

        extract_moving_object(grayFrame, elementErode, elementDilate, erodeFilter, dilateFilter, bsModel, roiRect);

#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
        //RNG rng(12345);
        for(list<Rect>::iterator rr=roiRect.begin();rr!=roiRect.end();++rr) {
            //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            rectangle( outFrame, rr->tl(), rr->br(), Scalar(0, 255, 0), 1, 8, 0 );
            //writeText( outFrame, "roi", rr->tl());
        }

#ifdef NEW_TARGET_RESTRICTION    
        Rect nr = tracker.NewTargetRestriction();
        rectangle( outFrame, nr.tl(), nr.br(), Scalar(127, 0, 127), 1, 8, 0 );
        writeText( outFrame, "New Target Restriction Area", Point(cx - 200, cy - 200));
#endif
        writeText( outFrame, currentDateTime(), Point(240, 40));
#endif
        line(outFrame, Point(0, (CAMERA_WIDTH * 4 / 5)), Point(CAMERA_HEIGHT, (CAMERA_WIDTH * 4 / 5)), Scalar(127, 127, 0), 1);

        tracker.Update(roiRect);

        list< Target > & targets = tracker.TargetList();

#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
        list< list< Rect > > & newTargetHistory = tracker.NewTargetHistory();
        for(auto & it : newTargetHistory) {
            for(auto & r : it) {
                rectangle( outFrame, r.tl(), r.br(), Scalar(127, 127, 0), 1, 8, 0 );
            }
        }        
#endif
        bool doTrigger = false;
        for(list< Target >::iterator t=targets.begin();t!=targets.end();++t) {
#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
            t->Draw(outFrame);
#endif
            if(t->TriggerCount() > 0 && t->TriggerCount() < MAX_NUM_TRIGGER) {
                doTrigger = true;
            } else if(t->ArcLength() > MIN_COURSE_LENGTH && 
                t->RectCount() > MIN_TARGET_TRACKED_COUNT) {
#ifdef VIDEO_INPUT_FILE
                if((t->BeginPoint().x > cx && t->EndPoint().x <= cx) ||
                    (t->BeginPoint().x < cx && t->EndPoint().x >= cx)) {
#else
                if((t->BeginPoint().y > cy && t->EndPoint().y <= cy) ||
                    (t->BeginPoint().y < cy && t->EndPoint().y >= cy)) {
#endif //VIDEO_INPUT_FILE
                    if(t->VectorCount() <= 8 &&
                        t->VectorDistortion() >= 40) { /* 最大位移向量值與最小位移向量值的比例 */
                        printf("Velocity distortion %f !!!\n", t->VectorDistortion());
                        t->Info();
                    } else if(t->AverageArea() < 200 &&
                        t->NormVelocity() > 50) {
                        printf("Bug detected !!! average area = %d, velocity = %f\n", t->AverageArea(), t->NormVelocity());
                        t->Info();
                    } else if(t->AverageArea() < 600 &&
                        t->NormVelocity() > 100) {
                        printf("Bug detected !!! average area = %d, velocity = %f\n", t->AverageArea(), t->NormVelocity());
                        t->Info();
                    } else if(t->TriggerCount() < MAX_NUM_TRIGGER) { /* Triggle 4 times maximum  */
                        doTrigger = true;
                    }
                }
            }

            if(doTrigger) {
#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
#ifdef VIDEO_INPUT_FILE
                line(outFrame, Point(cx, 0), Point(cx, cy), Scalar(0, 0, 255), 3);
#else
                line(outFrame, Point(0, cy), Point(cx, cy), Scalar(0, 0, 255), 3);
#endif //VIDEO_INPUT_FILE
#endif //VIDEO_OUTPUT_FRAME
                printf("T R I G G E R - %d\n", t->TriggerCount());
                t->Trigger();
                t->Info();

                break; /* Has been trigger, ignore other targets */           
            }
        }        

#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
        char str[32];
        snprintf(str, 32, "FPS : %.2lf", fps);
        writeText(outFrame, string(str), Point( 10, 40 ));
#if defined(VIDEO_OUTPUT_FILE)
        videoWriterQueue.push(outFrame.clone());
#endif
#ifdef VIDEO_OUTPUT_SCREEN
        resize(outFrame, outFrame, Size(outFrame.cols * 3 / 4, outFrame.rows * 3 / 4));
#ifdef VIDEO_INPUT_FILE        
        namedWindow(argv[1], WINDOW_AUTOSIZE);
        imshow(argv[1], outFrame);
#else
        namedWindow("Out Frame", WINDOW_AUTOSIZE);
        imshow("Out Frame", outFrame);
#endif
#endif
#endif
#ifdef VIDEO_OUTPUT_SCREEN
        int k = waitKey(1);
        if(k == 27) { /* Press key 'ESC' to quit */
            break;
        } else if(k == 'p' || doTrigger) { /* Press key 'p' to pause or resume */
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
        if(loopCount > 0 && loopCount % 30 == 0) {
            //std::cout << (dt_us / 1000.0) << " ms" << std::endl;
            fps = (1000000.0 / dt_us);
            std::cout << "FPS : " << fixed  << setprecision(2) <<  fps << std::endl;
        }
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
