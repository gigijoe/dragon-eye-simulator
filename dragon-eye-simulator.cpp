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
using std::chrono::milliseconds;

#ifndef DEBUG
#define DEBUG
#endif

#define CAMERA_WIDTH 1280
#define CAMERA_HEIGHT 720
#define CAMERA_FPS 30
#define MIN_TARGET_WIDTH 8
#define MIN_TARGET_HEIGHT 6
#define MAX_TARGET_WIDTH 320
#define MAX_TARGET_HEIGHT 320
#define MAX_TARGET_TRACKING_DISTANCE    360

#define MAX_NUM_TARGET                  9
#define MAX_NUM_TRIGGER                 1
#define MAX_NUM_FRAME_MISSING_TARGET    3

#define MIN_COURSE_LENGTH               16     /* Minimum course length of RF trigger after detection of cross line */
#define MIN_TARGET_TRACKED_COUNT        3      /* Minimum target tracked count of RF trigger after detection of cross line (3 * 33ms = 99ms) */

#define NEW_TARGET_RESTRICTION

#define FAKE_TARGET_DETECTION           true
#define BUG_TRIGGER                     true

#define HORIZON_RATIO                8 / 10

#define VIDEO_INPUT_FILE

#define VIDEO_OUTPUT_SCREEN
//#define VIDEO_OUTPUT_FILE "base"
#define VIDEO_OUTPUT_DIR "."

#define VIDEO_FRAME_DROP 30

//#define IMAGE_INFERENCE

#ifdef IMAGE_INFERENCE
#include "image_inference.hpp" 
#endif

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

/*
* The PSNR returns a float number, that if the two inputs are similar between 30 and 50 (higher is better).
*/

/* 
* Reference 
* https://www.w3cschool.cn/opencv/opencv-snrm2f01.html
*/

struct BufferPSNR                                     // Optimized CUDA versions
{   // Data allocations are very expensive on CUDA. Use a buffer to solve: allocate once reuse later.
    cuda::GpuMat gI1, gI2, gs, t1,t2;
    cuda::GpuMat buf;
};

double getPSNR_CUDA_optimized(const Mat& I1, const Mat& I2)
{
    static BufferPSNR b;

    b.gI1.upload(I1);
    b.gI2.upload(I2);
    b.gI1.convertTo(b.t1, CV_32F);
    b.gI2.convertTo(b.t2, CV_32F);
    cuda::absdiff(b.t1.reshape(1), b.t2.reshape(1), b.gs);
    cuda::multiply(b.gs, b.gs, b.gs);
    double sse = cuda::sum(b.gs, b.buf)[0];
    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse = sse /(double)(I1.channels() * I1.total());
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}

double getPSNR_CUDA(const Mat& I1, const Mat& I2)
{
    cuda::GpuMat gI1, gI2, gs, t1,t2;
    gI1.upload(I1);
    gI2.upload(I2);
    gI1.convertTo(t1, CV_32F);
    gI2.convertTo(t2, CV_32F);
    cuda::absdiff(t1.reshape(1), t2.reshape(1), gs);
    cuda::multiply(gs, gs, gs);
    Scalar s = cuda::sum(gs);
    double sse = s.val[0] + s.val[1] + s.val[2];
    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse =sse /(double)(gI1.channels() * I1.total());
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}

double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // 不能在8位矩陣上做平方運算
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         // 疊加每個通道的元素

    double sse = s.val[0] + s.val[1] + s.val[2]; // 疊加所有通道

    if( sse <= 1e-10) // 如果值太小就直接等於0
        return 0;
    else
    {
        double  mse =sse /(double)(I1.channels() * I1.total());
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}

/*
*
*/

struct BufferMSSIM                                     // Optimized CUDA versions
{   // Data allocations are very expensive on CUDA. Use a buffer to solve: allocate once reuse later.
    cuda::GpuMat gI1, gI2, gs, t1,t2;
    cuda::GpuMat I1_2, I2_2, I1_I2;
    vector<cuda::GpuMat> vI1, vI2;
    cuda::GpuMat mu1, mu2;
    cuda::GpuMat mu1_2, mu2_2, mu1_mu2;
    cuda::GpuMat sigma1_2, sigma2_2, sigma12;
    cuda::GpuMat t3;
    cuda::GpuMat ssim_map;
    cuda::GpuMat buf;
};

Scalar getMSSIM_CUDA_optimized( const Mat& i1, const Mat& i2)
{
    static BufferMSSIM b;

    const float C1 = 6.5025f, C2 = 58.5225f;
    /***************************** INITS **********************************/
    b.gI1.upload(i1);
    b.gI2.upload(i2);
    cuda::Stream stream;
    b.gI1.convertTo(b.t1, CV_32F, stream);
    b.gI2.convertTo(b.t2, CV_32F, stream);
    cuda::split(b.t1, b.vI1, stream);
    cuda::split(b.t2, b.vI2, stream);
    Scalar mssim;
    Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(b.vI1[0].type(), -1, Size(11, 11), 1.5);
    for( int i = 0; i < b.gI1.channels(); ++i )
    {
        cuda::multiply(b.vI2[i], b.vI2[i], b.I2_2, 1, -1, stream);        // I2^2
        cuda::multiply(b.vI1[i], b.vI1[i], b.I1_2, 1, -1, stream);        // I1^2
        cuda::multiply(b.vI1[i], b.vI2[i], b.I1_I2, 1, -1, stream);       // I1 * I2
        gauss->apply(b.vI1[i], b.mu1, stream);
        gauss->apply(b.vI2[i], b.mu2, stream);
        cuda::multiply(b.mu1, b.mu1, b.mu1_2, 1, -1, stream);
        cuda::multiply(b.mu2, b.mu2, b.mu2_2, 1, -1, stream);
        cuda::multiply(b.mu1, b.mu2, b.mu1_mu2, 1, -1, stream);
        gauss->apply(b.I1_2, b.sigma1_2, stream);
        cuda::subtract(b.sigma1_2, b.mu1_2, b.sigma1_2, cuda::GpuMat(), -1, stream);
        //b.sigma1_2 -= b.mu1_2;  - This would result in an extra data transfer operation
        gauss->apply(b.I2_2, b.sigma2_2, stream);
        cuda::subtract(b.sigma2_2, b.mu2_2, b.sigma2_2, cuda::GpuMat(), -1, stream);
        //b.sigma2_2 -= b.mu2_2;
        gauss->apply(b.I1_I2, b.sigma12, stream);
        cuda::subtract(b.sigma12, b.mu1_mu2, b.sigma12, cuda::GpuMat(), -1, stream);
        //b.sigma12 -= b.mu1_mu2;
        //here too it would be an extra data transfer due to call of operator*(Scalar, Mat)
        cuda::multiply(b.mu1_mu2, 2, b.t1, 1, -1, stream); //b.t1 = 2 * b.mu1_mu2 + C1;
        cuda::add(b.t1, C1, b.t1, cuda::GpuMat(), -1, stream);
        cuda::multiply(b.sigma12, 2, b.t2, 1, -1, stream); //b.t2 = 2 * b.sigma12 + C2;
        cuda::add(b.t2, C2, b.t2, cuda::GpuMat(), -12, stream);
        cuda::multiply(b.t1, b.t2, b.t3, 1, -1, stream);     // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
        cuda::add(b.mu1_2, b.mu2_2, b.t1, cuda::GpuMat(), -1, stream);
        cuda::add(b.t1, C1, b.t1, cuda::GpuMat(), -1, stream);
        cuda::add(b.sigma1_2, b.sigma2_2, b.t2, cuda::GpuMat(), -1, stream);
        cuda::add(b.t2, C2, b.t2, cuda::GpuMat(), -1, stream);
        cuda::multiply(b.t1, b.t2, b.t1, 1, -1, stream);     // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
        cuda::divide(b.t3, b.t1, b.ssim_map, 1, -1, stream);      // ssim_map =  t3./t1;
        stream.waitForCompletion();
        Scalar s = cuda::sum(b.ssim_map, b.buf);
        mssim.val[i] = s.val[0] / (b.ssim_map.rows * b.ssim_map.cols);
    }
    return mssim;
}

/*
*
*/

const Mat SubMat(const Mat & capFrame, const Point & center, const Size & size)
{
        Rect r(center.x - (size.width / 2), center.y - (size.height / 2), size.width, size.height);

        if(r.x < 0)
            r.x = 0;
        else if(r.x + size.width >= capFrame.cols)
            r.x = capFrame.cols - (size.width + 1);

        if(r.y < 0)
            r.y = 0;
        else if(r.y + size.height >= capFrame.rows)
            r.y = capFrame.rows - (size.height + 1);

        //printf("r (%d, %d) [%d, %d]\n", r.x, r.y, r.width, r.height);

        return capFrame(r);
}

/*
const Point & Center(Rect & r) {
    return std::move(Point(r.tl().x + (r.width / 2), r.tl().y + (r.height / 2)));
}
*/

static inline Point Center(Rect & r) {
    return Point(r.tl().x + (r.width / 2), r.tl().y + (r.height / 2));
}

bool isLineIntersection(Point & s0, Point & e0, Point & s1, Point & e1) {
    float d = (s0.x - e0.x)*(s1.y - e1.y) - (s0.y - e0.y)*(s1.x - e1.x);
    return d == 0 ? false : true;
}

class Target
{
protected:
    uint32_t m_id;
    double m_arcLength;
    double m_absLength;
    unsigned long m_lastFrameTick;
    uint8_t m_triggerCount;
    uint16_t m_bugTriggerCount;

    vector< unsigned long > m_frameTicks;
    vector< Rect > m_rects;
    vector< Point > m_vectors;
    double m_maxVector, m_minVector;
    Point m_velocity;
    Point m_acceleration;
    int m_averageArea;
    double m_normVelocity;
    double m_angleOfTurn;

    Mat roiFrame;

    static uint32_t s_id;

    double CosineAngle(const Point & v1, const Point & v2) {
        /* A.B = |A||B|cos() */
        /* cos() = A.B / |A||B| */
        return v1.dot(v2) / (norm(v1) * norm(v2));
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
    Target(Rect & roi, Mat & capFrame, unsigned long frameTick) : m_arcLength(0), m_absLength(0), m_lastFrameTick(frameTick), m_triggerCount(0), m_bugTriggerCount(0), 
            m_maxVector(0), m_minVector(0), m_averageArea(0), m_normVelocity(0), m_angleOfTurn(0) {
        m_id = s_id++;
        roiFrame = SubMat(capFrame, Center(roi), Size(16, 16));
        m_rects.push_back(roi);
        m_lastFrameTick = frameTick;
        m_frameTicks.push_back(frameTick);
        m_averageArea = roi.area();
    }

    void Reset() {
        Rect r = m_rects.back();
        m_rects.clear();
        m_rects.push_back(r);
        m_frameTicks.clear();
        m_frameTicks.push_back(m_lastFrameTick);
        m_triggerCount = 0;
        m_bugTriggerCount = 0;
        m_maxVector = 0;
        m_minVector = 0;
        m_averageArea = 0;
        m_normVelocity = 0;
        m_angleOfTurn = 0;
        //m_arcLength = 0; /* TODO : Do clear this ? */
        //m_absLength = 0;
        //m_lastFrameTick =
    }

    void Update(Rect & roi, Mat & capFrame, unsigned long frameTick) {
        if(frameTick <= m_lastFrameTick) /* Reverse tick ??? Illegal !!! */
            return;

        roiFrame = SubMat(capFrame, Center(roi), Size(16, 16));

        int itick = frameTick - m_lastFrameTick;

        if(m_rects.size() > 0) { /* We have 1 point now and will have 2 */
            Point p = (roi.tl() - m_rects.back().tl()) / itick;
            double v = norm(p) / itick;

            m_arcLength += v;
            m_vectors.push_back(p);

            m_absLength = norm(roi.tl() - m_rects[0].tl());

            if(m_rects.size() == 1) {
                m_maxVector = v;
                m_minVector = v;
            } else if(v > m_maxVector)
                m_maxVector = v;
            else if(v < m_minVector)
                m_minVector = v;

            m_averageArea = (m_averageArea + roi.area()) / 2;
        } else {
            m_averageArea = roi.area();
        }

        if(m_rects.size() == 1) { /* We have 2 point now */
            m_velocity.x = (roi.tl().x - m_rects.back().tl().x) / itick;
            m_velocity.y = (roi.tl().y - m_rects.back().tl().y) / itick;
        } else if(m_rects.size() > 1) { /* We have at latest 3 point now */
            m_velocity.x = (m_velocity.x + (roi.tl().x - m_rects.back().tl().x) / itick) / 2;
            m_velocity.y = (m_velocity.y + (roi.tl().y - m_rects.back().tl().y) / itick) / 2;

            size_t n = m_vectors.size() - 1;
            size_t n_1 = n - 1;

            m_acceleration.x = (m_acceleration.x + (m_vectors[n].x - m_vectors[n_1].x)) / 2;
            m_acceleration.y = (m_acceleration.y + (m_vectors[n].y - m_vectors[n_1].y)) / 2;

            double v = CosineAngle(m_vectors[n], m_vectors[n_1]);
            double radian;
            if(v <= -1.0f)
                radian = M_PI;
            else if(v >= 1.0f)
                radian = 0;
            else
                radian = acos(v);

            /* 
            * r = (v1.x * v2.y) - (v2.x * v1.y) 
            * If r > 0 v2 is located on the left side of v1. 
            * if r == 0 v2 and v1 on the same line.
            * If r < 0 v2 is located on the right side of v2.
            */ 
            v = (m_vectors[n].x * m_vectors[n_1].y) - (m_vectors[n_1].x * m_vectors[n].y);
            if(v < 0)
                radian *= -1.0;

            m_angleOfTurn += (radian * 180 / M_PI);
        }

        m_normVelocity = norm(m_velocity);

        m_rects.push_back(roi);
        m_lastFrameTick = frameTick;
        m_frameTicks.push_back(frameTick);
#if 1
        if(m_triggerCount >= MAX_NUM_TRIGGER)
            Reset();
#endif
    }

    void Update(Target & t, Mat & capFrame) {
        for(size_t i=0;i<t.m_rects.size();i++) {
            Update(t.m_rects[i], capFrame, t.m_frameTicks[i]);
        }
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

    double CosineAngleTl(const Point & p) {
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

    double CosineAngleBr(const Point & p) {
        size_t i = m_rects.size();
        if(i < 2)
            return 0;
        --i;
        Point v1, v2;
        v1.x = p.x - m_rects[i].br().x;
        v1.y = p.y - m_rects[i].br().y;
        v2.x = m_rects[i].br().x - m_rects[i-1].br().x;
        v2.y = m_rects[i].br().y - m_rects[i-1].br().y;

        /* A.B = |A||B|cos() */
        /* cos() = A.B / |A||B| */
        return v1.dot(v2) / (norm(v1) * norm(v2));
    }

    double CosineAngleCt(const Point & p) {
        size_t i = m_rects.size();
        if(i < 2)
            return 0;
        --i;
        Point v1, v2;
        Point ct0 = Center(m_rects[i]);
        Point ct1 = Center(m_rects[i-1]);
        v1.x = p.x - ct0.x;
        v1.y = p.y - ct0.y;
        v2.x = ct0.x - ct1.x;
        v2.y = ct0.y - ct1.y;

        /* A.B = |A||B|cos() */
        /* cos() = A.B / |A||B| */
        return v1.dot(v2) / (norm(v1) * norm(v2));
    }

    void Draw(Mat & outFrame, bool drawAll = false) {
        RNG rng(m_rects.front().area());
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        Rect r = m_rects.back();
        //rectangle( outFrame, r.tl(), r.br(), Scalar( 255, 0, 0 ), 1, 8, 0 );
        rectangle( outFrame, r.tl(), r.br(), color, 1, 8, 0 );

        if(m_rects.size() > 1) { /* Minimum 2 points ... */
            for(int i=0;i<m_rects.size()-1;i++) {
                //Point p0 = m_rects[i].tl();
                //Point p1 = m_rects[i+1].tl();
                //line(outFrame, p0, p1, Scalar(0, 0, 255), 1);
                line(outFrame, Center(m_rects[i]), Center(m_rects[i+1]), color, 1);
                //line(outFrame, p0, p1, color, 1);
                //Point v = p1 - p0;
                //printf("[%d,%d]\n", v.x, v.y);
                if(drawAll)
                    //rectangle( outFrame, m_rects[i].tl(), m_rects[i].br(), Scalar( 196, 0, 0 ), 1, 8, 0 );
                    //rectangle( outFrame, m_rects[i].tl(), m_rects[i].br(), Scalar( (m_rects[0].x + m_rects[0].y) % 255, 0, 0 ), 1, 8, 0 );
                    rectangle( outFrame, m_rects[i].tl(), m_rects[i].br(), color, 1, 8, 0 );
            }
        }        
    }

    void Info() {
#ifdef DEBUG
        printf("\033[0;31m"); /* Red */
        printf("\n= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
        printf("[%u] Target :\n\tsamples = %lu, area = %d, arc length = %.1f, abs length = %.1f, velocity = %.1f, angle of turn = %.1f\n", 
            m_id, m_rects.size(), m_averageArea, m_arcLength, m_absLength, m_normVelocity, m_angleOfTurn);
        printf("\nVectors : length\n");      
        for(auto v : m_vectors) {
            printf("(%d, %d)[%.1f]\t", v.x, v.y, norm(v));
        }
        printf("\n");
        printf("maximum = %.1f, minimum = %.1f\n", m_maxVector, m_minVector);      
        printf("\nTrajectory : [ Tick ] (x, y) area\n");
        /*
        for(auto p : m_rects) {
            printf("(%4d,%4d)<%5d>\t", p.tl().x, p.tl().y, p.area());
        }
        */
        for(size_t i=0;i<m_rects.size();i++) {
            printf("[%lu](%4d,%4d) %d\t", m_frameTicks[i], m_rects[i].tl().x, m_rects[i].tl().y, m_rects[i].area());
        }

        printf("\nAngle of turn :\n"); 
        if(m_vectors.size() > 1)
        for(size_t i=0;i<m_vectors.size()-1;i++) {
            double v = CosineAngle(m_vectors[i], m_vectors[i+1]);
            double radian;
            if(v <= -1.0f)
                radian = M_PI;
            else if(v >= 1.0f)
                radian = 0;
            else
                radian = acos(v);

            /* 
            * r = (v1.x * v2.y) - (v2.x * v1.y) 
            * If r > 0 v2 is located on the left side of v1. 
            * if r == 0 v2 and v1 on the same line.
            * If r < 0 v2 is located on the right side of v2.
            */ 
            v = (m_vectors[i+1].x * m_vectors[1].y) - (m_vectors[1].x * m_vectors[i+1].y);
            if(v < 0)
                radian *= -1.0;

            printf("<%f ", (radian * 180 / M_PI));
        }
        printf("\n= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
        printf("\033[0m"); /* Default color */
#endif
    }

    inline double VectorDistortion() {
        return m_minVector > 0 ? (m_maxVector / m_minVector) : 0;
    }

    inline double ArcLength() { return m_arcLength; }
    inline double AbsLength() { return m_absLength; }

    inline unsigned long FrameTick() { return m_lastFrameTick; }
    inline const Rect & LastRect() { return m_rects.back(); }

    inline const Point BeginCenterPoint() { return Center(m_rects[0]); }
    inline const Point EndCenterPoint() { return Center(m_rects.back()); }

    inline const Point CurrentCenterPoint() { return Center(m_rects.back()); }
    const Point PreviousCenterPoint() {
        if(m_rects.size() < 2)
            return std::move(Point(0, 0));

        auto it = m_rects.rbegin();
        return Center(*(++it)); 
    }
    
    bool Trigger(bool enableBugTrigger = false) {
        bool r = false;
        if(m_vectors.size() <= 8 &&
            VectorDistortion() >= 40) { /* 最大位移向量值與最小位移向量值的比例 */
#ifdef DEBUG
            printf("\033[0;31m"); /* Red */
            printf("Velocity distortion %f !!!\n", VectorDistortion());
            printf("\033[0m"); /* Default color */
#endif
        } else if(enableBugTrigger) { 
            if((m_averageArea < 144 && m_normVelocity > 30) || /* 12 x 12 */
                    (m_averageArea < 256 && m_normVelocity > 40) || /* 16 x 16 */
                    (m_averageArea < 324 && m_normVelocity > 50) || /* 18 x 18 */
                    (m_averageArea < 400 && m_normVelocity > 75) || /* 20 x 20 */
                    (m_averageArea < 576 && m_normVelocity > 100) || /* 24 x 24 */
                    (m_averageArea < 900 && m_normVelocity > 125) /* 30 x 30 */
                ) {
#ifdef DEBUG
                printf("\033[0;31m"); /* Red */
                printf("Bug detected !!! average area = %d, velocity = %f\n", m_averageArea, m_normVelocity);
                printf("\033[0m"); /* Default color */
#endif
                m_bugTriggerCount++;
            } else {
                if(m_bugTriggerCount > 0) {
#ifdef DEBUG
                    printf("\033[0;31m"); /* Red */
                    printf("False trigger due to bug trigger count is %u\n", m_bugTriggerCount);
                    printf("\033[0m"); /* Default color */
#endif
                    if(m_bugTriggerCount <= 3) /* To avoid false bug detection */
                        m_bugTriggerCount--;
                } else {
                    m_triggerCount++;
                    r = true;
                }
            }
        } else {
            m_triggerCount++;
            r = true;
        }
        if(r) {
#ifdef DEBUG
            printf("\033[0;31m"); /* Red */
            printf("[%u] T R I G G E R (%d)\n", m_id, m_triggerCount);
            printf("\033[0m"); /* Default color */
#endif
        }

        Info();

        return r;
    }

    inline uint8_t TriggerCount() { return m_triggerCount; }

    inline int AverageArea() { return m_averageArea; }
    inline size_t TrackedCount() { return m_rects.size(); }

    vector< Point > & Vectors() { return m_vectors; }

    friend class Tracker;
};

uint32_t Target::s_id = 0;

static inline bool TargetSortByArea(Target & a, Target & b)
{
    return a.LastRect().area() > b.LastRect().area();
}

static inline bool TargetSortByTrackedCount(Target & a, Target & b)
{
    return a.TrackedCount() > b.TrackedCount();
}

static Rect MergeRect(Rect & r1, Rect & r2) {
    Rect r;
    r.x = min(r1.x, r2.x);
    r.y = min(r1.y, r2.y);
    r.width = max(r1.x + r1.width, r2.x + r2.width) - r.x;
    r.height = max(r1.y + r1.height, r2.y + r2.height) - r.y;
    return r;
}

class RectSortByDistance {
    Rect const & t;
public:
    RectSortByDistance(Rect const & r) : t(r) {}

    bool operator()(Rect const & a, Rect const & b) {
        return cv::norm(a.tl() - t.tl()) > cv::norm(b.tl() - t.tl());
    }
};

class Tracker
{
private:
    int m_width, m_height;
    unsigned long m_lastFrameTick;
    list< Target > m_targets;
    Rect m_newTargetRestrictionRect;
    list< list< Rect > > m_newTargetsHistory;
    int m_horizonHeight;

    size_t MaxTrackedCountOfTargets() {
        size_t maxCount = 0;
        for(list< Target >::iterator t=m_targets.begin();t!=m_targets.end();t++) {
            if(t->TrackedCount() > maxCount)
                maxCount = t->TrackedCount();
        }
        return maxCount;
    }

public:
    Tracker(int width, int height) : m_width(width), m_height(height), m_lastFrameTick(0), m_horizonHeight(height * HORIZON_RATIO) {}
    
    int HorizonHeight() const { return m_horizonHeight; }

    void NewTargetRestriction(const Rect & r) {
        m_newTargetRestrictionRect = r;
    }

    Rect NewTargetRestriction() const {   return m_newTargetRestrictionRect; }

    void Update(list< Rect > & roiRect, Mat & capFrame, bool enableFakeTargetDetection = false) {
        ++m_lastFrameTick;
        for(list< Target >::iterator t=m_targets.begin();t!=m_targets.end();) { /* Try to find lost targets */
            list<Rect>::iterator rr;
            Rect r1 = t->m_rects.back();
            Rect r2 = r1;
            int f = m_lastFrameTick - t->FrameTick();
            if(t->m_vectors.size() > 0) {
                Point v = t->m_vectors.back();
                for(int i=0;i<f;i++) {
                    v.x += t->m_acceleration.x;
                    v.y += t->m_acceleration.y;
                    r2.x += v.x;
                    r2.y += v.y;
                }
            }

            if(t->m_triggerCount > 0 &&
                (r2.x < 0 || r2.x > m_width)) { 
#ifdef DEBUG
                Point p = t->m_rects.back().tl();
                printf("\033[0;31m"); /* Red */
                printf("<%u> Otu of range target : (%d, %d), samples : %lu\n", t->m_id, p.x, p.y, t->m_rects.size());
                printf("\033[0m"); /* Default color */
                //if(t->m_rects.size() >= 10)
                t->Info();
#endif
                t = m_targets.erase(t); /* Remove tracing target */
                continue;
            }

            double n0 = 0;
            if(t->m_vectors.size() > 0) {
                n0 = cv::norm(Center(r2) - Center(r1)); /* Moving distance of predict target */
            }

            //sort(roiRect.begin(), roiRect.end(), RectSortByDistance(r2));
            //roiRect.sort(RectSortByDistance(r2));

            for(rr=roiRect.begin();rr!=roiRect.end();++rr) {

                if(r1.area() > (rr->area() * 32) ||
                    rr->area() > (r1.area() * 32)) /* Object and target area difference */
                    continue;


                Point rrct = Center(*rr);
                double n1 = cv::norm(rrct - Center(r1)); /* Distance between object and target */

                if(n1 > MAX_TARGET_TRACKING_DISTANCE) /* 360 */
                    continue; /* Too far */
#if 0
                if(t->m_vectors.size() > 0) {
                    double n = cv::norm(t->m_vectors.back());
                    if(n1 > (n0 * 3) / 2)
                        continue; /* Too far */
                }
#endif
                if((r1 & *rr).area() > 0) { /* Target tracked ... */
                    //if(t->DotProduct(rr->tl()) >= 0) /* Two vector less than 90 degree */
                        break;                
                }

                if(t->m_vectors.size() > 0) {
                    Rect r = r1;
                    bool tracked = false;
                    Point v = t->m_vectors.back();
                    for(int i=0;i<f;i++) {
                        r.x += v.x;
                        r.y += v.y;
                        if((r & *rr).area() > 0) { /* Target tracked with velocity ... */
                            tracked = true;
                            break;
                        }
                    }
                    if(tracked)
                        break;
                }

                if(t->m_vectors.size() > 0) {
                    Rect r = r1;
                    bool tracked = false;
                    Point v = t->m_vectors.back();
                    for(int i=0;i<f;i++) {
                        v.x += t->m_acceleration.x;
                        v.y += t->m_acceleration.y;
                        r.x += v.x;
                        r.y += v.y;
                        if((r & *rr).area() > 0) { /* Target tracked with velocity ... */
                            tracked = true;
                            break;
                        }
                    }
                    if(tracked)
                        break;
                }

                if(t->m_vectors.size() == 0) { /* new target with zero velocity */
                    if(rr->y >= m_horizonHeight) {
                        if(n1 < (rr->width + rr->height)) /* Target tracked with Euclidean distance ... */
                            break;
                    } else {
                        int factor = 2;
                        if(n1 < (rr->width + rr->height) * factor) /* Target tracked with Euclidean distance ... */
                            break;
                    }
                } else if(n1 < (n0 * 3) / 2) { /* Target tracked with velocity and Euclidean distance ... */
                    if(rr->y >= m_horizonHeight) {
                        double a = t->CosineAngleCt(rrct);
                        if(a > 0.9659) /* cos(PI/12) */
                            break;
                    } else {
                        //double a = t->CosineAngleTl(rr->tl());
                        //if(a > 0.8587) /* cos(PI/6) */
                        //    break;
                    }
                }
            }

            if(rr == roiRect.end() && 
                t->m_vectors.size() > 0) { /* Target missing ... */
                for(rr=roiRect.begin();rr!=roiRect.end();++rr) { /* */

                    if(r1.area() > (rr->area() * 32) ||
                        rr->area() > (r1.area() * 32)) /* Object and target area difference */
                        continue;

                    Point rrct = Center(*rr);
                    double n1 = cv::norm(rrct - Center(r1)); /* Distance between object and target */

                    if(n1 > (MAX_TARGET_TRACKING_DISTANCE / 2)) /* 180 */
                        continue; /* Too far */

                    //if(n1 > (n0 * 3) / 2)
                    //    continue; /* Too far */

                    double a = t->CosineAngleCt(rrct);
                    double n2 = cv::norm(rrct - Center(r2));

                    if(a > 0.5 && /* cos(PI/3) */
                        n2 < (n0 * 3) / 2) { 
                        break;
                    }

                    if(a > 0.8587 && /* cos(PI/6) */
                        n1 < (t->m_normVelocity * f * 2)) {
                        break;
                    }
                }
            }

            if(rr == roiRect.end()) { /* Target missing ... */
                bool isTargetLost = false;
                if(t->m_vectors.size() > 1) {
                    uint32_t compensation = (t->TrackedCount() / 6); /* Tracking more frames with more sample */
                    if(compensation > 5)
                        compensation = 5;
                    if(f > (MAX_NUM_FRAME_MISSING_TARGET + compensation)) /* Target still missing for over X frames */
                        isTargetLost = true;
                } else { /* new target with zero velocity */
                    if(f > MAX_NUM_FRAME_MISSING_TARGET) 
                        isTargetLost = true;
                }
                if(isTargetLost) {
#ifdef DEBUG
                    Point p = t->m_rects.back().tl();
                    printf("\033[0;31m"); /* Red */
                    printf("<%u> Lost target : (%d, %d), samples : %lu\n", t->m_id, p.x, p.y, t->m_rects.size());
                    printf("\033[0m"); /* Default color */
                    //if(t->m_rects.size() >= 10)
                        t->Info();
#endif
                    t = m_targets.erase(t); /* Remove tracing target */
                    continue;
                } else {
                }
            } else { /* Target tracked ... */
#ifdef DEBUG
                Point p = t->m_rects.back().tl();
                printf("\033[0;32m"); /* Green */
                printf("<%u> Target tracked : [%lu](%d, %d) -> (%d, %d)[%d, %d]\n", t->m_id, m_lastFrameTick, p.x, p.y, 
                    rr->x, rr->y, rr->x - t->m_rects.back().x, rr->y - t->m_rects.back().y);
                printf("\033[0m"); /* Default color */
#if 0
                /* Target tracked, check PSNR */
#if 1
                //double psnr = getPSNR(t->roiFrame, SubMat(capFrame, Center(*rr), Size(16, 16)));
                //double psnr = getPSNR_CUDA(t->roiFrame, SubMat(capFrame, Center(*rr), Size(16, 16)));
                double psnr = getPSNR_CUDA_optimized(t->roiFrame, SubMat(capFrame, Center(*rr), Size(16, 16)));

                printf("\033[0;33m"); /* Yellow */
                printf("<%u> psnr = %f\n", t->m_id, psnr);
                printf("\033[0m"); /* Default color */
#else
                Scalar s = getMSSIM_CUDA_optimized(t->roiFrame, SubMat(capFrame, Center(*rr), Size(16, 16)));
                printf("\033[0;33m"); /* Yellow */
                printf("<%u> mssim = %f, %f, %f\n", t->m_id, s.val[0], s.val[1], s.val[2]);
                printf("\033[0m"); /* Default color */                
#endif

#endif

#endif
                t->Update(*rr, capFrame, m_lastFrameTick);
                roiRect.erase(rr);
            }
            ++t;
        }

        list< Rect > newTargetList;

        for(list<Rect>::iterator rr=roiRect.begin();rr!=roiRect.end();++rr) { /* New targets registration */
            if(!m_newTargetRestrictionRect.empty()) {
                if((m_newTargetRestrictionRect & *rr).area() > 0)
                    continue;
            }

            uint32_t overlap_count = 0;
            for(auto & l : m_newTargetsHistory) {
                for(auto & r : l) {
                    if(rr->y < m_horizonHeight)
                        continue;
                    if((r & *rr).area() > 0) { /* new target overlap previous new target */
                        ++overlap_count;
                    }
                }
            }
            if(rr->y >= m_horizonHeight) {
                if(overlap_count > 0) {
                    rr->x -= rr->width;
                    rr->y -= rr->height;
                    rr->width = rr->width << 1;
                    rr->height = rr->height << 1;
                }
                newTargetList.push_back(*rr);
            }

            if(enableFakeTargetDetection && 
                overlap_count >= 2) {
#ifdef DEBUG
                printf("<X> Fake target : (%u)\n", overlap_count);
#endif
            } else {
                m_targets.push_back(Target(*rr, capFrame, m_lastFrameTick));
#ifdef DEBUG
                printf("\033[0;32m"); /* Green */
                printf("<%u> New target : [%lu](%d, %d)\n", m_targets.back().m_id, m_lastFrameTick, rr->tl().x, rr->tl().y);
                printf("\033[0m"); /* Default color */
#endif
            }
        }

        m_newTargetsHistory.push_back(newTargetList);
        if(m_newTargetsHistory.size() >= 90) {
            m_newTargetsHistory.pop_front();
        }

        if(m_targets.size() > 1) {
            if(MaxTrackedCountOfTargets() > 6)
                m_targets.sort(TargetSortByTrackedCount);
            else
                m_targets.sort(TargetSortByArea);
        }
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

void contour_moving_object(Mat & frame, Mat & foregroundMask, list<Rect> & roiRect)
{
    uint32_t num_target = 0;

    vector< vector<Point> > contours;
    vector< Vec4i > hierarchy;
//  findContours(foregroundMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    findContours(foregroundMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
#if 1 /* Anti exposure burst */   
    if(contours.size() > 64)
        return;
#endif
    vector<Rect> boundRect;

    for(int i=0; i<contours.size(); i++) {
        Rect r = boundingRect(Mat(contours[i]));
        if(r.area() < 400 && r.br().y > CAMERA_WIDTH * HORIZON_RATIO) {
            Point ct = Center(r);
            bool isMerged = false;
            for(auto it=boundRect.begin();it!=boundRect.end();++it) {
                if(it->area() < 400) { /* Less than 20x20 */
                    if((r & *it).area() > 0) { /* Merge overlaped rect */
                        isMerged = true;
                    } else if(cv::norm(ct - Center(*it)) < 36) { /* Merge closely enough rect */
                        isMerged = true;
                    }
                }
                if(isMerged) {
                    *it = MergeRect(r, *it);
                    break;
                }
            }
            if(isMerged)
                continue;
        }
        boundRect.push_back(r);
    }

    sort(boundRect.begin(), boundRect.end(), [](const Rect & r1, const Rect & r2) {  
            //return (contour1.size() > contour2.size()); /* Outline length */
            return (r1.area() > r2.area()); /* Area */
        }); /* Rects sort by area, boundRect[0] is largest */

    for(int i=0; i<boundRect.size(); i++) {
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
        if((maxVal - minVal) < 16)
            continue; /* Too small, drop it. */
#endif
#if 1
        if(roiFrame.cols > roiFrame.rows && (roiFrame.cols >> 4) > roiFrame.rows)
            continue; /* Ignore thin object */
#endif
        roiRect.push_back(boundRect[i]);
        if(++num_target >= MAX_NUM_TARGET)
            break;
    }
}

void extract_moving_object(Mat & frame, 
    Mat & elementErode, Mat & elementDilate, 
    Ptr<cuda::Filter> & erodeFilter, Ptr<cuda::Filter> & dilateFilter, 
    Ptr<cuda::BackgroundSubtractorMOG2> & bsModel, 
    list<Rect> & roiRect)
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

    contour_moving_object(frame, foregroundMask, roiRect);
}

void color_image_equalize_histogram(cv::Mat & src)
{
    Mat imageq;
    Mat histImage;

    vector<Mat> bgr;
    split(src, bgr);

    //equalize image
    equalizeHist(bgr[0], bgr[0]);
    equalizeHist(bgr[1], bgr[1]);
    equalizeHist(bgr[2], bgr[2]);

    merge(bgr, imageq);

    namedWindow("Equalized image", WINDOW_AUTOSIZE);
    imshow( "Equalized image", imageq );
}

int main(int argc, char**argv)
{
    double fps = 0;

    if(signal(SIGINT, sig_handler) == SIG_ERR)
        printf("\ncan't catch SIGINT\n");

    cuda::printShortCudaDeviceInfo(cuda::getDevice());
    std::cout << cv::getBuildInformation() << std::endl;

#ifdef IMAGE_INFERENCE    
    ImageInference *ifer = new ImageInference;
    cout << "Load " << MODEL_ENGINE << endl;
    cout << "Please wait ..." << endl;
// TODO : Dynamic model engine file path ...
    if(ifer->AllocContext(MODEL_ENGINE) == 0)
        ifer->oneInference();
    else {
        cout << "Fail to load " << MODEL_ENGINE << " -->> Abort !!!" << endl;
        exit(1);
    }
#endif

    Mat capFrame, bgFrame;
#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
    Mat outFrame;
#endif

#ifdef VIDEO_INPUT_FILE
    VideoCapture cap;
    if(argc > 1)
        cap.open(argv[1], cv::CAP_FFMPEG);
    else {
        cout << "Usage : dragon-eye-simulator <vidoe file>" << endl;
        return 0;
    }
#else
    int index = 0;    
    if(argc > 1)
        index = atoi(argv[1]);
    else {
        cout << "Usage : dragon-eye-simulator <vidoe device>" << endl;
        return 0;
    }
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
    cap.set(CAP_PROP_FPS, CAMERA_FPS);

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

    Tracker tracker(capFrame.cols, capFrame.rows);

#ifdef NEW_TARGET_RESTRICTION    
    //tracker.NewTargetRestriction(Rect(160, 1080, 400, 200));
    tracker.NewTargetRestriction(Rect(cx - 200, cy - 200, 400, 200));
    //tracker.NewTargetRestriction(Rect(cx - 360, cy - 200, 720, 200));
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

    double varThreshold = 8;
    char *mog2_threshold = getenv("MOG2_THRESHOLD");
    if(mog2_threshold) {
        int len = strlen(mog2_threshold);
        int i = 0;
        for(i=0;i<len;i++) {
            if(!isdigit(mog2_threshold[i]))
                break;
        }
        if(i == len) { // string is number
            varThreshold = atoi(mog2_threshold);
            printf("MOG2_THRESHOLD = %f\n", varThreshold);
        }
    }

    /* background history count, varThreshold, shadow detection */
    Ptr<cuda::BackgroundSubtractorMOG2> bsModel = cuda::createBackgroundSubtractorMOG2(30, varThreshold, false);
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

    auto lastTriggerTime(steady_clock::now());

    while(cap.read(capFrame)) {

        //cout << "timestamp : " << cap.get(CAP_PROP_POS_MSEC) << endl;

        ++loopCount;
#if 0
        /* (contrast) alpha = 2.2, (brightness) beta = 50 */   
        capFrame.convertTo(capFrame, -1, 2.2, 50);
#endif

#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
        capFrame.copyTo(outFrame);
        line(outFrame, Point(cx, 0), Point(cx, cy), Scalar(0, 255, 0), 1);
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
        char textStr[16];
        snprintf(textStr, 16, "%ld", loopCount);
        writeText( outFrame, textStr, Point(560, 40));

        //line(outFrame, Point(0, (CAMERA_WIDTH * 4 / 5)), Point(CAMERA_HEIGHT, (CAMERA_WIDTH * 4 / 5)), Scalar(127, 127, 0), 1);
        line(outFrame, Point(0, (CAMERA_WIDTH * HORIZON_RATIO)), Point(CAMERA_HEIGHT, (CAMERA_WIDTH * HORIZON_RATIO)), Scalar(0, 255, 255), 1);
#endif
        tracker.Update(roiRect, grayFrame, FAKE_TARGET_DETECTION);

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
            t->Draw(outFrame, true);
#endif
            if(t->TriggerCount() > 0 && t->TriggerCount() < MAX_NUM_TRIGGER) {
                doTrigger = true;
            }

            if(t->ArcLength() > MIN_COURSE_LENGTH &&
                t->AbsLength() > MIN_COURSE_LENGTH && 
                t->TrackedCount() > MIN_TARGET_TRACKED_COUNT) {
                if((t->BeginCenterPoint().x > cx && t->EndCenterPoint().x <= cx) ||
                    (t->BeginCenterPoint().x < cx && t->EndCenterPoint().x >= cx) ||
                    (t->PreviousCenterPoint().x > cx && t->CurrentCenterPoint().x <= cx) ||
                    (t->PreviousCenterPoint().x < cx && t->CurrentCenterPoint().x >= cx)) {
                    bool tgr = t->Trigger(BUG_TRIGGER);
                    if(doTrigger == false) {
                        doTrigger = tgr;
#ifdef IMAGE_INFERENCE    
                        if(ifer->ready && 
                            doTrigger && (t->LastRect().width >= 32 || t->LastRect().height >= 32)) {
printf("I N F E R E N C E\n");
                            Rect roi = t->LastRect();
                            if(roi.width < 32) {
                                roi.width = 32;
                            }
                            if(roi.height < 32) {
                                roi.height = 32;
                            }
                            auto wh = std::max(roi.width, roi.height);

                            Point v = t->Vectors().back();
                            if(v.x > 0 && roi.width < roi.height) {
                                roi.x = roi.x + roi.width - roi.height;
                            }

                            roi.width = roi.height = wh;

                            if(roi.x < 0)
                                roi.x = 0;

                            if(roi.y < 0)
                                roi.y = 0;

                            if(roi.x + roi.width >= CAMERA_HEIGHT)
                                roi.x = CAMERA_HEIGHT - roi.width - 1;

                            if(roi.y + roi.height >= CAMERA_WIDTH)
                                roi.y = CAMERA_WIDTH - roi.height - 1;

                            Mat roiFrame = capFrame(roi);
                            int r = ifer->Inference(roiFrame);
                            printf("Inference index is %d\n", r);
#if 0
                            static uint32_t s_roi_index = 0;

                            string sp(argv[1]);
                            sp.erase(remove(sp.begin(), sp.end(), '.'), sp.end());
                            replace(sp.begin(), sp.end(), '/', '_');

                            char dname[128];
                            if(r == 0)  
                                snprintf(dname, 128, "../P/%s", sp.c_str());
                            else
                                snprintf(dname, 128, "../N/%s", sp.c_str());

                            DIR *d;
                            d = opendir(dname);
                            if(d) {
                                /* Directory exists. */
                                closedir(d);
                            } else {
                                mkdir(dname, S_IRWXU);
                            }
                                
                            char fn[128];
                            if(r == 0)
                                snprintf(fn, 128, "../P/%s/%03d_%dx%d.png", sp.c_str(), s_roi_index++, roiFrame.cols, roiFrame.rows);
                            else
                                snprintf(fn, 128, "../N/%s/%03d_%dx%d.png", sp.c_str(), s_roi_index++, roiFrame.cols, roiFrame.rows);
                            imwrite(fn, roiFrame);
                            //namedWindow("ROI", WINDOW_AUTOSIZE);
                            //imshow("ROI", roiFrame);
#endif
                        }
#endif
#if 0                        
                        if(doTrigger && (t->LastRect().width >= 16 || t->LastRect().height >= 16)) {
printf("R O I F R A M E\n");
                            Rect roi = t->LastRect();
                            printf("roi (%d, %d) / [%d, %d]\n", roi.x, roi.y, roi.width, roi.height);
                            Point c = Center(roi);
                            if(roi.width < 64) {
                                roi.x = (roi.x >= 32) ? (roi.x - (32 - (roi.width / 2))) : 0;
                                roi.width = 64;
                                if((roi.x + roi.width) > capFrame.cols)
                                    roi.x = capFrame.cols - 65;
                            }
                            if(roi.height < 64) {
                                roi.y = (roi.y >= 32) ? (roi.y - (32 - (roi.height / 2))) : 0;
                                roi.height = 64;
                                if((roi.y + roi.height) > capFrame.rows)
                                    roi.y = capFrame.rows - 65;
                            }                                

                            Mat roiFrame = capFrame(roi);
                            Rect r = roi;
                            r.x = 0;
                            r.y = 0;
                            Mat rFrame = outFrame(r);
                            roiFrame.copyTo(rFrame);

                            static uint32_t s_roi_index = 0;
                            string sp(argv[1]);
                            sp.erase(remove(sp.begin(), sp.end(), '.'), sp.end());
                            replace(sp.begin(), sp.end(), '/', '_');
                            char fn[128];
                            snprintf(fn, 128, "../64x64/%s_%dx%d_%03d.png", sp.c_str(), rFrame.cols, rFrame.rows, s_roi_index++);
                            imwrite(fn, rFrame);
                        }
#endif                        
                    }
                }
            }
        }

        bool isNewTrigger = false;
        if(doTrigger) {
#if defined(VIDEO_OUTPUT_SCREEN) || defined(VIDEO_OUTPUT_FILE)
            line(outFrame, Point(cx, 0), Point(cx, cy), Scalar(0, 0, 255), 3);
#endif //VIDEO_OUTPUT_FRAME
            long long duration = duration_cast<milliseconds>(steady_clock::now() - lastTriggerTime).count();
            printf("duration = %lld\n" , duration);
            if(duration > 800) { /* new trigger */
                isNewTrigger = true;
            }
            lastTriggerTime = steady_clock::now();
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
/*
        Rect roi;
        roi.x = 0;
        roi.y = (CAMERA_WIDTH * HORIZON_RATIO);
        roi.width = CAMERA_HEIGHT;
        roi.height = CAMERA_WIDTH - (CAMERA_WIDTH * HORIZON_RATIO);

        Mat m = capFrame(roi);

        color_image_equalize_histogram(m);
*/
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
#if 0            
        } else if(k == 'p') { /* Press key 'p' to pause or resume */
#else
        } else if(k == 'p' || isNewTrigger) { /* Press key 'p' to pause or resume */
#endif
            while(waitKey(1) != 'p') {
                if(bShutdown)
                    break;
            }
        }
#endif
        if(bShutdown)
            break;

        steady_clock::time_point t2(steady_clock::now());
        auto it = t2 - t1;
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
            //std::cout << "FPS : " << fixed  << setprecision(2) <<  fps << std::endl;
            std::cout << "FPS : " << fixed  << setprecision(2) <<  fps << " / " << duration_cast<milliseconds>(it).count() << " ms" << std::endl;
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
