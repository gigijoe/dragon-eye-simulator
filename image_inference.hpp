/*
*
*/
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>

#include <unistd.h>
#include <dirent.h> 
#include <sys/stat.h>

const char *MODEL_ENGINE = "../model.engine";

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 32;
static const int INPUT_W = 32;
static const int OUTPUT_SIZE = 2;

const char *INPUT_BLOB_NAME = "input";
const char *OUTPUT_BLOB_NAME = "output";

using namespace nvinfer1;

static Logger gLogger;

class ImageInference {
private:
    IRuntime *runtime;
    IExecutionContext *context;
    ICudaEngine *engine;

    void* buffers[2];
    int inputIndex;
    int outputIndex;

    cudaStream_t stream;

    void doInference(float* input, float* output, int batchSize)
    {
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context->enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

public:
    bool ready;

    ImageInference() : runtime(nullptr), context(nullptr), engine(nullptr), ready(false) {}

    int AllocContext(const char *engine_file) {
        // create a model using the API directly and serialize it to a stream
        char *trtModelStream{nullptr};
        size_t size{0};

        std::ifstream file(engine_file, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        } else {
            std::cout << "Error open engine file " << engine_file << " !!!" << std::endl;
            return -1;
        }

        runtime = createInferRuntime(gLogger);
        if(runtime == nullptr)
            return -1;
        engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
        delete[] trtModelStream;
        if(engine == nullptr) {
            ReleaseContext();
            return -1;
        }
        if(engine->getNbBindings() != 2) {
            printf("engine: number of binding is NOT 2 !!! %d\n", engine->getNbBindings());
            ReleaseContext();
            return -1;
        }
        context = engine->createExecutionContext();
        if(context == nullptr) {
            ReleaseContext();
            return -1;
        }

        printf("Bindings after deserializing:\n");
        for(int bi = 0; bi < engine->getNbBindings(); bi++) {
            if(engine->bindingIsInput(bi) == true) {
                printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
                printf("%s\n", engine->getBindingFormatDesc(bi));
            } else {
                printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
                printf("%s\n", engine->getBindingFormatDesc(bi));
            }
        }

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        //assert(engine->getNbBindings() == 2);
        //void* buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        const int batchSize = 1;

        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

        // Create stream
        CHECK(cudaStreamCreate(&stream));

        ready = true;

        return 0;
    }

    void ReleaseContext() {
        if(context != nullptr && engine != nullptr && runtime != nullptr) {
            // Release stream and buffers
            cudaStreamDestroy(stream);
            CHECK(cudaFree(buffers[inputIndex]));
            CHECK(cudaFree(buffers[outputIndex]));
        }

        if(context != nullptr)
            context->destroy();
        if(engine != nullptr)
            engine->destroy();
        if(runtime != nullptr)
            runtime->destroy();

        runtime = nullptr;
        context = nullptr;
        engine = nullptr;

        ready = false;
    }

    void oneInference()
    {
        const int batchSize = 1;
        // Do very first inference to avoid slow processing
        float prob[OUTPUT_SIZE];
        // Subtract mean from image
        float one[3 * INPUT_H * INPUT_W];
        for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
            one[i] = 1.0;

        doInference(one, prob, batchSize);    
    }

    int Inference(cv::Mat img)
    {
        const int batchSize = 1;
        float prob[OUTPUT_SIZE];

        auto start = std::chrono::system_clock::now();
        // Do inference now
        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32FC3, 1.f / 255.f);

    //transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    // https://blog.csdn.net/hello_dear_you/article/details/119863264
        cv::subtract(img, cv::Scalar(0.4914, 0.4822, 0.4465), img, cv::noArray(), -1);
        cv::divide(img, cv::Scalar(0.2023, 0.1994, 0.2010), img, 1, -1);

    // TensorRT requires your image data to be in NCHW order. But OpenCV reads this in the NHWC order.
    // https://pfnet-research.github.io/menoh/md_tutorial.html
        std::vector<float> chw(img.channels() * img.rows * img.cols);
        for(int y = 0; y < img.rows; ++y) {
            for(int x = 0; x < img.cols; ++x) {
                for(int c = 0; c < img.channels(); ++c) {
                    chw[c * (img.rows * img.cols) + y * img.cols + x] =
                      img.at<cv::Vec3f>(y, x)[c];
                }
            }
        }

        float *data = &chw[0];

        doInference(data, prob, batchSize);
        auto end = std::chrono::system_clock::now();

        if(prob[0] < prob[1])
            printf("\033[0;31m"); /* Red */
        
        for(unsigned int i = 0; i < OUTPUT_SIZE; i++) {
            std:cout << setprecision(5);
            std::cout << prob[i] << ", ";
        }
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        printf("\033[0m"); /* Default color */

        return prob[0] > prob[1] ? 0 : 1;
    }
};
