#ifndef HAND_ENGINE_H
#define HAND_ENGINE_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

struct HandResult {
    cv::Rect box;
    std::string label;
    cv::Scalar color;
};

class HandEngine {
public:
    // 构造函数：模型路径
    HandEngine(const std::string& onnxPath);
    
    // 核心接口
    std::vector<HandResult> detect(const cv::Mat& frame);

private:
    cv::dnn::Net handNet;
    std::vector<std::string> classNames;
    std::vector<cv::Scalar> colors;

    // 参数 (保留之前的极速版参数)
    const float CONF_THRES = 0.15f; 
    const float NMS_THRES = 0.45f;
    const int INPUT_SIZE = 320;

    void letterbox(const cv::Mat& src, cv::Mat& dst, float& ratio, float& dw, float& dh);
};

#endif // HAND_ENGINE_H