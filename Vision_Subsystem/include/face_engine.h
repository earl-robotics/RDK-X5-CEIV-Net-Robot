#ifndef FACE_ENGINE_H
#define FACE_ENGINE_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

struct FaceResult {
    cv::Rect box;
    std::string label;
};

class FaceEngine {
public:
    FaceEngine(const std::string& xmlPath, const std::string& onnxPath);
    
    // 核心接口：传入一帧，返回检测结果
    std::vector<FaceResult> detect(const cv::Mat& frame);

private:
    cv::CascadeClassifier faceCascade;
    cv::dnn::Net emotionNet;
    std::vector<std::string> emotionNames;
    
    // 内部状态 (用于防闪烁)
    cv::Rect last_face_box;
    std::string last_face_label;
    int face_missing_frames;
    const int MAX_MISSING_FRAMES = 2; // 短时记忆

    void adjustGamma(const cv::Mat& src, cv::Mat& dst, double gamma);
};

#endif // FACE_ENGINE_H