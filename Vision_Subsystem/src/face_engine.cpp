#include "face_engine.h"
#include <iostream>

using namespace cv;
using namespace std;

FaceEngine::FaceEngine(const string& xmlPath, const string& onnxPath) {
    if (!faceCascade.load(xmlPath)) cerr << "❌ FaceEngine: XML加载失败 " << xmlPath << endl;
    
    emotionNet = dnn::readNetFromONNX(onnxPath);
    if (emotionNet.empty()) cerr << "❌ FaceEngine: 模型加载失败 " << onnxPath << endl;
    else {
        emotionNet.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        emotionNet.setPreferableTarget(dnn::DNN_TARGET_CPU);
    }

    emotionNames = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};
    face_missing_frames = 0;
    last_face_label = "";
}

void FaceEngine::adjustGamma(const Mat& src, Mat& dst, double gamma) {
    static Mat lut(1, 256, CV_8U);
    static bool init = false;
    if (!init) {
        uchar* p = lut.ptr();
        for (int i = 0; i < 256; ++i) p[i] = saturate_cast<uchar>(pow(i / 255.0, 1.0 / gamma) * 255.0);
        init = true;
    }
    LUT(src, lut, dst);
}

vector<FaceResult> FaceEngine::detect(const Mat& frame) {
    vector<FaceResult> results;
    if (frame.empty()) return results;

    Mat enhanced, gray;
    adjustGamma(frame, enhanced, 1.5);
    cvtColor(enhanced, gray, COLOR_BGR2GRAY);

    vector<Rect> faces;
    // 宽松检测
    faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, Size(50, 50));

    bool found_valid = false;

    if (!faces.empty()) {
        Rect largest = faces[0];
        for (const auto& f : faces) {
            if (f.area() > largest.area()) largest = f;
        }

        // 核心逻辑：比例过滤防鼻孔
        float ratio = (float)largest.width / largest.height;
        if (ratio > 0.75 && ratio < 1.3) {
            found_valid = true;
            Mat roi = gray(largest);
            Mat blob;
            dnn::blobFromImage(roi, blob, 1.0/255.0, Size(48, 48), Scalar(), false, false);
            emotionNet.setInput(blob);
            Mat prob = emotionNet.forward();
            Point classId;
            minMaxLoc(prob, 0, 0, 0, &classId);

            last_face_box = largest;
            last_face_label = emotionNames[classId.x];
            face_missing_frames = 0;

            results.push_back({largest, last_face_label});
        }
    }

    // 防闪烁缓冲
    if (!found_valid) {
        if (face_missing_frames < MAX_MISSING_FRAMES && last_face_box.width > 0) {
            face_missing_frames++;
            results.push_back({last_face_box, last_face_label});
        }
    }
    return results;
}