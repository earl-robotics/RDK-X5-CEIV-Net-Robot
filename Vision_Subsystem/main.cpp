#include "face_engine.h"
#include "hand_engine.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <ifaddrs.h>
#include <arpa/inet.h>

using namespace std;
using namespace cv;

// === 全局状态 ===
Mat global_frame;
mutex data_mutex;
atomic<bool> is_running(true);

vector<FaceResult> g_face_results;
vector<HandResult> g_hand_results;

// === 【新增】UDP 发送相关变量 ===
int udp_sock_fd = -1;
struct sockaddr_in udp_server_addr;

// === 【新增】初始化 UDP 发送器 ===
void init_udp_sender() {
    udp_sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_sock_fd < 0) {
        cerr << "❌ UDP Socket 创建失败!" << endl;
        return;
    }
    udp_server_addr.sin_family = AF_INET;
    udp_server_addr.sin_port = htons(5005); // 目标端口 5005 (Python监听这个)
    udp_server_addr.sin_addr.s_addr = inet_addr("127.0.0.1"); // 本机
}

// === 【新增】发送视觉数据函数 ===
void send_vision_data(string gesture, string emotion) {
    if (udp_sock_fd < 0) return;
    // 格式: "Gesture:xxx|Emotion:xxx"
    string msg = "Gesture:" + gesture + "|Emotion:" + emotion;
    sendto(udp_sock_fd, msg.c_str(), msg.length(), 0, (struct sockaddr*)&udp_server_addr, sizeof(udp_server_addr));
}

// 图像增强
void adjustGammaDisplay(const Mat& src, Mat& dst) {
    if (src.empty()) return;
    static Mat lut(1, 256, CV_8U);
    static bool init = false;
    if (!init) {
        uchar* p = lut.ptr();
        for (int i = 0; i < 256; ++i) p[i] = saturate_cast<uchar>(pow(i / 255.0, 1.0 / 1.5) * 255.0);
        init = true;
    }
    LUT(src, lut, dst);
}

// === 线程 A: 面部 AI ===
void face_loop() {
    FaceEngine engine("haarcascade_frontalface_default.xml", "emotion.onnx");
    Mat frame;
    while(is_running) {
        {
            lock_guard<mutex> lock(data_mutex);
            if(global_frame.empty()) { this_thread::sleep_for(chrono::milliseconds(5)); continue; }
            global_frame.copyTo(frame);
        }
        
        if (!frame.empty()) {
            vector<FaceResult> results = engine.detect(frame);
            {
                lock_guard<mutex> lock(data_mutex);
                g_face_results = results;
            }
        }
        this_thread::sleep_for(chrono::milliseconds(30));
    }
}

// === 线程 B: 手势 AI ===
void hand_loop() {
    HandEngine engine("v5n.onnx");
    Mat frame;
    while(is_running) {
        {
            lock_guard<mutex> lock(data_mutex);
            if(global_frame.empty()) { this_thread::sleep_for(chrono::milliseconds(5)); continue; }
            global_frame.copyTo(frame);
        }
        
        if (!frame.empty()) {
            vector<HandResult> results = engine.detect(frame);
            {
                lock_guard<mutex> lock(data_mutex);
                g_hand_results = results;
            }
        }
        this_thread::sleep_for(chrono::milliseconds(15));
    }
}

// === 辅助: 打印 IP ===
void printIP(int port) {
    struct ifaddrs *interfaces = nullptr;
    getifaddrs(&interfaces);
    for(struct ifaddrs *t=interfaces; t!=nullptr; t=t->ifa_next) {
        if(t->ifa_addr && t->ifa_addr->sa_family==AF_INET) {
            void* p = &((struct sockaddr_in*)t->ifa_addr)->sin_addr;
            char b[INET_ADDRSTRLEN]; inet_ntop(AF_INET, p, b, INET_ADDRSTRLEN);
            if(string(b)!="127.0.0.1") cout << "👉 Web 观看地址: http://" << b << ":" << port << "/video" << endl;
        }
    }
}

// === 视频流处理 ===
void handleClient(int clientSocket, VideoCapture& cap) {
    char buffer[1024] = {0};
    read(clientSocket, buffer, 1024);
    string req(buffer);

    // 忽略模式切换指令，只响应视频流
    if (req.find("GET /video") != string::npos) {
        string header = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        send(clientSocket, header.c_str(), header.size(), MSG_NOSIGNAL);
        
        Mat frame, display, flipped;
        vector<uchar> buf;
        vector<FaceResult> faces;
        vector<HandResult> hands;

        while(is_running) {
            bool cam_ok = cap.isOpened();
            if (cam_ok) {
                cap >> frame;
                if (frame.empty()) cam_ok = false;
            }

            // 断线重连逻辑
            if (!cam_ok) {
                cap.release(); 
                // 优先尝试 MIPI (ID 8)
                int ids[] = {8, 0, 1, 4}; 
                for(int id : ids) {
                    int fd = dup(STDERR_FILENO); freopen("/dev/null", "w", stderr);
                    cap.open(id, CAP_V4L2);
                    dup2(fd, STDERR_FILENO); close(fd);
                    if(cap.isOpened()) {
                        cap.set(CAP_PROP_FRAME_WIDTH, 320);
                        cap.set(CAP_PROP_FRAME_HEIGHT, 240);
                        cap.set(CAP_PROP_FPS, 30);
                        cout << "✅ 摄像头重连成功 ID: " << id << endl;
                        break;
                    }
                }
                if (!cap.isOpened()) {
                    // 发送错误图
                    Mat errorImg(240, 320, CV_8UC3, Scalar(0,0,0));
                    putText(errorImg, "NO CAMERA", Point(80, 120), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 2);
                    imencode(".jpg", errorImg, buf);
                    string body = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + to_string(buf.size()) + "\r\n\r\n";
                    body.append(string(buf.begin(), buf.end()));
                    body += "\r\n";
                    send(clientSocket, body.c_str(), body.size(), MSG_NOSIGNAL);
                    
                    this_thread::sleep_for(chrono::seconds(1)); 
                    continue; 
                }
            }
            
            // 正常处理
            flip(frame, flipped, 1);
            if (flipped.empty()) continue;

            {
                lock_guard<mutex> lock(data_mutex);
                flipped.copyTo(global_frame);
                faces = g_face_results;
                hands = g_hand_results;
            }

            // ================= 【新增】融合逻辑：提取结果并发送 =================
            string current_emotion = "Neutral";
            string current_gesture = "None";

            // 简单策略：取第一个检测到的人脸表情
            if (!faces.empty()) {
                current_emotion = faces[0].label; 
            }
            // 简单策略：取第一个检测到的手势
            if (!hands.empty()) {
                current_gesture = hands[0].label;
            }

            // 发送 UDP 数据包给 Python
            send_vision_data(current_gesture, current_emotion);
            // ================================================================

            adjustGammaDisplay(flipped, display);
            if (display.empty()) continue;

            // 绘制
            for(const auto& f : faces) {
                 rectangle(display, f.box, Scalar(0,255,0), 2);
                 putText(display, f.label, Point(f.box.x, f.box.y-5), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);
            }
            for(const auto& h : hands) {
                 rectangle(display, h.box, h.color, 2);
                 putText(display, h.label, Point(h.box.x, h.box.y-5), FONT_HERSHEY_SIMPLEX, 0.6, h.color, 2);
            }

            putText(display, "System: Fusion Ready", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);

            if (!display.empty()) {
                vector<int> p = {IMWRITE_JPEG_QUALITY, 70};
                try {
                    imencode(".jpg", display, buf, p);
                } catch (...) {
                    continue;
                }

                string body = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + to_string(buf.size()) + "\r\n\r\n";
                body.append(string(buf.begin(), buf.end()));
                body += "\r\n";
                if(send(clientSocket, body.c_str(), body.size(), MSG_NOSIGNAL) < 0) break;
            }
            
            this_thread::sleep_for(chrono::milliseconds(30));
        }
    }
    close(clientSocket);
}

int main() {
    // 1. 初始化网络发送
    init_udp_sender();
    
    // 2. 初始化摄像头 (优先尝试 8号 MIPI)
    VideoCapture cap;
    int ids[] = {8, 0, 1, 4}; 
    for(int id : ids) {
        int fd = dup(STDERR_FILENO); freopen("/dev/null", "w", stderr);
        cap.open(id, CAP_V4L2);
        dup2(fd, STDERR_FILENO); close(fd);
        if(cap.isOpened()) {
            cout << "✅ 启动时检测到摄像头 ID: " << id << endl;
            cap.set(CAP_PROP_FRAME_WIDTH, 320);
            cap.set(CAP_PROP_FRAME_HEIGHT, 240);
            cap.set(CAP_PROP_FPS, 30);
            break;
        }
    }

    // 3. 启动 AI 线程
    thread t1(face_loop);
    thread t2(hand_loop);
    t1.detach();
    t2.detach();

    // 4. 启动 Web Server
    int server = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr; addr.sin_family = AF_INET; addr.sin_port = htons(5000); addr.sin_addr.s_addr = INADDR_ANY;
    int opt = 1; setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    bind(server, (struct sockaddr*)&addr, sizeof(addr));
    listen(server, 5);
    
    cout << "🚀 多模态融合系统启动! (Python端口: 5005)" << endl;
    printIP(5000);
    
    while(true) {
        int client = accept(server, nullptr, nullptr);
        if(client >= 0) {
            thread(handleClient, client, ref(cap)).detach();
        }
    }
}