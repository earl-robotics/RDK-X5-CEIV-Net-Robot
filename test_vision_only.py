#!/usr/bin/env python3
import sys
import os
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response

# ================= 1. 基础配置 =================
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.10/dist-packages")
sys.path.append("/usr/lib/python3.10/dist-packages")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_DIR = os.path.join(ROOT_DIR, "Vision__python_Cpp")
sys.path.append(VISION_DIR)

# 确保模型路径正确
FACE_XML = os.path.join(VISION_DIR, "models", "haarcascade_frontalface_default.xml")
EMOTION_ONNX = os.path.join(VISION_DIR, "models", "emotion.onnx")
HAND_ONNX = os.path.join(VISION_DIR, "models", "v5n.onnx")

try:
    from hobot_vio import libsrcampy as srcampy
except ImportError:
    from hobot_vio_rdkx5 import libsrcampy as srcampy

import rdk_ai_module

# ================= 2. 全局变量 =================
data_lock = threading.Lock()
latest_frame_data = None 
current_mode = 2  # 2 = Hand Gesture
ai_result_hand = "None"
ai_result_face = "None"
ai_hand_box = [] 

# ================= 3. 后台 AI 线程 =================
def ai_worker_thread():
    global latest_frame_data, ai_result_hand, ai_result_face, ai_hand_box, current_mode
    
    print("🧠 [AI] 正在初始化 C++ 引擎 (High Recall Mode)...")
    if not os.path.exists(HAND_ONNX):
        print(f"❌ 错误: 模型文件未找到: {HAND_ONNX}")
        return

    pipeline = rdk_ai_module.VideoPipeline(FACE_XML, EMOTION_ONNX, HAND_ONNX)
    print("✅ [AI] 引擎加载完毕")
    
    while True:
        frame_bytes = None
        mode_now = 2
        with data_lock:
            if latest_frame_data is not None:
                frame_bytes = latest_frame_data
            mode_now = current_mode
        
        if frame_bytes is None:
            time.sleep(0.01)
            continue
            
        try:
            # 调用 C++ 接口
            res = pipeline.process_frame(frame_bytes, mode_now)
            
            # res结构: (dummy_bytes, face_str, hand_str, hand_box_list)
            new_face = res[1]
            new_hand = res[2]
            new_box = res[3]
            
            with data_lock:
                ai_result_face = new_face
                ai_result_hand = new_hand
                ai_hand_box = new_box
                
        except Exception as e:
            print(f"❌ AI Loop Error: {e}")
            
        # 稍微休眠防止完全占满一个核，给视频流线程留点资源
        time.sleep(0.005)

# ================= 4. 视频流线程 =================
app = Flask(__name__)
cam = None

def init_camera():
    global cam
    # 杀掉可能占用摄像头的进程
    os.system("sudo fuser -k /dev/video* > /dev/null 2>&1")
    os.system("echo 1 > /sys/class/gpio/gpio353/value 2>/dev/null")
    
    cam = srcampy.Camera()
    # 0: pipe_id, -1: sensor_id, 30: fps, 640x480 resolution
    cam.open_cam(0, -1, 30, 640, 480)
    print("📷 [Camera] 摄像头已开启")

def generate_stream():
    global latest_frame_data
    
    while True:
        if cam is None: time.sleep(1); continue
        
        # 1. 获取 NV12 图像
        img = cam.get_img(2, 640, 480)
        if img is None: time.sleep(0.01); continue
        
        # 2. 同步数据给 AI 线程
        hand_text = "None"
        hand_rect = []
        face_text = "None"
        
        with data_lock:
            latest_frame_data = img
            hand_text = ai_result_hand
            hand_rect = ai_hand_box
            face_text = ai_result_face
            
        # 3. 转码用于显示
        img_np = np.frombuffer(img, dtype=np.uint8)
        img_yuv = img_np.reshape((720, 640))
        img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)
        img_bgr = cv2.flip(img_bgr, -1) 
        
        # 4. 绘制 UI
        if current_mode == 2: # 手势
            cv2.putText(img_bgr, "Mode: Hand Gesture", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if hand_text != "None":
                color = (0, 255, 0)
                # 画框
                if len(hand_rect) == 4:
                    x, y, w, h = hand_rect
                    # 边界保护
                    x = max(0, x); y = max(0, y)
                    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 3)
                    
                    label_y = max(y - 10, 20)
                    cv2.putText(img_bgr, hand_text, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                else:
                    cv2.putText(img_bgr, f"Result: {hand_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            else:
                cv2.putText(img_bgr, "Waiting...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        elif current_mode == 1: # 表情
            cv2.putText(img_bgr, "Mode: Face Emotion", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img_bgr, f"Emotion: {face_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        # 5. 推流
        ret, jpeg = cv2.imencode('.jpg', img_bgr)
        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        # 限制显示帧率，留出 CPU 给 AI
        time.sleep(0.03)

@app.route("/")
@app.route("/video")
def video_feed():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ================= 5. 主程序 =================
if __name__ == "__main__":
    if os.geteuid() != 0:
        print("❌ 请使用 sudo 运行！")
        sys.exit(1)

    init_camera()

    t_ai = threading.Thread(target=ai_worker_thread)
    t_ai.daemon = True
    t_ai.start()
    
    # 默认锁定在手势模式
    current_mode = 2

    print("\n" + "="*50)
    print("🚀 RDK X5 视觉测试 (High Performance)")
    print("📡 视频流: http://IP:5000")
    print("="*50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)