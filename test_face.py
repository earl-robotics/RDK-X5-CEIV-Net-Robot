#!/usr/bin/env python3
import sys
import os
import time
import threading
import cv2
import numpy as np
import serial
from flask import Flask, Response

# ================= 1. 路径配置 =================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_DIR = os.path.join(ROOT_DIR, "Vision__python_Cpp")

sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.10/dist-packages")
sys.path.append(VISION_DIR)

# 串口配置
SERIAL_PORT = "/dev/ttyS1"
BAUD_RATE = 115200

# 模型路径
FACE_XML = os.path.join(VISION_DIR, "models", "haarcascade_frontalface_default.xml")
EMOTION_ONNX = os.path.join(VISION_DIR, "models", "emotion.onnx")
HAND_ONNX = os.path.join(VISION_DIR, "models", "v5n.onnx")

try:
    from hobot_vio import libsrcampy as srcampy
except:
    pass

import rdk_ai_module

# ================= 2. 全局变量 =================
shared_data = {
    "frame_nv12": None,      
    "ai_face_res": "Init...",     
    "ai_box": [],        
    "ai_time": 0.0,          
    "lock": threading.Lock() 
}

# ================= 3. 串口控制 =================
class SerialManager:
    def __init__(self):
        self.ser = None
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"🔌 串口已连接: {SERIAL_PORT}")
        except Exception as e:
            print(f"❌ 串口连接失败: {e}")

    def send(self, char_cmd):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(char_cmd.encode('utf-8'))
            except:
                pass

display = SerialManager()

# ================= 4. 核心线程 =================

# --- 摄像头线程 ---
def thread_camera_capture():
    os.system("sudo fuser -k /dev/video* > /dev/null 2>&1")
    cam = srcampy.Camera()
    cam.open_cam(0, -1, 30, 640, 480)
    print("📷 摄像头启动成功")
    
    while True:
        img = cam.get_img(2, 640, 480)
        if img is not None:
            with shared_data["lock"]:
                shared_data["frame_nv12"] = img
        time.sleep(0.005)

# --- AI 推理线程 (含串口) ---
def thread_ai_inference():
    print("🧠 正在加载 AI 引擎...")
    pipeline = rdk_ai_module.VideoPipeline(FACE_XML, EMOTION_ONNX, HAND_ONNX)
    print("✅ AI 引擎加载完毕")
    
    emotion_map = {
        "Happy": "1", "Neutral": "2", "Surprise": "4",
        "Angry": "5", "Disgust": "6", "Fear": "7", "Sad": "8"
    }
    
    last_sent_emotion = "None"
    
    while True:
        curr_img = None
        with shared_data["lock"]:
            if shared_data["frame_nv12"] is not None:
                curr_img = shared_data["frame_nv12"]
        
        if curr_img is None:
            time.sleep(0.01); continue
            
        t0 = time.time()
        try:
            # mode=1 (面部)
            res = pipeline.process_frame(curr_img, 1)
            
            # 现在的 C++ 已经返回了真实的防抖结果和框
            face_label = res[1] 
            face_box = res[3]  # 获取人脸框
            
            # 串口联动
            if face_label in emotion_map and face_label != last_sent_emotion:
                cmd = emotion_map[face_label]
                display.send(cmd)
                print(f"👉 稳定表情: {face_label} -> 屏幕: {cmd}")
                last_sent_emotion = face_label
            elif face_label == "None" and last_sent_emotion != "None":
                display.send("2")
                last_sent_emotion = "None"
                
        except: 
            face_label = "Err"; face_box = []

        cost = (time.time() - t0) * 1000.0
        
        with shared_data["lock"]:
            shared_data["ai_face_res"] = face_label
            shared_data["ai_box"] = face_box # 存入真实的框
            shared_data["ai_time"] = shared_data["ai_time"] * 0.9 + cost * 0.1
        
        time.sleep(0.001)

# --- 显示线程 (使用真实框) ---
app = Flask(__name__)
import logging
log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)

def generate_frames():
    while True:
        time.sleep(0.03)
        img_raw = None
        text = ""; lat = 0.0; box = []
        
        with shared_data["lock"]:
            img_raw = shared_data["frame_nv12"]
            text = shared_data["ai_face_res"]
            box = shared_data["ai_box"]
            lat = shared_data["ai_time"]
            
        if img_raw is None: continue
        
        img_np = np.frombuffer(img_raw, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_np.reshape((720, 640)), cv2.COLOR_YUV2BGR_NV12)
        img_bgr = cv2.flip(img_bgr, -1) 
        
        # 🔥 绘制真实的 AI 框
        if len(box) == 4 and text != "None":
            x, y, w, h = box
            
            # 颜色根据情绪变
            color = (0, 255, 0) # 默认绿
            if text == "Angry": color = (0, 0, 255) # 生气红
            if text == "Happy": color = (0, 255, 255) # 开心黄
            
            # 直接画框 (坐标已由 C++ 对齐)
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_bgr, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        cv2.putText(img_bgr, f"Time: {lat:.1f}ms", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        (flag, jpg) = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if flag: yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(jpg) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<body style='background:#000; margin:0; display:flex; justify-content:center; align-items:center; height:100vh;'><img src='/video_feed' style='max-width:100%; max-height:100vh;'></body>"

if __name__ == "__main__":
    os.system("sudo fuser -k 5000/tcp > /dev/null 2>&1")
    threading.Thread(target=thread_camera_capture, daemon=True).start()
    threading.Thread(target=thread_ai_inference, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)