#!/usr/bin/env python3
import sys
import os
import time
import threading
import cv2
import numpy as np
import serial # 👈 新增：串口库
from flask import Flask, Response

# ================= 1. 配置区域 =================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_DIR = os.path.join(ROOT_DIR, "Vision__python_Cpp")

sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/usr/local/lib/python3.10/dist-packages")
sys.path.append(VISION_DIR)

# 串口配置 (对应你之前的设置)
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
    "ai_res": "Init...",     
    "ai_box": [],        
    "ai_time": 0.0,          
    "lock": threading.Lock() 
}

# ================= 3. 串口控制类 (新增) =================
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
                # print(f"📤 发送指令: {char_cmd}") # 调试用
            except:
                pass

# 初始化串口对象
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

# --- AI 推理 + 串口控制线程 ---
def thread_ai_inference():
    print("🧠 正在加载 AI 引擎...")
    pipeline = rdk_ai_module.VideoPipeline(FACE_XML, EMOTION_ONNX, HAND_ONNX)
    print("✅ AI 引擎加载完毕")
    
    # 映射关系：AI识别到的英文 -> 屏幕指令字符
    # 之前逻辑: Paper=a, Scissors=b, Rock=c
    hand_map = {
        "Paper": "a",
        "Scissors": "b",
        "Rock": "c"
    }
    
    last_sent_label = "None" # 用于防抖，防止重复发送
    
    while True:
        curr_img = None
        with shared_data["lock"]:
            if shared_data["frame_nv12"] is not None:
                curr_img = shared_data["frame_nv12"]
        
        if curr_img is None:
            time.sleep(0.01)
            continue
            
        t0 = time.time()
        try:
            # 强制模式 2 (手势)
            res = pipeline.process_frame(curr_img, 2)
            label = res[2]
            box = res[3]
            
            # === 🔥 串口联动逻辑 🔥 ===
            # 1. 只有当识别到有效手势 (在字典里)
            # 2. 并且手势跟上一次不一样 (防止刷屏)
            if label in hand_map and label != last_sent_label:
                cmd = hand_map[label]
                display.send(cmd) # 发送给屏幕
                print(f"👉 识别到: {label} -> 发送屏幕: {cmd}")
                last_sent_label = label # 更新记录
            
            # 如果手消失了，是否要重置？
            # 这里的逻辑是保持最后一个手势，或者你可以取消注释下面两行来发送"眨眼(2)"恢复默认
            # if label == "None" and last_sent_label != "None":
            #     display.send("2") 
            #     last_sent_label = "None"
                
        except: 
            label = "Err"; box = []

        cost = (time.time() - t0) * 1000.0
        
        with shared_data["lock"]:
            shared_data["ai_res"] = label
            shared_data["ai_box"] = box
            shared_data["ai_time"] = shared_data["ai_time"] * 0.9 + cost * 0.1
        
        time.sleep(0.001)

# --- 显示线程 (保持不变) ---
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
            text = shared_data["ai_res"]
            box = shared_data["ai_box"]
            lat = shared_data["ai_time"]
            
        if img_raw is None: continue
        
        img_np = np.frombuffer(img_raw, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_np.reshape((720, 640)), cv2.COLOR_YUV2BGR_NV12)
        img_bgr = cv2.flip(img_bgr, -1)
        
        if len(box) == 4 and text != "None":
            x, y, w, h = box
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        color = (0, 255, 0) if lat < 50 else (0, 0, 255)
        cv2.putText(img_bgr, f"Time: {lat:.1f}ms", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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