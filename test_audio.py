#!/usr/bin/env python3
import os
import time
import subprocess
import requests # 这里的测试用同步库，方便看日志
import sys

# ================= 配置 =================
# 和你的 run_system.py 保持一致
MIC_DEVICE = "plughw:1,0" 
AUDIO_SAVE_PATH = "./debug_audio.flac"
WHISPER_SERVER_URL = "https://unbigoted-subumbonal-cordia.ngrok-free.dev/transcribe"

print("========================================")
print("🔍 系统音频与网络诊断工具")
print("========================================")

# 1. 检查麦克风设备
print(f"\n[1/4] 检查麦克风设备 ({MIC_DEVICE})...")
cmd_check = "arecord -l"
os.system(cmd_check)

# 2. 测试录音 (强制录 5 秒，不使用静音检测，排除 sox 逻辑干扰)
print(f"\n[2/4] 测试强制录音 5秒...")
print(f"👉 请对着麦克风大声说话！")
kill_cmd = "fuser -k /dev/snd/* >/dev/null 2>&1"
os.system(kill_cmd) # 清理占用

# 使用最简单的 arecord 命令，不通过 sox 管道，排除 sox 问题
cmd_rec = f"arecord -D {MIC_DEVICE} -f S16_LE -c 1 -r 16000 -d 5 -t wav {AUDIO_SAVE_PATH}"
print(f"执行指令: {cmd_rec}")

try:
    # 去掉 stderr=subprocess.DEVNULL 以便看到报错
    subprocess.run(cmd_rec, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ 录音失败！错误代码: {e}")
    sys.exit(1)

# 检查文件是否存在及大小
if os.path.exists(AUDIO_SAVE_PATH):
    size = os.path.getsize(AUDIO_SAVE_PATH)
    print(f"✅ 录音结束。文件路径: {AUDIO_SAVE_PATH}, 大小: {size/1024:.2f} KB")
    if size < 100:
        print("❌ 警告：文件过小，可能麦克风没有录入声音！")
else:
    print("❌ 错误：录音文件未生成！")
    sys.exit(1)

# 3. 测试 Whisper 连接
print(f"\n[3/4] 测试 Whisper 服务连接...")
print(f"目标地址: {WHISPER_SERVER_URL}")
try:
    with open(AUDIO_SAVE_PATH, 'rb') as f:
        t0 = time.time()
        response = requests.post(WHISPER_SERVER_URL, files={'file': f}, timeout=10)
        t1 = time.time()
        
    print(f"HTTP状态码: {response.status_code}")
    print(f"耗时: {t1-t0:.2f}秒")
    
    if response.status_code == 200:
        print(f"✅ 识别结果: {response.json()}")
    else:
        print(f"❌ 服务端报错: {response.text}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    print("可能原因：ngrok 地址过期了，或者服务器未启动。")

# 4. 完成
print("\n========================================")
print("诊断结束。")