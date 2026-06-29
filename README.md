# RDK-X5-CEIV-Net-Robot
Edge control &amp; multi-threading scheduling code for 'Zhiban Tongxin' on RDK X5. Features: 1. Asyncio &amp; ThreadPoolExecutor concurrent architecture; 2. Real-time face, emotion &amp; gesture tracking (C++ &amp; ONNX); 3. Edge-cloud integration (Qwen-VL, Whisper, Tencent TTS); 4. Serial adaptive expression mirroring &amp; Rock-Paper-Scissors game logic.

# Zhiban Tongxin: Embodied Companion Robot &amp; Multimodal Dual-System Affective Computing Optimization Based on Horizon RDK X5

This repository contains the edge intelligent computing master scheduling hub and complete diagnostic suite for the **"Zhiban Tongxin" Embodied Companion Robot**, an award-winning project submitted to the National Embedded Chip and System Design Competition. Tailored for smart home companionship and primary psychological care scenarios, the system leverages the Horizon RDK X5 edge AI platform and its multi-core heterogeneous architecture to achieve a system-level closed-loop flow control, integrating high-efficiency visual perception, edge-cloud collaborative bilingual voice interaction, large language model (LLM) cognition, and embodied serial hardware control.

---

## 🚀 Core Technical Features

The edge master scheduling engine (anchored by `run_system.py`) delivers four distinct industrial-grade design highlights:

1. **Multi-Threaded Asynchronous High-Concurrency Architecture (Concurrent Execution)**
   * Built upon a hybrid concurrency control flow combining Python `asyncio` event loops with a `ThreadPoolExecutor` worker pool.
   * Physically isolates and concurrently schedules high-frequency camera frame capture (30FPS throughput), local AI engine inference inference pipelines, a Flask real-time streaming server, and the main interaction loop, achieving zero-deadlock thread execution under high-load multi-dimensional data streaming.

2. **Lightweight Edge Multimodal Perception &amp; Tracking (Edge ML Inference)**
   * Deeply integrates OpenCV DNN with a custom C++ inference framework, interfacing directly with the local execution engine via the `rdk_ai_module` dynamic library.
   * Implements a dynamic operation-mode hot-swapping mechanism, supporting high-recall single-camera face bounding-box detection, real-time 7-class facial emotion decoding (stabilized via an anti-shake queue to提纯 true state probability), and gesture target state-machine extraction.

3. **Edge-Cloud Collaborative High-Semantic Cognitive Loop (Edge-Cloud Collaboration)**
   * **Hearing (VAD &amp; ASR)**: Interfaces with the ALSA architecture via `arecord` and `sox` to filter out domestic ambient noise for lightweight Voice Activity Detection (VAD), streaming raw audio to a remote asynchronous **Whisper ASR Server** for millisecond-level textual decoding.
   * **Thinking (LLM)**: Utilizes Alibaba's DashScope framework to asynchronously call the **Qwen-VL-Flash Multimodal Large Language Model**, managing dense multi-turn conversation histories locally while keeping the end-to-end cloud roundtrip latency below 1.45 seconds.
   * **Speaking (TTS)**: Seamlessly mounts the **Tencent Cloud TTS** asynchronous long-task API, dynamically pulling raw PCM audio streams over the local audio bus to realize fluid, human-like verbal responses.

4. **Embodied Serial Kinematics &amp; Game Logic (Embodied Interaction)**
   * Abstracts a `SerialDisplayManager` bus controller to drive an external audiovisual expressions screen in real time over the `/dev/ttyS1` serial link at a 115200 baud rate.
   * **Active Follower Mirroring Task (`emotion_task`)**: Automatically triggers every 60 seconds for a 30-second window of real-time micro-expression tracking, mapping the user's explicit emotions instantly into 9 anthropomorphic gaze patterns and facial expressions to establish deep embodied empathy.
   * **Human-Robot Rock-Paper-Scissors Game (`play_game_logic`)**: Features a fully self-contained interactive loop combining voice prompts, vision-tokenized hand gesture recognition, randomized algorithm-based gaming matchups, and serial-based motor/display feedback.
  
---

## 🛠️ System Self-Diagnostic &amp; Hardware Validation Matri

To ensure the stability of the heterogeneous platform under extreme workloads, a comprehensive suite of hardware smoke-testing and diagnostic tools is deployed in the root directory. Execute these sequentially before launching the main control pipeline:

### 1. Microphone Hardware &amp; Cloud Connectivity Diagnostics (`test_audio.py`)
* **Objective**: Targets and resolves ALSA device deadlocks, microphone ADC sampling mismatches, and ngrok tunnel infrastructure stability.
* **Mechanism**:
  * **[Process Cleansing]**: Automatically invokes `fuser -k /dev/snd/*` to forcibly terminate zombie processes holding audio locks.
  * **[Low-Level Recording]**: Bypasses the VAD pipeline to execute a raw 5-second `arecord` session, verifying file integrity boundaries to eliminate null hardware capture.
  * **[Network Penetration]**: Initiates a synchronous `requests.post` pipeline to the remote Whisper API endpoint, measuring the ngrok tunnel's throughput and text translation response times.

---

### 2. Local Emotion Network &amp; Serial Display Co-Validation (`test_face.py`)
* **Objective**: Benchmarks the edge execution latency of the local ONNX emotion model and calibrates serial transmission control frame boundaries.
* **Mechanism**:
  * **[Execution Mode Locking]**: Drives the camera via `srcampy.Camera` and locks the `rdk_ai_module.VideoPipeline` to Mode 1 (Facial Affective Perception).
  * **[Serial Pipeline Mapping]**: Maps the 7 decoded textual emotion outputs into discrete control tokens (e.g., `Happy -> "1"`, `Sad -> "8"`) and streams them down `/dev/ttyS1` with state-differential filtering to prevent serial bus flooding.
  * **[Visualized Streaming]**: Spawns a local Flask instance to render dynamic color-coded bounding boxes on the remote dashboard (e.g., Red for Angry, Yellow for Happy), plotting rolling inference latency metrics (ms).

---

### 3. Local Hand Tracking &amp; Anti-Shake Game Diagnostics (`test_hand.py`)
* **Objective**: Isolates the hand gesture inference stream and stress-tests the hardware execution stability of the embodied mini-game.
* **Mechanism**:
  * **[Gesture Detection Engine]**: Forces the system to load C++ Execution Mode 2, utilizing the optimized `v5n.onnx` network on the edge processor.
  * **[Command Serializer]**: Standardizes the mapping of `Paper -> "a"`, `Scissors -> "b"`, `Rock -> "c"` into immediate hardware instructions.
  * **[Bus Stabilization Lock]**: Implements a `last_sent_label` state-lock mutex. Commands are only dispatched to the serial bus upon catching a stable, state-shifted gesture token, eliminating system hangs caused by bus collision under high current or dim lighting.

---

### 4. High-Throughput Computational Edge Stress-Test (`test_vision_only.py`)
* **Objective**: High Performance / High Recall limit test. Evaluates the maximum concurrent throughput of the asynchronous I/O thread system without peripheral serial overhead.
* **Mechanism**:
  * **[Zero-Copy Sync]**: Allocates a secure thread-safe data passage via a `data_lock` mutex, achieving zero-copy memory pointer swapping of high-frequency NV12 video frames between the camera worker thread and the ML worker thread.
  * **[Compute Optimization]**: Finetunes the Flask video thread streaming interval (`time.sleep(0.03)`). While guaranteeing a fluid 30FPS tracking monitor for the remote supervisor PC, it yields maximal CPU cycles back to the C++ hardware tensor pipeline to prevent kernel panics.
---

## 🛠️ Deployment &amp; Execution Guide

### 1. Environment Configuration &amp; Credentials De-sensitization
⚠️ **Security Notice**: Before pushing this system to a public GitHub repository or external evaluation servers, credentials must be cleansed. Open `run_system.py` and matching diagnostic tools to swap the raw tokens out with secure environment placeholders:

```python
# 1. Edge-Cloud Whisper Speech Recognition Server URL
WHISPER_SERVER_URL = "YOUR_LOCAL_OR_TUNNELED_WHISPER_SERVER_URL"

# 2. Horizon / Alibaba DashScope Large Language Model API Token
DASH_API_KEY = "YOUR_DASHSCOPE_API_KEY_HERE"

# 3. Tencent Cloud Online Speech Synthesis (TTS) Credentials
SECRET_ID = "YOUR_TENCENT_CLOUD_SECRET_ID_HERE"
SECRET_KEY = "YOUR_TENCENT_CLOUD_SECRET_KEY_HERE"
```

---

### 2. Local Dependencies & Model Verification

The architecture enforces a strict boot-time validation engine (check_models). Ensure the compiled model weights are nested correctly within the Vision__python_Cpp/models/ directory prior to initialization:

```bash
python3 -c "from run_system import check_models; check_models()"
```
---

### 3. Launching the System Main Loop

For independent debugging and testing of the voice interaction subsystem, you can directly activate the pre-configured isolated Python sandbox environment to execute the standalone voice kernel:

```bash
cd Voise_Subsystem
source bin/activate
python3 code/bot\ test.py
```

Once all individual peripheral hardware diagnostic validations pass successfully, exit the virtual environment and execute the multi-threaded master scheduling hub with sudo privileges under the global heterogeneous high-frequency clock domain:

```bash
sudo python3 run_system.py
```

## 📂 Project Multi-Level Directory Architecture

The system's modular layout decouples hardware primitives from cloud cognitive architectures according to industry standards. The cascade repository tree is laid out as follows:

```text
MULTIMODAL_INTERACTION_SYSTEM/         # Root Directory of the Embodied System
│
├── run_system.py                     # 🚀 Main: Multi-threaded Edge Master Controller & Closed-Loop Hub
├── test_audio.py                     # Diagnostic: Hardware Recording, ALSA Purge, & Whisper Connectivity
├── test_face.py                      # Diagnostic: Edge Emotion Inference & Serial Mirroring Integration
├── test_hand.py                      # Diagnostic: Edge Gesture Tracking & Rock-Paper-Scissors Bus Lock
├── test_vision_only.py               # Stress-Test: Max-Throughput Asynchronous Video Pipeline Validation
│
├── 📂 Vision__python_Cpp/             # 1. Python-C++ Hybrid Vision Inference Framework
│   ├── build/                        # Local Binaries and Compiled Dynamic Libraries
│   ├── include/                      # Core C++ Vision Optimization Headers
│   ├── models/                       # Local Network Weights (Face XML, Emotion ONNX, and Gesture ONNX)
│   │   ├── haarcascade_frontalface_default.xml
│   │   ├── emotion.onnx
│   │   └── v5n.onnx
│   ├── src/                          # Computational Feature Extraction C++ Sources
│   ├── CMakeLists.txt                # CMake Build System Configuration Profile
│   └── main.py                       # Local Single-Module Diagnostic Entry for Python-C++ Wrapper
│
├── 📂 Vision_Subsystem/               # 2. Pure C++ Low-Level Image Processing Core
│   ├── build/                        # Local C++ Compilation Cache
│   ├── include/                      # Base Image Conversion and Alignment Primitives
│   ├── src/                          # Fast NV12 Matrix Formatting Source Files
│   ├── CMakeLists.txt                # Subsystem Build and Project Link Management Profile
│   └── main.cpp                      # Pure C++ Native Program Standalone Entry
│
└── 📂 Voice_Subsystem/                # 3. Runtime Voice Stream I/O Exchange Buffer
    └── code/
        └── audio/ 
            └── live.flac              # High-Frequency Asynchronous Audio Buffer File
```

