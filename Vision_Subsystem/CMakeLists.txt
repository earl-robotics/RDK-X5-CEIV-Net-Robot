cmake_minimum_required(VERSION 3.10)
project(ModularSystem)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 关键：要把 include 文件夹加进去
include_directories(include)

# 开启多线程
find_package(Threads REQUIRED)

# 添加所有源文件
add_executable(final_app 
    main.cpp 
    src/face_engine.cpp 
    src/hand_engine.cpp
)

target_link_libraries(final_app ${OpenCV_LIBS} Threads::Threads)