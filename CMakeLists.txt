cmake_minimum_required(VERSION 3.15)
project(test)

set(CMAKE_CXX_STANDARD 20)
set(OpenVINO_DIR "/opt/intel/openvino_2024.4.0/runtime/cmake")



find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED) 

add_executable(main main.cpp)

add_subdirectory(openvino)


include_directories(
        #OpenVINO推理引擎的头文件
        "/opt/intel/openvino_2024.4.0/runtime/include"
)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        
)

include_directories(
        "/opt/intel/openvino_2024.4.0/runtime/3rdparty/tbb/include"
)

link_directories(
        #OpenVINO推理引擎的库文件
        "/opt/intel/openvino_2024.4.0/runtime/lib/intel64"
)

link_directories(
        "/opt/intel/openvino_2024.4.0/runtime/3rdparty/tbb/lib"
)

target_link_libraries(main PUBLIC openvino::runtime) #加了这一行就可以了

# link_libraries(libopenvino.so)

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenVINO_LIBS: ${OpenVINO_LIBS}")
message(STATUS "OpenVINO_DIR: ${OpenVINO_DIRS}")
message(STATUS "OpenVINO_LIBRARIES: ${InferenceEngine_LIBRARIES}")
message(STATUS "OpenVINO_INCLUDE_DIRS: ${InferenceEngine_INCLUDE_DIRS}")
message(STATUS "OpenVINO_INCLUDE_DIRS: ${OpenVINO_INCLUDES}")

target_link_libraries(
        main
        PUBLIC
        ${OpenCV_LIBS}
        openvino
)