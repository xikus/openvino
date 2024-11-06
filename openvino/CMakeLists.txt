# 查找当前目录下的src文件夹中的所有源文件
aux_source_directory(./src openvino_DIR)

# 生成共享库
add_library(openvino ${openvino_DIR})

# 设置库的包含目录#
target_include_directories(openvino PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED) 

target_include_directories(
        openvino PUBLIC
        "/opt/intel/openvino_2024.4.0/runtime/include"
)#OpenVINO推理引擎的头文件

target_include_directories(
        openvino PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        
)

target_include_directories(
        openvino PUBLIC
        "/opt/intel/openvino_2024.4.0/runtime/3rdparty/tbb/include"
)

target_link_directories(
        openvino PUBLIC
        #OpenVINO推理引擎的库文件
        "/opt/intel/openvino_2024.4.0/runtime/lib/intel64"
)

target_link_directories(
        openvino PUBLIC
        "/opt/intel/openvino_2024.4.0/runtime/3rdparty/tbb/lib"
)

target_link_libraries(openvino PUBLIC openvino::runtime) #加了这一行就可以了

target_link_libraries(openvino PUBLIC ${OpenCV_LIBS})