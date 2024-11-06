#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include<string>
#include<iostream>

static const cv::Scalar mean(0.3000, 0.3020, 0.4224);
static const cv::Scalar stddev(0.2261, 0.2384, 0.2214);

int main() {
    std::string IMAGE_PATH = "../1504.png";
    std::string MODEL_PATH = "../model_num.xml";
    std::string DEVICE = "CPU";

    ov::Core core;

    ov::CompiledModel compiled_model = core.compile_model(MODEL_PATH, DEVICE);

    ov::InferRequest infer_request = compiled_model.create_infer_request();

    ov::Tensor input_tensor = infer_request.get_input_tensor(0);
    ov::Shape tensor_shape = input_tensor.get_shape();

    size_t channel = tensor_shape[1];
    size_t height = tensor_shape[2];
    size_t width = tensor_shape[3];
    std::cout << "channel: " << channel << std::endl;
    std::cout << "height: " << height << std::endl;
    std::cout << "width: " << width << std::endl;

    cv::Mat blob_image = cv::imread(IMAGE_PATH);
    // cv::Mat blob_image;
    // std::cout << "src size: " << src.size() << std::endl;
    // cv::resize(src, blob_image, cv::Size(width, height)); //注释这一行会segmentation fault
    // std::cout << "blob_image size: " << blob_image.size() << std::endl;
    cv::dnn::blobFromImage(blob_image, blob_image, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);

    blob_image = blob_image - mean;
    blob_image = blob_image / stddev;
    float* image_data = input_tensor.data<float>();
    float* data = blob_image.ptr<float>();

    auto start1 = std::chrono::high_resolution_clock::now();
    std::memcpy(image_data, data, channel * height * width * sizeof(float));
    auto end1 = std::chrono::high_resolution_clock::now();
    // for (size_t c = 0; c < channel; c++) {
    //     for (size_t h = 0; h < height; h++) {
    //         for (size_t w = 0; w < width; w++) {
    //             size_t index = c * width * height + h * width + w;
    //             image_data[index] = blob_image.at<cv::Vec3f>(h, w)[c];
    //             std::cout << "image:" << image_data[index] << std::endl;
    //             image_datalibrary[index] = data[index];
    //             std::cout <<"iamge_"<< image_data[index] << std::endl;
    //         }
    //     }
    // }
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

    auto start = std::chrono::high_resolution_clock::now();
    //同步推理
    infer_request.infer();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //异步推理

    //从输出节点自动获取推理结果，适用于单输出模型；
    auto output1 = infer_request.get_output_tensor();

    //根据输出节点编号获取推理结果，适用于多输出模型；
    //auto output1 = infer_request.get_output_tensor(0);

    //根据输出节点名称获取推理结果，适用于多输出模型；
    //auto output1 = infer_request.get_tensor("20");

    const float* output_buffer = output1.data<const float>();

    //Method1: 采用for loop比较
    // float max_pb = output_buffer[0];
    // int predict = 0;
    // for (int i = 1; i < 10; i++) {
    //     if (output_buffer[i] > max_pb) {
    //         max_pb = output_buffer[i];
    //         predict = i;
    //     }
    // }

    //Method2: max_element
    for (int i = 0; i < 6; i++) {
        std::cout << "The probability of number " << i << " is:" << output_buffer[i] << std::endl;
    }
    int predict = std::max_element(output_buffer, output_buffer + 6) - output_buffer;
    std::cout << "The prediction number is:" << predict << std::endl;
    std::cout << "The inference time is:" << duration.count() << "ms" << std::endl;
    std::cout << "The memcpy time is:" << duration1.count() << "ms" << std::endl;
}

