#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>


using namespace std;
using namespace cv;


struct Box
{
    float bbox[4];//上下左右
    float conf;//置信度
    int color;//类别
    vector<float> kps;//关键点
};


//输出单元维度
struct Out_dimension
{
    int model_bboxes;//输出框数
    int data_frame_size;//输出数据帧大小
};

struct In_dimension
{
    int channel;//输入图片通道，3
    int input_h;//输入图片高640
    int input_w;//输入图片宽640
};

struct Metrics{
    int topk = 100;
    float score_thres = 0.2f;
    float iou_thres = 0.65f;
    float dw;
    float dh;
    float r;
    int origin_h;
    int origin_w;
    int enemy;      //0 for red, 1 for blue
};
class openvino
{
private:
    const std::vector<std::vector<unsigned int>> SKELETON = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}
    };
    Mat test_img;
    std::string MODEL_PATH;                                                     //模型路径
    std::string DEVICE = "CPU";                                                           //设备

    ov::Core core;                                                                        //openvino核心

    ov::CompiledModel compiled_model/*= core.compile_model(MODEL_PATH, DEVICE)*/;           //编译模型

    ov::InferRequest infer_request;                                                         //创建推理请求

    Out_dimension outdimension;                                                             //输出维度
    In_dimension indimension;                                                               //输入维度

    Metrics metrics;                                                                        //评价指标

    int BatchSize = 1;                                                                        //batchsize默认为1
    int num_class = 2;                                                                            //输出的类别数
    int max_outbboxes = 50;                                                                //最多输出的框数目

    float* input_buffer_host = nullptr;                                                       //输入内存缓存空间
    float* decoded_buffer = nullptr;                                                           //解码结果内存缓存空间

    uchar* mat_buffer = nullptr;                                                              //保存uchar格式的图片数据
    uchar* mat_buffer_beforeresize = nullptr;                                                 //保存resize后的uchar格式的图片数据
    //图片resize后会保持比例放在左上角
    float ratiox;                                                                           //x方向与原图的比例
    float ratioy;                                                                           //y方向与原图的比例
    int px;                                                                                 //x填充
    int py;                                                                                 //y填充


public:
    Mat cpy;                                                                                //输出图片                   
    vector<Box> Objects;                                                              //输出结果的数组
    float* output_buffer_host = nullptr;                                                      //输出内存缓存空间

private:
    void prepare_engine(std::string& MODEL_PATH, std::string& DEVICE, ov::Core& core, ov::CompiledModel& compiled_model, ov::InferRequest& infer_request);         //准备引擎

    void get_input_size(ov::InferRequest& infer_request, In_dimension& in_dims) ;

    void resize(const Mat& inputimg, Mat& outputimg, int input_h, int input_w, int output_h, int output_w);

    void preprocess(Mat& inputimg, float* inputbuffer, In_dimension& indimension);

    void test_input(ov::InferRequest& infer_request, Mat& inputimg, int input_h, int input_w, Metrics& metrics);

    void infer(ov::InferRequest& infer_request);

    void get_output(ov::InferRequest& infer_request, float* output_buffer_host, Out_dimension& out_dims);

    void postprocess(Out_dimension& out_dims, float* output_buffer_host, vector<Box>& objects, Metrics& metrics, int enemy);

    void get_output_size(ov::InferRequest& infer_request, Out_dimension& out_dims);

    void mem_check(float* data, Out_dimension& out_dims);

    void get_metrics(int input_h, int input_w, int output_h, int output_w, Metrics& metrics);

    void draw_box(Mat& cpy, Box& objects, Metrics& metrics, const std::vector<std::vector<unsigned int>>& SKELETON);
                                                //绘制结果到原图

    void draw(Mat& cpy, std::vector<Box>& Objects, Metrics& metrics, const std::vector<std::vector<unsigned int>>& SKELETON);                                 //绘制结果到指定图

public:
    void detect(float* data, Mat& img_show);       //直接输入预处理后的数据来推理
    void detect(Mat &inputimage);

    void draw(Mat& cpy);    //绘制结果到制定图

    void data_check();

    openvino(std::string& MODEL_PATH, Mat& test_img, int enemy);
    openvino(std::string& MODEL_PATH, std::string& DEVICE, Mat& test_img, int enemy);

    ~openvino();
};
