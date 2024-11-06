#include "openvino.hpp"

void openvino::prepare_engine(std::string& MODEL_PATH, std::string& DEVICE, ov::Core& core, ov::CompiledModel& compiled_model, ov::InferRequest& infer_request) {
    compiled_model = core.compile_model(MODEL_PATH, DEVICE);
    infer_request = compiled_model.create_infer_request();
    ov::Tensor input_tensor = infer_request.get_input_tensor(0);
    this->input_buffer_host = input_tensor.data<float>();
}

//这个函数顺便完成了input_buffer_host的初始化
void openvino::get_input_size(ov::InferRequest& infer_request, In_dimension& in_dims) {
    ov::Tensor input_tensor = infer_request.get_input_tensor(0);
    this->input_buffer_host = input_tensor.data<float>();
    ov::Shape tensor_shape = input_tensor.get_shape();
    in_dims.channel = tensor_shape[1];
    in_dims.input_h = tensor_shape[2];
    in_dims.input_w = tensor_shape[3];
}

void openvino::get_metrics(int input_h, int input_w, int output_h, int output_w, Metrics& metrics) {
    float r = std::min(output_h / input_h, output_w / input_w);
    int padw = std::round(input_w * r);
    int padh = std::round(input_h * r);
    float dw = output_w - padw;
    float dh = output_h - padh;
    dw /= 2.0f;
    dh /= 2.0f;
    metrics.dw = dw;
    metrics.dh = dh;
    metrics.r = r;
    metrics.origin_h = input_h;
    metrics.origin_w = input_w;
}

void openvino::resize(const Mat& inputimg, Mat& outputimg, int input_h, int input_w, int output_h, int output_w) {
    float r = std::min((float)output_h / input_h, (float)output_w / input_w);
    int padw = std::round(input_w * r);
    int padh = std::round(input_h * r);
    cv::Mat tmp1;
    if (input_w != padw || input_h != padh) {
        cv::resize(inputimg, tmp1, cv::Size(padw, padh));
    }
    else {
        tmp1 = inputimg.clone();
    }

    
    float dw = output_w - padw;
    float dh = output_h - padh;
    
    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw-0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp1, outputimg, top, bottom, left, right, cv::BORDER_CONSTANT, { 0, 0, 0 });

    if (outputimg.size() != cv::Size(output_w, output_h)) {
        cv::resize(outputimg, outputimg, cv::Size(output_w, output_h));
    }
}

void openvino::preprocess(Mat& inputimg, float* inputbuffer, In_dimension& indimension) {
    cv::imshow("input", inputimg);  
    waitKey(0);
    cv::Mat normalizedImage, tmp;
    int h = inputimg.rows;
    int w = inputimg.cols;
    resize(inputimg, tmp, h, w, indimension.input_h, indimension.input_w);
    tmp.convertTo(normalizedImage, CV_32F, 1.0 / 255.0);
    float* data = normalizedImage.ptr<float>();
    std::memcpy(inputbuffer, data, indimension.channel * indimension.input_h * indimension.input_w * sizeof(float));
}

void openvino::infer(ov::InferRequest& infer_request) {
    infer_request.infer();
}

void openvino::get_output(ov::InferRequest& infer_request, float* output_buffer_host, Out_dimension& out_dims) {
    //TODO: 这里的get_output_tensor(0)只能在infer之后使用吗？
    ov::Tensor output_tensor = infer_request.get_output_tensor();
// FIXME: 在detect时这里的memcpy并没有成功, output_buffer_host的值并没有改变
    std::memcpy(output_buffer_host, output_tensor.data<float>(), out_dims.model_bboxes * out_dims.data_frame_size * sizeof(float));
}

//这个函数顺便完成了output_buffer_host的初始化
void openvino::get_output_size(ov::InferRequest& infer_request, Out_dimension& out_dims) {
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    ov::Shape tensor_shape = output_tensor.get_shape();
    this->output_buffer_host = output_tensor.data<float>();
    out_dims.model_bboxes = tensor_shape[1];
    out_dims.data_frame_size = tensor_shape[2];
}
void openvino::test_input(ov::InferRequest& infer_request, Mat& inputimg, int input_h, int input_w, Metrics& metrics) {
    //TODO: 不知道out_dims是否有用
    get_input_size(infer_request, this->indimension);
    preprocess(inputimg, this->input_buffer_host, this->indimension);
    int h = inputimg.rows;
    int w = inputimg.cols;
    get_metrics(h, w, input_h, input_w, metrics);
    infer(infer_request);
    get_output_size(infer_request, this->outdimension);

}

void openvino::postprocess(Out_dimension& out_dims, float* output_buffer_host, vector<Box>& objects, Metrics& metrics, int enemy) {
    objects.clear();
    int bboxes_num = out_dims.model_bboxes;
    int data_frame_size = out_dims.data_frame_size;

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    std::vector<std::vector<float>> kpss;

    for (size_t i = 0; i < bboxes_num; i++)
    {
        float* head_ptr = output_buffer_host;
        float* bboxes_ptr = head_ptr + i;
        float* scores_ptr_red = head_ptr + 4 * bboxes_num + i;
        float* scores_ptr_blue = head_ptr + 5 * bboxes_num + i;
        float* scores_ptr = ((*scores_ptr_red) > (*scores_ptr_blue)) ? scores_ptr_red : scores_ptr_blue;
        //TODO: 颜色信息不一定要储存
        int color = ((*scores_ptr_red) > (*scores_ptr_blue)) ? 0 : 1;
        float* kps_ptr = head_ptr + 6 * bboxes_num + i;

        float score = *scores_ptr;

        if (score > metrics.score_thres && color == enemy) {
            auto clamp = [](float val, float min, float max) {
                return val > min ? (val < max ? val : max) : min;
            }; // 保证坐标位于图像内

            float x = *bboxes_ptr - metrics.dw;
            float y = *(bboxes_ptr + bboxes_num) - metrics.dh;
            float w = *(bboxes_ptr + 2 * bboxes_num) / metrics.r;
            float h = *(bboxes_ptr + 3 * bboxes_num) / metrics.r;

            float x0 = clamp((x - 0.5f * w), 0.f, metrics.origin_w);
            float y0 = clamp((y - 0.5f * h), 0.f, metrics.origin_h);
            float x1 = clamp((x + 0.5f * w), 0.f, metrics.origin_w);
            float y1 = clamp((y + 0.5f * h), 0.f, metrics.origin_h);

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = abs(x1 - x0);
            bbox.height = abs(y1 - y0);

            std::vector<float> kps;
            for (int k = 0; k < 4; k++) {
                //FIXME: 问题应该出在下两行, metrics有问题
                float kps_x = (*(kps_ptr + 2 * k * bboxes_num) - metrics.dw) / metrics.r;
                float kps_y = (*(kps_ptr + (2 * k + 1) * bboxes_num) - metrics.dh) / metrics.r;
                kps_x = clamp(kps_x, 0.f, metrics.origin_w);
                kps_y = clamp(kps_y, 0.f, metrics.origin_h);
                kps.push_back(kps_x);
                kps.push_back(kps_y);
            }

            bboxes.push_back(bbox);
            //TODO: 颜色信息不一定要储存
            labels.push_back(color);
            scores.push_back(score);
            kpss.push_back(kps);
        }
    }

    cv::dnn::NMSBoxes(bboxes, scores, metrics.score_thres, metrics.iou_thres, indices);
    
    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= metrics.topk) {
            break;
        }
        Box obj;
        obj.bbox[0] = bboxes[i].y;
        obj.bbox[1] = bboxes[i].y + bboxes[i].height;
        obj.bbox[2] = bboxes[i].x;
        obj.bbox[3] = bboxes[i].x + bboxes[i].width;
        obj.conf = scores[i];
        obj.color = labels[i];
        obj.kps = kpss[i];
        objects.push_back(obj);
        cnt += 1;
    }
}

void openvino::detect(Mat& inputimg) { 
    preprocess(inputimg, this->input_buffer_host, this->indimension);
    infer(this->infer_request);
    //TODO: 每次都要get outDims, 想要在正式推理前进行一次示例推理，然后获取outDims并存在类中
    get_output(this->infer_request, this->output_buffer_host, this->outdimension);
    postprocess(outdimension, this->output_buffer_host, this->Objects, this->metrics, this->metrics.enemy);
    Mat img_show = inputimg.clone();
    draw(img_show, this->Objects, this->metrics, this->SKELETON);
    cv::imshow("result", img_show);
    cv::waitKey(0);

}

void openvino::detect(float* data, Mat& img_show) {
    std::memcpy(this->input_buffer_host, data, this->indimension.channel * this->indimension.input_h * this->indimension.input_w * sizeof(float));
    infer(this->infer_request);
    get_output(this->infer_request, this->output_buffer_host,  this->outdimension);
    postprocess(outdimension, this->output_buffer_host, this->Objects, this->metrics, this->metrics.enemy);
    draw(img_show, this->Objects, this->metrics, this->SKELETON);
    cv::imshow("result", img_show);
    cv::waitKey(0);
}

void openvino::mem_check(float* data, Out_dimension& out_dims) {
    int bboxes = out_dims.model_bboxes;
    int data_frame_size = out_dims.data_frame_size;
    for (size_t i = 0; i < bboxes; i++)
    {
        for (size_t i = 0; i < data_frame_size ; i++)
        {
            std::cout << data[i] <<' ';
        }
        std::cout << std::endl;
    }
    
}

void openvino::data_check() {
    mem_check(this->output_buffer_host, this->outdimension);
}

void openvino::draw_box(Mat& cpy, Box& obj, Metrics& metrics, const std::vector<std::vector<unsigned int>>& SKELETON){
    cv::rectangle(cpy, cv::Point(obj.bbox[0], obj.bbox[2]), cv::Point(obj.bbox[1],obj.bbox[3]), cv::Scalar(0, 255, 0), 2);
    char text[256];
    int      baseLine = 0;
    sprintf(text, "%.1f%%", obj.conf * 100);
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::putText(cpy, text, cv::Point(obj.bbox[2], obj.bbox[0]),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    auto& kps = obj.kps;
    int num_kpss = kps.size() / 2;

    for (size_t i = 0; i < num_kpss; i++)
    {
        auto& ske = SKELETON[i];
        int kps_x1 = std::round(kps[ske[0] * 2]);
        int kps_y1 = std::round(kps[ske[0] * 2 + 1]);

        int kps_x2 = std::round(kps[ske[1] * 2]);
        int kps_y2 = std::round(kps[ske[1] * 2 + 1]);

        cv::line(cpy, cv::Point(kps_x1, kps_y1), cv::Point(kps_x2, kps_y2), cv::Scalar(255, 0, 0), 2);

    }

}
void openvino::draw(Mat& cpy, std::vector<Box>& Objects, Metrics& metrics, const std::vector<std::vector<unsigned int>>& SKELETON) {
    for (auto& obj : Objects) {
        draw_box(cpy, obj, metrics, SKELETON);
    }
}

void openvino::draw(Mat& cpy) {
    draw(cpy, this->Objects, this->metrics, this->SKELETON);
}
openvino::openvino(std::string& MODEL_PATH, Mat& test_img, int enemy) {
    this->MODEL_PATH = MODEL_PATH;
    this->test_img = test_img.clone();
    this->metrics.enemy = enemy;
    prepare_engine(this->MODEL_PATH, this->DEVICE, this->core, this->compiled_model, this->infer_request);

    //FIXME: 这里的input_h和input_w传到test_input后就错了
    test_input(this->infer_request, this->test_img, this->indimension.input_h, this->indimension.input_w, this->metrics);
}

openvino::openvino(std::string& MODEL_PATH, std::string& DEVICE, Mat& test_img, int enemy) {
    this->DEVICE = DEVICE;
    openvino(MODEL_PATH, test_img, enemy);
}

openvino::~openvino() {}
