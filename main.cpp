#include"openvino.hpp"

using namespace std;
using namespace cv;

int main()
{
    std::string MODEL_PATH = "../model/best.xml";
    Mat test_img = imread("../test.jpg");
    Mat test_cpy = test_img.clone();
    int enemy = 0;
    openvino openvino(MODEL_PATH, test_img, enemy);
    openvino.detect(test_img);
    openvino.draw(test_cpy);
    return 0;
}