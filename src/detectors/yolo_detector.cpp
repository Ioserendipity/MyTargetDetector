#include "mytargetdetector/detectors/yolo_detector.hpp"
#include "mytargetdetector/common/constants.hpp"
#include <iostream>

namespace my_detector {

YoloDetector::YoloDetector(const std::string& model_path, float conf_threshold, float nms_threshold)
    : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {
    
    net_ = cv::dnn::readNetFromONNX(model_path);
    if (net_.empty()) {
        std::cerr << "FATAL: Failed to load ONNX model from " << model_path << std::endl;
        // 在无法加载模型时抛出异常是一种更好的实践
        throw std::runtime_error("Could not load ONNX model");
    }
    
    // 尝试使用CUDA后端，如果OpenCV编译时支持的话
    // 这能极大提升在NVIDIA GPU上的性能
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    
    std::cout << "[YoloDetector] Model loaded successfully." << std::endl;
    
    input_width_ = 640;
    input_height_ = 640;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame) {
    cv::Mat blob;
    // 将图像转换为YOLO模型需要的格式 (blob)
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(input_width_, input_height_), cv::Scalar(), true, false);

    net_.setInput(blob);
    std::vector<cv::Mat> outs;
    // 执行前向传播
    net_.forward(outs, net_.getUnconnectedOutLayersNames());

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    // YOLOv5的输出尺寸是 [1, 25200, 85]
    // 25200 是检测框的数量
    // 85 = 4 (cx, cy, w, h) + 1 (objectness) + 80 (class scores)
    const int dimensions = 5 + COCO_CLASSES.size();
    const int rows = outs[0].size[1];
    float* data = (float*)outs[0].data;

    // 将检测结果从模型输出尺寸缩放到原始图像尺寸
    float x_factor = (float)frame.cols / input_width_;
    float y_factor = (float)frame.rows / input_height_;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4]; // 物体置信度
        if (confidence >= conf_threshold_) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, COCO_CLASSES.size(), CV_32F, classes_scores);
            cv::Point class_id_point;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

            // 结合物体置信度和类别置信度
            if (max_class_score * confidence > conf_threshold_) {
                confidences.push_back(confidence * (float)max_class_score);
                class_ids.push_back(class_id_point.x);
                
                float cx = data[0]; float cy = data[1]; float w = data[2]; float h = data[3];
                int left = (int)((cx - w / 2) * x_factor);
                int top = (int)((cy - h / 2) * y_factor);
                int width = (int)(w * x_factor);
                int height = (int)(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += dimensions;
    }

    // 使用非极大值抑制 (NMS) 消除重叠的框
    std::vector<int> nms_result_indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, nms_result_indices);

    std::vector<Detection> detections;
    for (int idx : nms_result_indices) {
        Detection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }
    
    return detections;
}

} // namespace my_detector
