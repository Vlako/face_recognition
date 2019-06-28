#ifndef FACERECOGNIZER_H
#define FACERECOGNIZER_H

#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

#include "face.h"

class FaceRecognizer
{
public:
    FaceRecognizer();

    Face recognize(cv::Mat frame);

private:
    cv::dnn::Net detector;
    cv::dnn::Net recognizer;

    std::vector<cv::Mat> embeddings;
    std::vector<std::string> labels;
};

#endif // FACERECOGNIZER_H
