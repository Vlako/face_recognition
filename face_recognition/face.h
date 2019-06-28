#ifndef FACE_H
#define FACE_H

#include <string>
#include "opencv2/opencv.hpp"

class Face
{
public:
    Face(std::string name, cv::Point point1, cv::Point point2);
    std::string name;
    cv::Point point1, point2;
};

#endif // FACE_H
