#include <iostream>
#include <string>
#include <fstream>
#include <glob.h>
#include <vector>
#include <codecvt>
#include <sys/stat.h>
#include "opencv2/opencv.hpp"

using namespace std;

vector<string> globVector(const string& pattern){
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    vector<string> files;
    for(unsigned int i=0; i<glob_result.gl_pathc; ++i){
        files.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

string getNameFromFilename(string filename){
    wstring_convert<codecvt_utf8_utf16<wchar_t> > converter;
    wstring wfile = converter.from_bytes(filename);

    int start_symbol = wfile.find_last_of(converter.from_bytes("/")) + 1;
    int end_symbol = wfile.find_last_of(converter.from_bytes("."));
    string name = converter.to_bytes(wfile.substr(start_symbol, end_symbol - start_symbol));
    return name;
}

int main()
{
    const char* folder = "../data";
    struct stat sb;

    if (!(stat(folder, &sb) == 0 && S_ISDIR(sb.st_mode))) {
        mkdir(folder, 0777);
    }

    ofstream embeddings;
    embeddings.open ("../data/embeddings");

    ofstream labels;
    labels.open("../data/labels");

    auto recognizer = cv::dnn::readNetFromCaffe("../models/resnet50_256.prototxt", "../models/resnet50_256.caffemodel");
    auto detector = cv::dnn::readNetFromTensorflow("../models/opencv_face_detector_uint8.pb", "../models/opencv_face_detector.pbtxt");

    std::string face_dir = "../faces/";

    vector<string> files = globVector(face_dir + "*");

    for (auto file : files){

        string name = getNameFromFilename(file);

        cv::Mat image = cv::imread(file);

        cv::Mat inputBlob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300), {104, 177, 123});

        detector.setInput(inputBlob, "data");
        cv::Mat detection = detector.forward("detection_out");

        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        cv::Mat embed;

        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if(confidence > 0.95)
            {
                int x1 = max(0, static_cast<int>(detectionMat.at<float>(i, 3) * image.size().width));
                int y1 = max(0, static_cast<int>(detectionMat.at<float>(i, 4) * image.size().height));
                int x2 = min(image.size().width - 1, static_cast<int>(detectionMat.at<float>(i, 5) * image.size().width));
                int y2 = min(image.size().height - 1, static_cast<int>(detectionMat.at<float>(i, 6) * image.size().height));

                cv::Point point1 = cv::Point(x1, y1), point2 = cv::Point(x2, y2);

                cv::Mat face = image(cv::Rect(point1, point2));
                cv::Mat inputBlob = cv::dnn::blobFromImage(face, 1.0, cv::Size(224, 224), {93.5940, 104.7624, 129.1863});

                recognizer.setInput(inputBlob, "data");
                int new_dims[] = {256, 1};
                embed = recognizer.forward("feat_extract").reshape(1, 2, new_dims);

                break;
            }
        }

        cout << name << endl;
        labels << name << endl;
        for(int i = 0; i < 256; ++i){
            embeddings << embed.at<float>(0,i) << " ";
        }
        embeddings << std::endl;
    }
    return 0;
}
