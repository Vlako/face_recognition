#include <fstream>

#include "facerecognizer.h"

using namespace std;
using namespace cv;

FaceRecognizer::FaceRecognizer()
{
    detector = dnn::readNetFromTensorflow("../models/opencv_face_detector_uint8.pb", "../models/opencv_face_detector.pbtxt");
    recognizer = dnn::readNetFromCaffe("../models/resnet50_256.prototxt", "../models/resnet50_256.caffemodel");

    ifstream labelsStream;
    labelsStream.open("../data/labels");
    while(!labelsStream.eof()){
        string line, name;
        labelsStream >> line;
        name += line;
        labelsStream >> line;
        name += " " + line;
        labels.push_back(name);
    }
    labelsStream.close();

    ifstream embeddingsStream;
    embeddingsStream.open("../data/embeddings");
    while(!embeddingsStream.eof()){
        vector<float> embed;
        for(int i = 0; i < 256; ++i){
            float num;
            embeddingsStream >> num;
            embed.push_back(num);
        }
        embeddings.push_back(cv::Mat(embed, true));
    }
    embeddings.pop_back();
    embeddingsStream.close();
}

Face FaceRecognizer::recognize(Mat frame){
    Mat inputBlob = dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), {104, 177, 123});

    detector.setInput(inputBlob, "data");
    Mat detection = detector.forward("detection_out");

    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > 0.95)
        {
            int x1 = max(0, static_cast<int>(detectionMat.at<float>(i, 3) * frame.size().width));
            int y1 = max(0, static_cast<int>(detectionMat.at<float>(i, 4) * frame.size().height));
            int x2 = min(frame.size().width - 1, static_cast<int>(detectionMat.at<float>(i, 5) * frame.size().width));
            int y2 = min(frame.size().height - 1, static_cast<int>(detectionMat.at<float>(i, 6) * frame.size().height));

            Point point1 = Point(x1, y1), point2 = Point(x2, y2);

            Mat face = frame(Rect(point1, point2));
            Mat inputBlob = dnn::blobFromImage(face, 1.0, Size(224, 224), {93.5940, 104.7624, 129.1863});

            recognizer.setInput(inputBlob, "data");
            int new_dims[] = {256, 1};
            Mat embed = recognizer.forward("feat_extract").reshape(1, 2, new_dims);

            int num = -1;
            double minLen = 0.6;
            for(int j = 0; j < embeddings.size(); ++j){
                double len = embed.dot(embeddings[j]) / norm(embed) / norm(embeddings[j]);
                if(len > minLen){
                    num = j;
                    minLen = len;
                }
            }

            if(num != -1){
                string name = labels[num];
                return Face(name, point1, point2);
            }
        }
    }

    return Face("", Point(0, 0), Point(0, 0));
}
