#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <algorithm>
#include <set>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->graphicsView->setScene(new QGraphicsScene(this));
    ui->graphicsView->scene()->addItem(&pixmap);

    faceRecognizer = FaceRecognizer();
    webcamNum = 0;
}


void MainWindow::closeEvent(QCloseEvent *event)
{
    if(video.isOpened())
    {
        video.release();
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::start()
{
    if(!video.open(webcamNum))
    {
        QMessageBox::critical(this,
                              "Camera Error",
                              "Make sure you entered a correct camera index,"
                              "<br>or that the camera is not being accessed by another program!");
        return;
    }

    set<string> peopleSet;
    ofstream peopleFile;

    std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buf[20];
    strftime(buf, 20, "%d.%m.%Y %H:%M:%S", localtime(&t));
    peopleFile.open("../people " + std::string(buf));

    video.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    video.set(CV_CAP_PROP_FRAME_HEIGHT, 1024);

    Face nowFace = Face("", Point(0, 0), Point(0, 0));
    Face prevFace = Face("", Point(0, 0), Point(0, 0));
    Mat frame;

    for(int i = 0; video.isOpened(); ++i)
    {
        video >> frame;
        if(!frame.empty())
        {
            if(i % 3 == 0){
                prevFace = nowFace;
                nowFace = faceRecognizer.recognize(frame);
            }

            if(nowFace.name != "" && prevFace.name == nowFace.name){
                string line = "Hello " + nowFace.name;
                ui->lineEdit->setText(line.c_str());

                rectangle(frame, nowFace.point1, nowFace.point2, cv::Scalar(0, 255, 0), 2, 4);

                if(peopleSet.find(nowFace.name) == peopleSet.end()){
                    peopleFile << nowFace.name << endl;
                    peopleSet.insert(nowFace.name);
                }
            }
            else {
                ui->lineEdit->setText("");
            }

            QImage qimg(frame.data,
                        frame.cols,
                        frame.rows,
                        frame.step,
                        QImage::Format_RGB888);
            pixmap.setPixmap( QPixmap::fromImage(qimg.rgbSwapped()) );
            ui->graphicsView->fitInView(&pixmap, Qt::KeepAspectRatio);
            ui->graphicsView->resize(this->width(), this->height() - 100);
            ui->lineEdit->resize(this->width(), 100);
            ui->lineEdit->move(0, this->height() - 100);
        }
        qApp->processEvents();
    }

}
