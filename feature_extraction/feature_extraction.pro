TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += $(HOME)/opencv/include/
LIBS += -L$(HOME)/opencv/build/lib

LIBS += -lopencv_core \
        -lopencv_imgproc \
        -lopencv_imgcodecs \
        -lopencv_dnn

SOURCES += \
        main.cpp
