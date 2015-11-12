#-------------------------------------------------
#
# Project created by QtCreator 2014-10-07T20:15:38
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = Program1
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += flowGL.cpp \
    oclFlow.cpp


INCLUDEPATH+=/usr/lib/x86_64-linux-gnu
LIBS += -L/usr/lib/x86_64-linux-gnu
LIBS += -lGL -lGLU -lglut #-llapack -lblas -lgfortran
LIBS += -L /home/lineo/NVIDIA_GPU_Computing_SDK/C/common/lib/linux -lGLEW

INCLUDEPATH += /usr/include/opencv
LIBS += -L/usr/local/lib
LIBS += -L/usr/lib/x86_64-linux-gnu
LIBS += -lm
LIBS += -lopencv_core
LIBS += -lopencv_imgproc
LIBS += -lopencv_highgui
LIBS += -lopencv_objdetect
LIBS += -lopencv_calib3d

INCLUDEPATH += /home/lineo/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc
INCLUDEPATH += /home/lineo/NVIDIA_GPU_Computing_SDK/shared/inc

LIBS += -L/home/lineo/NVIDIA_GPU_Computing_SDK/shared/lib
LIBS += -lshrutil_x86_64

LIBS += -L/usr/lib/x86_64-linux-gnu
LIBS += -lOpenCL

LIBS += -L/home/lineo/NVIDIA_GPU_Computing_SDK/OpenCL/common/lib
LIBS += -loclUtil_x86_64

OTHER_FILES += \
    filters.cl\
    lkflow.cl\
    motion.cl


