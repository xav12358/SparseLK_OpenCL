#ifndef OCLPYRAMID
#define OCLPYRAMID

#include <CL/cl.hpp>
#include <QDebug>
#include <iostream>

#include <cv.hpp>
#include <opencv2/highgui/highgui.hpp>

#define SINGLE_CHANNEL_TYPE CL_INTENSITY


const char* oclErrorString(cl_int error);
size_t DivUp(size_t dividend, size_t divisor);

struct ocl_image {
    cl::Image2D mem;
    unsigned int w;
    unsigned int h;
    cl::ImageFormat image_format;
} ;


struct ocl_buffer {
    cl::Buffer mem;
    unsigned int w;
    unsigned int h;
    cl::ImageFormat image_format; // the type of image data this temp buffer reflects
};


class ocl_pyramid {

    cl_channel_order channel_order;
    cl_channel_type data_type;
    int ilevel;
public:
    ocl_image imgLvl[3];
    ocl_buffer scratchBuf;
    ocl_image scratchImg;

    ocl_pyramid( cl_channel_order channel_order,cl_channel_type data_type);
    cl_int init(cl::Context &ctx,int w, int h, const char *name  = NULL );
    cl_int fill(ocl_image&, cl::Kernel downfilter_x, cl::Kernel downfilter_y,cl::CommandQueue &cmdq);
    cl_event gpu_timer;
};








#endif // OCLPYRAMID

