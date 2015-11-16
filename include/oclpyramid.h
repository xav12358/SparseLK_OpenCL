#ifndef OCLPYRAMID
#define OCLPYRAMID

#include <CL/cl.hpp>
#include <QDebug>
#include <iostream>

#include <cv.hpp>
#include <opencv2/highgui/highgui.hpp>

#define SINGLE_CHANNEL_TYPE CL_INTENSITY



const char* oclErrorString(cl_int error)
{
    static const char* errorString[] = {
        "CL_SUCCESS",
        "CL_DEVICE_NOT_FOUND",
        "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES",
        "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE",
        "CL_MAP_FAILURE",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "CL_INVALID_VALUE",
        "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE",
        "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM",
        "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION",
        "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS",
        "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION",
        "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
    };

    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

    const int index = -error;

    return (index >= 0 && index < errorCount) ? errorString[index] : "Unspecified Error";
}

// Helper to get next up value for integer division
static inline size_t DivUp(size_t dividend, size_t divisor)
{
    return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}


template<cl_channel_order co, cl_channel_type dt>
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


template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
class ocl_pyramid {
public:
    ocl_image<channel_order, data_type> imgLvl[3];
    ocl_buffer scratchBuf;
    ocl_image<channel_order, data_type> scratchImg;

    ocl_pyramid(/*cl::Context &, cl::CommandQueue &*/);
    cl_int init(cl::Context &ctx,int w, int h, const char *name );
    cl_int fill(ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> &, cl::Kernel downfilter_x, cl::Kernel downfilter_y,cl::CommandQueue &cmdq);
    cl_event gpu_timer;
};


template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
ocl_pyramid<lvls,channel_order,data_type>::ocl_pyramid(/*cl::Context &context_in, cl::command_queue &command_queue_in*/)
{
    //    ctx = context_in;
    //    cmdq = command_queue_in;
}


///////////////
/// \brief ocl_pyramid<lvls, channel_order, data_type>::init
/// \param w
/// \param h
/// \param name
/// \return
///
template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
cl_int ocl_pyramid<lvls,channel_order,data_type>::init(cl::Context &ctx,int w, int h, const char *name = NULL )
{
    cl_int err;
    cl_mem_flags memflag;

    memflag = CL_MEM_READ_ONLY;

    // store ubytes as RGBA packed
    for( int i=0 ; i<3 ; i++ ) {
        std::cout << "level " << i << "  ";
        imgLvl[i].w = w>>i;
        imgLvl[i].h = h>>i;
        imgLvl[i].image_format.image_channel_data_type  = data_type;
        imgLvl[i].image_format.image_channel_order      = channel_order;

        //        cl::ImageFormat format (CL_R, CL_UNORM_INT8);
        imgLvl[i].mem = cl::Image2D( ctx, memflag, imgLvl[i].image_format, imgLvl[i].w, imgLvl[i].h);
        if( err != CL_SUCCESS ) return err;

        cl::Image2D * pt = &imgLvl[i].mem;
        cl::NDRange dims1(pt->getImageInfo<CL_IMAGE_WIDTH>(),
                          pt->getImageInfo<CL_IMAGE_HEIGHT>());
        std::cout << "size: " << dims1[0] << " " << dims1[1] << std::endl;

    }

    // initialize a scratch buffer
    // on Mac, it is a linear buffer used between texture passes
    // on LInux it is a R/W texture, presumably faster

    memflag = CL_MEM_READ_WRITE;
    scratchImg.w = imgLvl[0].w;
    scratchImg.h = imgLvl[0].h;
    scratchImg.image_format.image_channel_data_type = data_type;
    scratchImg.image_format.image_channel_order     = channel_order;
    scratchImg.mem = cl::Image2D( ctx, memflag,scratchImg.image_format, scratchImg.w, scratchImg.h );


    scratchBuf.w = imgLvl[0].w;
    scratchBuf.h = imgLvl[0].h;
    scratchBuf.image_format.image_channel_data_type = data_type;
    scratchBuf.image_format.image_channel_order = channel_order;

    int size = scratchBuf.w * scratchBuf.h*16;
    scratchBuf.mem = cl::Buffer( ctx, CL_MEM_READ_WRITE, size, NULL, &err );

    return err;

}

template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
cl_int ocl_pyramid<lvls,channel_order,data_type>::fill(ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> &img,
                                                       cl::Kernel downfilter_kernel_x,
                                                       cl::Kernel downfilter_kernel_y,
                                                       cl::CommandQueue &cmdq)
{
    static cl_int err = CL_SUCCESS;

    cl::size_t<3> src_origin;
    cl::size_t<3> dst_origin;
    cl::size_t<3> region;
    static int i;

    src_origin[0] = 0;
    src_origin[1] = 0;
    src_origin[2] = 0;

    dst_origin[0] = 0;
    dst_origin[1] = 0;
    dst_origin[2] = 0;

    region[0] = img.w;
    region[1] = img.h;
    region[2] = 1;
    // copy level 0 image (full size)

    //    std::cout << "Fill: copy image" << std::endl;
//    cl::NDRange dims1(img.mem.getImageInfo<CL_IMAGE_WIDTH>(),
//                      img.mem.getImageInfo<CL_IMAGE_HEIGHT>());
//    std::cout << "Fill: copy image size1 " << dims1[0] << " " << dims1[1] << std::endl;

    //    (imgLvl[0]).mem.getImageInfo<CL_IMAGE_WIDTH>();
    //    cl::NDRange dims2(imgLvl[0].mem.getImageInfo<CL_IMAGE_WIDTH>(),imgLvl[0].mem.getImageInfo<CL_IMAGE_HEIGHT>());
    //    std::cout << "Fill: copy image size2 " << dims2[0] << " " << dims2[1] << std::endl;
    //    imgLvl[0].mem = img.mem;
    cmdq.enqueueCopyImage( img.mem,
                           imgLvl[0].mem,
            src_origin,
            dst_origin,
            region );

    region[0] = imgLvl[0].w;
    region[1] = imgLvl[0].h;
    region[2] = 1;

    cv::Mat ImageScaledinit(imgLvl[0].h,imgLvl[0].w,CV_8UC1);
    cmdq.enqueueReadImage( imgLvl[0].mem, CL_TRUE, src_origin, region, ImageScaledinit.cols, 0, ImageScaledinit.data);
    cv::imwrite("Image_L1.png",ImageScaledinit);
//    std::cout << "Fill: copy image fin" << std::endl;

    for(  i=1 ; i<3 ; i++ ) {

        std::cout << "Level " << i << std::endl;
        cl::NDRange local_work_sizes(32,32);
        cl::NDRange global_work_sizes = cl::NDRange( 32 * DivUp( imgLvl[i-1].w, 32),
                32 * DivUp( imgLvl[i-1].h, 32)) ;
        cl_int err;
        int argCnt = 0;

        //        std::cout << "Level-1 size " << imgLvl[i-1].w << " " << imgLvl[i-1].h << std::endl;

        err = downfilter_kernel_x.setArg(argCnt++,  imgLvl[i-1].mem );
                std::cout << "downfilter_kernel_x " << oclErrorString(err) << std::endl;
        err = downfilter_kernel_x.setArg(argCnt++,  scratchBuf.mem );
                std::cout << "downfilter_kernel_x " << oclErrorString(err) << std::endl;
        err = downfilter_kernel_x.setArg(argCnt++,  imgLvl[i-1].w );
                std::cout << "downfilter_kernel_x " << oclErrorString(err) << std::endl;
        err = downfilter_kernel_x.setArg(argCnt++,  imgLvl[i-1].h );
                std::cout << "downfilter_kernel_x " << oclErrorString(err) << std::endl;

        cmdq.enqueueNDRangeKernel(downfilter_kernel_x, cl::NullRange,global_work_sizes, local_work_sizes, 0);
        cmdq.enqueueCopyBufferToImage(  scratchBuf.mem, scratchImg.mem,0, dst_origin, region );


        region[0] = imgLvl[i-1].w;
        region[1] = imgLvl[i-1].h;
        region[2] = 1;

        //        std::cout << "ImageScaled_x " << imgLvl[i].w << " " << imgLvl[i].h << std::endl;
        //        cv::Mat ImageScaled_x(imgLvl[0].h,imgLvl[0].w,CV_8UC1);
        //        cmdq.enqueueReadImage( scratchImg.mem, CL_TRUE, src_origin, region, ImageScaledinit.cols, 0, ImageScaled_x.data);
        //        cv::imwrite("Imaged.png",ImageScaled_x);



        global_work_sizes = cl::NDRange(32 * DivUp( imgLvl[i].w, 32),
                                        32 * DivUp( imgLvl[i].h, 32 )) ;
        argCnt = 0;

        // send the imgLvl[i] texture holding teh first pass as input
        // send the scratch buffer as output
        err = downfilter_kernel_y.setArg( argCnt++,  scratchImg.mem );
                std::cout << "downfilter_kernel_y " << oclErrorString(err) << std::endl;
        err = downfilter_kernel_y.setArg( argCnt++,  scratchBuf.mem );
                std::cout << "downfilter_kernel_y " << oclErrorString(err) << std::endl;
        err = downfilter_kernel_y.setArg( argCnt++,  imgLvl[i].w );
                std::cout << "downfilter_kernel_y " << oclErrorString(err) << std::endl;
        err = downfilter_kernel_y.setArg( argCnt++,  imgLvl[i].h );
                std::cout << "downfilter_kernel_y " << oclErrorString(err) << std::endl;

        region[0] = imgLvl[i].w;
        region[1] = imgLvl[i].h;
        region[2] = 1;
        cmdq.enqueueNDRangeKernel(downfilter_kernel_y,  cl::NullRange, global_work_sizes, local_work_sizes);
        cmdq.enqueueCopyBufferToImage( scratchBuf.mem, imgLvl[i].mem,0, dst_origin, region);


        //        cl::size_t<3> src_origin;
        //        cl::size_t<3> region;

        src_origin[0] = 0;
        src_origin[1] = 0;
        src_origin[2] = 0;

        region[0] = imgLvl[i].w;
        region[1] = imgLvl[i].h;
        region[2] = 1;

        if(i == 1)
        {
//            qDebug() << "<<<<<<<<<<<<<<<<<<<<<<<";
//            std::cout << "scale size " << imgLvl[i].w << " " << imgLvl[i].h << std::endl;
            cv::Mat ImageScaled(imgLvl[i].h,imgLvl[i].w,CV_8UC1);
            cmdq.enqueueReadImage( imgLvl[i].mem, CL_TRUE, src_origin, region, ImageScaled.cols, 0, ImageScaled.data);
            cv::imwrite("ImageScaled_L1.png",ImageScaled);

        }else if(i==2)
        {
//            std::cout << "scale size " << imgLvl[i].w << " " << imgLvl[i].h << std::endl;
            cv::Mat ImageScaled(imgLvl[i].h,imgLvl[i].w,CV_8UC1);
            cmdq.enqueueReadImage( imgLvl[i].mem, CL_TRUE, src_origin, region, ImageScaled.cols, 0, ImageScaled.data);
            cv::imwrite("ImageScaled_L2.png",ImageScaled);
        }

    }
    return err;
}





#endif // OCLPYRAMID

