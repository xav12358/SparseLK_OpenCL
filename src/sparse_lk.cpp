#include <include/sparse_lk.h>
#include <CL/cl.hpp>
#include <GL/gl.h>


#include <cv.h>
#include <opencv2/highgui/highgui.hpp>

#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <string>

#include <include/oclpyramid.h>



cl::Platform *current_platform;
cl::Device *current_device;
cl::Context *context;
cl::CommandQueue *queue;
cl::Program *Program_lkFLow;
cl::Program *Program_filter;
cl::Kernel *kernel_lkflow;
cl::Kernel *downfilter_kernel_x;
cl::Kernel *downfilter_kernel_y;

ocl_pyramid*I;
ocl_pyramid*J ;
ocl_image *Images[2];

int opencl_init() {


    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    current_platform = new cl::Platform(all_platforms[0]);
    std::cout << "Using platform: "<<current_platform->getInfo<CL_PLATFORM_NAME>()<<"\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    current_platform->getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    current_device = new cl::Device( all_devices[0]);
    std::cout<< "Using device: "<< current_device->getInfo<CL_DEVICE_NAME>()<<"\n";


    context = new cl::Context(*current_device);

    queue = new cl::CommandQueue(*context);
}



///////////////////////////////
/// \brief buildProgramFromFile
///
void buildProgramFromFile()
{

    cl_int err;
    cl::size_t<3> src_origin;
    cl::size_t<3> region;

    std::vector<cl::Device> curVecDevice;
    curVecDevice.push_back(*current_device);
    // load opencl source
    std::ifstream cl_file("/home/xavier/Bureau/Developpement/SparseLK_OpenCL/opencl/lkflow.cl");
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                  cl_string.length() + 1));

    // create program
    Program_lkFLow = new cl::Program(*context, source,&err);



    // compile opencl source
    Program_lkFLow->build(curVecDevice);

    // load named kernel from opencl source
    kernel_lkflow = new cl::Kernel(*Program_lkFLow, "lkflow",&err);
    std::cout <<  "--->lkflow err " << oclErrorString(err) << std::endl;

    if(err != CL_SUCCESS){
        std::cout << "Build Status: " << Program_lkFLow->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*current_device) << std::endl;
        std::cout << "Build Options:\t" << Program_lkFLow->getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(*current_device) << std::endl;
        std::cout << "Build Log:\t " << Program_lkFLow->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*current_device) << std::endl;
    }

    // load opencl source
    std::ifstream cl_file2("/home/xavier/Bureau/Developpement/SparseLK_OpenCL/opencl/filters.cl");
    std::string cl_string2(std::istreambuf_iterator<char>(cl_file2), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source2(1, std::make_pair(cl_string2.c_str(),
                                                   cl_string2.length() + 1));

    // create program
    Program_filter = new cl::Program(*context, source2);

    // compile opencl source
    Program_filter->build(curVecDevice);

    // load named kernel from opencl source
    downfilter_kernel_x = new cl::Kernel(*Program_filter, "downfilter_x_g",&err);
    std::cout <<  "--->downfilter_kernel_x err " << oclErrorString(err) << std::endl;
    downfilter_kernel_y = new cl::Kernel(*Program_filter, "downfilter_y_g",&err);
    std::cout <<  "--->downfilter_kernel_y err " << oclErrorString(err) << std::endl;

    I  = new ocl_pyramid(SINGLE_CHANNEL_TYPE,CL_UNSIGNED_INT8);
    J  = new ocl_pyramid(SINGLE_CHANNEL_TYPE,CL_UNSIGNED_INT8);
    Images[0] = new ocl_image();
    Images[1] = new ocl_image();

    cl_mem_flags memflag;
    memflag = CL_MEM_READ_WRITE;

    Images[0]->image_format.image_channel_data_type = CL_UNSIGNED_INT8;
    Images[0]->image_format.image_channel_order     = SINGLE_CHANNEL_TYPE;
    (Images[0]->mem) = cl::Image2D(*context,memflag,Images[0]->image_format,640,480);
    Images[0]->h = 480;
    Images[0]->w = 640;


    Images[1]->image_format.image_channel_data_type = CL_UNSIGNED_INT8;
    Images[1]->image_format.image_channel_order     = SINGLE_CHANNEL_TYPE;
    (Images[1]->mem) = cl::Image2D(*context,memflag,Images[1]->image_format,640,480);
    Images[1]->h = 480;
    Images[1]->w = 640;

    std::cout << "load image 1" << std::endl;
    cv::Mat Image1;
    Image1 = cv::imread("/home/xavier/Bureau/Developpement/SparseLK_OpenCL/data/minicooper/frame10.png",0);

    std::cout << "load image 2" << std::endl;
    cv::Mat Image2;
    Image2 = cv::imread("/home/xavier/Bureau/Developpement/SparseLK_OpenCL/data/minicooper/frame11.png",0);

    src_origin[0] = 0;
    src_origin[1] = 0;
    src_origin[2] = 0;

    region[0] = Image1.cols;
    region[1] = Image1.rows;
    region[2] = 1;

    queue->enqueueWriteImage( Images[0]->mem, CL_TRUE, src_origin, region, Image1.cols, 0, Image1.data);
    queue->enqueueWriteImage( Images[1]->mem, CL_TRUE, src_origin, region, Image1.cols, 0, Image2.data);

}



float calc_flow()
{
    cl_int err = CL_SUCCESS;

    float t_flow = 0.0f;


    cl_mem_flags memflag;
    memflag = CL_MEM_READ_WRITE;

    std::cout << "**calcl_flow in"<< std::endl;

    //    (Images[0]->mem) = cl::Image2D(*context,memflag,Images[0]->image_format,640,480);
    //    std::cout << "**calcl_flow in2"<< std::endl;
    //    Images[0]->h = 640;
    //    Images[0]->w = 480;

    std::cout <<std::endl << "Init image"<< std::endl;
    I->init(*context,640,480);
    J->init(*context,640,480);


//    cl::size_t<3> src_origin;
//    cl::size_t<3> region;

//    src_origin[0] = 0;
//    src_origin[1] = 0;
//    src_origin[2] = 0;

//    region[0] = Images[0]->w;
//    region[1] = Images[0]->h;
//    region[2] = 1;


//    std::cout << "region size " << Images[0]->w << " " << Images[0]->h << std::endl;
//    cv::Mat ImageScaledinit(480,640,CV_8UC1);
//    queue->enqueueReadImage( (Images[0])->mem, CL_TRUE, src_origin, region, ImageScaledinit.cols, 0, ImageScaledinit.data);
//    cv::imwrite("Image2.png",ImageScaledinit);


//    cv::Mat ImageScaledinit(480,640,CV_8UC1);
//    queue->enqueueReadImage( Images[0]->mem, CL_TRUE, src_origin, region, ImageScaledinit.cols, 0, ImageScaledinit.data);
//    cv::imwrite("Image2.png",ImageScaledinit);


//    std::cout <<std::endl<< "Fill image"<< std::endl;
    I->fill(*Images[0],*downfilter_kernel_x,*downfilter_kernel_y,*queue);
    J->fill(*Images[1],*downfilter_kernel_x,*downfilter_kernel_y,*queue);

    //    for( int i=lvls-1; i>=0 ; i-- ) {
    //        int argCnt = 0;
    //        int use_guess = 0;
    //        if( i <lvls-1 ) use_guess = 1;

    //        cl::NDRange local(16,8);
    //        cl::NDRange global(1,1);

    //        kernel_lkflow->setArg(0, &I.imgLvl[i].mem );
    //        kernel_lkflow->setArg(1, &use_guess );

    //        queue->enqueueNDRangeKernel( lkflow_kernel, cl::NullRange, global, local);

    //        clWaitForEvents(1, &flow_timer );

    //    }
    return t_flow;
}





float computeOCLFlow(int curr, int next)
{
    std::clock_t start;
    double duration;
    start = std::clock();

    float t_flow = 0;
    // todo: don't need to refill both images, only the new one.
    cl_int err;
    I->fill( *Images[0], *downfilter_kernel_x, *downfilter_kernel_y,*queue);
    //    J->fill( Images[next], *downfilter_kernel_x, *downfilter_kernel_y,*queue);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    //    std::cout<<"Fill duration : "<< duration <<'\n';
    //    start = std::clock();

    //    t_flow = calc_flow( *I, *J, lkflow_kernel, command_queue );

    //    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    //    std::cout<<"calc_flow duration : "<< duration <<'\n';

    return t_flow;
}

