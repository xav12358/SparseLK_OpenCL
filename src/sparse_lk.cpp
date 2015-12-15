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
cv::Mat *ImageConcat;

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

    if(err != CL_SUCCESS){
        std::cout << "Build Status: " << Program_lkFLow->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*current_device) << std::endl;
        std::cout << "Build Options:\t" << Program_lkFLow->getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(*current_device) << std::endl;
        std::cout << "Build Log:\t " << Program_lkFLow->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*current_device) << std::endl;
    }

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


    //////////////////////////////////////////////////

    cv::vector<cv::Point2f> PrevPt;
    cv::vector<cv::Point2f> NextPt;
    cv::vector<uchar> Status;
    cv::Mat error;
    int indice= 0;
    int iNbPt = 20;
    for(int i=0;i<1 && indice<iNbPt;i++)
        for(int j=0;j<10 && indice<iNbPt;j++)
        {
            PrevPt.push_back(cv::Point2f(200+i*40.0,100+j*40.0));
            NextPt.push_back(cv::Point2f(200+i*40.0,100+j*40.0));
            Status.push_back( 1);
            indice++;
        }


    cv::calcOpticalFlowPyrLK(Image1,Image2,PrevPt,NextPt,Status,error);

    /// ///////////////////////
    cv::Size sz1 = Image1.size();
    cv::Size sz2 = Image2.size();
    cv::Mat im3(sz1.height, sz1.width+sz2.width, CV_8U);
    ImageConcat = new cv::Mat(sz1.height, sz1.width+sz2.width, CV_8U);

    cv::Mat left(*ImageConcat, cv::Rect(0, 0, Image2.cols, Image2.rows));
    Image1.copyTo(left);
    cv::Mat right(*ImageConcat, cv::Rect(sz1.width, 0, Image2.cols,Image2.rows));
    Image2.copyTo(right);

    cv::Mat im3concat;

    ImageConcat->copyTo(im3concat);

    for(int j=0;j<NextPt.size();j++)
    {
        cv::line( im3concat, cv::Point( PrevPt[j].x, PrevPt[j].y ), cv::Point( NextPt[j].x+640, NextPt[j].y ) ,cv::Scalar(255,255,255));
        std::cout<< "Prev " << PrevPt[j].x << " " << PrevPt[j].y << " Next " << NextPt[j].x<< " " << NextPt[j].y << std::endl;
    }

    cv::imwrite("ImageConcat.png",im3concat);

    src_origin[0] = 0;
    src_origin[1] = 0;
    src_origin[2] = 0;

    region[0] = Image1.cols;
    region[1] = Image1.rows;
    region[2] = 1;

    queue->enqueueWriteImage( Images[0]->mem, CL_TRUE, src_origin, region, Image1.cols, 0, Image1.data);
    queue->enqueueWriteImage( Images[1]->mem, CL_TRUE, src_origin, region, Image1.cols, 0, Image2.data);

}



void calc_flow()
{

    std::cout <<std::endl << "Init image"<< std::endl;
    I->init(*context,640,480);
    J->init(*context,640,480);

    I->fill(*Images[0],*downfilter_kernel_x,*downfilter_kernel_y,*queue);
    J->fill(*Images[1],*downfilter_kernel_x,*downfilter_kernel_y,*queue);

    //////////////////////////////////////////////
    //////////////////////////////////////////////

    int iNbPt = 10;
    cl_float2 *PrevPt = (cl_float2*)malloc(iNbPt*sizeof(cl_float2));
    cl_float2 *NextPt = (cl_float2*)malloc(iNbPt*sizeof(cl_float2));
    cl_uchar *uStatus = (cl_uchar*)malloc(iNbPt*sizeof(cl_uchar));
    cl_float2 *ftmp = (cl_float2*)malloc((2*5+1)*(2*5+1)*sizeof(cl_float2));

    cl_int err;

    cl::Buffer PrevPtBuffer(*context,CL_MEM_READ_WRITE,iNbPt*sizeof(cl_float2),NULL,&err);
    cl::Buffer NextPtBuffer(*context,CL_MEM_READ_WRITE,iNbPt*sizeof(cl_float2),NULL,&err);
    cl::Buffer uStatusBuffer(*context,CL_MEM_READ_WRITE,iNbPt*sizeof(cl_uchar),NULL,&err);
    cl::Buffer ftmpBuffer(*context,CL_MEM_READ_WRITE,(2*5+1)*(2*5+1)*sizeof(cl_float2),NULL,&err);


    int indice = 0;
    for(int i=0;i<1 && indice<iNbPt;i++)
        for(int j=0;j<10 && indice<iNbPt;j++)
        {
            PrevPt[indice].x = (200+i*40.0);
            PrevPt[indice].y = (100+j*40.0);

            NextPt[indice].x = (200+i*40.0)/(1<<(3));
            NextPt[indice].y = (100+j*40.0)/(1<<(3));
            uStatus[i]= 1;
            indice++;
        }


    queue->enqueueWriteBuffer(PrevPtBuffer,TRUE,0,iNbPt*sizeof(cl_float2),PrevPt);
    queue->enqueueWriteBuffer(NextPtBuffer,TRUE,0,iNbPt*sizeof(cl_float2),NextPt);
    queue->enqueueWriteBuffer(uStatusBuffer,TRUE,0,iNbPt*sizeof(cl_uchar),uStatus);

    int lvls = 3;
    err = CL_SUCCESS;
    cl::NDRange global_work_size(16*iNbPt,8);
    cl::NDRange local_work_size(16,8);

    std::cout << "<<<<<<<<<<kernel_lkflow " << oclErrorString(err) << std::endl;

    //    local_work_size[0] =16;
    //    local_work_size[1] =8;

    //    global_work_size[0] = local_work_size[0];
    //    global_work_size[1] = local_work_size[1];

    for( int i=lvls-1; i>=0 ; i-- )
    {
        int argCnt = 0;
        err = kernel_lkflow->setArg(  argCnt++, (I->imgLvl[i].mem));
        std::cout << "kernel_lkflow " << oclErrorString(err) << std::endl;
        err = kernel_lkflow->setArg(  argCnt++, (J->imgLvl[i].mem) );
        std::cout << "kernel_lkflow " << oclErrorString(err) << std::endl;

        err = kernel_lkflow->setArg(  argCnt++, PrevPtBuffer );
        std::cout << "kernel_lkflow " << oclErrorString(err) << std::endl;
        err = kernel_lkflow->setArg(  argCnt++, NextPtBuffer );
        std::cout << "kernel_lkflow " << oclErrorString(err) << std::endl;
        err = kernel_lkflow->setArg(  argCnt++, uStatusBuffer );
        std::cout << "kernel_lkflow " << oclErrorString(err) << std::endl;

        err = kernel_lkflow->setArg(  argCnt++, ftmpBuffer );
        std::cout << "kernel_lkflow " << oclErrorString(err) << std::endl;


        err = kernel_lkflow->setArg(  argCnt++, (I->imgLvl[i].h) ); // cols
        std::cout << "kernel_lkflow " << oclErrorString(err) << std::endl;
        err = kernel_lkflow->setArg(  argCnt++, (I->imgLvl[i].w) ); // rows
        std::cout << "kernel_lkflow " << oclErrorString(err) << std::endl;
        err = kernel_lkflow->setArg(  argCnt++, 10 ); // number of iter
        std::cout << "kernel_lkflow " << oclErrorString(err) << std::endl;
        err = kernel_lkflow->setArg(  argCnt++, i );  // level
        std::cout << "kernel_lkflow " << oclErrorString(err) << std::endl;


        err = queue->enqueueNDRangeKernel(*kernel_lkflow,cl::NullRange,global_work_size,local_work_size);
        std::cout << "kernel_lkflow  enqueue " << oclErrorString(err) << std::endl;
        //        std::cout << "found next:  " << NextPt[0].x << " " << NextPt[0].y << std::endl;

        queue->enqueueReadBuffer(NextPtBuffer,TRUE,NULL,iNbPt*sizeof(cl_float2),NextPt);
        queue->enqueueReadBuffer(ftmpBuffer,TRUE,NULL,(5*2+1)*(5*2+1)*sizeof(cl_float2),ftmp);

//        for(int il=0;il<iNbPt;il++)
//        {
//            std::cout << "prev :  "<< PrevPt[il].x/(1<<(i)) << " " << PrevPt[il].y/(1<<(i)) << " Next " << NextPt[il].x << " " << NextPt[il].y << std::endl;
//        }


//        for(int il=0;il<(2*5+1)*(2*5+1);il++)
//        {
//            if(il%(2*5+1) == 0)
//                std::cout << std::endl;

//            std::cout << (float)ftmp[il].x << " ";
//        }


//        std::cout << "----------------------------- ";
//        for(int il=0;il<(2*5+1)*(2*5+1);il++)
//        {
//            if(il%(2*5+1) == 0)
//                std::cout << std::endl;

//            std::cout << (int)ftmp[il].y << " ";
//        }
        //        __global  float2 *prevPt,
        //        __global  float2 *nextPt,
        //        __global uchar *status,

        //        err = clEnqueueNDRangeKernel( cmdq, lkflow_kernel, 2, 0,
        //            global_work_size, local_work_size, 0, NULL, &flow_timer );
        //        checkErr( err, __LINE__, "lkflowKernel");

        //        clWaitForEvents(1, &flow_timer );

    }

    for(int j=0;j<iNbPt;j++)
    {
        cv::line( *ImageConcat, cv::Point( PrevPt[j].x, PrevPt[j].y ), cv::Point( NextPt[j].x+640, NextPt[j].y ) ,cv::Scalar(255,255,255));
        std::cout << "prev :  "<< PrevPt[j].x << " " << PrevPt[j].y << " Next " << NextPt[j].x << " " << NextPt[j].y << std::endl;
    }

    cv::imwrite("ImageConcatResult.png",*ImageConcat);
}





