#define Kernel_SIZE 5


__kernel void lkflow( 
__read_only image2d_t I,
__read_only image2d_t J,
__global  float2 *prevPt,
__global  float2 *nextPt,
//    __global float2 *guess_in,
//    int guess_in_w,
//    __global float2 *guess_out,
//    int guess_out_w,
//    int guess_out_h,
//    int use_guess,
int level)
{
// declare some shared memory
__local int smem[Kernel_SIZE*2+1][Kernel_SIZE*2+1] ;
__local int smemIy[Kernel_SIZE*2+1][Kernel_SIZE*2+1] ;
__local int smemIx[Kernel_SIZE*2+1][Kernel_SIZE*2+1] ;

	// Create sampler objects.  One is for nearest neighbour, the other fo
        // bilinear interpolation
        sampler_t bilinSampler = CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_LINEAR ;
        sampler_t nnSampler = CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST ;

//	// Image indices. Note for the texture, we offset by 0.5 to use the centre
//	// of the texel.
//    int2 iIidx = { get_global_id(0), get_global_id(1)};
//    float2 Iidx = { get_global_id(0)+0.5, get_global_id(1)+0.5 };

    float2 PrevPtL = prevPt[get_global_id(0)]/(1<<level);
    // copy into local memory
    for(int y= get_local_id(1) ;y< Kernel_SIZE*2+3;y = y+get_local_size(1))
    for(int x= get_local_id(0) ;x< Kernel_SIZE*2+3;x = x+get_local_size(0))
    {
    smem[y][x] = read_imageui( I, nnSampler, PrevPtL+(float2)(x,y) ).x;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

}
