// launched over downsampled area
// first pass sampling from larger level, so x2 the coordinates
__kernel void downfilter_x_g(
    __read_only image2d_t src,
    __global uchar *dst, int dst_w, int dst_h )
{

    sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST ;

    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    //printf("ix %d iy %d",ix,iy);
    float x0 = read_imageui( src, srcSampler, (int2)(ix-2, iy ) ).x/16.0f;
    float x1 = read_imageui( src, srcSampler, (int2)(ix-1, iy ) ).x/4.0f;
    float x2 = (3*read_imageui( src, srcSampler, (int2)(ix, iy )).x)/8.0f;
    float x3 = read_imageui( src, srcSampler, (int2)(ix+1, iy ) ).x/4.0f;
    float x4 = read_imageui( src, srcSampler, (int2)(ix+2, iy ) ).x/16.0f;

    int output = round( x0 + x1 + x2 + x3 + x4 );

    if( ix < dst_w && iy < dst_h ) {
        dst[iy*dst_w + ix ] = (uchar)output;  // uncoalesced when writing to memory object
    }
}

// Simultaneously does a Y smoothing filter and downsampling (i.e. only does filter at
// downsampled points.  Writes to the next smaller pyramid level whose max dimensions are
// given by dst_w/dst_h
__kernel void downfilter_y_g(
    __read_only image2d_t src,
    __global uchar *dst, int dst_w, int dst_h )
{
    sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST ;

    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    float x0 = read_imageui( src, srcSampler, (int2)(2*ix, 2*iy -2 ) ).x/16.0f;
    float x1 = read_imageui( src, srcSampler, (int2)(2*ix, 2*iy -1 ) ).x/4.0f;
    float x2 = (3*read_imageui( src, srcSampler, (int2)(2*ix, 2*iy ) ).x)/8.0f;
    float x3 = read_imageui( src, srcSampler, (int2)(2*ix, 2*iy +1) ).x/4.0f;
    float x4 = read_imageui( src, srcSampler, (int2)(2*ix, 2*iy +2) ).x/16.0f;

    int output = round(x0 + x1 + x2 + x3 + x4);

    if( ix < dst_w && iy < dst_h ) {
        dst[iy*dst_w + ix ] = (uchar)output;
    }

}
