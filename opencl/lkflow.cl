//#define HALF_WIN        7
//#define LOCAL_SIZE_X    16
//#define LOCAL_SIZE_Y    8
//#define THRESHOLD       0.01

//sampler_t bilinSampler = CLK_NORMALIZED_COORDS_FALSE |
//CLK_ADDRESS_CLAMP_TO_EDGE |
//CLK_FILTER_LINEAR ;

//float GetPixel(image2d_t Img, float2 pixLoc)
//{
//    float2 pix_TL;
//    //pixLoc = (float2)(145.25f,145.25f);
//    pix_TL.x = floor(pixLoc.x);
//    pix_TL.y = floor(pixLoc.y);

//    //if(pix_TL.x == pixLoc.x && pix_TL.y == pixLoc.y)
//        return read_imageui( Img, bilinSampler, pixLoc ).x;

//    float pixTL_Value = read_imageui( Img, bilinSampler, pix_TL ).x;
//    float pixTR_Value = read_imageui( Img, bilinSampler, pix_TL + (float2)(1.0,0) ).x;
//    float pixBL_Value = read_imageui( Img, bilinSampler, pix_TL + (float2)(0,1.0)).x;
//    float pixBR_Value = read_imageui( Img, bilinSampler, pix_TL + (float2)(1.0,1.0)).x;

//    float x1 = pix_TL.x;
//    float y1 = pix_TL.y;
//    float x2 = x1+1.0f;
//    float y2 = y1+1.0f;

//    float  x2x, y2y, yy1, xx1;
//    x2x = x2 - pixLoc.x;
//    y2y = y2 - pixLoc.y;
//    yy1 = pixLoc.y - y1;
//    xx1 = pixLoc.x - x1;

//    float BL = pixBL_Value * x2x * yy1;
//    float BR = pixBR_Value * xx1 * yy1;
//    float TL = pixTL_Value * x2x * y2y;
//    float TR = pixTR_Value * xx1 * y2y;
//    //return  (  (pixBL_Value * x2x * yy1) +   (pixBR_Value * xx1 * yy1) +  (pixTR_Value * xx1 * y2y) +    (pixTL_Value * x2x * y2y) );
//    return (BL+BR+TL+TR);
//}

//__kernel void lkflow(
//    __read_only image2d_t I,
//    __read_only image2d_t J,
//    __global  float2 *prevPt,
//    __global  float2 *nextPt,
//    __global uchar *status,
//    __global float2 *ftmp,
//    int rows,
//    int cols,
//    int iiter,
//    int ilevel)
//{
//    // declare some shared memory
//    __local float smem[HALF_WIN*2+3][HALF_WIN*2+3] ;
//    __local float smemIy[HALF_WIN*2+1][HALF_WIN*2+1] ;
//    __local float smemIx[HALF_WIN*2+1][HALF_WIN*2+1] ;
//    __local float smemA11[LOCAL_SIZE_Y][LOCAL_SIZE_X];
//    __local float smemA21[LOCAL_SIZE_Y][LOCAL_SIZE_X];
//    __local float smemA22[LOCAL_SIZE_Y][LOCAL_SIZE_X];
//    __local float smemb1[LOCAL_SIZE_Y][LOCAL_SIZE_X];
//    __local float smemb2[LOCAL_SIZE_Y][LOCAL_SIZE_X];


//    for(int y= get_local_id(1) ;y< LOCAL_SIZE_Y;y = y+get_local_size(1))
//    {
//        for(int x= get_local_id(0) ;x< LOCAL_SIZE_X;x = x+get_local_size(0))
//        {
//            smemA11[y][x] = 0;
//            smemA21[y][x] = 0;
//            smemA22[y][x] = 0;
//            smemb1[y][x] = 0;
//            smemb2[y][x] = 0;
//        }
//    }


//    // copy into local memory
//    __local float2 PrevPtL;

//    if (get_local_id(0) ==0 && get_local_id(1) == 0)
//    {
//        PrevPtL = prevPt[get_group_id(0)]*1/(1<<(ilevel))- (float2)(HALF_WIN,HALF_WIN);
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);


//    for(int y= get_local_id(1)-1 ;y< HALF_WIN*2+3-1;y = y+get_local_size(1))
//    {
//        for(int x= get_local_id(0)-1 ;x< HALF_WIN*2+3-1;x = x+get_local_size(0))
//        {
//            smem[y+1][x+1] = GetPixel(I,PrevPtL+(float2)(x+0.5f,y+0.5f));// read_imageui( I, bilinSampler, PrevPtL+(float2)(x,y) ).x;
////            ftmp[(y+1)*(5*2+1) +x+1].x = read_imageui( I, bilinSampler, PrevPtL+(float2)(x,y) ).x;
//           // ftmp[(y+1)*(5*2+1) +x+1].y = get_local_size(1);//smem[y+1][x+1];
//        }
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);

//    //////////////////////////////////////////////////////
//    // Compute derivative

//    float ValY_1X_1;
//    float ValY_1X;
//    float ValY_1Xp1;

//    float ValYX_1;
//    float ValYXp1;

//    float ValYp1X_1;
//    float ValYp1X;
//    float ValYp1Xp1;

//    for(int y= get_local_id(1)+1 ;y< HALF_WIN*2+1;y = y+get_local_size(1))
//    {
//        for(int x= get_local_id(0)+1 ;x< HALF_WIN*2+1;x = x+get_local_size(0))
//        {

////         ValY_1X_1  = smem[y-1][x-1];
////         ValY_1X    = smem[y-1][x];
////         ValY_1Xp1  = smem[y-1][x+1];

////         ValYX_1    = smem[y][x-1];
////         ValYXp1    = smem[y][x+1];

////         ValYp1X_1  = smem[y+1][x-1];
////         ValYp1X    = smem[y+1][x];
////         ValYp1Xp1  = smem[y+1][x+1];

//         ValY_1X_1  =  GetPixel(I,PrevPtL+(float2)(x-1+0.5f,y-1+0.5f));
//         ValY_1X    =  GetPixel(I,PrevPtL+(float2)(x+0.5f,y-1+0.5f));
//         ValY_1Xp1  =  GetPixel(I,PrevPtL+(float2)(x+1+0.5f,y-1+0.5f));

//         ValYX_1    =  GetPixel(I,PrevPtL+(float2)(x-1+0.5f,y+0.5f));
//         ValYXp1    =  GetPixel(I,PrevPtL+(float2)(x+1+0.5f,y+0.5f));

//         ValYp1X_1  =  GetPixel(I,PrevPtL+(float2)(x-1+0.5f,y+1+0.5f));
//         ValYp1X    =  GetPixel(I,PrevPtL+(float2)(x+0.5f,y+1+0.5f));
//         ValYp1Xp1  =  GetPixel(I,PrevPtL+(float2)(x+1+0.5f,y+1+0.5f));



//         smemIx[y-1][x-1] = 3.0*( ValY_1Xp1 +  ValYp1Xp1 - ValY_1X_1 -ValYp1X_1 ) + 10.0*(ValYXp1 - ValYX_1);
//         smemIy[y-1][x-1] = 3.0*( ValYp1X_1 +  ValYp1Xp1 - ValY_1X_1 -ValY_1Xp1 ) + 10.0*(ValYp1X - ValY_1X);

//        }
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);

//    ////////////////////////////////////////
//    // Calculated A (only on one thread)
//    __local float A11,A12,A22;

//    if (get_local_id(0) ==0 && get_local_id(1) == 0)
//    {
//        A11 = 0;
//        A12 = 0;
//        A22 = 0;
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);

//   for(int y= get_local_id(1) ;y< HALF_WIN*2+1;y = y+get_local_size(1))
//      {
//            for(int x= get_local_id(0) ;x< HALF_WIN*2+1;x = x+get_local_size(0))
//            {
//                smemA11[get_local_id(1)][get_local_id(0)] +=smemIx[y][x]*smemIx[y][x];
//                smemA21[get_local_id(1)][get_local_id(0)] +=smemIx[y][x]*smemIy[y][x];
//                smemA22[get_local_id(1)][get_local_id(0)] +=smemIy[y][x]*smemIy[y][x];
//            }
//        }

//    barrier(CLK_LOCAL_MEM_FENCE);

//    __local float D;

//    if (get_local_id(0) ==0 && get_local_id(1) == 0)
//    {
//        for(int y= 0 ;y< get_local_size(1);y++)
//        {
//            for(int x= 0 ;x< get_local_size(0);x++)
//            {
//                A11 +=smemA11[y][x];
//                A12 +=smemA21[y][x];
//                A22 +=smemA22[y][x];
//            }
//        }

//        float D = A11 * A22 - A12 * A12;



//        A11 /= D;
//        A12 /= D;
//        A22 /= D;

//    }

//    barrier(CLK_LOCAL_MEM_FENCE);
//           /* if (D < 1.192092896e-07f)
//            {
//                if (get_group_id(0) == 0 && ilevel == 0)
//                    status[get_group_id(0)] = 0;

//                return;
//            }*/

//    /////////////////////////////////////////
//    // Compute optical flow
//    __local float2 NextPtL;

//    if (get_local_id(0) ==0 && get_local_id(1) == 0)
//    {
//        NextPtL = nextPt[get_group_id(0)]*2.0 - (float2)(HALF_WIN,HALF_WIN);
//    }

//    barrier(CLK_LOCAL_MEM_FENCE);

//     for (int k = 0; k < iiter; k++)
//     {
//        if (NextPtL.x < -HALF_WIN || NextPtL.x >= cols || NextPtL.y < -HALF_WIN || NextPtL.y >= rows)
//        {
//        //                if (tid == 0 && level == 0)
//            status[get_group_id(0)] = 0;
//            return;
//        }



//            for(int y= get_local_id(1) ;y< HALF_WIN*2+1;y = y+get_local_size(1))
//            {
//                for(int x= get_local_id(0) ;x< HALF_WIN*2+1;x = x+get_local_size(0))
//                {
////                float I_val = GetPixel(I,PrevPtL+(float2)(x+0.5f,y+0.5f));//smem[y+1][x+1];
////                float J_val = GetPixel(J,NextPtL+(float2)(x+0.5f,y+0.5f));//read_imageui( J, bilinSampler, NextPtL+(float2)(x,y) ).x;
////                float diff = (J_val - I_val)*32.0;


////         ValY_1X_1  =  GetPixel(I,PrevPtL+(float2)(x-1+0.5f,y-1+0.5f));
////         ValY_1X    =  GetPixel(I,PrevPtL+(float2)(x+0.5f,y-1+0.5f));
////         ValY_1Xp1  =  GetPixel(I,PrevPtL+(float2)(x+1+0.5f,y-1+0.5f));

////         ValYX_1    =  GetPixel(I,PrevPtL+(float2)(x-1+0.5f,y+0.5f));
////         ValYXp1    =  GetPixel(I,PrevPtL+(float2)(x+1+0.5f,y+0.5f));

////         ValYp1X_1  =  GetPixel(I,PrevPtL+(float2)(x-1+0.5f,y+1+0.5f));
////         ValYp1X    =  GetPixel(I,PrevPtL+(float2)(x+0.5f,y+1+0.5f));
////         ValYp1Xp1  =  GetPixel(I,PrevPtL+(float2)(x+1+0.5f,y+1+0.5f));

//                float I_val = GetPixel(I,PrevPtL+(float2)(x,y));//smem[y+1][x+1];
//                float J_val = GetPixel(J,NextPtL+(float2)(x,y));//read_imageui( J, bilinSampler, NextPtL+(float2)(x,y) ).x;
//                float diff = (J_val - I_val);//*32.0;


//         ValY_1X_1  =  GetPixel(I,PrevPtL+(float2)(x-1,y-1));
//         ValY_1X    =  GetPixel(I,PrevPtL+(float2)(x,y-1));
//         ValY_1Xp1  =  GetPixel(I,PrevPtL+(float2)(x+1,y-1));

//         ValYX_1    =  GetPixel(I,PrevPtL+(float2)(x-1,y));
//         ValYXp1    =  GetPixel(I,PrevPtL+(float2)(x+1,y));

//         ValYp1X_1  =  GetPixel(I,PrevPtL+(float2)(x-1,y+1));
//         ValYp1X    =  GetPixel(I,PrevPtL+(float2)(x,y+1));
//         ValYp1Xp1  =  GetPixel(I,PrevPtL+(float2)(x+1,y+1));

//         float smemIxt = 3.0*( ValY_1Xp1 +  ValYp1Xp1 - ValY_1X_1 -ValYp1X_1 ) + 10.0*(ValYXp1 - ValYX_1);
//         float smemIyt = 3.0*( ValYp1X_1 +  ValYp1Xp1 - ValY_1X_1 -ValY_1Xp1 ) + 10.0*(ValYp1X - ValY_1X);




//            //ftmp[(y)*(5*2+1) +x].x = smemIxt;
//            //ftmp[(y)*(5*2+1) +x].y = smemIyt;
////                  smemb1[get_local_id(1)][get_local_id(0)] +=diff*smemIx[y][x];
////                  smemb2[get_local_id(1)][get_local_id(0)] +=diff*smemIy[y][x];

//                  smemb1[get_local_id(1)][get_local_id(0)] +=diff*smemIxt;
//                  smemb2[get_local_id(1)][get_local_id(0)] +=diff*smemIyt;
//                }
//             }


//            barrier(CLK_LOCAL_MEM_FENCE);
//            __local float2 delta;
//            if (get_local_id(0) ==0 && get_local_id(1) == 0)
//            {
//            float b1 = 0;
//            float b2 = 0;
//            b1 = 0;
//            b2 = 0;
//                for(int y= 0 ;y< get_local_size(1);y++)
//                {
//                    for(int x= 0 ;x< get_local_size(0);x++)
//                    {
//                        b1 +=smemb1[y][x];
//                        b2 +=smemb2[y][x];
//                    }
//                }


//                delta.x = A12 * b2 - A22 * b1;
//                delta.y = A12 * b1 - A11 * b2;

//                NextPtL.x += delta.x;
//                NextPtL.y += delta.y;
//                //ftmp[k].x = sqrt(delta.x*delta.x+delta.y*delta.y);
//                //ftmp[k].y = delta.y;
//            }

//            barrier(CLK_LOCAL_MEM_FENCE);

//            if(fabs(delta.x) < THRESHOLD && fabs(delta.y) < THRESHOLD)
//                break;
//    }


//    if (get_local_id(0) ==0 && get_local_id(1) == 0)
//    {

//        NextPtL.x += HALF_WIN;
//        NextPtL.y += HALF_WIN;

//        nextPt[get_group_id(0)] = NextPtL;
//     }

//}


#define HALF_WIN        10
#define LOCAL_SIZE_X    16
#define LOCAL_SIZE_Y    8
#define THRESHOLD       0.01

sampler_t bilinSampler = CLK_NORMALIZED_COORDS_FALSE |
CLK_ADDRESS_CLAMP_TO_EDGE |
CLK_FILTER_LINEAR ;

float GetPixel(image2d_t Img, float2 pixLoc)
{
    int2 pix_TL;
    //pixLoc = (float2)(145.5f,145.5f);
    pix_TL.x = (int)pixLoc.x;
    pix_TL.y = (int)pixLoc.y;

    if(pix_TL.x == pixLoc.x && pix_TL.y == pixLoc.y)
        return read_imageui( Img, bilinSampler, pixLoc ).x;

    float pixTL_Value = read_imageui( Img, bilinSampler, pix_TL ).x;
    float pixTR_Value = read_imageui( Img, bilinSampler, pix_TL + (int2)(1,0) ).x;
    float pixBL_Value = read_imageui( Img, bilinSampler, pix_TL + (int2)(0,1)).x;
    float pixBR_Value = read_imageui( Img, bilinSampler, pix_TL + (int2)(1,1)).x;

    float x1 = pix_TL.x;
    float y1 = pix_TL.y;
    float x2 = x1+1.0f;
    float y2 = y1+1.0f;

    float x2x1, y2y1, x2x, y2y, yy1, xx1;
    x2x = x2 - pixLoc.x;
    y2y = y2 - pixLoc.y;
    yy1 = pixLoc.y - y1;
    xx1 = pixLoc.x - x1;

    float BL = pixBL_Value * x2x * yy1;
    float BR = pixBR_Value * xx1 * yy1;
    float TL = pixTL_Value * x2x * y2y;
    float TR = pixTR_Value * xx1 * y2y;
    //return  (  (pixBL_Value * x2x * yy1) +   (pixBR_Value * xx1 * yy1) +  (pixTR_Value * xx1 * y2y) +    (pixTL_Value * x2x * y2y) );
    return (BL+BR+TL+TR);
}

__kernel void lkflow(
    __read_only image2d_t I,
    __read_only image2d_t J,
    __global  float2 *prevPt,
    __global  float2 *nextPt,
    __global uchar *status,
    __global float2 *ftmp,
    int rows,
    int cols,
    int iiter,
    int ilevel)
{
    // declare some shared memory
    __local float smem[HALF_WIN*2+3][HALF_WIN*2+3] ;
    __local float smemIy[HALF_WIN*2+1][HALF_WIN*2+1] ;
    __local float smemIx[HALF_WIN*2+1][HALF_WIN*2+1] ;
    __local float smemA11[LOCAL_SIZE_Y][LOCAL_SIZE_X];
    __local float smemA21[LOCAL_SIZE_Y][LOCAL_SIZE_X];
    __local float smemA22[LOCAL_SIZE_Y][LOCAL_SIZE_X];
    __local float smemb1[LOCAL_SIZE_Y][LOCAL_SIZE_X];
    __local float smemb2[LOCAL_SIZE_Y][LOCAL_SIZE_X];


    for(int y= get_local_id(1) ;y< LOCAL_SIZE_Y;y = y+get_local_size(1))
    {
        for(int x= get_local_id(0) ;x< LOCAL_SIZE_X;x = x+get_local_size(0))
        {
            smemA11[y][x] = 0;
            smemA21[y][x] = 0;
            smemA22[y][x] = 0;
            smemb1[y][x] = 0;
            smemb2[y][x] = 0;
        }
    }


    // copy into local memory
    __local float2 PrevPtL;

    if (get_local_id(0) ==0 && get_local_id(1) == 0)
    {
        PrevPtL = prevPt[get_group_id(0)]*1.0/(1<<(ilevel))- (float2)(HALF_WIN,HALF_WIN);
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    for(int y= get_local_id(1)-1 ;y< HALF_WIN*2+3-1;y = y+get_local_size(1))
    {
        for(int x= get_local_id(0)-1 ;x< HALF_WIN*2+3-1;x = x+get_local_size(0))
        {
            smem[y+1][x+1] = GetPixel(I,PrevPtL+(float2)(x,y));//read_imageui( I, bilinSampler, PrevPtL+(float2)(x,y) ).x;
//            ftmp[(y+1)*(5*2+1) +x+1].x = read_imageui( I, bilinSampler, PrevPtL+(float2)(x,y) ).x;
//            ftmp[(y+1)*(5*2+1) +x+1].y = smem[y+1][x+1];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //////////////////////////////////////////////////////
    // Compute derivative

    int ValY_1X_1;
    int ValY_1X;
    int ValY_1Xp1;

    int ValYX_1;
    int ValYXp1;

    int ValYp1X_1;
    int ValYp1X;
    int ValYp1Xp1;

    for(int y= get_local_id(1)+1 ;y<= HALF_WIN*2+1;y = y+get_local_size(1))
    {
        for(int x= get_local_id(0)+1 ;x<= HALF_WIN*2+1;x = x+get_local_size(0))
        {

         ValY_1X_1  = smem[y-1][x-1];
         ValY_1X    = smem[y-1][x];
         ValY_1Xp1  = smem[y-1][x+1];

         ValYX_1    = smem[y][x-1];
         ValYXp1    = smem[y][x+1];

         ValYp1X_1  = smem[y+1][x-1];
         ValYp1X    = smem[y+1][x];
         ValYp1Xp1  = smem[y+1][x+1];

         smemIx[y-1][x-1] = 3.0*( ValY_1Xp1 +  ValYp1Xp1 - ValY_1X_1 -ValYp1X_1 ) + 10.0*(ValYXp1 - ValYX_1);
         smemIy[y-1][x-1] = 3.0*( ValYp1X_1 +  ValYp1Xp1 - ValY_1X_1 -ValY_1Xp1 ) + 10.0*(ValYp1X - ValY_1X);

        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    ////////////////////////////////////////
    // Calculated A (only on one thread)
    __local float A11,A12,A22;

    if (get_local_id(0) ==0 && get_local_id(1) == 0)
    {
        A11 = 0;
        A12 = 0;
        A22 = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

   for(int y= get_local_id(1) ;y< HALF_WIN*2+1;y = y+get_local_size(1))
      {
            for(int x= get_local_id(0) ;x< HALF_WIN*2+1;x = x+get_local_size(0))
            {
                smemA11[get_local_id(1)][get_local_id(0)] +=smemIx[y][x]*smemIx[y][x];
                smemA21[get_local_id(1)][get_local_id(0)] +=smemIx[y][x]*smemIy[y][x];
                smemA22[get_local_id(1)][get_local_id(0)] +=smemIy[y][x]*smemIy[y][x];
            }
        }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) ==0 && get_local_id(1) == 0)
    {
        for(int y= 0 ;y< get_local_size(1);y++)
        {
            for(int x= 0 ;x< get_local_size(0);x++)
            {
                A11 +=smemA11[y][x];
                A12 +=smemA21[y][x];
                A22 +=smemA22[y][x];
            }
        }

        float D = A11 * A22 - A12 * A12;

//        if (D < 1.192092896e-07f)
//        {
//            if (tid == 0 && level == 0)
//                status[gid] = 0;

//            return;
//        }

        A11 /= D;
        A12 /= D;
        A22 /= D;

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /////////////////////////////////////////
    // Compute optical flow
    __local float2 NextPtL;

    if (get_local_id(0) ==0 && get_local_id(1) == 0)
    {
        NextPtL = nextPt[get_group_id(0)]*2.0 - (float2)(HALF_WIN,HALF_WIN);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

     for (int k = 0; k < iiter; k++)
     {
        if (NextPtL.x < -HALF_WIN || NextPtL.x >= cols || NextPtL.y < -HALF_WIN || NextPtL.y >= rows)
        {
        //                if (tid == 0 && level == 0)
            status[get_group_id(0)] = 0;
            return;
        }

            float b1 = 0;
            float b2 = 0;

            for(int y= get_local_id(1) ;y< HALF_WIN*2+1;y = y+get_local_size(1))
            {
                for(int x= get_local_id(0) ;x< HALF_WIN*2+1;x = x+get_local_size(0))
                {
                float I_val = smem[y+1][x+1];
                float J_val = GetPixel(J,NextPtL+(float2)(x,y));//read_imageui( J, bilinSampler, NextPtL+(float2)(x,y) ).x;
                float diff = (J_val - I_val);//*32.0;


                  smemb1[get_local_id(1)][get_local_id(0)] +=diff*smemIx[y][x];
                  smemb2[get_local_id(1)][get_local_id(0)] +=diff*smemIy[y][x];
                }
             }


            barrier(CLK_LOCAL_MEM_FENCE);
            __local float2 delta;
            if (get_local_id(0) ==0 && get_local_id(1) == 0)
            {
            b1 = 0;
            b2 = 0;
                for(int y= 0 ;y< get_local_size(1);y++)
                {
                    for(int x= 0 ;x< get_local_size(0);x++)
                    {
                        b1 +=smemb1[y][x];
                        b2 +=smemb2[y][x];
                    }
                }


                delta.x = A12 * b2 - A22 * b1;
                delta.y = A12 * b1 - A11 * b2;

                NextPtL.x += delta.x;
                NextPtL.y += delta.y;
                ftmp[k].x = sqrt(delta.x*delta.x+delta.y*delta.y);
                ftmp[k].y = delta.y;
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            if(fabs(delta.x) < THRESHOLD && fabs(delta.y) < THRESHOLD)
                break;
    }


    if (get_local_id(0) ==0 && get_local_id(1) == 0)
    {

        NextPtL.x += HALF_WIN;
        NextPtL.y += HALF_WIN;

        nextPt[get_group_id(0)] = NextPtL;
     }

}
