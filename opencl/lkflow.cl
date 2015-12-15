#define HALF_WIN 10


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
    __local float smemA11[HALF_WIN*2+1][HALF_WIN*2+1] ;
    __local float smemA21[HALF_WIN*2+1][HALF_WIN*2+1] ;
    __local float smemA22[HALF_WIN*2+1][HALF_WIN*2+1] ;
    __local float smemb1[HALF_WIN*2+1][HALF_WIN*2+1] ;
    __local float smemb2[HALF_WIN*2+1][HALF_WIN*2+1] ;




    // Create sampler objects.  One is for nearest neighbour, the other fo
    // bilinear interpolation
    sampler_t bilinSampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE |
    CLK_FILTER_LINEAR ;
    sampler_t nnSampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE |
    CLK_FILTER_NEAREST ;

    // copy into local memory
    __local int maxXY;
    __local float2 PrevPtL;

    if (get_local_id(0) ==0 && get_local_id(1) == 0)
    {
        maxXY = HALF_WIN*2+3;
        PrevPtL = prevPt[get_group_id(0)]*1.0/(1<<(ilevel))- (float2)(HALF_WIN+1,HALF_WIN+1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    for(int y= get_local_id(1) ;y< maxXY;y = y+get_local_size(1))
    {
        for(int x= get_local_id(0) ;x< maxXY;x = x+get_local_size(0))
        {
            smem[y][x] = read_imageui( I, nnSampler, PrevPtL+(float2)(x,y) ).x;
//            if(y < maxXY-2 && x< maxXY -2)
//            {

//            ftmp[(y)*(2*5+1) +(x)].x = smem[y][x];//PrevPtL.x+x;//nextPt[get_group_id(0)].x*2.0;//PrevPtL.x;//+x;//I_val;
//            ftmp[(y)*(2*5+1) +(x)].y = PrevPtL.y+y;//nextPt[get_group_id(0)].y*2.0;//PrevPtL.y;//+y;//J_val;
//            }
//            if(y >0 && x>0 && y < maxXY-1 && x< maxXY -1)
//            {
//               ftmp[(y-1)*(2*5+1) +(x-1)].x = nextPt[get_group_id(0)].x*2.0;//PrevPtL.x;//+x;//I_val;
//               ftmp[(y-1)*(2*5+1) +(x-1)].y = nextPt[get_group_id(0)].y*2.0;//PrevPtL.y;//+y;//J_val;
//            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    // Compute derivative
    maxXY = HALF_WIN*2+1;
    int ValY_1X_1;
    int ValY_1X;
    int ValY_1Xp1;

    int ValYX_1;
    int ValYXp1;

    int ValYp1X_1;
    int ValYp1X;
    int ValYp1Xp1;

    for(int y= get_local_id(1)+1 ;y< maxXY;y = y+get_local_size(1))
    {
        for(int x= get_local_id(0)+1 ;x< maxXY;x = x+get_local_size(0))
        {

         ValY_1X_1  = smem[y-1][x-1];
         ValY_1X    = smem[y-1][x];
         ValY_1Xp1  = smem[y-1][x+1];

         ValYX_1    = smem[y][x-1];
         ValYXp1    = smem[y][x+1];

         ValYp1X_1  = smem[y+1][x-1];
         ValYp1X    = smem[y+1][x];
         ValYp1Xp1  = smem[y+1][x+1];

    //     smemIx[y-1][x-1] = 3.0*ValY_1Xp1 + 10.0*ValYXp1 + 3.0*ValYp1Xp1-
    //                        (3.0*ValY_1X_1 + 10.0*ValYX_1 + 3.0*ValY_1Xp1);

    //     smemIy[y-1][x-1] = 3.0*ValYp1X_1 + 10.0*ValYp1X + 3.0*ValYp1Xp1-
    //                        (3.0*ValY_1X_1 + 10.0*ValY_1X + 3.0*ValY_1Xp1);

         smemIx[y-1][x-1] = 3.0*( ValY_1Xp1 +  ValYp1Xp1 - ValY_1X_1 -ValY_1Xp1 ) + 10.0*(ValYXp1 - ValYX_1);
         smemIy[y-1][x-1] = 3.0*( ValYp1X_1 +  ValYp1Xp1 - ValY_1X_1 -ValY_1Xp1 ) + 10.0*(ValYp1X - ValY_1X);
//         ftmp[y*(2*5+1) +x].x = smemIx[y-1][x-1];
//         ftmp[y*(2*5+1) +x].y = smemIy[y-1][x-1];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculated A (only on one thread)
    __local float A11,A12,A22;

    if (get_local_id(0) ==0 && get_local_id(1) == 0)
    {
        A11 = 0;
        A12 = 0;
        A22 = 0;

        maxXY = HALF_WIN*2+1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

        for(int y= get_local_id(1) ;y< maxXY;y = y+get_local_size(1))
        {
            for(int x= get_local_id(0) ;x< maxXY;x = x+get_local_size(0))
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

            for(int y= get_local_id(1) ;y< maxXY;y = y+get_local_size(1))
            {
                for(int x= get_local_id(0) ;x< maxXY;x = x+get_local_size(0))
                {
                float I_val = smem[y+1][x+1];//read_imageui( J, nnSampler, PrevPtL+(float2)(x,y) ).x;
                float J_val = read_imageui( J, bilinSampler, NextPtL+(float2)(x,y) ).x;
                float diff = (J_val - I_val) * ilevel;//2.0f;

//                ftmp[y*(2*5+1) +x].x = I_val;//PrevPtL.x;//NextPtL.x;//+x;//I_val;
//                ftmp[y*(2*5+1) +x].y = J_val;//PrevPtL.y;//NextPtL.y;//+y;//J_val;

//                ftmp[y*(2*5+1) +x].x = I_val;
//                ftmp[y*(2*5+1) +x].y = J_val;

                    smemb1[get_local_id(1)][get_local_id(0)] +=diff*smemIx[y][x];
                    smemb2[get_local_id(1)][get_local_id(0)] +=diff*smemIy[y][x];

                }
             }



            if (get_local_id(0) ==0 && get_local_id(1) == 0)
            {
                for(int y= 0 ;y< get_local_size(1);y++)
                {
                    for(int x= 0 ;x< get_local_size(0);x++)
                    {
                        b1 +=smemb1[y][x];
                        b2 +=smemb2[y][x];
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);


            if (get_local_id(0) ==0 && get_local_id(1) == 0)
            {
                float2 delta;
                delta.x = A12 * b2 - A22 * b1;
                delta.y = A12 * b1 - A11 * b2;

                NextPtL.x += delta.x;
                NextPtL.y += delta.y;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
    }


    if (get_local_id(0) ==0 && get_local_id(1) == 0)
    {
        NextPtL.x += HALF_WIN;
        NextPtL.y += HALF_WIN;

        nextPt[get_group_id(0)] = NextPtL;
     }

}
