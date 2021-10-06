#include "ccsd_t_common.hpp"

extern double* t3_s_d; // does not work correctly.
extern double* t3_d;

#define FUSION_SIZE_SLICE_1_H3  4
#define FUSION_SIZE_SLICE_1_H2  4
#define FUSION_SIZE_SLICE_1_H1  4
#define FUSION_SIZE_SLICE_1_P6  4
#define FUSION_SIZE_SLICE_1_P5  4
#define FUSION_SIZE_SLICE_1_P4  4
#define FUSION_SIZE_SLICE_1_P7  16

#define FUSION_SIZE_SLICE_2_H3  4
#define FUSION_SIZE_SLICE_2_H2  4
#define FUSION_SIZE_SLICE_2_H1  4
#define FUSION_SIZE_SLICE_2_P6  4
#define FUSION_SIZE_SLICE_2_P5  4
#define FUSION_SIZE_SLICE_2_P4  4
#define FUSION_SIZE_SLICE_2_P7  16

#define FUSION_SIZE_INT_UNIT 	FUSION_SIZE_SLICE_1_P7

#define FUSION_SIZE_TB_1_X 	    FUSION_SIZE_SLICE_1_H3 * FUSION_SIZE_SLICE_1_H2
#define FUSION_SIZE_TB_1_Y 	    FUSION_SIZE_SLICE_1_P6 * FUSION_SIZE_SLICE_1_H1
#define FUSION_SIZE_REG_1_X 	FUSION_SIZE_SLICE_1_P5
#define FUSION_SIZE_REG_1_Y 	FUSION_SIZE_SLICE_1_P4

#define FUSION_SIZE_TB_2_X 	    FUSION_SIZE_SLICE_2_H3 * FUSION_SIZE_SLICE_2_H2
#define FUSION_SIZE_TB_2_Y 	    FUSION_SIZE_SLICE_2_P4 * FUSION_SIZE_SLICE_2_H1
#define FUSION_SIZE_REG_2_X 	FUSION_SIZE_SLICE_2_P5
#define FUSION_SIZE_REG_2_Y     FUSION_SIZE_SLICE_2_P6

#define CEIL(a, b)              (((a) + (b) - 1) / (b))

//
__constant__ int list_stride_t2[9];
__constant__ int list_stride_v2[9];

#define DEBUG_ENALBLE_ALL_KERNEL

/*
    doubles (d1)
*/
// kernel: d1_1
__global__ void kernel_ccsdT_sd1_1(double* t3, 
    double* d_t2_1, double* d_v2_1, 
    int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
    int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
    int stride_reg_x, int stride_reg_y,
    int size_internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    int internal_upperbound   = 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_2_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_2_H3;
    int idx_p4 = threadIdx.y % FUSION_SIZE_SLICE_2_P4;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_2_P4;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }

    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }

    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 +  
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < size_internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p4 < rng_p4 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_1[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p6; ll++)
        {
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_1[blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h2) * size_h3 + (threadIdx.y + l) * list_stride_v2[6]];
        }
        __syncthreads();

        // Cross-Product: -1
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            //temp_bv[0] = sm_b[ll][d_v2_1_offset[l_idx_t3] + 0];
            temp_bv[0] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 0];
            temp_bv[1] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 16];
            temp_bv[2] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 32];
            temp_bv[3] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_2_H3 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                //temp_av = sm_a[ll][d_t2_1_offset[l_idx_t3] + (xx * 16)];
                temp_av = sm_a[ll][idx_p4 + (idx_h1) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

                reg_tile[0][xx] -= temp_av * temp_bv[0];
                reg_tile[1][xx] -= temp_av * temp_bv[1];
                reg_tile[2][xx] -= temp_av * temp_bv[2];
                reg_tile[3][xx] -= temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
    {
        if (rng_p6 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
    {
        if (rng_p6 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p6 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p6 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p6 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d1_1
void jk_ccsd_t_d1_1(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_h7, double* host_t3, double* host_t2, double* host_v2)
{
	// # of Blocks for Each Kernel
    // int	num_blocks_kernel_1;
    int num_blocks_kernel_2;
    int size_internal = (int)size_h7;
    
	// Device Memory for Inputs and Output
    double *dev_t3;
	double *dev_t2;
	double *dev_v2;

    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);	
	// cudaMalloc((void**) &dev_t2, sizeof(double) * size_h1 * size_p5 * size_p4 * size_h7);
	// cudaMalloc((void**) &dev_v2, sizeof(double) * size_h7 * size_p6 * size_h2 * size_h3);

    size_t size_t2 = (sizeof(double) * size_h1 * size_p5 * size_p4 * size_h7);
	size_t size_v2 = (sizeof(double) * size_h7 * size_p6 * size_h2 * size_h3);

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);

    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h1 * size_p5 * size_p4 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h7 * size_p6 * size_h2 * size_h3, cudaMemcpyHostToDevice);

    // num_blocks_kernel_1 = CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);
    num_blocks_kernel_2 = CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

	// Depends on # of Fused Kernel
	// dim3 gridsize_1(num_blocks_kernel_1);
	// dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

	dim3 gridsize_2(num_blocks_kernel_2);
	dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

	int	str_sd2_t3_h3 = 1;
	int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
	int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
	int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
	int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
	// int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

	// int str_reg_x_1 = str_sd2_t3_p5;
	// int str_reg_y_1 = str_sd2_t3_p4;
	int str_reg_x_2 = str_sd2_t3_p5;
	int str_reg_y_2 = str_sd2_t3_p6;

    int* list_stride_sd1_v2_1 = (int*)malloc(sizeof(int) * 9);
    list_stride_sd1_v2_1[0] = size_p4 * size_h2 * size_h3;
	list_stride_sd1_v2_1[1] = size_p4 * size_h1 * size_h3;
	list_stride_sd1_v2_1[2] = size_p4 * size_h1 * size_h2; 
	list_stride_sd1_v2_1[3] = size_p5 * size_h2 * size_h3;
	list_stride_sd1_v2_1[4] = size_p5 * size_h1 * size_h3;
    list_stride_sd1_v2_1[5] = size_p5 * size_h1 * size_h2;

    list_stride_sd1_v2_1[6] = size_p6 * size_h2 * size_h3;
    list_stride_sd1_v2_1[7] = size_p6 * size_h1 * size_h3;
    list_stride_sd1_v2_1[8] = size_p6 * size_h1 * size_h2;

    cudaMemcpyToSymbol(list_stride_v2, list_stride_sd1_v2_1, sizeof(int) * 9);

    // 
    dev_t3 = t3_d;

#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd1_1<<<gridsize_2, blocksize_2>>>(dev_t3, 
    dev_t2, dev_v2, 
    (int)size_h3, (int)size_h2, (int)size_h1, (int)size_p6, (int)size_p5, (int)size_p4, (int)size_h7,
    CEIL(size_h3, FUSION_SIZE_SLICE_2_H3),CEIL(size_h2, FUSION_SIZE_SLICE_2_H2),CEIL(size_h1, FUSION_SIZE_SLICE_2_H1),
    CEIL(size_p6, FUSION_SIZE_SLICE_2_P6),CEIL(size_p5, FUSION_SIZE_SLICE_2_P5),CEIL(size_p4, FUSION_SIZE_SLICE_2_P4),
    str_reg_x_2, str_reg_y_2,
    size_internal);
#endif

    // Copy the Result from Device to Host
	// cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d1_2
__global__ void kernel_ccsdT_sd1_2(double* t3, 
    double* d_t2_2, double* d_v2_2, 
    int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
    int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
    int stride_reg_x, int stride_reg_y,
    int size_internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    int internal_upperbound   = 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_2_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_2_H3;
    int idx_p4 = threadIdx.y % FUSION_SIZE_SLICE_2_P4;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_2_P4;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }

    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }

    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 +  
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < size_internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p4 < rng_p4 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_2[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p6; ll++)
        {
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_2[(blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[7]];
        }
        __syncthreads();

        // Cross-Product: -1
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            //temp_bv[0] = sm_b[ll][d_v2_2_offset[l_idx_t3] + 0];
            temp_bv[0] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 0];
            temp_bv[1] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 16];
            temp_bv[2] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 32];
            temp_bv[3] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_2_H3 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                //temp_av = sm_a[ll][d_t2_2_offset[l_idx_t3] + (xx * 16)];
                temp_av = sm_a[ll][idx_p4 + (idx_h2) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

                reg_tile[0][xx] += temp_av * temp_bv[0];
                reg_tile[1][xx] += temp_av * temp_bv[1];
                reg_tile[2][xx] += temp_av * temp_bv[2];
                reg_tile[3][xx] += temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
    {
        if (rng_p6 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
    {
        if (rng_p6 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p6 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p6 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p6 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d1_2
void jk_ccsd_t_d1_2(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_h7, double* host_t3, double* host_t2, double* host_v2)
{
	// # of Blocks for Each Kernel
    // int	num_blocks_kernel_1;
    int num_blocks_kernel_2;
    int size_internal = (int)size_h7;
    
	// Device Memory for Inputs and Output
    double *dev_t3;
	double *dev_t2;
	double *dev_v2;

    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
	// cudaMalloc((void**) &dev_t2, sizeof(double) * size_h2 * size_p5 * size_p4 * size_h7);
	// cudaMalloc((void**) &dev_v2, sizeof(double) * size_h7 * size_p6 * size_h1 * size_h3);

    size_t size_t2 = (sizeof(double) * size_h2 * size_p5 * size_p4 * size_h7);
    size_t size_v2 = (sizeof(double) * size_h7 * size_p6 * size_h1 * size_h3);

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);

    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h2 * size_p5 * size_p4 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h7 * size_p6 * size_h1 * size_h3, cudaMemcpyHostToDevice);

    // num_blocks_kernel_1 = CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);
    num_blocks_kernel_2 = CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

	// Depends on # of Fused Kernel
	// dim3 gridsize_1(num_blocks_kernel_1);
	// dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

	dim3 gridsize_2(num_blocks_kernel_2);
	dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

	int	str_sd2_t3_h3 = 1;
	int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
	int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
	int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
	int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
	// int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

	// int str_reg_x_1 = str_sd2_t3_p5;
	// int str_reg_y_1 = str_sd2_t3_p4;
	int str_reg_x_2 = str_sd2_t3_p5;
	int str_reg_y_2 = str_sd2_t3_p6;

    int* list_stride_sd1_v2_1 = (int*)malloc(sizeof(int) * 9);
    list_stride_sd1_v2_1[0] = size_p4 * size_h2 * size_h3;
	list_stride_sd1_v2_1[1] = size_p4 * size_h1 * size_h3;
	list_stride_sd1_v2_1[2] = size_p4 * size_h1 * size_h2; 
	list_stride_sd1_v2_1[3] = size_p5 * size_h2 * size_h3;
	list_stride_sd1_v2_1[4] = size_p5 * size_h1 * size_h3;
    list_stride_sd1_v2_1[5] = size_p5 * size_h1 * size_h2;

    list_stride_sd1_v2_1[6] = size_p6 * size_h2 * size_h3;
    list_stride_sd1_v2_1[7] = size_p6 * size_h1 * size_h3;
    list_stride_sd1_v2_1[8] = size_p6 * size_h1 * size_h2;

    cudaMemcpyToSymbol(list_stride_v2, list_stride_sd1_v2_1, sizeof(int) * 9);

    // 
    dev_t3 = t3_d;

#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd1_2<<<gridsize_2, blocksize_2>>>(dev_t3, 
    dev_t2, dev_v2, 
    (int)size_h3, (int)size_h2, (int)size_h1, (int)size_p6, (int)size_p5, (int)size_p4, (int)size_h7,
    CEIL(size_h3, FUSION_SIZE_SLICE_2_H3),CEIL(size_h2, FUSION_SIZE_SLICE_2_H2),CEIL(size_h1, FUSION_SIZE_SLICE_2_H1),
    CEIL(size_p6, FUSION_SIZE_SLICE_2_P6),CEIL(size_p5, FUSION_SIZE_SLICE_2_P5),CEIL(size_p4, FUSION_SIZE_SLICE_2_P4),
    str_reg_x_2, str_reg_y_2,
    size_internal);
#endif

    // Copy the Result from Device to Host
	// cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d1_3
__global__ void kernel_ccsdT_sd1_3(double* t3, 
    double* d_t2_3, double* d_v2_3, 
    int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
    int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
    int stride_reg_x, int stride_reg_y,
    int size_internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    int internal_upperbound   = 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_2_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_2_H3;
    int idx_p4 = threadIdx.y % FUSION_SIZE_SLICE_2_P4;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_2_P4;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }

    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }

    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 +  
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < size_internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p4 < rng_p4 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_3[(blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h1) * size_p5) * size_p4) * size_h7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p6; ll++)
        {
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_2_X] = d_v2_3[(blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h2 + (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[8]];
        }
        __syncthreads();

        // Cross-Product: -1
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            //temp_bv[0] = sm_b[ll][d_v2_3_offset[l_idx_t3] + 0];
            temp_bv[0] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 0];
            temp_bv[1] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 16];
            temp_bv[2] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 32];
            temp_bv[3] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_2_H2 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                //temp_av = sm_a[ll][d_t2_3_offset[l_idx_t3] + (xx * 16)];
                temp_av = sm_a[ll][idx_p4 + (idx_h3) * FUSION_SIZE_SLICE_2_P4 + (xx * 16)];

                reg_tile[0][xx] -= temp_av * temp_bv[0];
                reg_tile[1][xx] -= temp_av * temp_bv[1];
                reg_tile[2][xx] -= temp_av * temp_bv[2];
                reg_tile[3][xx] -= temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
    {
        if (rng_p6 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
    {
        if (rng_p6 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p6 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p6 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p6 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d1_3
void jk_ccsd_t_d1_3(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_h7, double* host_t3, double* host_t2, double* host_v2)
{
	// # of Blocks for Each Kernel
    // int	num_blocks_kernel_1;
    int num_blocks_kernel_2;
    int size_internal = (int)size_h7;
    
	// Device Memory for Inputs and Output
    double *dev_t3;
	double *dev_t2;
	double *dev_v2;

    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
	// cudaMalloc((void**) &dev_t2, sizeof(double) * size_h3 * size_p5 * size_p4 * size_h7);
    // cudaMalloc((void**) &dev_v2, sizeof(double) * size_h7 * size_p6 * size_h1 * size_h2);

    size_t size_t2 = (sizeof(double) * size_h3 * size_p5 * size_p4 * size_h7);
    size_t size_v2 = (sizeof(double) * size_h7 * size_p6 * size_h1 * size_h2);

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);

    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h3 * size_p5 * size_p4 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h7 * size_p6 * size_h1 * size_h2, cudaMemcpyHostToDevice);

    // num_blocks_kernel_1 = CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);
    num_blocks_kernel_2 = CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

	// Depends on # of Fused Kernel
	// dim3 gridsize_1(num_blocks_kernel_1);
	// dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

	dim3 gridsize_2(num_blocks_kernel_2);
	dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

	int	str_sd2_t3_h3 = 1;
	int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
	int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
	int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
	int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
	// int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

	// int str_reg_x_1 = str_sd2_t3_p5;
	// int str_reg_y_1 = str_sd2_t3_p4;
	int str_reg_x_2 = str_sd2_t3_p5;
	int str_reg_y_2 = str_sd2_t3_p6;

    int* list_stride_sd1_v2_1 = (int*)malloc(sizeof(int) * 9);
    list_stride_sd1_v2_1[0] = size_p4 * size_h2 * size_h3;
	list_stride_sd1_v2_1[1] = size_p4 * size_h1 * size_h3;
	list_stride_sd1_v2_1[2] = size_p4 * size_h1 * size_h2; 
	list_stride_sd1_v2_1[3] = size_p5 * size_h2 * size_h3;
	list_stride_sd1_v2_1[4] = size_p5 * size_h1 * size_h3;
    list_stride_sd1_v2_1[5] = size_p5 * size_h1 * size_h2;

    list_stride_sd1_v2_1[6] = size_p6 * size_h2 * size_h3;
    list_stride_sd1_v2_1[7] = size_p6 * size_h1 * size_h3;
    list_stride_sd1_v2_1[8] = size_p6 * size_h1 * size_h2;

    cudaMemcpyToSymbol(list_stride_v2, list_stride_sd1_v2_1, sizeof(int) * 9);

    // 
    dev_t3 = t3_d;

#ifdef DEBUG_ENALBLE_ALL_KERNEL 
    kernel_ccsdT_sd1_3<<<gridsize_2, blocksize_2>>>(dev_t3, 
    dev_t2, dev_v2, 
    (int)size_h3, (int)size_h2, (int)size_h1, (int)size_p6, (int)size_p5, (int)size_p4, (int)size_h7,
    CEIL(size_h3, FUSION_SIZE_SLICE_2_H3),CEIL(size_h2, FUSION_SIZE_SLICE_2_H2),CEIL(size_h1, FUSION_SIZE_SLICE_2_H1),
    CEIL(size_p6, FUSION_SIZE_SLICE_2_P6),CEIL(size_p5, FUSION_SIZE_SLICE_2_P5),CEIL(size_p4, FUSION_SIZE_SLICE_2_P4),
    str_reg_x_2, str_reg_y_2,
    size_internal);
#endif
    // Copy the Result from Device to Host
	// cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d1_4
__global__ void kernel_ccsdT_sd1_4(double* t3, 
    double* d_t2_4, double* d_v2_4, 
    int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
    int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
    int stride_reg_x, int stride_reg_y,
    int size_internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    //int l_idx_t3                = threadIdx.x + threadIdx.y * FUSION_SIZE_TB_1_X;
    int internal_upperbound     = 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
    int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3)) >= FUSION_SIZE_SLICE_1_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_1_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_1_H3;
    }

    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2)) >= FUSION_SIZE_SLICE_1_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_1_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_1_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1)) >= FUSION_SIZE_SLICE_1_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_1_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_1_H1;
    }

    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6)) >= FUSION_SIZE_SLICE_1_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_1_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_1_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5)) >= FUSION_SIZE_SLICE_1_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_1_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_1_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4)) >= FUSION_SIZE_SLICE_1_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_1_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_1_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < size_internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p6 < rng_p6 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_4[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
        
        }

        // Load Input Tensor to Shared Memory
        if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_4[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[0]];
        }
        __syncthreads();

        // Cross-Product: -1
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_H3 + 0];
            temp_bv[1] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_H3 + 16];
            temp_bv[2] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_H3 + 32];
            temp_bv[3] = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_H3 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_a[ll][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6 + (xx * 16)];

                reg_tile[0][xx] -= temp_av * temp_bv[0];
                reg_tile[1][xx] -= temp_av * temp_bv[1];
                reg_tile[2][xx] -= temp_av * temp_bv[2];
                reg_tile[3][xx] -= temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    // 
    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d1_4
void jk_ccsd_t_d1_4(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_h7, double* host_t3, double* host_t2, double* host_v2)
{
	// # of Blocks for Each Kernel
    int	num_blocks_kernel_1;
    // int num_blocks_kernel_2;
    int size_internal = (int)size_h7;
    
	// Device Memory for Inputs and Output
    double *dev_t3;
	double *dev_t2;
	double *dev_v2;

    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
	// cudaMalloc((void**) &dev_t2, sizeof(double) * size_h1 * size_p6 * size_p5 * size_h7);
	// cudaMalloc((void**) &dev_v2, sizeof(double) * size_h7 * size_p4 * size_h2 * size_h3);

    size_t size_t2 = (sizeof(double) * size_h1 * size_p6 * size_p5 * size_h7);
	size_t size_v2 = (sizeof(double) * size_h7 * size_p4 * size_h2 * size_h3);

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
	
    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h1 * size_p6 * size_p5 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h7 * size_p4 * size_h2 * size_h3, cudaMemcpyHostToDevice);

    num_blocks_kernel_1 = CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);
    // num_blocks_kernel_2 = CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

	// Depends on # of Fused Kernel
	dim3 gridsize_1(num_blocks_kernel_1);
	dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

	// dim3 gridsize_2(num_blocks_kernel_2);
	// dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

	int	str_sd2_t3_h3 = 1;
	int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
	int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
	int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
	int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
	int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

	int str_reg_x_1 = str_sd2_t3_p5;
	int str_reg_y_1 = str_sd2_t3_p4;
	// int str_reg_x_2 = str_sd2_t3_p5;
	// int str_reg_y_2 = str_sd2_t3_p6;

    int* list_stride_sd1_v2_1 = (int*)malloc(sizeof(int) * 9);
    list_stride_sd1_v2_1[0] = size_p4 * size_h2 * size_h3;
	list_stride_sd1_v2_1[1] = size_p4 * size_h1 * size_h3;
	list_stride_sd1_v2_1[2] = size_p4 * size_h1 * size_h2; 
	list_stride_sd1_v2_1[3] = size_p5 * size_h2 * size_h3;
	list_stride_sd1_v2_1[4] = size_p5 * size_h1 * size_h3;
    list_stride_sd1_v2_1[5] = size_p5 * size_h1 * size_h2;

    list_stride_sd1_v2_1[6] = size_p6 * size_h2 * size_h3;
    list_stride_sd1_v2_1[7] = size_p6 * size_h1 * size_h3;
    list_stride_sd1_v2_1[8] = size_p6 * size_h1 * size_h2;

    cudaMemcpyToSymbol(list_stride_v2, list_stride_sd1_v2_1, sizeof(int) * 9);

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd1_4<<<gridsize_1, blocksize_1>>>(dev_t3, 
    dev_t2, dev_v2, 
    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3),CEIL(size_h2, FUSION_SIZE_SLICE_1_H2),CEIL(size_h1, FUSION_SIZE_SLICE_1_H1),
    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6),CEIL(size_p5, FUSION_SIZE_SLICE_1_P5),CEIL(size_p4, FUSION_SIZE_SLICE_1_P4),
    str_reg_x_1, str_reg_y_1,
    size_internal);
#endif
    // Copy the Result from Device to Host
	// cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d1_5
__global__ void kernel_ccsdT_sd1_5(double* t3, 
    double* d_t2_5, double* d_v2_5, 
    int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
    int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
    int stride_reg_x, int stride_reg_y,
    int size_internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    //int l_idx_t3                = threadIdx.x + threadIdx.y * FUSION_SIZE_TB_1_X;
    int internal_upperbound     = 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
    int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3)) >= FUSION_SIZE_SLICE_1_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_1_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_1_H3;
    }

    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2)) >= FUSION_SIZE_SLICE_1_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_1_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_1_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1)) >= FUSION_SIZE_SLICE_1_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_1_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_1_H1;
    }

    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6)) >= FUSION_SIZE_SLICE_1_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_1_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_1_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5)) >= FUSION_SIZE_SLICE_1_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_1_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_1_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4)) >= FUSION_SIZE_SLICE_1_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_1_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_1_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < size_internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p6 < rng_p6 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_5[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)]; 
        }

        // Load Input Tensor to Shared Memory
        if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_5[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[1]];
        }
        __syncthreads();

        // Cross-Product: -1
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_1_H3 + 0];
            temp_bv[1] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_1_H3 + 16];
            temp_bv[2] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_1_H3 + 32];
            temp_bv[3] = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_1_H3 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_a[ll][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6 + (xx * 16)];

                reg_tile[0][xx] += temp_av * temp_bv[0];
                reg_tile[1][xx] += temp_av * temp_bv[1];
                reg_tile[2][xx] += temp_av * temp_bv[2];
                reg_tile[3][xx] += temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d1_5
void jk_ccsd_t_d1_5(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_h7, double* host_t3, double* host_t2, double* host_v2)
{
	// # of Blocks for Each Kernel
    int	num_blocks_kernel_1;
    // int num_blocks_kernel_2;
    int size_internal = (int)size_h7;
    
	// Device Memory for Inputs and Output
    double *dev_t3;
	double *dev_t2;
	double *dev_v2;

    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
    // cudaMalloc((void**) &dev_t2, sizeof(double) * size_h2 * size_p6 * size_p5 * size_h7);
	// cudaMalloc((void**) &dev_v2, sizeof(double) * size_h7 * size_p4 * size_h1 * size_h3);

    size_t size_t2 = (sizeof(double) * size_h2 * size_p6 * size_p5 * size_h7);
	size_t size_v2 = (sizeof(double) * size_h7 * size_p4 * size_h1 * size_h3);

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
	
    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h2 * size_p6 * size_p5 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h7 * size_p4 * size_h1 * size_h3, cudaMemcpyHostToDevice);

    num_blocks_kernel_1 = CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);
    // num_blocks_kernel_2 = CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

	// Depends on # of Fused Kernel
	dim3 gridsize_1(num_blocks_kernel_1);
	dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

	// dim3 gridsize_2(num_blocks_kernel_2);
	// dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

	int	str_sd2_t3_h3 = 1;
	int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
	int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
	int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
	int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
	int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

	int str_reg_x_1 = str_sd2_t3_p5;
	int str_reg_y_1 = str_sd2_t3_p4;
	// int str_reg_x_2 = str_sd2_t3_p5;
	// int str_reg_y_2 = str_sd2_t3_p6;

    int* list_stride_sd1_v2_1 = (int*)malloc(sizeof(int) * 9);
    list_stride_sd1_v2_1[0] = size_p4 * size_h2 * size_h3;
	list_stride_sd1_v2_1[1] = size_p4 * size_h1 * size_h3;
	list_stride_sd1_v2_1[2] = size_p4 * size_h1 * size_h2; 
	list_stride_sd1_v2_1[3] = size_p5 * size_h2 * size_h3;
	list_stride_sd1_v2_1[4] = size_p5 * size_h1 * size_h3;
    list_stride_sd1_v2_1[5] = size_p5 * size_h1 * size_h2;

    list_stride_sd1_v2_1[6] = size_p6 * size_h2 * size_h3;
    list_stride_sd1_v2_1[7] = size_p6 * size_h1 * size_h3;
    list_stride_sd1_v2_1[8] = size_p6 * size_h1 * size_h2;

    cudaMemcpyToSymbol(list_stride_v2, list_stride_sd1_v2_1, sizeof(int) * 9);

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd1_5<<<gridsize_1, blocksize_1>>>(dev_t3, 
    dev_t2, dev_v2, 
    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3),CEIL(size_h2, FUSION_SIZE_SLICE_1_H2),CEIL(size_h1, FUSION_SIZE_SLICE_1_H1),
    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6),CEIL(size_p5, FUSION_SIZE_SLICE_1_P5),CEIL(size_p4, FUSION_SIZE_SLICE_1_P4),
    str_reg_x_1, str_reg_y_1,
    size_internal);
#endif
    // Copy the Result from Device to Host
	// cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d1_6
__global__ void kernel_ccsdT_sd1_6(double* t3, 
    double* d_t2_6, double* d_v2_6, 
    int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
    int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
    int stride_reg_x, int stride_reg_y,
    int size_internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    //int l_idx_t3                = threadIdx.x + threadIdx.y * FUSION_SIZE_TB_1_X;
    int internal_upperbound     = 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
    int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3)) >= FUSION_SIZE_SLICE_1_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_1_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_1_H3;
    }

    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2)) >= FUSION_SIZE_SLICE_1_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_1_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_1_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1)) >= FUSION_SIZE_SLICE_1_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_1_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_1_H1;
    }

    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6)) >= FUSION_SIZE_SLICE_1_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_1_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_1_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5)) >= FUSION_SIZE_SLICE_1_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_1_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_1_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4)) >= FUSION_SIZE_SLICE_1_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_1_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_1_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < size_internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1 //63, 21
        if (idx_p6 < rng_p6 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_6[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p5) * size_h7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_6[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[2]];
        }
        __syncthreads();

        // Cross-Product: -1
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_1_H2 + 0];
            temp_bv[1] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_1_H2 + 16];
            temp_bv[2] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_1_H2 + 32];
            temp_bv[3] = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_1_H2 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_a[ll][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6 + (xx * 16)];

                reg_tile[0][xx] -= temp_av * temp_bv[0];
                reg_tile[1][xx] -= temp_av * temp_bv[1];
                reg_tile[2][xx] -= temp_av * temp_bv[2];
                reg_tile[3][xx] -= temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d1_6
void jk_ccsd_t_d1_6(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_h7, double* host_t3, double* host_t2, double* host_v2)
{
	// # of Blocks for Each Kernel
    int	num_blocks_kernel_1;
    // int num_blocks_kernel_2;
    int size_internal = (int)size_h7;
    
	// Device Memory for Inputs and Output
    double *dev_t3;
	double *dev_t2;
	double *dev_v2;

    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
	// cudaMalloc((void**) &dev_t2, sizeof(double) * size_h3 * size_p6 * size_p5 * size_h7);
	// cudaMalloc((void**) &dev_v2, sizeof(double) * size_h7 * size_p4 * size_h1 * size_h2);

    size_t size_t2 = (sizeof(double) * size_h3 * size_p6 * size_p5 * size_h7);
	size_t size_v2 = (sizeof(double) * size_h7 * size_p4 * size_h1 * size_h2);

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
	
    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h3 * size_p6 * size_p5 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h7 * size_p4 * size_h1 * size_h2, cudaMemcpyHostToDevice);

    num_blocks_kernel_1 = CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);
    // num_blocks_kernel_2 = CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

	// Depends on # of Fused Kernel
	dim3 gridsize_1(num_blocks_kernel_1);
	dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

	// dim3 gridsize_2(num_blocks_kernel_2);
	// dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

	int	str_sd2_t3_h3 = 1;
	int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
	int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
	int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
	int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
	int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

	int str_reg_x_1 = str_sd2_t3_p5;
	int str_reg_y_1 = str_sd2_t3_p4;
	// int str_reg_x_2 = str_sd2_t3_p5;
	// int str_reg_y_2 = str_sd2_t3_p6;

    int* list_stride_sd1_v2_1 = (int*)malloc(sizeof(int) * 9);
    list_stride_sd1_v2_1[0] = size_p4 * size_h2 * size_h3;
	list_stride_sd1_v2_1[1] = size_p4 * size_h1 * size_h3;
	list_stride_sd1_v2_1[2] = size_p4 * size_h1 * size_h2; 
	list_stride_sd1_v2_1[3] = size_p5 * size_h2 * size_h3;
	list_stride_sd1_v2_1[4] = size_p5 * size_h1 * size_h3;
    list_stride_sd1_v2_1[5] = size_p5 * size_h1 * size_h2;

    list_stride_sd1_v2_1[6] = size_p6 * size_h2 * size_h3;
    list_stride_sd1_v2_1[7] = size_p6 * size_h1 * size_h3;
    list_stride_sd1_v2_1[8] = size_p6 * size_h1 * size_h2;

    cudaMemcpyToSymbol(list_stride_v2, list_stride_sd1_v2_1, sizeof(int) * 9);

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd1_6<<<gridsize_1, blocksize_1>>>(dev_t3, 
    dev_t2, dev_v2, 
    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3),CEIL(size_h2, FUSION_SIZE_SLICE_1_H2),CEIL(size_h1, FUSION_SIZE_SLICE_1_H1),
    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6),CEIL(size_p5, FUSION_SIZE_SLICE_1_P5),CEIL(size_p4, FUSION_SIZE_SLICE_1_P4),
    str_reg_x_1, str_reg_y_1,
    size_internal);
#endif
    // Copy the Result from Device to Host
	// cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d1_7
__global__ void kernel_ccsdT_sd1_7(double* t3, 
    double* d_t2_7, double* d_v2_7, 
    int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
    int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
    int stride_reg_x, int stride_reg_y,
    int size_internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    //int l_idx_t3                = threadIdx.x + threadIdx.y * FUSION_SIZE_TB_1_X;
    int internal_upperbound     = 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
    int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3)) >= FUSION_SIZE_SLICE_1_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_1_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_1_H3;
    }

    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2)) >= FUSION_SIZE_SLICE_1_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_1_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_1_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1)) >= FUSION_SIZE_SLICE_1_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_1_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_1_H1;
    }

    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6)) >= FUSION_SIZE_SLICE_1_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_1_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_1_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5)) >= FUSION_SIZE_SLICE_1_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_1_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_1_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4)) >= FUSION_SIZE_SLICE_1_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_1_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_1_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < size_internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p6 < rng_p6 && idx_h1 < rng_h1 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_7[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_7[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h2) * size_h3) + (threadIdx.y + l) * list_stride_v2[3]];
        }
        __syncthreads();

        // Cross-Product: -1
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_a[ll][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6 + 0];
            temp_bv[1] = sm_a[ll][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6 + 16];
            temp_bv[2] = sm_a[ll][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6 + 32];
            temp_bv[3] = sm_a[ll][idx_p6 + (idx_h1) * FUSION_SIZE_SLICE_1_P6 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_b[ll][idx_h3 + (idx_h2) * FUSION_SIZE_SLICE_1_H3 + (xx * 16)];

                reg_tile[0][xx] += temp_av * temp_bv[0];
                reg_tile[1][xx] += temp_av * temp_bv[1];
                reg_tile[2][xx] += temp_av * temp_bv[2];
                reg_tile[3][xx] += temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d1_7
void jk_ccsd_t_d1_7(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_h7, double* host_t3, double* host_t2, double* host_v2)
{
	// # of Blocks for Each Kernel
    int	num_blocks_kernel_1;
    // int num_blocks_kernel_2;
    int size_internal = (int)size_h7;
    
	// Device Memory for Inputs and Output
    double *dev_t3;
	double *dev_t2;
	double *dev_v2;

    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
    // cudaMalloc((void**) &dev_t2, sizeof(double) * size_h1 * size_p6 * size_p4 * size_h7);
	// cudaMalloc((void**) &dev_v2, sizeof(double) * size_h7 * size_p5 * size_h2 * size_h3);

    size_t size_t2 = (sizeof(double) * size_h1 * size_p6 * size_p4 * size_h7);
	size_t size_v2 = (sizeof(double) * size_h7 * size_p5 * size_h2 * size_h3);

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h1 * size_p6 * size_p4 * size_h7, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h7 * size_p5 * size_h2 * size_h3, cudaMemcpyHostToDevice);
    
    num_blocks_kernel_1 = CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);
    // num_blocks_kernel_2 = CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

	// Depends on # of Fused Kernel
	dim3 gridsize_1(num_blocks_kernel_1);
	dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

	// dim3 gridsize_2(num_blocks_kernel_2);
	// dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

	int	str_sd2_t3_h3 = 1;
	int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
	int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
	int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
	int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
	int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

	int str_reg_x_1 = str_sd2_t3_p5;
	int str_reg_y_1 = str_sd2_t3_p4;
	// int str_reg_x_2 = str_sd2_t3_p5;
	// int str_reg_y_2 = str_sd2_t3_p6;

    int* list_stride_sd1_v2_1 = (int*)malloc(sizeof(int) * 9);
    list_stride_sd1_v2_1[0] = size_p4 * size_h2 * size_h3;
	list_stride_sd1_v2_1[1] = size_p4 * size_h1 * size_h3;
	list_stride_sd1_v2_1[2] = size_p4 * size_h1 * size_h2; 
	list_stride_sd1_v2_1[3] = size_p5 * size_h2 * size_h3;
	list_stride_sd1_v2_1[4] = size_p5 * size_h1 * size_h3;
    list_stride_sd1_v2_1[5] = size_p5 * size_h1 * size_h2;

    list_stride_sd1_v2_1[6] = size_p6 * size_h2 * size_h3;
    list_stride_sd1_v2_1[7] = size_p6 * size_h1 * size_h3;
    list_stride_sd1_v2_1[8] = size_p6 * size_h1 * size_h2;

    cudaMemcpyToSymbol(list_stride_v2, list_stride_sd1_v2_1, sizeof(int) * 9);

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd1_7<<<gridsize_1, blocksize_1>>>(dev_t3, 
    dev_t2, dev_v2, 
    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3),CEIL(size_h2, FUSION_SIZE_SLICE_1_H2),CEIL(size_h1, FUSION_SIZE_SLICE_1_H1),
    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6),CEIL(size_p5, FUSION_SIZE_SLICE_1_P5),CEIL(size_p4, FUSION_SIZE_SLICE_1_P4),
    str_reg_x_1, str_reg_y_1,
    size_internal);
#endif
    // Copy the Result from Device to Host
	// cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d1_8
__global__ void kernel_ccsdT_sd1_8(double* t3, 
    double* d_t2_8, double* d_v2_8, 
    int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
    int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
    int stride_reg_x, int stride_reg_y,
    int size_internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    //int l_idx_t3                = threadIdx.x + threadIdx.y * FUSION_SIZE_TB_1_X;
    int internal_upperbound     = 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
    int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3)) >= FUSION_SIZE_SLICE_1_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_1_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_1_H3;
    }

    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2)) >= FUSION_SIZE_SLICE_1_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_1_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_1_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1)) >= FUSION_SIZE_SLICE_1_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_1_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_1_H1;
    }

    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6)) >= FUSION_SIZE_SLICE_1_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_1_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_1_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5)) >= FUSION_SIZE_SLICE_1_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_1_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_1_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4)) >= FUSION_SIZE_SLICE_1_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_1_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_1_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < size_internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p6 < rng_p6 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_8[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_8[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h3) + (threadIdx.y + l) * list_stride_v2[4]];
        }
        __syncthreads();

        // Cross-Product: -1
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_a[ll][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6 + 0];
            temp_bv[1] = sm_a[ll][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6 + 16];
            temp_bv[2] = sm_a[ll][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6 + 32];
            temp_bv[3] = sm_a[ll][idx_p6 + (idx_h2) * FUSION_SIZE_SLICE_1_P6 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_b[ll][idx_h3 + (idx_h1) * FUSION_SIZE_SLICE_1_H3 + (xx * 16)];

                reg_tile[0][xx] -= temp_av * temp_bv[0];
                reg_tile[1][xx] -= temp_av * temp_bv[1];
                reg_tile[2][xx] -= temp_av * temp_bv[2];
                reg_tile[3][xx] -= temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    // 
    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d1_8
void jk_ccsd_t_d1_8(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_h7, double* host_t3, double* host_t2, double* host_v2)
{
	// # of Blocks for Each Kernel
    int	num_blocks_kernel_1;
    // int num_blocks_kernel_2;
    int size_internal = (int)size_h7;
    
	// Device Memory for Inputs and Output
    double *dev_t3;
	double *dev_t2;
	double *dev_v2;

    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
    // cudaMalloc((void**) &dev_t2, sizeof(double) * size_h2 * size_p6 * size_p4 * size_h7);
	// cudaMalloc((void**) &dev_v2, sizeof(double) * size_h7 * size_p5 * size_h1 * size_h3);

    size_t size_t2 = (sizeof(double) * size_h2 * size_p6 * size_p4 * size_h7);
	size_t size_v2 = (sizeof(double) * size_h7 * size_p5 * size_h1 * size_h3);

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h2 * size_p6 * size_p4 * size_h7, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h7 * size_p5 * size_h1 * size_h3, cudaMemcpyHostToDevice);    

    num_blocks_kernel_1 = CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);
    // num_blocks_kernel_2 = CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

	// Depends on # of Fused Kernel
	dim3 gridsize_1(num_blocks_kernel_1);
	dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

	// dim3 gridsize_2(num_blocks_kernel_2);
	// dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

	int	str_sd2_t3_h3 = 1;
	int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
	int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
	int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
	int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
	int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

	int str_reg_x_1 = str_sd2_t3_p5;
	int str_reg_y_1 = str_sd2_t3_p4;
	// int str_reg_x_2 = str_sd2_t3_p5;
	// int str_reg_y_2 = str_sd2_t3_p6;

    int* list_stride_sd1_v2_1 = (int*)malloc(sizeof(int) * 9);
    list_stride_sd1_v2_1[0] = size_p4 * size_h2 * size_h3;
	list_stride_sd1_v2_1[1] = size_p4 * size_h1 * size_h3;
	list_stride_sd1_v2_1[2] = size_p4 * size_h1 * size_h2; 
	list_stride_sd1_v2_1[3] = size_p5 * size_h2 * size_h3;
	list_stride_sd1_v2_1[4] = size_p5 * size_h1 * size_h3;
    list_stride_sd1_v2_1[5] = size_p5 * size_h1 * size_h2;

    list_stride_sd1_v2_1[6] = size_p6 * size_h2 * size_h3;
    list_stride_sd1_v2_1[7] = size_p6 * size_h1 * size_h3;
    list_stride_sd1_v2_1[8] = size_p6 * size_h1 * size_h2;

    cudaMemcpyToSymbol(list_stride_v2, list_stride_sd1_v2_1, sizeof(int) * 9);

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd1_8<<<gridsize_1, blocksize_1>>>(dev_t3, 
    dev_t2, dev_v2, 
    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3),CEIL(size_h2, FUSION_SIZE_SLICE_1_H2),CEIL(size_h1, FUSION_SIZE_SLICE_1_H1),
    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6),CEIL(size_p5, FUSION_SIZE_SLICE_1_P5),CEIL(size_p4, FUSION_SIZE_SLICE_1_P4),
    str_reg_x_1, str_reg_y_1,
    size_internal);
#endif
    // Copy the Result from Device to Host
	// cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d1_9
__global__ void kernel_ccsdT_sd1_9(double* t3, 
    double* d_t2_9, double* d_v2_9, 
    int size_h3,    int size_h2,    int size_h1,    int size_p6,    int size_p5,    int size_p4,    int size_h7, 
    int numBlk_h3,  int numBlk_h2,  int numBlk_h1,  int numBlk_p6,  int numBlk_p5,  int numBlk_p4,
    int stride_reg_x, int stride_reg_y,
    int size_internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    //int l_idx_t3                = threadIdx.x + threadIdx.y * FUSION_SIZE_TB_1_X;
    int internal_upperbound     = 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
    int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = blockIdx.x % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3)) >= FUSION_SIZE_SLICE_1_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_1_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_1_H3;
    }

    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2)) >= FUSION_SIZE_SLICE_1_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_1_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_1_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1)) >= FUSION_SIZE_SLICE_1_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_1_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_1_H1;
    }

    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6)) >= FUSION_SIZE_SLICE_1_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_1_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_1_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5)) >= FUSION_SIZE_SLICE_1_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_1_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_1_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4)) >= FUSION_SIZE_SLICE_1_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_1_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_1_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < size_internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - size_internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p6 < rng_p6 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_9[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_p6) * size_p4) * size_h7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && threadIdx.y < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.y][threadIdx.x + ll * FUSION_SIZE_TB_1_X] = d_v2_9[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h3 + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_h2 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_h1) * size_h2) + (threadIdx.y + l) * list_stride_v2[5]];
        }
        __syncthreads();

        // Cross-Product: -1
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_a[ll][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6 + 0];
            temp_bv[1] = sm_a[ll][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6 + 16];
            temp_bv[2] = sm_a[ll][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6 + 32];
            temp_bv[3] = sm_a[ll][idx_p6 + (idx_h3) * FUSION_SIZE_SLICE_1_P6 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_b[ll][idx_h2 + (idx_h1) * FUSION_SIZE_SLICE_1_H2 + (xx * 16)];

                reg_tile[0][xx] += temp_av * temp_bv[0];
                reg_tile[1][xx] += temp_av * temp_bv[1];
                reg_tile[2][xx] += temp_av * temp_bv[2];
                reg_tile[3][xx] += temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d1_9
void jk_ccsd_t_d1_9(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_h7, double* host_t3, double* host_t2, double* host_v2)
{
	// # of Blocks for Each Kernel
    int	num_blocks_kernel_1;
    // int num_blocks_kernel_2;
    int size_internal = (int)size_h7;
    
	// Device Memory for Inputs and Output
    double *dev_t3;
	double *dev_t2;
	double *dev_v2;

    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
	// cudaMalloc((void**) &dev_t2, sizeof(double) * size_h3 * size_p6 * size_p4 * size_h7);
    // cudaMalloc((void**) &dev_v2, sizeof(double) * size_h7 * size_p5 * size_h1 * size_h2);

    size_t size_t2 = (sizeof(double) * size_h3 * size_p6 * size_p4 * size_h7);
    size_t size_v2 = (sizeof(double) * size_h7 * size_p5 * size_h1 * size_h2);

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);    
    cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h3 * size_p6 * size_p4 * size_h7, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h7 * size_p5 * size_h1 * size_h2, cudaMemcpyHostToDevice);

    num_blocks_kernel_1 = CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);
    // num_blocks_kernel_2 = CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

	// Depends on # of Fused Kernel
	dim3 gridsize_1(num_blocks_kernel_1);
	dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

	// dim3 gridsize_2(num_blocks_kernel_2);
	// dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

	int	str_sd2_t3_h3 = 1;
	int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
	int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
	int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
	int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
	int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

	int str_reg_x_1 = str_sd2_t3_p5;
	int str_reg_y_1 = str_sd2_t3_p4;
	// int str_reg_x_2 = str_sd2_t3_p5;
	// int str_reg_y_2 = str_sd2_t3_p6;

    int* list_stride_sd1_v2_1 = (int*)malloc(sizeof(int) * 9);
    list_stride_sd1_v2_1[0] = size_p4 * size_h2 * size_h3;
	list_stride_sd1_v2_1[1] = size_p4 * size_h1 * size_h3;
	list_stride_sd1_v2_1[2] = size_p4 * size_h1 * size_h2; 
	list_stride_sd1_v2_1[3] = size_p5 * size_h2 * size_h3;
	list_stride_sd1_v2_1[4] = size_p5 * size_h1 * size_h3;
    list_stride_sd1_v2_1[5] = size_p5 * size_h1 * size_h2;

    list_stride_sd1_v2_1[6] = size_p6 * size_h2 * size_h3;
    list_stride_sd1_v2_1[7] = size_p6 * size_h1 * size_h3;
    list_stride_sd1_v2_1[8] = size_p6 * size_h1 * size_h2;

    cudaMemcpyToSymbol(list_stride_v2, list_stride_sd1_v2_1, sizeof(int) * 9);

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd1_9<<<gridsize_1, blocksize_1>>>(dev_t3, 
    dev_t2, dev_v2, 
    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7,
    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3),CEIL(size_h2, FUSION_SIZE_SLICE_1_H2),CEIL(size_h1, FUSION_SIZE_SLICE_1_H1),
    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6),CEIL(size_p5, FUSION_SIZE_SLICE_1_P5),CEIL(size_p4, FUSION_SIZE_SLICE_1_P4),
    str_reg_x_1, str_reg_y_1,
    size_internal);
#endif
    // Copy the Result from Device to Host
	// cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost);

	// cudaFree()
    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

/*
    doubles (d2)
*/
// kernel: d2_1
__global__ void kernel_ccsdT_sd2_1(double* t3, 
    double* d_t2_1, double* d_v2_1, 
    int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
    int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
    int stride_reg_x, int stride_reg_y, int internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    int internal_upperbound   	= 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
    int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    // Common for Threads within a Thread Block
    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }
    
    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }
    
    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p6 < rng_h1 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_1[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_h1) * size_p4) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h3 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_v2_1[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_p6 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h1 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_p6) * size_h3) * size_p7 + (threadIdx.x + l)];
        }
        __syncthreads();

        // Cross-Product: 16
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {   
            temp_bv[0] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_1_H1 + 0];
            temp_bv[1] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_1_H1 + 16];
            temp_bv[2] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_1_H1 + 32];
            temp_bv[3] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_1_H1 + 48];
            
            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_b[ll][idx_h3 + (idx_p6) * FUSION_SIZE_SLICE_1_H3 + (xx * 16)];

                reg_tile[0][xx] -= temp_av * temp_bv[0];
                reg_tile[1][xx] -= temp_av * temp_bv[1];
                reg_tile[2][xx] -= temp_av * temp_bv[2];
                reg_tile[3][xx] -= temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    // 
    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d2_1
void jk_ccsd_t_d2_1(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_p7, double* host_t3, double* host_t2, double* host_v2)
{
    // # of Blocks for Each Kernel
    int	num_blocks_kernel_1;
    // int num_blocks_kernel_2;
    int internal = size_p7;

    // Device Memory for Inputs and Output
    double *dev_t3;
    double *dev_t2;
    double *dev_v2;
    
    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
    // cudaMalloc((void**) &dev_t2, sizeof(double) * size_h2 * size_h1 * size_p4 * size_p7);
    // cudaMalloc((void**) &dev_v2, sizeof(double) * size_p5 * size_p6 * size_h3 * size_p7);

    size_t size_t2 = sizeof(double) * size_h2 * size_h1 * size_p4 * size_p7;
    size_t size_v2 = sizeof(double) * size_p5 * size_p6 * size_h3 * size_p7;

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);

    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h2 * size_h1 * size_p4 * size_p7, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_p5 * size_p6 * size_h3 * size_p7, cudaMemcpyHostToDevice);
    
    num_blocks_kernel_1 =   CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * 
                            CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);

    // num_blocks_kernel_2 =   CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * 
    //                         CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

    
    // (5) launch kernel(s)
    // Depends on # of Fused Kernel
    dim3 gridsize_1(num_blocks_kernel_1);
    dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

    // dim3 gridsize_2(num_blocks_kernel_2);
    // dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

    int	str_sd2_t3_h3 = 1;
    int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
    int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
    int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
    int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
    int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

    int str_reg_x_1 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    int str_reg_y_1 = str_sd2_t3_p4;	// STR_SD2_T3_P4
    // int str_reg_x_2 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    // int str_reg_y_2 = str_sd2_t3_p6;	// SDT_SD2_T3_P6

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd2_1<<<gridsize_1, blocksize_1>>>(dev_t3, 
    dev_t2, dev_v2, 
    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_p7,
    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3), CEIL(size_h2, FUSION_SIZE_SLICE_1_H2), CEIL(size_h1, FUSION_SIZE_SLICE_1_H1), 
    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6), CEIL(size_p5, FUSION_SIZE_SLICE_1_P5), CEIL(size_p4, FUSION_SIZE_SLICE_1_P4), 
    str_reg_x_1, str_reg_y_1,
    internal);
#endif
    // 
    // cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyDeviceToHost);

    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d2_2
__global__ void kernel_ccsdT_sd2_2(double* t3, 
    double* d_t2_2, double* d_v2_2, 
    int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
    int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
    int stride_reg_x, int stride_reg_y, int internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    int internal_upperbound   	= 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
    int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    // Common for Threads within a Thread Block
    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }
    
    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }
    
    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p6 < rng_h2 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_2[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_h2) * size_p4) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h1 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_v2_2[(blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_p6 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h1 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_p6) * size_h1) * size_p7 + (threadIdx.x + l)];
        }
        __syncthreads();

        // Cross-Product: 16
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_1_H2 + 0];
            temp_bv[1] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_1_H2 + 16];
            temp_bv[2] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_1_H2 + 32];
            temp_bv[3] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_1_H2 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_b[ll][idx_h1 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + (xx * 16)];

                reg_tile[0][xx] -= temp_av * temp_bv[0];
                reg_tile[1][xx] -= temp_av * temp_bv[1];
                reg_tile[2][xx] -= temp_av * temp_bv[2];
                reg_tile[3][xx] -= temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    // 
    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d2_2
void jk_ccsd_t_d2_2(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_p7, double* host_t3, double* host_t2, double* host_v2)
{
    // # of Blocks for Each Kernel
    int	num_blocks_kernel_1;
    // int num_blocks_kernel_2;
    int internal = size_p7;

    // Device Memory for Inputs and Output
    double *dev_t3;
    double *dev_t2;
    double *dev_v2;
    
    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
    // cudaMalloc((void**) &dev_t2, sizeof(double) * size_h3 * size_h2 * size_p4 * size_p7);
    // cudaMalloc((void**) &dev_v2, sizeof(double) * size_p5 * size_p6 * size_h1 * size_p7);

    size_t size_t2 = sizeof(double) * size_h3 * size_h2 * size_p4 * size_p7;
    size_t size_v2 = sizeof(double) * size_p5 * size_p6 * size_h1 * size_p7;

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);

    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h3 * size_h2 * size_p4 * size_p7, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_p5 * size_p6 * size_h1 * size_p7, cudaMemcpyHostToDevice);
    
    num_blocks_kernel_1 =   CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * 
                            CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);

    // num_blocks_kernel_2 =   CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * 
    //                         CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

    
    // (5) launch kernel(s)
    // Depends on # of Fused Kernel
    dim3 gridsize_1(num_blocks_kernel_1);
    dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

    // dim3 gridsize_2(num_blocks_kernel_2);
    // dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

    int	str_sd2_t3_h3 = 1;
    int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
    int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
    int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
    int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
    int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

    int str_reg_x_1 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    int str_reg_y_1 = str_sd2_t3_p4;	// STR_SD2_T3_P4
    // int str_reg_x_2 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    // int str_reg_y_2 = str_sd2_t3_p6;	// SDT_SD2_T3_P6

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd2_2<<<gridsize_1, blocksize_1>>>(dev_t3, 
    dev_t2, dev_v2, 
    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_p7,
    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3), CEIL(size_h2, FUSION_SIZE_SLICE_1_H2), CEIL(size_h1, FUSION_SIZE_SLICE_1_H1), 
    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6), CEIL(size_p5, FUSION_SIZE_SLICE_1_P5), CEIL(size_p4, FUSION_SIZE_SLICE_1_P4), 
    str_reg_x_1, str_reg_y_1,
    internal);
#endif
    // 
    // cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyDeviceToHost);

    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d2_3
__global__ void kernel_ccsdT_sd2_3(double* t3, 
    double* d_t2_3, double* d_v2_3, 
    int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
    int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
    int stride_reg_x, int stride_reg_y, int internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    int internal_upperbound   	= 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
    int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    // Common for Threads within a Thread Block
    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }
    
    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }
    
    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p6 < rng_h1 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_3[(blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_h1) * size_p4) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h2 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_v2_3[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_p6 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h1 + (blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll) * size_p6) * size_h2) * size_p7 + (threadIdx.x + l)];
        }
        __syncthreads();

        // Cross-Product: 16
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_1_H1 + 0];
            temp_bv[1] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_1_H1 + 16];
            temp_bv[2] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_1_H1 + 32];
            temp_bv[3] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_1_H1 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_b[ll][idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2 + (xx * 16)];

                reg_tile[0][xx] += temp_av * temp_bv[0];
                reg_tile[1][xx] += temp_av * temp_bv[1];
                reg_tile[2][xx] += temp_av * temp_bv[2];
                reg_tile[3][xx] += temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    // 
    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d2_3
void jk_ccsd_t_d2_3(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_p7, double* host_t3, double* host_t2, double* host_v2)
{
    // # of Blocks for Each Kernel
    int	num_blocks_kernel_1;
    // int num_blocks_kernel_2;
    int internal = size_p7;

    // Device Memory for Inputs and Output
    double *dev_t3;
    double *dev_t2;
    double *dev_v2;
    
    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
    // cudaMalloc((void**) &dev_t2, sizeof(double) * size_h3 * size_h1 * size_p4 * size_p7);
    // cudaMalloc((void**) &dev_v2, sizeof(double) * size_p5 * size_p6 * size_h2 * size_p7);

    size_t size_t2 = sizeof(double) * size_h3 * size_h1 * size_p4 * size_p7;
    size_t size_v2 = sizeof(double) * size_p5 * size_p6 * size_h2 * size_p7;

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);

    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h3 * size_h1 * size_p4 * size_p7, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_p5 * size_p6 * size_h2 * size_p7, cudaMemcpyHostToDevice);
    
    num_blocks_kernel_1 =   CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * 
                            CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);

    // num_blocks_kernel_2 =   CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * 
    //                         CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);
    
    // (5) launch kernel(s)
    // Depends on # of Fused Kernel
    dim3 gridsize_1(num_blocks_kernel_1);
    dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

    // dim3 gridsize_2(num_blocks_kernel_2);
    // dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

    int	str_sd2_t3_h3 = 1;
    int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
    int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
    int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
    int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
    int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

    int str_reg_x_1 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    int str_reg_y_1 = str_sd2_t3_p4;	// STR_SD2_T3_P4
    // int str_reg_x_2 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    // int str_reg_y_2 = str_sd2_t3_p6;	// SDT_SD2_T3_P6

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd2_3<<<gridsize_1, blocksize_1>>>(dev_t3, 
    dev_t2, dev_v2, 
    size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_p7,
    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3), CEIL(size_h2, FUSION_SIZE_SLICE_1_H2), CEIL(size_h1, FUSION_SIZE_SLICE_1_H1), 
    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6), CEIL(size_p5, FUSION_SIZE_SLICE_1_P5), CEIL(size_p4, FUSION_SIZE_SLICE_1_P4), 
    str_reg_x_1, str_reg_y_1,
    internal);
#endif
    // 
    // cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyDeviceToHost);

    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d2_4
__global__ void kernel_ccsdT_sd2_4(double* t3, 
    double* d_t2_4, double* d_v2_4, 
    int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
    int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
    int stride_reg_x, int stride_reg_y, int internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    int internal_upperbound   	= 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
    int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    // Common for Threads within a Thread Block
    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }
    
    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }
    
    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p6 < rng_h1 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_4[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_p6 + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_h1) * size_h1) * size_p5) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h3 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_v2_4[(blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_p6 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h1 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_p6) * size_h3) * size_p7 + (threadIdx.x + l)];
        }
        __syncthreads();

        // Cross-Product: 16
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_b[ll][idx_h3 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 0];
            temp_bv[1] = sm_b[ll][idx_h3 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 16];
            temp_bv[2] = sm_b[ll][idx_h3 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 32];
            temp_bv[3] = sm_b[ll][idx_h3 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_1_H1 + (xx * 16)];

                reg_tile[0][xx] += temp_av * temp_bv[0];
                reg_tile[1][xx] += temp_av * temp_bv[1];
                reg_tile[2][xx] += temp_av * temp_bv[2];
                reg_tile[3][xx] += temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    // 
    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d2_4
void jk_ccsd_t_d2_4(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_p7, double* host_t3, double* host_t2, double* host_v2)
{
    // # of Blocks for Each Kernel
    int	num_blocks_kernel_1;
    // int num_blocks_kernel_2;
    int internal = size_p7;

    // Device Memory for Inputs and Output
    double *dev_t3;
    double *dev_t2;
    double *dev_v2;
    
    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
    // cudaMalloc((void**) &dev_t2, sizeof(double) * size_h2 * size_h1 * size_p5 * size_p7);
    // cudaMalloc((void**) &dev_v2, sizeof(double) * size_p4 * size_p6 * size_h3 * size_p7);

    size_t size_t2 = sizeof(double) * size_h2 * size_h1 * size_p5 * size_p7;
    size_t size_v2 = sizeof(double) * size_p4 * size_p6 * size_h3 * size_p7;

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);

    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h2 * size_h1 * size_p5 * size_p7, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_p4 * size_p6 * size_h3 * size_p7, cudaMemcpyHostToDevice);
    
    num_blocks_kernel_1 =   CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * 
                            CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);

    // num_blocks_kernel_2 =   CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * 
    //                         CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

    // (5) launch kernel(s)
    // Depends on # of Fused Kernel
    dim3 gridsize_1(num_blocks_kernel_1);
    dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

    // dim3 gridsize_2(num_blocks_kernel_2);
    // dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

    int	str_sd2_t3_h3 = 1;
    int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
    int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
    int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
    int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
    int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

    int str_reg_x_1 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    int str_reg_y_1 = str_sd2_t3_p4;	// STR_SD2_T3_P4
    // int str_reg_x_2 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    // int str_reg_y_2 = str_sd2_t3_p6;	// SDT_SD2_T3_P6

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd2_4<<<gridsize_1, blocksize_1>>>(dev_t3, 
    dev_t2, dev_v2, 
    (int)size_h3, (int)size_h2, (int)size_h1, (int)size_p6, (int)size_p5, (int)size_p4, (int)size_p7,
    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3), CEIL(size_h2, FUSION_SIZE_SLICE_1_H2), CEIL(size_h1, FUSION_SIZE_SLICE_1_H1), 
    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6), CEIL(size_p5, FUSION_SIZE_SLICE_1_P5), CEIL(size_p4, FUSION_SIZE_SLICE_1_P4), 
    str_reg_x_1, str_reg_y_1,
    internal);
#endif
    // 
    // cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyDeviceToHost);

    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d2_5
__global__ void kernel_ccsdT_sd2_5(double* t3, 
    double* d_t2_5, double* d_v2_5, 
    int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
    int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
    int stride_reg_x, int stride_reg_y, int internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    int internal_upperbound   	= 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
    int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    // Common for Threads within a Thread Block
    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }
    
    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }
    
    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p6 < rng_h2 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_5[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P5 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_h2) * size_p5) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h1 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_v2_5[(blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_p6 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h1 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_p6) * size_h1) * size_p7 + (threadIdx.x + l)];
        }
        __syncthreads();

        // Cross-Product: 16
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_b[ll][idx_h1 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 0];
            temp_bv[1] = sm_b[ll][idx_h1 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 16];
            temp_bv[2] = sm_b[ll][idx_h1 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 32];
            temp_bv[3] = sm_b[ll][idx_h1 + (idx_p6) * FUSION_SIZE_SLICE_1_H1 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_1_H2 + (xx * 16)];

                reg_tile[0][xx] += temp_av * temp_bv[0];
                reg_tile[1][xx] += temp_av * temp_bv[1];
                reg_tile[2][xx] += temp_av * temp_bv[2];
                reg_tile[3][xx] += temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    // 
    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d2_5
void jk_ccsd_t_d2_5(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_p7, double* host_t3, double* host_t2, double* host_v2)
{
    // # of Blocks for Each Kernel
    int	num_blocks_kernel_1;
    // int num_blocks_kernel_2;
    int internal = size_p7;

    // Device Memory for Inputs and Output
    double *dev_t3;
    double *dev_t2;
    double *dev_v2;
    
    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
    // cudaMalloc((void**) &dev_t2, sizeof(double) * size_h3 * size_h2 * size_p5 * size_p7);
    // cudaMalloc((void**) &dev_v2, sizeof(double) * size_p4 * size_p6 * size_h1 * size_p7);

    size_t size_t2 = sizeof(double) * size_h3 * size_h2 * size_p5 * size_p7;
    size_t size_v2 = sizeof(double) * size_p4 * size_p6 * size_h1 * size_p7;

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);

    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h3 * size_h2 * size_p5 * size_p7, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_p4 * size_p6 * size_h1 * size_p7, cudaMemcpyHostToDevice);
    
    num_blocks_kernel_1 =   CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * 
                            CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);

    // num_blocks_kernel_2 =   CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * 
    //                         CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

    // (5) launch kernel(s)
    // Depends on # of Fused Kernel
    dim3 gridsize_1(num_blocks_kernel_1);
    dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

    // dim3 gridsize_2(num_blocks_kernel_2);
    // dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

    int	str_sd2_t3_h3 = 1;
    int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
    int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
    int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
    int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
    int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

    int str_reg_x_1 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    int str_reg_y_1 = str_sd2_t3_p4;	// STR_SD2_T3_P4
    // int str_reg_x_2 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    // int str_reg_y_2 = str_sd2_t3_p6;	// SDT_SD2_T3_P6

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd2_5<<<gridsize_1, blocksize_1>>>(dev_t3, 
    dev_t2, dev_v2, 
    (int)size_h3, (int)size_h2, (int)size_h1, (int)size_p6, (int)size_p5, (int)size_p4, (int)size_p7,
    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3), CEIL(size_h2, FUSION_SIZE_SLICE_1_H2), CEIL(size_h1, FUSION_SIZE_SLICE_1_H1), 
    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6), CEIL(size_p5, FUSION_SIZE_SLICE_1_P5), CEIL(size_p4, FUSION_SIZE_SLICE_1_P4), 
    str_reg_x_1, str_reg_y_1,
    internal);
#endif
    // 
    // cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyDeviceToHost);

    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d2_6
__global__ void kernel_ccsdT_sd2_6(double* t3, 
    double* d_t2_6, double* d_v2_6, 
    int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
    int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
    int stride_reg_x, int stride_reg_y, int internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    int internal_upperbound   	= 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_1_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_1_H3;
    int idx_p6 = threadIdx.y % FUSION_SIZE_SLICE_1_P6;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_1_P6;

    // Common for Threads within a Thread Block
    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }
    
    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }
    
    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + idx_p6 + 
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p6 < rng_h1 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_t2_6[(blk_idx_p5 * FUSION_SIZE_SLICE_1_P6 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_1_H1 + idx_p6 + (blk_idx_h3 * FUSION_SIZE_SLICE_1_H3 + idx_h1) * size_h1) * size_p5) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p6 < rng_h2 && idx_h1 < rng_p6 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p4; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_1_Y] = d_v2_6[(blk_idx_h2 * FUSION_SIZE_SLICE_1_H2 + idx_p6 + (blk_idx_p6 * FUSION_SIZE_SLICE_1_P6 + idx_h1 + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + ll) * size_p6) * size_h2) * size_p7 + (threadIdx.x + l)];
        }
        __syncthreads();

        // Cross-Product: 16
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_b[ll][idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2 + 0];
            temp_bv[1] = sm_b[ll][idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2 + 16];
            temp_bv[2] = sm_b[ll][idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2 + 32];
            temp_bv[3] = sm_b[ll][idx_h2 + (idx_p6) * FUSION_SIZE_SLICE_1_H2 + 48];

            for (int xx = 0; xx < 4; xx++)	// 4 -> rng_p4: Local Transactions...
            {
                temp_av = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_1_H1 + (xx * 16)];

                reg_tile[0][xx] -= temp_av * temp_bv[0];
                reg_tile[1][xx] -= temp_av * temp_bv[1];
                reg_tile[2][xx] -= temp_av * temp_bv[2];
                reg_tile[3][xx] -= temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    // 
    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_h1)
    {
        if (rng_p4 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p4 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p4 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p4 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d2_6
void jk_ccsd_t_d2_6(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_p7, double* host_t3, double* host_t2, double* host_v2)
{
    // # of Blocks for Each Kernel
    int	num_blocks_kernel_1;
    // int num_blocks_kernel_2;
    int internal = size_p7;

    // Device Memory for Inputs and Output
    double *dev_t3;
    double *dev_t2;
    double *dev_v2;
    
    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
    // cudaMalloc((void**) &dev_t2, sizeof(double) * size_h3 * size_h1 * size_p5 * size_p7);
    // cudaMalloc((void**) &dev_v2, sizeof(double) * size_p4 * size_p6 * size_h2 * size_p7);

    size_t size_t2 = sizeof(double) * size_h3 * size_h1 * size_p5 * size_p7;
    size_t size_v2 = sizeof(double) * size_p4 * size_p6 * size_h2 * size_p7;

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);

    // 
    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h3 * size_h1 * size_p5 * size_p7, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_p4 * size_p6 * size_h2 * size_p7, cudaMemcpyHostToDevice);
    
    num_blocks_kernel_1 =   CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * 
                            CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);

    // num_blocks_kernel_2 =   CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * 
    //                         CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

    // (5) launch kernel(s)
    // Depends on # of Fused Kernel
    dim3 gridsize_1(num_blocks_kernel_1);
    dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

    // dim3 gridsize_2(num_blocks_kernel_2);
    // dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

    int	str_sd2_t3_h3 = 1;
    int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
    int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
    int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
    int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
    int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

    int str_reg_x_1 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    int str_reg_y_1 = str_sd2_t3_p4;	// STR_SD2_T3_P4
    // int str_reg_x_2 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    // int str_reg_y_2 = str_sd2_t3_p6;	// SDT_SD2_T3_P6

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd2_6<<<gridsize_1, blocksize_1>>>(dev_t3, 
    dev_t2, dev_v2, 
    (int)size_h3, (int)size_h2, (int)size_h1, (int)size_p6, (int)size_p5, (int)size_p4, (int)size_p7,
    CEIL(size_h3, FUSION_SIZE_SLICE_1_H3), CEIL(size_h2, FUSION_SIZE_SLICE_1_H2), CEIL(size_h1, FUSION_SIZE_SLICE_1_H1), 
    CEIL(size_p6, FUSION_SIZE_SLICE_1_P6), CEIL(size_p5, FUSION_SIZE_SLICE_1_P5), CEIL(size_p4, FUSION_SIZE_SLICE_1_P4), 
    str_reg_x_1, str_reg_y_1,
    internal);
#endif
    // 
    // cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyDeviceToHost);

    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d2_7
__global__ void kernel_ccsdT_sd2_7(double* t3, 
    double* d_t2_7, double* d_v2_7, 
    int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
    int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
    int stride_reg_x, int stride_reg_y, int internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    int internal_upperbound   = 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_2_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_2_H3;
    int idx_p4 = threadIdx.y % FUSION_SIZE_SLICE_2_P4;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_2_P4;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }
    
    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }
    
    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 +  
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p4 < rng_h1 && idx_h1 < rng_h2 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p6; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_7[(blk_idx_p6 *  FUSION_SIZE_SLICE_2_P6 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_p4 + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h1) * size_h1) * size_p6) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p4 < rng_h3 && idx_h1 < rng_p4 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_v2_7[(blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_h1) * size_p5) * size_h3) * size_p7 + (threadIdx.x + l)];
        }
        __syncthreads();

        // Cross-Product: 16
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_2_H1 + 0];
            temp_bv[1] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_2_H1 + 16];
            temp_bv[2] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_2_H1 + 32];
            temp_bv[3] = sm_a[ll][idx_h1 + (idx_h2) * FUSION_SIZE_SLICE_2_H1 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_b[ll][idx_h3 + (idx_p4) * FUSION_SIZE_SLICE_2_H3 + (xx * 16)];

                reg_tile[0][xx] -= temp_av * temp_bv[0];
                reg_tile[1][xx] -= temp_av * temp_bv[1];
                reg_tile[2][xx] -= temp_av * temp_bv[2];
                reg_tile[3][xx] -= temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    // 
    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
    {
        if (rng_p6 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
    {
        if (rng_p6 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p6 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p6 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p6 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d2_7
void jk_ccsd_t_d2_7(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_p7, double* host_t3, double* host_t2, double* host_v2)
{
    // # of Blocks for Each Kernel
    // int	num_blocks_kernel_1;
    int num_blocks_kernel_2;
    int internal = size_p7;

    // Device Memory for Inputs and Output
    double *dev_t3;
    double *dev_t2;
    double *dev_v2;
    
    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
    // cudaMalloc((void**) &dev_t2, sizeof(double) * size_h2 * size_h1 * size_p6 * size_p7);
    // cudaMalloc((void**) &dev_v2, sizeof(double) * size_p4 * size_p5 * size_h3 * size_p7);

    size_t size_t2 = sizeof(double) * size_h2 * size_h1 * size_p6 * size_p7;
    size_t size_v2 = sizeof(double) * size_p4 * size_p5 * size_h3 * size_p7;

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);

    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h2 * size_h1 * size_p6 * size_p7, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_p4 * size_p5 * size_h3 * size_p7, cudaMemcpyHostToDevice);
    
    // num_blocks_kernel_1 =   CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * 
    //                         CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);

    num_blocks_kernel_2 =   CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * 
                            CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

    // (5) launch kernel(s)
    // Depends on # of Fused Kernel
    // dim3 gridsize_1(num_blocks_kernel_1);
    // dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

    dim3 gridsize_2(num_blocks_kernel_2);
    dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

    int	str_sd2_t3_h3 = 1;
    int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
    int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
    int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
    int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
    // int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

    // int str_reg_x_1 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    // int str_reg_y_1 = str_sd2_t3_p4;	// STR_SD2_T3_P4
    int str_reg_x_2 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    int str_reg_y_2 = str_sd2_t3_p6;	// SDT_SD2_T3_P6

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd2_7<<<gridsize_2, blocksize_2>>>(dev_t3, 
    dev_t2, dev_v2, 
    (int)size_h3, (int)size_h2, (int)size_h1, (int)size_p6, (int)size_p5, (int)size_p4, (int)size_p7,
    CEIL(size_h3, FUSION_SIZE_SLICE_2_H3), CEIL(size_h2, FUSION_SIZE_SLICE_2_H2), CEIL(size_h1, FUSION_SIZE_SLICE_2_H1), 
    CEIL(size_p6, FUSION_SIZE_SLICE_2_P6), CEIL(size_p5, FUSION_SIZE_SLICE_2_P5), CEIL(size_p4, FUSION_SIZE_SLICE_2_P4), 
    str_reg_x_2, str_reg_y_2,
    internal);
#endif
    // 
    // cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyDeviceToHost);

    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d2_8
__global__ void kernel_ccsdT_sd2_8(double* t3, 
    double* d_t2_8, double* d_v2_8, 
    int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
    int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
    int stride_reg_x, int stride_reg_y, int internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    int internal_upperbound   = 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_2_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_2_H3;
    int idx_p4 = threadIdx.y % FUSION_SIZE_SLICE_2_P4;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_2_P4;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }
    
    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }
    
    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 +  
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p4 < rng_h2 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p6; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_8[(blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll + (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_p4 + (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h1) * size_h2) * size_p6) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p4 < rng_h1 && idx_h1 < rng_p4 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_v2_8[(blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_p4 * FUSION_SIZE_SLICE_1_P4 + idx_h1) * size_p5) * size_h1) * size_p7 + (threadIdx.x + l)];
        }
        __syncthreads();

        // Cross-Product: 16
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_2_H2 + 0];
            temp_bv[1] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_2_H2 + 16];
            temp_bv[2] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_2_H2 + 32];
            temp_bv[3] = sm_a[ll][idx_h2 + (idx_h3) * FUSION_SIZE_SLICE_2_H2 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_b[ll][idx_h1 + (idx_p4) * FUSION_SIZE_SLICE_2_H1 + (xx * 16)];

                reg_tile[0][xx] -= temp_av * temp_bv[0];
                reg_tile[1][xx] -= temp_av * temp_bv[1];
                reg_tile[2][xx] -= temp_av * temp_bv[2];
                reg_tile[3][xx] -= temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
    {
        if (rng_p6 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
    {
        if (rng_p6 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p6 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p6 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p6 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d2_8
void jk_ccsd_t_d2_8(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_p7, double* host_t3, double* host_t2, double* host_v2)
{
    // # of Blocks for Each Kernel
    // int	num_blocks_kernel_1;
    int num_blocks_kernel_2;
    int internal = size_p7;

    // Device Memory for Inputs and Output
    double *dev_t3;
    double *dev_t2;
    double *dev_v2;
    
    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);    
    // cudaMalloc((void**) &dev_t2, sizeof(double) * size_h3 * size_h2 * size_p6 * size_p7);
    // cudaMalloc((void**) &dev_v2, sizeof(double) * size_p4 * size_p5 * size_h1 * size_p7);

    size_t size_t2 = sizeof(double) * size_h3 * size_h2 * size_p6 * size_p7;
    size_t size_v2 = sizeof(double) * size_p4 * size_p5 * size_h1 * size_p7;

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h3 * size_h2 * size_p6 * size_p7, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_p4 * size_p5 * size_h1 * size_p7, cudaMemcpyHostToDevice);
    
    // num_blocks_kernel_1 =   CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * 
    //                         CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);

    num_blocks_kernel_2 =   CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * 
                            CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

    // (5) launch kernel(s)
    // Depends on # of Fused Kernel
    // dim3 gridsize_1(num_blocks_kernel_1);
    // dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

    dim3 gridsize_2(num_blocks_kernel_2);
    dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

    int	str_sd2_t3_h3 = 1;
    int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
    int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
    int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
    int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
    // int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

    // int str_reg_x_1 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    // int str_reg_y_1 = str_sd2_t3_p4;	// STR_SD2_T3_P4
    int str_reg_x_2 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    int str_reg_y_2 = str_sd2_t3_p6;	// SDT_SD2_T3_P6

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd2_8<<<gridsize_2, blocksize_2>>>(dev_t3, 
    dev_t2, dev_v2, 
    (int)size_h3, (int)size_h2, (int)size_h1, (int)size_p6, (int)size_p5, (int)size_p4, (int)size_p7,
    CEIL(size_h3, FUSION_SIZE_SLICE_2_H3), CEIL(size_h2, FUSION_SIZE_SLICE_2_H2), CEIL(size_h1, FUSION_SIZE_SLICE_2_H1), 
    CEIL(size_p6, FUSION_SIZE_SLICE_2_P6), CEIL(size_p5, FUSION_SIZE_SLICE_2_P5), CEIL(size_p4, FUSION_SIZE_SLICE_2_P4), 
    str_reg_x_2, str_reg_y_2,
    internal);
#endif
    // 
    // cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyDeviceToHost);

    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: d2_9
__global__ void kernel_ccsdT_sd2_9(double* t3, 
    double* d_t2_9, double* d_v2_9, 
    int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
    int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
    int stride_reg_x, int stride_reg_y, int internal)
{
    // For Shared Memory,
    __shared__ double sm_a[16][64 + 1];
    __shared__ double sm_b[16][64 + 1];

    int internal_upperbound   = 0;
    int internal_offset;

    // should support for non-full tiles
    int idx_h3 = threadIdx.x % FUSION_SIZE_SLICE_2_H3;
    int idx_h2 = threadIdx.x / FUSION_SIZE_SLICE_2_H3;
    int idx_p4 = threadIdx.y % FUSION_SIZE_SLICE_2_P4;
    int idx_h1 = threadIdx.y / FUSION_SIZE_SLICE_2_P4;

    int tmp_blkIdx;        
    int blk_idx_p4  = blockIdx.x / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);
    tmp_blkIdx      = blockIdx.x % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6 * numBlk_p5);

    int blk_idx_p5  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1 * numBlk_p6);

    int blk_idx_p6  = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2 * numBlk_h1);
    tmp_blkIdx      = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2 * numBlk_h1);

    int blk_idx_h1 = (tmp_blkIdx) / (numBlk_h3 * numBlk_h2);
    tmp_blkIdx     = (tmp_blkIdx) % (numBlk_h3 * numBlk_h2);

    int blk_idx_h2 = (tmp_blkIdx) / (numBlk_h3);
    int blk_idx_h3 = (tmp_blkIdx) % (numBlk_h3);

    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;

    if ((size_h3 - (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3)) >= FUSION_SIZE_SLICE_2_H3)
    {
        rng_h3 = FUSION_SIZE_SLICE_2_H3;
    }
    else
    {
        rng_h3 = size_h3 % FUSION_SIZE_SLICE_2_H3;
    }
    
    if ((size_h2 - (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2)) >= FUSION_SIZE_SLICE_2_H2)
    {
        rng_h2 = FUSION_SIZE_SLICE_2_H2;
    }
    else
    {
        rng_h2 = size_h2 % FUSION_SIZE_SLICE_2_H2;
    }

    if ((size_h1 - (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1)) >= FUSION_SIZE_SLICE_2_H1)
    {
        rng_h1 = FUSION_SIZE_SLICE_2_H1;
    }
    else
    {
        rng_h1 = size_h1 % FUSION_SIZE_SLICE_2_H1;
    }
    
    if ((size_p6 - (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6)) >= FUSION_SIZE_SLICE_2_P6)
    {
        rng_p6 = FUSION_SIZE_SLICE_2_P6;
    }
    else
    {
        rng_p6 = size_p6 % FUSION_SIZE_SLICE_2_P6;
    }

    if ((size_p5 - (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5)) >= FUSION_SIZE_SLICE_2_P5)
    {
        rng_p5 = FUSION_SIZE_SLICE_2_P5;
    }
    else
    {
        rng_p5 = size_p5 % FUSION_SIZE_SLICE_2_P5;
    }

    if ((size_p4 - (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4)) >= FUSION_SIZE_SLICE_2_P4)
    {
        rng_p4 = FUSION_SIZE_SLICE_2_P4;
    }
    else
    {
        rng_p4 = size_p4 % FUSION_SIZE_SLICE_2_P4;
    }

    int t3_base_thread = blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h3 + 
                        (blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_h2 + 
                        (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_h1 + 
                        (blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 +  
                        (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + 
                        (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_p4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;


    double temp_av;
    double temp_bv[4];
    double reg_tile[4][4];

    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    reg_tile[i][j] = 0.0;

    // tensor contraction
    #pragma unroll 1
    for (int l = 0; l < internal; l+= FUSION_SIZE_INT_UNIT)
    {
        // Part: Generalized Contraction Index (p7b)
        internal_offset = (l + FUSION_SIZE_INT_UNIT) - internal;
        if (internal_offset > 0) internal_upperbound = internal_offset;

        // Load Input Tensor to Shared Memory: 16:16
        // # of Internal Indices: 1
        if (idx_p4 < rng_h1 && idx_h1 < rng_h3 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p6; ll++)
        {
            sm_a[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_t2_9[(blk_idx_p6 * FUSION_SIZE_SLICE_2_P6 + ll + (blk_idx_h1 * FUSION_SIZE_SLICE_2_H1 + idx_p4 + (blk_idx_h3 * FUSION_SIZE_SLICE_2_H3 + idx_h1) * size_h1) * size_p6) * size_p7 + (threadIdx.x + l)];
        }

        // Load Input Tensor to Shared Memory
        if (idx_p4 < rng_h2 && idx_h1 < rng_p4 && threadIdx.x < FUSION_SIZE_INT_UNIT - internal_upperbound)
        for (int ll = 0; ll < rng_p5; ll++)
        {
            sm_b[threadIdx.x][threadIdx.y + ll * FUSION_SIZE_TB_2_Y] = d_v2_9[(blk_idx_h2 * FUSION_SIZE_SLICE_2_H2 + idx_p4 + (blk_idx_p5 * FUSION_SIZE_SLICE_2_P5 + ll + (blk_idx_p4 * FUSION_SIZE_SLICE_2_P4 + idx_h1) * size_p5) * size_h2) * size_p7 + (threadIdx.x + l)];
        }
        __syncthreads();

        // Cross-Product: 16
        // Part: Generalized Threads
        for (int ll = 0; ll < FUSION_SIZE_INT_UNIT - internal_upperbound; ll++)
        {
            temp_bv[0] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_2_H1 + 0];
            temp_bv[1] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_2_H1 + 16];
            temp_bv[2] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_2_H1 + 32];
            temp_bv[3] = sm_a[ll][idx_h1 + (idx_h3) * FUSION_SIZE_SLICE_2_H1 + 48];

            for (int xx = 0 ; xx < 4; xx++)
            {
                temp_av = sm_b[ll][idx_h2 + (idx_p4) * FUSION_SIZE_SLICE_2_H2 + (xx * 16)];

                reg_tile[0][xx] += temp_av * temp_bv[0];
                reg_tile[1][xx] += temp_av * temp_bv[1];
                reg_tile[2][xx] += temp_av * temp_bv[2];
                reg_tile[3][xx] += temp_av * temp_bv[3];
            }
        }
        __syncthreads();
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
    {
        if (rng_p6 == 1)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 2)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 3)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
        if (rng_p6 == 4)
        {
            if (rng_p5 == 1)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
            }
            if (rng_p5 == 2)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
            }
            if (rng_p5 == 3)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
            }
            if (rng_p5 == 4)
            {
                reg_tile[0][0] += t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[0][1] += t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[0][2] += t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[0][3] += t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[1][0] += t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[1][1] += t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[1][2] += t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[1][3] += t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[2][0] += t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[2][1] += t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[2][2] += t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[2][3] += t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)];

                reg_tile[3][0] += t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)];
                reg_tile[3][1] += t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)];
                reg_tile[3][2] += t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)];
                reg_tile[3][3] += t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)];
            }
        }
    }

    if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p4 < rng_p4 && idx_h1 < rng_h1)
    {
        if (rng_p6 == 1)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];
            }
        }
        if (rng_p6 == 2)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];
            }
        }
        if (rng_p6 == 3)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];
            }
        }
        if (rng_p6 == 4)
        {
            if (rng_p5 == 1)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
            }
            if (rng_p5 == 2)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
            }
            if (rng_p5 == 3)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
            }
            if (rng_p5 == 4)
            {
                t3[t3_base_thread + (0 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[0][0];
                t3[t3_base_thread + (0 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[0][1];
                t3[t3_base_thread + (0 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[0][2];
                t3[t3_base_thread + (0 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[0][3];

                t3[t3_base_thread + (1 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[1][0];
                t3[t3_base_thread + (1 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[1][1];
                t3[t3_base_thread + (1 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[1][2];
                t3[t3_base_thread + (1 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[1][3];

                t3[t3_base_thread + (2 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[2][0];
                t3[t3_base_thread + (2 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[2][1];
                t3[t3_base_thread + (2 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[2][2];
                t3[t3_base_thread + (2 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[2][3];

                t3[t3_base_thread + (3 * stride_reg_y) + (0 * stride_reg_x)] = reg_tile[3][0];
                t3[t3_base_thread + (3 * stride_reg_y) + (1 * stride_reg_x)] = reg_tile[3][1];
                t3[t3_base_thread + (3 * stride_reg_y) + (2 * stride_reg_x)] = reg_tile[3][2];
                t3[t3_base_thread + (3 * stride_reg_y) + (3 * stride_reg_x)] = reg_tile[3][3];
            }
        }
    }
}

// caller: d2_9
void jk_ccsd_t_d2_9(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_p7, double* host_t3, double* host_t2, double* host_v2)
{
    // # of Blocks for Each Kernel
    // int	 num_blocks_kernel_1;
    int num_blocks_kernel_2;
    int internal = size_p7;

    // Device Memory for Inputs and Output
    double *dev_t3;
    double *dev_t2;
    double *dev_v2;
    
    // cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4);
    // cudaMalloc((void**) &dev_t2, sizeof(double) * size_h3 * size_h1 * size_p6 * size_p7);
    // cudaMalloc((void**) &dev_v2, sizeof(double) * size_p4 * size_p5 * size_h2 * size_p7);
    size_t size_t2 = sizeof(double) * size_h3 * size_h1 * size_p6 * size_p7;
    size_t size_v2 = sizeof(double) * size_p4 * size_p5 * size_h2 * size_p7;

    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);

    // cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h3 * size_h1 * size_p6 * size_p7, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_p4 * size_p5 * size_h2 * size_p7, cudaMemcpyHostToDevice);
    
    // num_blocks_kernel_1 =   CEIL(size_h3, FUSION_SIZE_SLICE_1_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_1_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_1_H1) * 
    //                         CEIL(size_p6, FUSION_SIZE_SLICE_1_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_1_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_1_P4);

    num_blocks_kernel_2 =   CEIL(size_h3, FUSION_SIZE_SLICE_2_H3) * CEIL(size_h2, FUSION_SIZE_SLICE_2_H2) * CEIL(size_h1, FUSION_SIZE_SLICE_2_H1) * 
                            CEIL(size_p6, FUSION_SIZE_SLICE_2_P6) * CEIL(size_p5, FUSION_SIZE_SLICE_2_P5) * CEIL(size_p4, FUSION_SIZE_SLICE_2_P4);

    // (5) launch kernel(s)
    // Depends on # of Fused Kernel
    // dim3 gridsize_1(num_blocks_kernel_1);
    // dim3 blocksize_1(FUSION_SIZE_TB_1_X, FUSION_SIZE_TB_1_Y);

    dim3 gridsize_2(num_blocks_kernel_2);
    dim3 blocksize_2(FUSION_SIZE_TB_2_X, FUSION_SIZE_TB_2_Y);

    int	str_sd2_t3_h3 = 1;
    int str_sd2_t3_h2 = str_sd2_t3_h3 * size_h3;
    int str_sd2_t3_h1 = str_sd2_t3_h2 * size_h2;
    int str_sd2_t3_p6 = str_sd2_t3_h1 * size_h1;
    int str_sd2_t3_p5 = str_sd2_t3_p6 * size_p6;
    // int str_sd2_t3_p4 = str_sd2_t3_p5 * size_p5;

    // int str_reg_x_1 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    // int str_reg_y_1 = str_sd2_t3_p4;	// STR_SD2_T3_P4
    int str_reg_x_2 = str_sd2_t3_p5;	// STR_SD2_T3_P5
    int str_reg_y_2 = str_sd2_t3_p6;	// SDT_SD2_T3_P6

    // 
    dev_t3 = t3_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    kernel_ccsdT_sd2_9<<<gridsize_2, blocksize_2>>>(dev_t3, 
    dev_t2, dev_v2, 
    (int)size_h3, (int)size_h2, (int)size_h1, (int)size_p6, (int)size_p5, (int)size_p4, (int)size_p7,
    CEIL(size_h3, FUSION_SIZE_SLICE_2_H3), CEIL(size_h2, FUSION_SIZE_SLICE_2_H2), CEIL(size_h1, FUSION_SIZE_SLICE_2_H1), 
    CEIL(size_p6, FUSION_SIZE_SLICE_2_P6), CEIL(size_p5, FUSION_SIZE_SLICE_2_P5), CEIL(size_p4, FUSION_SIZE_SLICE_2_P4), 
    str_reg_x_2, str_reg_y_2,
    internal);
#endif
    // 
    // cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyDeviceToHost);

    // cudaFree(dev_t3);
    // cudaFree(dev_t2); cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

/*  
    singles
*/
#define JK_CCSD_T_FUSED_S1_SIZE_TILE_H1     4
#define JK_CCSD_T_FUSED_S1_SIZE_TILE_H2     4
#define JK_CCSD_T_FUSED_S1_SIZE_TILE_H3     4
#define JK_CCSD_T_FUSED_S1_SIZE_TILE_P4     4
#define JK_CCSD_T_FUSED_S1_SIZE_TILE_P5     4
#define JK_CCSD_T_FUSED_S1_SIZE_TILE_P6     4

#define JK_CCSD_T_FUSED_S1_SIZE_TB_X        64  // 3 indices mapped to threadIdx.x
#define JK_CCSD_T_FUSED_S1_SIZE_TB_Y        4   // 1 index mapped to threadIdx.y

#define JK_CCSD_T_FUSED_S1_SIZE_REG_X       4   // p5
#define JK_CCSD_T_FUSED_S1_SIZE_REG_Y       4   // p4

// kernel: s1_1
__global__ void jk_ccsd_t_s1_1(double* d_t3, 
double* d_t2_1, double* d_v2_1, 
size_t size_h3,     size_t size_h2,     size_t size_h1,     size_t size_p6,     size_t size_p5,     size_t size_p4, 
size_t num_blks_h3, size_t num_blks_h2, size_t num_blks_h1, size_t num_blks_p6, size_t num_blks_p5, size_t num_blks_p4, 
size_t stride_reg_x, size_t stride_reg_y)
{
    //  Shared Memory
    __shared__ double sm_a[16];     // "T_p4" * T_h1
    __shared__ double sm_b[256];    // T_h3 * T_h2 * T_p6 * "T_p5"

    //  offset-indices
    int idx_p6  = threadIdx.x / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int tmp_idx = threadIdx.x % (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int idx_h2  = tmp_idx     / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    int idx_h3  = threadIdx.x % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    int idx_h1  = threadIdx.y;

    //  blk-indices
    int blk_idx_p4  = blockIdx.x / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    tmp_idx         = blockIdx.x % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    int blk_idx_p5  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    int blk_idx_p6  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1);
    int blk_idx_h1  = tmp_idx    / (num_blks_h3 * num_blks_h2);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2);
    int blk_idx_h2  = tmp_idx    / (num_blks_h3);
    int blk_idx_h3  = blockIdx.x % num_blks_h3;

    //  boundary-checks
    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
    // 
    if ((size_h3 - (blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H3) 
        rng_h3 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    else
        rng_h3 = size_h3 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;

    //
    if ((size_h2 - (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) 
        rng_h2 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;
    else
        rng_h2 = size_h2 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;

    //
    if ((size_h1 - (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) 
        rng_h1 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;
    else
        rng_h1 = size_h1 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;

    //
    if ((size_p6 - (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) 
        rng_p6 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;
    else
        rng_p6 = size_p6 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;

    //
    if ((size_p5 - (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) 
        rng_p5 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;
    else
        rng_p5 = size_p5 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;

    //
    if ((size_p4 - (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) 
        rng_p4 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;
    else
        rng_p4 = size_p4 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;

    //
    int t3_based_addr =  blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                        (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2 + 
                        (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h1 + 
                        (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                        (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + 
                        (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    //
    double tmp_av[4];
    double tmp_bv[4];    
    double reg_tile[4][4];

    //
    for (int i = 0; i < 4; i++) // i -> p4
    for (int j = 0; j < 4; j++) // j -> p5
    reg_tile[i][j] = 0.0; 

    //                                        "x"         "x"
    //  >> s1_1:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h1] * v2[h3,h2,p6,p5]
    //
    {
        if (idx_h3 < rng_p4 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[idx_h3 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4] = d_t2_1[blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4 + idx_h3 + (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h2) * size_p4];

        if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_p5)
        sm_b[idx_h3 + (idx_h2 + (idx_p6 + (idx_h1) * 4) * 4) * 4] = d_v2_1[blk_idx_h3 * 4 + idx_h3 + (blk_idx_h2 * 4 + idx_h2 + (blk_idx_p6 * 4 + idx_p6 + (blk_idx_p5 * 4 + idx_h1) * size_p6) * size_h2) * size_h3];
        __syncthreads();

        //  "p4"
        tmp_av[0] = sm_a[0 + (idx_h1) * 4];
        tmp_av[1] = sm_a[1 + (idx_h1) * 4];
        tmp_av[2] = sm_a[2 + (idx_h1) * 4];
        tmp_av[3] = sm_a[3 + (idx_h1) * 4];

        //  "p5"
        tmp_bv[0] = sm_b[idx_h3 + (idx_h2 + (idx_p6 + (0) * 4) * 4) * 4];
        tmp_bv[1] = sm_b[idx_h3 + (idx_h2 + (idx_p6 + (1) * 4) * 4) * 4];
        tmp_bv[2] = sm_b[idx_h3 + (idx_h2 + (idx_p6 + (2) * 4) * 4) * 4];
        tmp_bv[3] = sm_b[idx_h3 + (idx_h2 + (idx_p6 + (3) * 4) * 4) * 4];

        //  "p4 x p5"
        reg_tile[0][0] += tmp_av[0] * tmp_bv[0];// * reg_tile[0][0];
        reg_tile[0][1] += tmp_av[0] * tmp_bv[1];// * reg_tile[0][1];
        reg_tile[0][2] += tmp_av[0] * tmp_bv[2];// * reg_tile[0][2];
        reg_tile[0][3] += tmp_av[0] * tmp_bv[3];// * reg_tile[0][3];

        reg_tile[1][0] += tmp_av[1] * tmp_bv[0];// * reg_tile[1][0];
        reg_tile[1][1] += tmp_av[1] * tmp_bv[1];// * reg_tile[1][1];
        reg_tile[1][2] += tmp_av[1] * tmp_bv[2];// * reg_tile[1][2];
        reg_tile[1][3] += tmp_av[1] * tmp_bv[3];// * reg_tile[1][3];

        reg_tile[2][0] += tmp_av[2] * tmp_bv[0];// * reg_tile[2][0];
        reg_tile[2][1] += tmp_av[2] * tmp_bv[1];// * reg_tile[2][1];
        reg_tile[2][2] += tmp_av[2] * tmp_bv[2];// * reg_tile[2][2];
        reg_tile[2][3] += tmp_av[2] * tmp_bv[3];// * reg_tile[2][3];

        reg_tile[3][0] += tmp_av[3] * tmp_bv[0];// * reg_tile[3][0];
        reg_tile[3][1] += tmp_av[3] * tmp_bv[1];// * reg_tile[3][1];
        reg_tile[3][2] += tmp_av[3] * tmp_bv[2];// * reg_tile[3][2];
        reg_tile[3][3] += tmp_av[3] * tmp_bv[3];// * reg_tile[3][3];
    }

    //
    //  to store the output
    //
    if (idx_h1 < rng_h1 && idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6)
    for (int i = 0; i < 4; i++) // p4
    {
        for (int j = 0; j < 4; j++) // p5
        {
            if (i < rng_p4 && j < rng_p5)
            d_t3[t3_based_addr + (i * stride_reg_x) + (j * stride_reg_y)] += reg_tile[i][j];
        }
    }
}

// caller: s1_1
void jk_ccsd_t_s1_1(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, double* host_t3, double* host_t2, double* host_v2)
{
    double *dev_t3;
	double *dev_t2; 
	double *dev_v2; 

    // t3
    // size_t size_t3   = sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;

    // s1_1:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h1] * v2[h3,h2,p6,p5]
    size_t size_t2 = sizeof(double) * size_p4 * size_h1;
    size_t size_v2 = sizeof(double) * size_h3 * size_h2 * size_p6 * size_p5;

    // cudaMalloc((void**) &dev_t3, size_t3); 
    // cudaMalloc((void**) &dev_t2, size_t2); 
    // cudaMalloc((void**) &dev_v2, size_v2); 
    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, size_t3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, size_t2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, size_v2, cudaMemcpyHostToDevice);

    // 
    size_t num_blks_h3 = CEIL(size_h3, JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    size_t num_blks_h2 = CEIL(size_h2, JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    size_t num_blks_h1 = CEIL(size_h1, JK_CCSD_T_FUSED_S1_SIZE_TILE_H1);
    size_t num_blks_p6 = CEIL(size_p6, JK_CCSD_T_FUSED_S1_SIZE_TILE_P6);
    size_t num_blks_p5 = CEIL(size_p5, JK_CCSD_T_FUSED_S1_SIZE_TILE_P5);
    size_t num_blks_p4 = CEIL(size_p4, JK_CCSD_T_FUSED_S1_SIZE_TILE_P4);

    //
    size_t num_blks_kernel = num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5 * num_blks_p4; 

    //
    dim3 gridsize(num_blks_kernel);
    dim3 blocksize(JK_CCSD_T_FUSED_S1_SIZE_TB_X, JK_CCSD_T_FUSED_S1_SIZE_TB_Y);

    // p4 (x) and p5 (y)
    int stride_reg_x = size_h3 * size_h2 * size_h1 * size_p6 * size_p5;
    int stride_reg_y = size_h3 * size_h2 * size_h1 * size_p6;

    // 
    dev_t3 = t3_s_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    //
    jk_ccsd_t_s1_1<<<gridsize, blocksize>>>(dev_t3, dev_t2, dev_v2, 
                                            size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, 
                                            num_blks_h3, num_blks_h2, num_blks_h1, num_blks_p6, num_blks_p5, num_blks_p4, 
                                            stride_reg_x, stride_reg_y);
#endif
    // cudaMemcpy(host_t3, dev_t3, size_t3, cudaMemcpyDeviceToHost);
    
    // cudaFree(dev_t3);
    // cudaFree(dev_t2);
    // cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: s1_2
__global__ void jk_ccsd_t_s1_2(double* d_t3, 
double* d_t2_2, double* d_v2_2, 
size_t size_h3,     size_t size_h2,     size_t size_h1,     size_t size_p6,     size_t size_p5,     size_t size_p4, 
size_t num_blks_h3, size_t num_blks_h2, size_t num_blks_h1, size_t num_blks_p6, size_t num_blks_p5, size_t num_blks_p4, 
size_t stride_reg_x, size_t stride_reg_y)
{
    //  Shared Memory
    __shared__ double sm_a[16];     // "T_p4" * T_h1
    __shared__ double sm_b[256];    // T_h3 * T_h2 * T_p6 * "T_p5"

    //  offset-indices
    int idx_p6  = threadIdx.x / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int tmp_idx = threadIdx.x % (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int idx_h2  = tmp_idx     / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    int idx_h3  = threadIdx.x % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    int idx_h1  = threadIdx.y;

    //  blk-indices
    int blk_idx_p4  = blockIdx.x / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    tmp_idx         = blockIdx.x % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    int blk_idx_p5  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    int blk_idx_p6  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1);
    int blk_idx_h1  = tmp_idx    / (num_blks_h3 * num_blks_h2);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2);
    int blk_idx_h2  = tmp_idx    / (num_blks_h3);
    int blk_idx_h3  = blockIdx.x % num_blks_h3;

    //  boundary-checks
    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
    // 
    if ((size_h3 - (blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H3) 
        rng_h3 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    else
        rng_h3 = size_h3 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;

    //
    if ((size_h2 - (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) 
        rng_h2 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;
    else
        rng_h2 = size_h2 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;

    //
    if ((size_h1 - (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) 
        rng_h1 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;
    else
        rng_h1 = size_h1 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;

    //
    if ((size_p6 - (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) 
        rng_p6 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;
    else
        rng_p6 = size_p6 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;

    //
    if ((size_p5 - (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) 
        rng_p5 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;
    else
        rng_p5 = size_p5 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;

    //
    if ((size_p4 - (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) 
        rng_p4 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;
    else
        rng_p4 = size_p4 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;

    //
    int t3_based_addr =  blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                        (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2 + 
                        (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h1 + 
                        (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                        (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + 
                        (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    //
    double tmp_av[4];
    double tmp_bv[4];    
    double reg_tile[4][4];

    //
    for (int i = 0; i < 4; i++) // i -> p4
    for (int j = 0; j < 4; j++) // j -> p5
    reg_tile[i][j] = 0.0; 

    //                                        "x1,x2"     "x1,x2,x3,y1"
    //  >> s1_2:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h2] * v2[h3,h1,p6,p5] (h3,h2,p6), (h1)
    //
    {
        if (idx_h3 < rng_p4 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[idx_h3 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4] = d_t2_2[blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4 + idx_h3 + (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2) * size_p4];

        if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p5)
        sm_b[idx_h3 + (idx_h2 + (idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3] 
        = d_v2_2[blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h2 + 
                (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + idx_h1) * size_p6) * size_h1) * size_h3];
        __syncthreads();

        //  "p4"
        tmp_av[0] = sm_a[0 + (idx_h2) * 4];
        tmp_av[1] = sm_a[1 + (idx_h2) * 4];
        tmp_av[2] = sm_a[2 + (idx_h2) * 4];
        tmp_av[3] = sm_a[3 + (idx_h2) * 4];

        //  "p5"
        tmp_bv[0] = sm_b[idx_h3 + (idx_h1 + (idx_p6 + (0) * 4) * 4) * 4];
        tmp_bv[1] = sm_b[idx_h3 + (idx_h1 + (idx_p6 + (1) * 4) * 4) * 4];
        tmp_bv[2] = sm_b[idx_h3 + (idx_h1 + (idx_p6 + (2) * 4) * 4) * 4];
        tmp_bv[3] = sm_b[idx_h3 + (idx_h1 + (idx_p6 + (3) * 4) * 4) * 4];

        //  "p4 x p5"
        reg_tile[0][0] -= tmp_av[0] * tmp_bv[0];
        reg_tile[0][1] -= tmp_av[0] * tmp_bv[1];
        reg_tile[0][2] -= tmp_av[0] * tmp_bv[2];
        reg_tile[0][3] -= tmp_av[0] * tmp_bv[3];

        reg_tile[1][0] -= tmp_av[1] * tmp_bv[0];
        reg_tile[1][1] -= tmp_av[1] * tmp_bv[1];
        reg_tile[1][2] -= tmp_av[1] * tmp_bv[2];
        reg_tile[1][3] -= tmp_av[1] * tmp_bv[3];

        reg_tile[2][0] -= tmp_av[2] * tmp_bv[0];
        reg_tile[2][1] -= tmp_av[2] * tmp_bv[1];
        reg_tile[2][2] -= tmp_av[2] * tmp_bv[2];
        reg_tile[2][3] -= tmp_av[2] * tmp_bv[3];

        reg_tile[3][0] -= tmp_av[3] * tmp_bv[0];
        reg_tile[3][1] -= tmp_av[3] * tmp_bv[1];
        reg_tile[3][2] -= tmp_av[3] * tmp_bv[2];
        reg_tile[3][3] -= tmp_av[3] * tmp_bv[3];
    }

    //
    //  to store the output
    //
    if (idx_h1 < rng_h1 && idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6)
    for (int i = 0; i < 4; i++) // p4
    {
        for (int j = 0; j < 4; j++) // p5
        {
            if (i < rng_p4 && j < rng_p5)
            d_t3[t3_based_addr + (i * stride_reg_x) + (j * stride_reg_y)] += reg_tile[i][j];
        }
    }
}
    
// caller:s1_2
void jk_ccsd_t_s1_2(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, double* host_t3, double* host_t2, double* host_v2)
{
    double *dev_t3;
    double *dev_t2; 
    double *dev_v2; 

    // t3
    // size_t size_t3   = sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;

    // s1_2:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h2] * v2[h3,h1,p6,p5]
    size_t size_t2 = sizeof(double) * size_p4 * size_h2;
    size_t size_v2 = sizeof(double) * size_h3 * size_h1 * size_p6 * size_p5;
    
    // cudaMalloc((void**) &dev_t3, size_t3); 
    // cudaMalloc((void**) &dev_t2, size_t2); 
    // cudaMalloc((void**) &dev_v2, size_v2); 
    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, size_t3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, size_t2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, size_v2, cudaMemcpyHostToDevice);

    // 
    size_t num_blks_h3 = CEIL(size_h3, JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    size_t num_blks_h2 = CEIL(size_h2, JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    size_t num_blks_h1 = CEIL(size_h1, JK_CCSD_T_FUSED_S1_SIZE_TILE_H1);
    size_t num_blks_p6 = CEIL(size_p6, JK_CCSD_T_FUSED_S1_SIZE_TILE_P6);
    size_t num_blks_p5 = CEIL(size_p5, JK_CCSD_T_FUSED_S1_SIZE_TILE_P5);
    size_t num_blks_p4 = CEIL(size_p4, JK_CCSD_T_FUSED_S1_SIZE_TILE_P4);

    //
    size_t num_blks_kernel = num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5 * num_blks_p4; 

    //
    dim3 gridsize(num_blks_kernel);
    dim3 blocksize(JK_CCSD_T_FUSED_S1_SIZE_TB_X, JK_CCSD_T_FUSED_S1_SIZE_TB_Y);

    // p4 (x) and p5 (y)
    int stride_reg_x = size_h3 * size_h2 * size_h1 * size_p6 * size_p5;
    int stride_reg_y = size_h3 * size_h2 * size_h1 * size_p6;

    // 
    dev_t3 = t3_s_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    //
    jk_ccsd_t_s1_2<<<gridsize, blocksize>>>(dev_t3, dev_t2, dev_v2, 
                                            size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, 
                                            num_blks_h3, num_blks_h2, num_blks_h1, num_blks_p6, num_blks_p5, num_blks_p4, 
                                            stride_reg_x, stride_reg_y);

    // cudaMemcpy(host_t3, dev_t3, size_t3, cudaMemcpyDeviceToHost);
#endif  
    // cudaFree(dev_t3);
    // cudaFree(dev_t2);
    // cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}
    
// kernel: s1_3
__global__ void jk_ccsd_t_s1_3(double* d_t3, 
double* d_t2_3, double* d_v2_3,
size_t size_h3,     size_t size_h2,     size_t size_h1,     size_t size_p6,     size_t size_p5,     size_t size_p4, 
size_t num_blks_h3, size_t num_blks_h2, size_t num_blks_h1, size_t num_blks_p6, size_t num_blks_p5, size_t num_blks_p4, 
size_t stride_reg_x, size_t stride_reg_y)
{
    //  Shared Memory
    __shared__ double sm_a[16];     // "T_p4" * T_h1
    __shared__ double sm_b[256];    // T_h3 * T_h2 * T_p6 * "T_p5"

    //  offset-indices
    int idx_p6  = threadIdx.x / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int tmp_idx = threadIdx.x % (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int idx_h2  = tmp_idx     / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    int idx_h3  = threadIdx.x % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    int idx_h1  = threadIdx.y;

    //  blk-indices
    int blk_idx_p4  = blockIdx.x / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    tmp_idx         = blockIdx.x % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    int blk_idx_p5  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    int blk_idx_p6  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1);
    int blk_idx_h1  = tmp_idx    / (num_blks_h3 * num_blks_h2);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2);
    int blk_idx_h2  = tmp_idx    / (num_blks_h3);
    int blk_idx_h3  = blockIdx.x % num_blks_h3;

    //  boundary-checks
    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
    // 
    if ((size_h3 - (blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H3) 
        rng_h3 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    else
        rng_h3 = size_h3 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;

    //
    if ((size_h2 - (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) 
        rng_h2 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;
    else
        rng_h2 = size_h2 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;

    //
    if ((size_h1 - (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) 
        rng_h1 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;
    else
        rng_h1 = size_h1 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;

    //
    if ((size_p6 - (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) 
        rng_p6 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;
    else
        rng_p6 = size_p6 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;

    //
    if ((size_p5 - (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) 
        rng_p5 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;
    else
        rng_p5 = size_p5 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;

    //
    if ((size_p4 - (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) 
        rng_p4 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;
    else
        rng_p4 = size_p4 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;

    //
    int t3_based_addr =  blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                        (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2 + 
                        (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h1 + 
                        (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                        (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + 
                        (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    //
    double tmp_av[4];
    double tmp_bv[4];    
    double reg_tile[4][4];

    //
    for (int i = 0; i < 4; i++) // i -> p4
    for (int j = 0; j < 4; j++) // j -> p5
    reg_tile[i][j] = 0.0; 

    //
    //  >> s1_3:   t3[h3,h2,h1,p6,p5,p4] -= t2[p4,h1] * v2[h3,h2,p6,p5] ??
    //  >> s1_3:   t3[h3,h2,h1,p6,p5,p4] += t1[p4,h3] * v2[h2,h1,p6,p5]
    //
    {
        if (idx_h3 < rng_p4 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[idx_h3 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4] = d_t2_3[blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4 + idx_h3 + (blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h2) * size_p4];

        if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p5)
        sm_b[idx_h3 + (idx_h2 + (idx_p6 + (idx_h1) * 4) * 4) * 4] = d_v2_3[blk_idx_h2 * 4 + idx_h3 + (blk_idx_h1 * 4 + idx_h2 + (blk_idx_p6 * 4 + idx_p6 + (blk_idx_p5 * 4 + idx_h1) * size_p6) * size_h1) * size_h2];
        __syncthreads();

        //  "p4"
        tmp_av[0] = sm_a[0 + (idx_h3) * 4];
        tmp_av[1] = sm_a[1 + (idx_h3) * 4];
        tmp_av[2] = sm_a[2 + (idx_h3) * 4];
        tmp_av[3] = sm_a[3 + (idx_h3) * 4];

        //  "p5"
        tmp_bv[0] = sm_b[idx_h2 + (idx_h1 + (idx_p6 + (0) * 4) * 4) * 4];
        tmp_bv[1] = sm_b[idx_h2 + (idx_h1 + (idx_p6 + (1) * 4) * 4) * 4];
        tmp_bv[2] = sm_b[idx_h2 + (idx_h1 + (idx_p6 + (2) * 4) * 4) * 4];
        tmp_bv[3] = sm_b[idx_h2 + (idx_h1 + (idx_p6 + (3) * 4) * 4) * 4];

        //  "p4 x p5"
        reg_tile[0][0] += tmp_av[0] * tmp_bv[0];
        reg_tile[0][1] += tmp_av[0] * tmp_bv[1];
        reg_tile[0][2] += tmp_av[0] * tmp_bv[2];
        reg_tile[0][3] += tmp_av[0] * tmp_bv[3];

        reg_tile[1][0] += tmp_av[1] * tmp_bv[0];
        reg_tile[1][1] += tmp_av[1] * tmp_bv[1];
        reg_tile[1][2] += tmp_av[1] * tmp_bv[2];
        reg_tile[1][3] += tmp_av[1] * tmp_bv[3];

        reg_tile[2][0] += tmp_av[2] * tmp_bv[0];
        reg_tile[2][1] += tmp_av[2] * tmp_bv[1];
        reg_tile[2][2] += tmp_av[2] * tmp_bv[2];
        reg_tile[2][3] += tmp_av[2] * tmp_bv[3];

        reg_tile[3][0] += tmp_av[3] * tmp_bv[0];
        reg_tile[3][1] += tmp_av[3] * tmp_bv[1];
        reg_tile[3][2] += tmp_av[3] * tmp_bv[2];
        reg_tile[3][3] += tmp_av[3] * tmp_bv[3];
    }

    //
    //  to store the output
    //
    if (idx_h1 < rng_h1 && idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6)
    for (int i = 0; i < 4; i++) // p4
    {
        for (int j = 0; j < 4; j++) // p5
        {
            if (i < rng_p4 && j < rng_p5)
            d_t3[t3_based_addr + (i * stride_reg_x) + (j * stride_reg_y)] += reg_tile[i][j];
        }
    }
}

// caller: s1_3
void jk_ccsd_t_s1_3(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, double* host_t3, double* host_t2, double* host_v2)
{
    double *dev_t3;
    double *dev_t2; 
    double *dev_v2; 

    // t3
    // size_t size_t3   = sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;

    // s1_3:   t3[h3,h2,h1,p6,p5,p4] += t1[p4,h3] * v2[h2,h1,p6,p5]
    size_t size_t2 = sizeof(double) * size_p4 * size_h3;
    size_t size_v2 = sizeof(double) * size_h2 * size_h1 * size_p6 * size_p5;
    
    // cudaMalloc((void**) &dev_t3, size_t3); 
    // cudaMalloc((void**) &dev_t2, size_t2); 
    // cudaMalloc((void**) &dev_v2, size_v2); 
    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, size_t3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, size_t2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, size_v2, cudaMemcpyHostToDevice);

    // 
    size_t num_blks_h3 = CEIL(size_h3, JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    size_t num_blks_h2 = CEIL(size_h2, JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    size_t num_blks_h1 = CEIL(size_h1, JK_CCSD_T_FUSED_S1_SIZE_TILE_H1);
    size_t num_blks_p6 = CEIL(size_p6, JK_CCSD_T_FUSED_S1_SIZE_TILE_P6);
    size_t num_blks_p5 = CEIL(size_p5, JK_CCSD_T_FUSED_S1_SIZE_TILE_P5);
    size_t num_blks_p4 = CEIL(size_p4, JK_CCSD_T_FUSED_S1_SIZE_TILE_P4);

    //
    size_t num_blks_kernel = num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5 * num_blks_p4; 

    //
    dim3 gridsize(num_blks_kernel);
    dim3 blocksize(JK_CCSD_T_FUSED_S1_SIZE_TB_X, JK_CCSD_T_FUSED_S1_SIZE_TB_Y);

    // p4 (x) and p5 (y)
    int stride_reg_x = size_h3 * size_h2 * size_h1 * size_p6 * size_p5;
    int stride_reg_y = size_h3 * size_h2 * size_h1 * size_p6;

    // 
    dev_t3 = t3_s_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    //
    jk_ccsd_t_s1_3<<<gridsize, blocksize>>>(dev_t3, dev_t2, dev_v2, 
                                            size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, 
                                            num_blks_h3, num_blks_h2, num_blks_h1, num_blks_p6, num_blks_p5, num_blks_p4, 
                                            stride_reg_x, stride_reg_y);

    // cudaMemcpy(host_t3, dev_t3, size_t3, cudaMemcpyDeviceToHost);
#endif
    // cudaFree(dev_t3);
    // cudaFree(dev_t2);
    // cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: s1_4
__global__ void jk_ccsd_t_s1_4(double* d_t3, 
double* d_t2_4, double* d_v2_4, 
size_t size_h3,     size_t size_h2,     size_t size_h1,     size_t size_p6,     size_t size_p5,     size_t size_p4, 
size_t num_blks_h3, size_t num_blks_h2, size_t num_blks_h1, size_t num_blks_p6, size_t num_blks_p5, size_t num_blks_p4, 
size_t stride_reg_x, size_t stride_reg_y)
{
    //  Shared Memory
    __shared__ double sm_a[16];     // "T_p4" * T_h1
    __shared__ double sm_b[256];    // T_h3 * T_h2 * T_p6 * "T_p5"

    //  offset-indices
    int idx_p6  = threadIdx.x / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int tmp_idx = threadIdx.x % (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int idx_h2  = tmp_idx     / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    int idx_h3  = threadIdx.x % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    int idx_h1  = threadIdx.y;

    //  blk-indices
    int blk_idx_p4  = blockIdx.x / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    tmp_idx         = blockIdx.x % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    int blk_idx_p5  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    int blk_idx_p6  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1);
    int blk_idx_h1  = tmp_idx    / (num_blks_h3 * num_blks_h2);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2);
    int blk_idx_h2  = tmp_idx    / (num_blks_h3);
    int blk_idx_h3  = blockIdx.x % num_blks_h3;

    //  boundary-checks
    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
    // 
    if ((size_h3 - (blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H3) 
        rng_h3 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    else
        rng_h3 = size_h3 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;

    //
    if ((size_h2 - (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) 
        rng_h2 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;
    else
        rng_h2 = size_h2 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;

    //
    if ((size_h1 - (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) 
        rng_h1 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;
    else
        rng_h1 = size_h1 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;

    //
    if ((size_p6 - (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) 
        rng_p6 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;
    else
        rng_p6 = size_p6 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;

    //
    if ((size_p5 - (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) 
        rng_p5 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;
    else
        rng_p5 = size_p5 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;

    //
    if ((size_p4 - (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) 
        rng_p4 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;
    else
        rng_p4 = size_p4 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;

    //
    int t3_based_addr =  blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                        (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2 + 
                        (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h1 + 
                        (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                        (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + 
                        (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    //
    double tmp_av[4];
    double tmp_bv[4];    
    double reg_tile[4][4];

    //
    for (int i = 0; i < 4; i++) // i -> p4
    for (int j = 0; j < 4; j++) // j -> p5
    reg_tile[i][j] = 0.0; 

    //
    //  >> s1_4:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h1] * v2[h3,h2,p6,p4] (h3,h2,p6), (h1)
    //
    {
        if (idx_h3 < rng_p5 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[idx_h3 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5] 
        = d_t2_4[blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + idx_h3 + 
                (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h2) * size_p5];

        if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6 && idx_h1 < rng_p4)
        sm_b[idx_h3 + (idx_h2 + (idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3] 
        = d_v2_4[blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2 + 
                (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4 + idx_h1) * size_p6) * size_h2) * size_h3];
        __syncthreads();

        //  "p5"
        tmp_av[0] = sm_a[0 + (idx_h1) * 4];
        tmp_av[1] = sm_a[1 + (idx_h1) * 4];
        tmp_av[2] = sm_a[2 + (idx_h1) * 4];
        tmp_av[3] = sm_a[3 + (idx_h1) * 4];

        //  "p4"
        tmp_bv[0] = sm_b[idx_h3 + (idx_h2 + (idx_p6 + (0) * 4) * 4) * 4];
        tmp_bv[1] = sm_b[idx_h3 + (idx_h2 + (idx_p6 + (1) * 4) * 4) * 4];
        tmp_bv[2] = sm_b[idx_h3 + (idx_h2 + (idx_p6 + (2) * 4) * 4) * 4];
        tmp_bv[3] = sm_b[idx_h3 + (idx_h2 + (idx_p6 + (3) * 4) * 4) * 4];

        //  "p4 x p5"
        reg_tile[0][0] -= tmp_av[0] * tmp_bv[0];
        reg_tile[0][1] -= tmp_av[1] * tmp_bv[0];
        reg_tile[0][2] -= tmp_av[2] * tmp_bv[0];
        reg_tile[0][3] -= tmp_av[3] * tmp_bv[0];

        reg_tile[1][0] -= tmp_av[0] * tmp_bv[1];
        reg_tile[1][1] -= tmp_av[1] * tmp_bv[1];
        reg_tile[1][2] -= tmp_av[2] * tmp_bv[1];
        reg_tile[1][3] -= tmp_av[3] * tmp_bv[1];

        reg_tile[2][0] -= tmp_av[0] * tmp_bv[2];
        reg_tile[2][1] -= tmp_av[1] * tmp_bv[2];
        reg_tile[2][2] -= tmp_av[2] * tmp_bv[2];
        reg_tile[2][3] -= tmp_av[3] * tmp_bv[2];

        reg_tile[3][0] -= tmp_av[0] * tmp_bv[3];
        reg_tile[3][1] -= tmp_av[1] * tmp_bv[3];
        reg_tile[3][2] -= tmp_av[2] * tmp_bv[3];
        reg_tile[3][3] -= tmp_av[3] * tmp_bv[3];
    }

    //
    //  to store the output
    //
    if (idx_h1 < rng_h1 && idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6)
    for (int i = 0; i < 4; i++) // p4
    {
        for (int j = 0; j < 4; j++) // p5
        {
            if (i < rng_p4 && j < rng_p5)
            d_t3[t3_based_addr + (i * stride_reg_x) + (j * stride_reg_y)] += reg_tile[i][j];
        }
    }
}

// caller: s1_4
void jk_ccsd_t_s1_4(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, double* host_t3, double* host_t2, double* host_v2)
{
    double *dev_t3;
    double *dev_t2; 
    double *dev_v2; 

    // t3
    // size_t size_t3   = sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;

    // s1_4:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h1] * v2[h3,h2,p6,p4]
    size_t size_t2 = sizeof(double) * size_p5 * size_h1;
    size_t size_v2 = sizeof(double) * size_h3 * size_h2 * size_p6 * size_p4;
    
    // cudaMalloc((void**) &dev_t3, size_t3); 
    // cudaMalloc((void**) &dev_t2, size_t2); 
    // cudaMalloc((void**) &dev_v2, size_v2); 
    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, size_t3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, size_t2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, size_v2, cudaMemcpyHostToDevice);

    // 
    size_t num_blks_h3 = CEIL(size_h3, JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    size_t num_blks_h2 = CEIL(size_h2, JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    size_t num_blks_h1 = CEIL(size_h1, JK_CCSD_T_FUSED_S1_SIZE_TILE_H1);
    size_t num_blks_p6 = CEIL(size_p6, JK_CCSD_T_FUSED_S1_SIZE_TILE_P6);
    size_t num_blks_p5 = CEIL(size_p5, JK_CCSD_T_FUSED_S1_SIZE_TILE_P5);
    size_t num_blks_p4 = CEIL(size_p4, JK_CCSD_T_FUSED_S1_SIZE_TILE_P4);

    //
    size_t num_blks_kernel = num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5 * num_blks_p4; 

    //
    dim3 gridsize(num_blks_kernel);
    dim3 blocksize(JK_CCSD_T_FUSED_S1_SIZE_TB_X, JK_CCSD_T_FUSED_S1_SIZE_TB_Y);

    // p4 (x) and p5 (y)
    int stride_reg_x = size_h3 * size_h2 * size_h1 * size_p6 * size_p5;
    int stride_reg_y = size_h3 * size_h2 * size_h1 * size_p6;

    // 
    dev_t3 = t3_s_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    //
    jk_ccsd_t_s1_4<<<gridsize, blocksize>>>(dev_t3, dev_t2, dev_v2, 
                                            size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, 
                                            num_blks_h3, num_blks_h2, num_blks_h1, num_blks_p6, num_blks_p5, num_blks_p4, 
                                            stride_reg_x, stride_reg_y);

    // cudaMemcpy(host_t3, dev_t3, size_t3, cudaMemcpyDeviceToHost);
#endif
    // cudaFree(dev_t3);
    // cudaFree(dev_t2);
    // cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: s1_5
__global__ void jk_ccsd_t_s1_5(double* d_t3, 
double* d_t2_5, double* d_v2_5, 
size_t size_h3,     size_t size_h2,     size_t size_h1,     size_t size_p6,     size_t size_p5,     size_t size_p4, 
size_t num_blks_h3, size_t num_blks_h2, size_t num_blks_h1, size_t num_blks_p6, size_t num_blks_p5, size_t num_blks_p4, 
size_t stride_reg_x, size_t stride_reg_y)
{
    //  Shared Memory
    __shared__ double sm_a[16];     // "T_p4" * T_h1
    __shared__ double sm_b[256];    // T_h3 * T_h2 * T_p6 * "T_p5"

    //  offset-indices
    int idx_p6  = threadIdx.x / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int tmp_idx = threadIdx.x % (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int idx_h2  = tmp_idx     / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    int idx_h3  = threadIdx.x % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    int idx_h1  = threadIdx.y;

    //  blk-indices
    int blk_idx_p4  = blockIdx.x / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    tmp_idx         = blockIdx.x % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    int blk_idx_p5  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    int blk_idx_p6  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1);
    int blk_idx_h1  = tmp_idx    / (num_blks_h3 * num_blks_h2);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2);
    int blk_idx_h2  = tmp_idx    / (num_blks_h3);
    int blk_idx_h3  = blockIdx.x % num_blks_h3;

    //  boundary-checks
    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
    // 
    if ((size_h3 - (blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H3) 
        rng_h3 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    else
        rng_h3 = size_h3 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;

    //
    if ((size_h2 - (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) 
        rng_h2 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;
    else
        rng_h2 = size_h2 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;

    //
    if ((size_h1 - (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) 
        rng_h1 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;
    else
        rng_h1 = size_h1 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;

    //
    if ((size_p6 - (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) 
        rng_p6 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;
    else
        rng_p6 = size_p6 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;

    //
    if ((size_p5 - (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) 
        rng_p5 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;
    else
        rng_p5 = size_p5 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;

    //
    if ((size_p4 - (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) 
        rng_p4 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;
    else
        rng_p4 = size_p4 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;

    //
    int t3_based_addr =  blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                        (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2 + 
                        (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h1 + 
                        (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                        (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + 
                        (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    //
    double tmp_av[4];
    double tmp_bv[4];    
    double reg_tile[4][4];

    //
    for (int i = 0; i < 4; i++) // i -> p4
    for (int j = 0; j < 4; j++) // j -> p5
    reg_tile[i][j] = 0.0; 

    //
    //  >> s1_5:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h2] * v2[h3,h1,p6,p4] (h3,h2,p6), (h1)
    //
    {
        if (idx_h3 < rng_p5 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[idx_h3 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5] 
        = d_t2_5[blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + idx_h3 + 
                (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2) * size_p5];

        if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p4)
        sm_b[idx_h3 + (idx_h2 + (idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3] 
        = d_v2_5[blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h2 + 
                (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4 + idx_h1) * size_p6) * size_h1) * size_h3];
        __syncthreads();

        //  "p5"
        tmp_av[0] = sm_a[0 + (idx_h2) * 4];
        tmp_av[1] = sm_a[1 + (idx_h2) * 4];
        tmp_av[2] = sm_a[2 + (idx_h2) * 4];
        tmp_av[3] = sm_a[3 + (idx_h2) * 4];

        //  "p4"
        tmp_bv[0] = sm_b[idx_h3 + (idx_h1 + (idx_p6 + (0) * 4) * 4) * 4];
        tmp_bv[1] = sm_b[idx_h3 + (idx_h1 + (idx_p6 + (1) * 4) * 4) * 4];
        tmp_bv[2] = sm_b[idx_h3 + (idx_h1 + (idx_p6 + (2) * 4) * 4) * 4];
        tmp_bv[3] = sm_b[idx_h3 + (idx_h1 + (idx_p6 + (3) * 4) * 4) * 4];

        //  "p4 x p5"
        reg_tile[0][0] += tmp_av[0] * tmp_bv[0];
        reg_tile[0][1] += tmp_av[1] * tmp_bv[0];
        reg_tile[0][2] += tmp_av[2] * tmp_bv[0];
        reg_tile[0][3] += tmp_av[3] * tmp_bv[0];

        reg_tile[1][0] += tmp_av[0] * tmp_bv[1];
        reg_tile[1][1] += tmp_av[1] * tmp_bv[1];
        reg_tile[1][2] += tmp_av[2] * tmp_bv[1];
        reg_tile[1][3] += tmp_av[3] * tmp_bv[1];

        reg_tile[2][0] += tmp_av[0] * tmp_bv[2];
        reg_tile[2][1] += tmp_av[1] * tmp_bv[2];
        reg_tile[2][2] += tmp_av[2] * tmp_bv[2];
        reg_tile[2][3] += tmp_av[3] * tmp_bv[2];

        reg_tile[3][0] += tmp_av[0] * tmp_bv[3];
        reg_tile[3][1] += tmp_av[1] * tmp_bv[3];
        reg_tile[3][2] += tmp_av[2] * tmp_bv[3];
        reg_tile[3][3] += tmp_av[3] * tmp_bv[3];
    }

    //
    //  to store the output
    //
    if (idx_h1 < rng_h1 && idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6)
    for (int i = 0; i < 4; i++) // p4
    {
        for (int j = 0; j < 4; j++) // p5
        {
            if (i < rng_p4 && j < rng_p5)
            d_t3[t3_based_addr + (i * stride_reg_x) + (j * stride_reg_y)] += reg_tile[i][j];
        }
    }
}

// caller: s1_5
void jk_ccsd_t_s1_5(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, double* host_t3, double* host_t2, double* host_v2)
{
    double *dev_t3;
    double *dev_t2; 
    double *dev_v2; 

    // t3
    // size_t size_t3   = sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;

    // s1_5:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h2] * v2[h3,h1,p6,p4]
    size_t size_t2 = sizeof(double) * size_p5 * size_h2;
    size_t size_v2 = sizeof(double) * size_h3 * size_h1 * size_p6 * size_p4;
    
    // cudaMalloc((void**) &dev_t3, size_t3); 
    // cudaMalloc((void**) &dev_t2, size_t2); 
    // cudaMalloc((void**) &dev_v2, size_v2); 
    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, size_t3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, size_t2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, size_v2, cudaMemcpyHostToDevice);

    // 
    size_t num_blks_h3 = CEIL(size_h3, JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    size_t num_blks_h2 = CEIL(size_h2, JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    size_t num_blks_h1 = CEIL(size_h1, JK_CCSD_T_FUSED_S1_SIZE_TILE_H1);
    size_t num_blks_p6 = CEIL(size_p6, JK_CCSD_T_FUSED_S1_SIZE_TILE_P6);
    size_t num_blks_p5 = CEIL(size_p5, JK_CCSD_T_FUSED_S1_SIZE_TILE_P5);
    size_t num_blks_p4 = CEIL(size_p4, JK_CCSD_T_FUSED_S1_SIZE_TILE_P4);

    //
    size_t num_blks_kernel = num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5 * num_blks_p4; 

    //
    dim3 gridsize(num_blks_kernel);
    dim3 blocksize(JK_CCSD_T_FUSED_S1_SIZE_TB_X, JK_CCSD_T_FUSED_S1_SIZE_TB_Y);

    // p4 (x) and p5 (y)
    int stride_reg_x = size_h3 * size_h2 * size_h1 * size_p6 * size_p5;
    int stride_reg_y = size_h3 * size_h2 * size_h1 * size_p6;

    // 
    dev_t3 = t3_s_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    //
    jk_ccsd_t_s1_5<<<gridsize, blocksize>>>(dev_t3, dev_t2, dev_v2, 
                                            size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, 
                                            num_blks_h3, num_blks_h2, num_blks_h1, num_blks_p6, num_blks_p5, num_blks_p4, 
                                            stride_reg_x, stride_reg_y);

    // cudaMemcpy(host_t3, dev_t3, size_t3, cudaMemcpyDeviceToHost);
#endif
    // cudaFree(dev_t3);
    // cudaFree(dev_t2);
    // cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: s1_6
__global__ void jk_ccsd_t_s1_6(double* d_t3, 
double* d_t2_6, double* d_v2_6,
size_t size_h3,     size_t size_h2,     size_t size_h1,     size_t size_p6,     size_t size_p5,     size_t size_p4, 
size_t num_blks_h3, size_t num_blks_h2, size_t num_blks_h1, size_t num_blks_p6, size_t num_blks_p5, size_t num_blks_p4, 
size_t stride_reg_x, size_t stride_reg_y)
{
    //  Shared Memory
    __shared__ double sm_a[16];     // "T_p4" * T_h1
    __shared__ double sm_b[256];    // T_h3 * T_h2 * T_p6 * "T_p5"

    //  offset-indices
    int idx_p6  = threadIdx.x / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int tmp_idx = threadIdx.x % (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int idx_h2  = tmp_idx     / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    int idx_h3  = threadIdx.x % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    int idx_h1  = threadIdx.y;

    //  blk-indices
    int blk_idx_p4  = blockIdx.x / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    tmp_idx         = blockIdx.x % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    int blk_idx_p5  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    int blk_idx_p6  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1);
    int blk_idx_h1  = tmp_idx    / (num_blks_h3 * num_blks_h2);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2);
    int blk_idx_h2  = tmp_idx    / (num_blks_h3);
    int blk_idx_h3  = blockIdx.x % num_blks_h3;

    //  boundary-checks
    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
    // 
    if ((size_h3 - (blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H3) 
        rng_h3 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    else
        rng_h3 = size_h3 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;

    //
    if ((size_h2 - (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) 
        rng_h2 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;
    else
        rng_h2 = size_h2 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;

    //
    if ((size_h1 - (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) 
        rng_h1 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;
    else
        rng_h1 = size_h1 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;

    //
    if ((size_p6 - (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) 
        rng_p6 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;
    else
        rng_p6 = size_p6 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;

    //
    if ((size_p5 - (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) 
        rng_p5 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;
    else
        rng_p5 = size_p5 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;

    //
    if ((size_p4 - (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) 
        rng_p4 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;
    else
        rng_p4 = size_p4 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;

    //
    int t3_based_addr =  blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                        (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2 + 
                        (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h1 + 
                        (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                        (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + 
                        (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    //
    double tmp_av[4];
    double tmp_bv[4];    
    double reg_tile[4][4];

    //
    for (int i = 0; i < 4; i++) // i -> p4
    for (int j = 0; j < 4; j++) // j -> p5
    reg_tile[i][j] = 0.0; 

    //
    //  >> s1_6:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h3] * v2[h2,h1,p6,p4] (h3,h2,p6), (h1)
    //
    {
        if (idx_h3 < rng_p5 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[idx_h3 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5] 
        = d_t2_6[blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + idx_h3 + 
                (blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h2) * size_p5];

        if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && idx_p6 < rng_p6 && idx_h1 < rng_p4)
        sm_b[idx_h3 + (idx_h2 + (idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2] 
        = d_v2_6[blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h3 + 
                (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h2 + 
                (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4 + idx_h1) * size_p6) * size_h1) * size_h2];
        __syncthreads();

        //  "p5"
        tmp_av[0] = sm_a[0 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5];
        tmp_av[1] = sm_a[1 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5];
        tmp_av[2] = sm_a[2 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5];
        tmp_av[3] = sm_a[3 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5];

        //  "p4"
        tmp_bv[0] = sm_b[idx_h2 + (idx_h1 + (idx_p6 + (0) * 4) * 4) * 4];
        tmp_bv[1] = sm_b[idx_h2 + (idx_h1 + (idx_p6 + (1) * 4) * 4) * 4];
        tmp_bv[2] = sm_b[idx_h2 + (idx_h1 + (idx_p6 + (2) * 4) * 4) * 4];
        tmp_bv[3] = sm_b[idx_h2 + (idx_h1 + (idx_p6 + (3) * 4) * 4) * 4];

        //  "p4 x p5"
        reg_tile[0][0] -= tmp_av[0] * tmp_bv[0];
        reg_tile[0][1] -= tmp_av[1] * tmp_bv[0];
        reg_tile[0][2] -= tmp_av[2] * tmp_bv[0];
        reg_tile[0][3] -= tmp_av[3] * tmp_bv[0];

        reg_tile[1][0] -= tmp_av[0] * tmp_bv[1];
        reg_tile[1][1] -= tmp_av[1] * tmp_bv[1];
        reg_tile[1][2] -= tmp_av[2] * tmp_bv[1];
        reg_tile[1][3] -= tmp_av[3] * tmp_bv[1];

        reg_tile[2][0] -= tmp_av[0] * tmp_bv[2];
        reg_tile[2][1] -= tmp_av[1] * tmp_bv[2];
        reg_tile[2][2] -= tmp_av[2] * tmp_bv[2];
        reg_tile[2][3] -= tmp_av[3] * tmp_bv[2];

        reg_tile[3][0] -= tmp_av[0] * tmp_bv[3];
        reg_tile[3][1] -= tmp_av[1] * tmp_bv[3];
        reg_tile[3][2] -= tmp_av[2] * tmp_bv[3];
        reg_tile[3][3] -= tmp_av[3] * tmp_bv[3];
    }

    //
    //  to store the output
    //
    if (idx_h1 < rng_h1 && idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6)
    for (int i = 0; i < 4; i++) // p4
    {
        for (int j = 0; j < 4; j++) // p5
        {
            if (i < rng_p4 && j < rng_p5)
            d_t3[t3_based_addr + (i * stride_reg_x) + (j * stride_reg_y)] += reg_tile[i][j];
        }
    }
}

// caller: s1_6
void jk_ccsd_t_s1_6(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, double* host_t3, double* host_t2, double* host_v2)
{
    double *dev_t3;
    double *dev_t2; 
    double *dev_v2; 

    // t3
    // size_t size_t3   = sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;

    // s1_6:   t3[h3,h2,h1,p6,p5,p4] -= t2[p5,h3] * v2[h2,h1,p6,p4]
    size_t size_t2 = sizeof(double) * size_p5 * size_h3;
    size_t size_v2 = sizeof(double) * size_h2 * size_h1 * size_p6 * size_p4;
    
    // cudaMalloc((void**) &dev_t3, size_t3); 
    // cudaMalloc((void**) &dev_t2, size_t2); 
    // cudaMalloc((void**) &dev_v2, size_v2); 
    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, size_t3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, size_t2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, size_v2, cudaMemcpyHostToDevice);

    // 
    size_t num_blks_h3 = CEIL(size_h3, JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    size_t num_blks_h2 = CEIL(size_h2, JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    size_t num_blks_h1 = CEIL(size_h1, JK_CCSD_T_FUSED_S1_SIZE_TILE_H1);
    size_t num_blks_p6 = CEIL(size_p6, JK_CCSD_T_FUSED_S1_SIZE_TILE_P6);
    size_t num_blks_p5 = CEIL(size_p5, JK_CCSD_T_FUSED_S1_SIZE_TILE_P5);
    size_t num_blks_p4 = CEIL(size_p4, JK_CCSD_T_FUSED_S1_SIZE_TILE_P4);

    //
    size_t num_blks_kernel = num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5 * num_blks_p4; 

    //
    dim3 gridsize(num_blks_kernel);
    dim3 blocksize(JK_CCSD_T_FUSED_S1_SIZE_TB_X, JK_CCSD_T_FUSED_S1_SIZE_TB_Y);

    // p4 (x) and p5 (y)
    int stride_reg_x = size_h3 * size_h2 * size_h1 * size_p6 * size_p5;
    int stride_reg_y = size_h3 * size_h2 * size_h1 * size_p6;

    // 
    dev_t3 = t3_s_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    //
    jk_ccsd_t_s1_6<<<gridsize, blocksize>>>(dev_t3, dev_t2, dev_v2, 
                                            size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, 
                                            num_blks_h3, num_blks_h2, num_blks_h1, num_blks_p6, num_blks_p5, num_blks_p4, 
                                            stride_reg_x, stride_reg_y);

    // cudaMemcpy(host_t3, dev_t3, size_t3, cudaMemcpyDeviceToHost);
#endif
    // cudaFree(dev_t3);
    // cudaFree(dev_t2);
    // cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: s1_7
__global__ void jk_ccsd_t_s1_7(double* d_t3, 
double* d_t2_7, double* d_v2_7, 
size_t size_h3,     size_t size_h2,     size_t size_h1,     size_t size_p6,     size_t size_p5,     size_t size_p4, 
size_t num_blks_h3, size_t num_blks_h2, size_t num_blks_h1, size_t num_blks_p6, size_t num_blks_p5, size_t num_blks_p4, 
size_t stride_reg_x, size_t stride_reg_y)
{
    //  Shared Memory
    __shared__ double sm_a[16];     // "T_p4" * T_h1
    __shared__ double sm_b[256];    // T_h3 * T_h2 * T_p6 * "T_p5"

    //  offset-indices
    int idx_p6  = threadIdx.x / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int tmp_idx = threadIdx.x % (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int idx_h2  = tmp_idx     / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    int idx_h3  = threadIdx.x % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    int idx_h1  = threadIdx.y;

    //  blk-indices
    int blk_idx_p4  = blockIdx.x / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    tmp_idx         = blockIdx.x % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    int blk_idx_p5  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    int blk_idx_p6  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1);
    int blk_idx_h1  = tmp_idx    / (num_blks_h3 * num_blks_h2);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2);
    int blk_idx_h2  = tmp_idx    / (num_blks_h3);
    int blk_idx_h3  = blockIdx.x % num_blks_h3;

    //  boundary-checks
    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
    // 
    if ((size_h3 - (blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H3) 
        rng_h3 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    else
        rng_h3 = size_h3 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;

    //
    if ((size_h2 - (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) 
        rng_h2 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;
    else
        rng_h2 = size_h2 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;

    //
    if ((size_h1 - (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) 
        rng_h1 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;
    else
        rng_h1 = size_h1 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;

    //
    if ((size_p6 - (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) 
        rng_p6 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;
    else
        rng_p6 = size_p6 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;

    //
    if ((size_p5 - (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) 
        rng_p5 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;
    else
        rng_p5 = size_p5 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;

    //
    if ((size_p4 - (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) 
        rng_p4 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;
    else
        rng_p4 = size_p4 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;

    //
    int t3_based_addr =  blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                        (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2 + 
                        (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h1 + 
                        (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                        (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + 
                        (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    //
    // double tmp_av[4];
    // double tmp_bv[4];    
    double reg_tile[4][4];

    //
    for (int i = 0; i < 4; i++) // i -> p4
    for (int j = 0; j < 4; j++) // j -> p5
    reg_tile[i][j] = 0.0; 

    //
    //  >> s1_7:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h1] * v2[h3,h2,p5,p4] (h3,h2,p6), (h1)
    //
    {
        if (idx_h3 < rng_p6 && idx_h2 < rng_h1 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[idx_h3 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] 
        = d_t2_7[blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_h3 + 
                (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h2) * size_p6];

        if (idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p5 && idx_h1 < rng_p4)
        sm_b[idx_h3 + (idx_h2 + (idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3] 
        = d_v2_7[blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2 + 
                (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + idx_p6 + 
                (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4 + idx_h1) * size_p5) * size_h2) * size_h3];
        __syncthreads();

        //  "p4" x "p5"
        reg_tile[0][0] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (0 + (0) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[0][1] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (1 + (0) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[0][2] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (2 + (0) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[0][3] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (3 + (0) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];

        reg_tile[1][0] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (0 + (1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[1][1] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (1 + (1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[1][2] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (2 + (1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[1][3] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (3 + (1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];

        reg_tile[2][0] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (0 + (2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[2][1] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (1 + (2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[2][2] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (2 + (2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[2][3] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (3 + (2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];

        reg_tile[3][0] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (0 + (3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[3][1] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (1 + (3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[3][2] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (2 + (3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[3][3] += sm_a[idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h2 + (3 + (3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        
    }

    //
    //  to store the output
    //
    if (idx_h1 < rng_h1 && idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6)
    for (int i = 0; i < 4; i++) // p4
    {
        for (int j = 0; j < 4; j++) // p5
        {
            if (i < rng_p4 && j < rng_p5)
            d_t3[t3_based_addr + (i * stride_reg_x) + (j * stride_reg_y)] += reg_tile[i][j];
        }
    }
}

// caller: s1_7
void jk_ccsd_t_s1_7(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, double* host_t3, double* host_t2, double* host_v2)
{
    double *dev_t3;
    double *dev_t2; 
    double *dev_v2; 

    // t3
    // size_t size_t3   = sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;

    // s1_7:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h1] * v2[h3,h2,p5,p4]
    size_t size_t2 = sizeof(double) * size_p6 * size_h1;
    size_t size_v2 = sizeof(double) * size_h3 * size_h2 * size_p5 * size_p4;
    
    // cudaMalloc((void**) &dev_t3, size_t3); 
    // cudaMalloc((void**) &dev_t2, size_t2); 
    // cudaMalloc((void**) &dev_v2, size_v2); 
    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, size_t3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, size_t2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, size_v2, cudaMemcpyHostToDevice);

    // 
    size_t num_blks_h3 = CEIL(size_h3, JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    size_t num_blks_h2 = CEIL(size_h2, JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    size_t num_blks_h1 = CEIL(size_h1, JK_CCSD_T_FUSED_S1_SIZE_TILE_H1);
    size_t num_blks_p6 = CEIL(size_p6, JK_CCSD_T_FUSED_S1_SIZE_TILE_P6);
    size_t num_blks_p5 = CEIL(size_p5, JK_CCSD_T_FUSED_S1_SIZE_TILE_P5);
    size_t num_blks_p4 = CEIL(size_p4, JK_CCSD_T_FUSED_S1_SIZE_TILE_P4);

    //
    size_t num_blks_kernel = num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5 * num_blks_p4; 

    //
    dim3 gridsize(num_blks_kernel);
    dim3 blocksize(JK_CCSD_T_FUSED_S1_SIZE_TB_X, JK_CCSD_T_FUSED_S1_SIZE_TB_Y);

    // p4 (x) and p5 (y)
    int stride_reg_x = size_h3 * size_h2 * size_h1 * size_p6 * size_p5;
    int stride_reg_y = size_h3 * size_h2 * size_h1 * size_p6;

    // 
    dev_t3 = t3_s_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    //
    jk_ccsd_t_s1_7<<<gridsize, blocksize>>>(dev_t3, dev_t2, dev_v2, 
                                            size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, 
                                            num_blks_h3, num_blks_h2, num_blks_h1, num_blks_p6, num_blks_p5, num_blks_p4, 
                                            stride_reg_x, stride_reg_y);

    // cudaMemcpy(host_t3, dev_t3, size_t3, cudaMemcpyDeviceToHost);
#endif
    // cudaFree(dev_t3);
    // cudaFree(dev_t2);
    // cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: s1_8
__global__ void jk_ccsd_t_s1_8(double* d_t3, 
double* d_t2_8, double* d_v2_8, 
size_t size_h3,     size_t size_h2,     size_t size_h1,     size_t size_p6,     size_t size_p5,     size_t size_p4, 
size_t num_blks_h3, size_t num_blks_h2, size_t num_blks_h1, size_t num_blks_p6, size_t num_blks_p5, size_t num_blks_p4, 
size_t stride_reg_x, size_t stride_reg_y)
{
    //  Shared Memory
    __shared__ double sm_a[16];     // "T_p4" * T_h1
    __shared__ double sm_b[256];    // T_h3 * T_h2 * T_p6 * "T_p5"

    //  offset-indices
    int idx_p6  = threadIdx.x / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int tmp_idx = threadIdx.x % (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int idx_h2  = tmp_idx     / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    int idx_h3  = threadIdx.x % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    int idx_h1  = threadIdx.y;

    //  blk-indices
    int blk_idx_p4  = blockIdx.x / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    tmp_idx         = blockIdx.x % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    int blk_idx_p5  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    int blk_idx_p6  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1);
    int blk_idx_h1  = tmp_idx    / (num_blks_h3 * num_blks_h2);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2);
    int blk_idx_h2  = tmp_idx    / (num_blks_h3);
    int blk_idx_h3  = blockIdx.x % num_blks_h3;

    //  boundary-checks
    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
    // 
    if ((size_h3 - (blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H3) 
        rng_h3 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    else
        rng_h3 = size_h3 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;

    //
    if ((size_h2 - (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) 
        rng_h2 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;
    else
        rng_h2 = size_h2 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;

    //
    if ((size_h1 - (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) 
        rng_h1 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;
    else
        rng_h1 = size_h1 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;

    //
    if ((size_p6 - (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) 
        rng_p6 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;
    else
        rng_p6 = size_p6 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;

    //
    if ((size_p5 - (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) 
        rng_p5 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;
    else
        rng_p5 = size_p5 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;

    //
    if ((size_p4 - (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) 
        rng_p4 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;
    else
        rng_p4 = size_p4 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;

    //
    int t3_based_addr =  blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                        (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2 + 
                        (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h1 + 
                        (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                        (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + 
                        (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    //
    // double tmp_av[4];
    // double tmp_bv[4];    
    double reg_tile[4][4];

    //
    for (int i = 0; i < 4; i++) // i -> p4
    for (int j = 0; j < 4; j++) // j -> p5
    reg_tile[i][j] = 0.0; 

    //
    //  >> s1_8:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h2] * v2[h3,h1,p5,p4] (h3,h2,p6), (h1)
    //
    {
        if (idx_h3 < rng_p6 && idx_h2 < rng_h2 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[idx_h3 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] 
        = d_t2_8[blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_h3 + 
                (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2) * size_p6];
                
        if (idx_h3 < rng_h3 && idx_h2 < rng_h1 && idx_p6 < rng_p5 && idx_h1 < rng_p4)
        sm_b[idx_h3 + (idx_h2 + (idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3] 
        = d_v2_8[blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h2 + 
                (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + idx_p6 + 
                (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4 + idx_h1) * size_p5) * size_h1) * size_h3];
        __syncthreads();

        //  "p4" x "p5"
        reg_tile[0][0] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (0 + (0) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[0][1] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (1 + (0) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[0][2] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (2 + (0) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[0][3] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (3 + (0) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];

        reg_tile[1][0] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (0 + (1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[1][1] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (1 + (1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[1][2] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (2 + (1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[1][3] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (3 + (1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];

        reg_tile[2][0] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (0 + (2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[2][1] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (1 + (2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[2][2] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (2 + (2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[2][3] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (3 + (2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];

        reg_tile[3][0] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (0 + (3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[3][1] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (1 + (3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[3][2] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (2 + (3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
        reg_tile[3][3] -= sm_a[idx_p6 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h3 + (idx_h1 + (3 + (3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3];
    }

    //
    //  to store the output
    //
    if (idx_h1 < rng_h1 && idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6)
    for (int i = 0; i < 4; i++) // p4
    {
        for (int j = 0; j < 4; j++) // p5
        {
            if (i < rng_p4 && j < rng_p5)
            d_t3[t3_based_addr + (i * stride_reg_x) + (j * stride_reg_y)] += reg_tile[i][j];
        }
    }
}

// caller: s1_8
void jk_ccsd_t_s1_8(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, double* host_t3, double* host_t2, double* host_v2)
{
    double *dev_t3;
    double *dev_t2; 
    double *dev_v2; 

    // t3
    // size_t size_t3   = sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;

    // s1_8:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h2] * v2[h3,h1,p5,p4]
    size_t size_t2 = sizeof(double) * size_p6 * size_h2;
    size_t size_v2 = sizeof(double) * size_h3 * size_h1 * size_p5 * size_p4;
    
    // cudaMalloc((void**) &dev_t3, size_t3); 
    // cudaMalloc((void**) &dev_t2, size_t2); 
    // cudaMalloc((void**) &dev_v2, size_v2); 
    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, size_t3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, size_t2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, size_v2, cudaMemcpyHostToDevice);

    // 
    size_t num_blks_h3 = CEIL(size_h3, JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    size_t num_blks_h2 = CEIL(size_h2, JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    size_t num_blks_h1 = CEIL(size_h1, JK_CCSD_T_FUSED_S1_SIZE_TILE_H1);
    size_t num_blks_p6 = CEIL(size_p6, JK_CCSD_T_FUSED_S1_SIZE_TILE_P6);
    size_t num_blks_p5 = CEIL(size_p5, JK_CCSD_T_FUSED_S1_SIZE_TILE_P5);
    size_t num_blks_p4 = CEIL(size_p4, JK_CCSD_T_FUSED_S1_SIZE_TILE_P4);

    //
    size_t num_blks_kernel = num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5 * num_blks_p4; 

    //
    dim3 gridsize(num_blks_kernel);
    dim3 blocksize(JK_CCSD_T_FUSED_S1_SIZE_TB_X, JK_CCSD_T_FUSED_S1_SIZE_TB_Y);

    // p4 (x) and p5 (y)
    int stride_reg_x = size_h3 * size_h2 * size_h1 * size_p6 * size_p5;
    int stride_reg_y = size_h3 * size_h2 * size_h1 * size_p6;

    // 
    dev_t3 = t3_s_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    //
    jk_ccsd_t_s1_8<<<gridsize, blocksize>>>(dev_t3, dev_t2, dev_v2, 
                                            size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, 
                                            num_blks_h3, num_blks_h2, num_blks_h1, num_blks_p6, num_blks_p5, num_blks_p4, 
                                            stride_reg_x, stride_reg_y);

    // cudaMemcpy(host_t3, dev_t3, size_t3, cudaMemcpyDeviceToHost);
#endif
    // cudaFree(dev_t3);
    // cudaFree(dev_t2);
    // cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// kernel: s1_9
__global__ void jk_ccsd_t_s1_9(double* d_t3, 
double* d_t2_9, double* d_v2_9,
size_t size_h3,     size_t size_h2,     size_t size_h1,     size_t size_p6,     size_t size_p5,     size_t size_p4, 
size_t num_blks_h3, size_t num_blks_h2, size_t num_blks_h1, size_t num_blks_p6, size_t num_blks_p5, size_t num_blks_p4, 
size_t stride_reg_x, size_t stride_reg_y)
{
    //  Shared Memory
    __shared__ double sm_a[16];     // "T_p4" * T_h1
    __shared__ double sm_b[256];    // T_h3 * T_h2 * T_p6 * "T_p5"

    //  offset-indices
    int idx_p6  = threadIdx.x / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int tmp_idx = threadIdx.x % (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    int idx_h2  = tmp_idx     / (JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    int idx_h3  = threadIdx.x % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    int idx_h1  = threadIdx.y;

    //  blk-indices
    int blk_idx_p4  = blockIdx.x / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    tmp_idx         = blockIdx.x % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5);
    int blk_idx_p5  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6);
    int blk_idx_p6  = tmp_idx    / (num_blks_h3 * num_blks_h2 * num_blks_h1);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2 * num_blks_h1);
    int blk_idx_h1  = tmp_idx    / (num_blks_h3 * num_blks_h2);
    tmp_idx         = tmp_idx    % (num_blks_h3 * num_blks_h2);
    int blk_idx_h2  = tmp_idx    / (num_blks_h3);
    int blk_idx_h3  = blockIdx.x % num_blks_h3;

    //  boundary-checks
    int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
    // 
    if ((size_h3 - (blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H3) 
        rng_h3 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;
    else
        rng_h3 = size_h3 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H3;

    //
    if ((size_h2 - (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H2) 
        rng_h2 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;
    else
        rng_h2 = size_h2 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H2;

    //
    if ((size_h1 - (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) 
        rng_h1 = JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;
    else
        rng_h1 = size_h1 % JK_CCSD_T_FUSED_S1_SIZE_TILE_H1;

    //
    if ((size_p6 - (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P6) 
        rng_p6 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;
    else
        rng_p6 = size_p6 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P6;

    //
    if ((size_p5 - (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) 
        rng_p5 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;
    else
        rng_p5 = size_p5 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P5;

    //
    if ((size_p4 - (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4)) >= JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) 
        rng_p4 = JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;
    else
        rng_p4 = size_p4 % JK_CCSD_T_FUSED_S1_SIZE_TILE_P4;

    //
    int t3_based_addr =  blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h3 + 
                        (blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h2 + 
                        (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h1 + 
                        (blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_p6 + 
                        (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + 
                        (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

    //
    // double tmp_av[4];
    // double tmp_bv[4];    
    double reg_tile[4][4];

    //
    for (int i = 0; i < 4; i++) // i -> p4
    for (int j = 0; j < 4; j++) // j -> p5
    reg_tile[i][j] = 0.0; 

    //
    //  >> s1_9:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h3] * v2[h2,h1,p5,p4] (h3,h2,p6), (h1)
    //
    {
        if (idx_h3 < rng_p6 && idx_h2 < rng_h3 && idx_p6 == 0 && idx_h1 == 0)
        sm_a[idx_h3 + (idx_h2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] 
        = d_t2_9[blk_idx_p6 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6 + idx_h3 + 
                (blk_idx_h3 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H3 + idx_h2) * size_p6];

        if (idx_h3 < rng_h2 && idx_h2 < rng_h1 && idx_p6 < rng_p5 && idx_h1 < rng_p4)
        sm_b[idx_h3 + (idx_h2 + (idx_p6 + (idx_h1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2] 
        = d_v2_9[blk_idx_h2 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2 + idx_h3 + 
                (blk_idx_h1 * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1 + idx_h2 + 
                (blk_idx_p5 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5 + idx_p6 + 
                (blk_idx_p4 * JK_CCSD_T_FUSED_S1_SIZE_TILE_P4 + idx_h1) * size_p5) * size_h1) * size_h2];
        __syncthreads();

        //  "p4" x "p5"
        reg_tile[0][0] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (0 + (0) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
        reg_tile[0][1] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (1 + (0) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
        reg_tile[0][2] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (2 + (0) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
        reg_tile[0][3] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (3 + (0) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];

        reg_tile[1][0] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (0 + (1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
        reg_tile[1][1] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (1 + (1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
        reg_tile[1][2] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (2 + (1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
        reg_tile[1][3] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (3 + (1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];

        reg_tile[2][0] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (0 + (2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
        reg_tile[2][1] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (1 + (2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
        reg_tile[2][2] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (2 + (2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
        reg_tile[2][3] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (3 + (2) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];

        reg_tile[3][0] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (0 + (3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
        reg_tile[3][1] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (1 + (3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
        reg_tile[3][2] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (2 + (3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
        reg_tile[3][3] += sm_a[idx_p6 + (idx_h3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P6] * sm_b[idx_h2 + (idx_h1 + (3 + (3) * JK_CCSD_T_FUSED_S1_SIZE_TILE_P5) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H1) * JK_CCSD_T_FUSED_S1_SIZE_TILE_H2];
    }

    //
    //  to store the output
    //
    if (idx_h1 < rng_h1 && idx_h3 < rng_h3 && idx_h2 < rng_h2 && idx_p6 < rng_p6)
    for (int i = 0; i < 4; i++) // p4
    {
        for (int j = 0; j < 4; j++) // p5
        {
            if (i < rng_p4 && j < rng_p5)
            d_t3[t3_based_addr + (i * stride_reg_x) + (j * stride_reg_y)] += reg_tile[i][j];
        }
    }
}

// caller: s1_9
void jk_ccsd_t_s1_9(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, double* host_t3, double* host_t2, double* host_v2)
{
    double *dev_t3;
    double *dev_t2; 
    double *dev_v2; 

    // t3
    // size_t size_t3   = sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;

    // s1_9:   t3[h3,h2,h1,p6,p5,p4] -= t2[p6,h3] * v2[h2,h1,p5,p4]
    size_t size_t2 = sizeof(double) * size_p6 * size_h3;
    size_t size_v2 = sizeof(double) * size_h2 * size_h1 * size_p5 * size_p4;

    // cudaMalloc((void**) &dev_t3, size_t3); 
    // cudaMalloc((void**) &dev_t2, size_t2); 
    // cudaMalloc((void**) &dev_v2, size_v2); 
    dev_t2 = (double *) getGpuMem(size_t2);
	dev_v2 = (double *) getGpuMem(size_v2);
    
    // cudaMemcpy(dev_t3, host_t3, size_t3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_t2, host_t2, size_t2, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v2, host_v2, size_v2, cudaMemcpyHostToDevice);

    // 
    size_t num_blks_h3 = CEIL(size_h3, JK_CCSD_T_FUSED_S1_SIZE_TILE_H3);
    size_t num_blks_h2 = CEIL(size_h2, JK_CCSD_T_FUSED_S1_SIZE_TILE_H2);
    size_t num_blks_h1 = CEIL(size_h1, JK_CCSD_T_FUSED_S1_SIZE_TILE_H1);
    size_t num_blks_p6 = CEIL(size_p6, JK_CCSD_T_FUSED_S1_SIZE_TILE_P6);
    size_t num_blks_p5 = CEIL(size_p5, JK_CCSD_T_FUSED_S1_SIZE_TILE_P5);
    size_t num_blks_p4 = CEIL(size_p4, JK_CCSD_T_FUSED_S1_SIZE_TILE_P4);

    //
    size_t num_blks_kernel = num_blks_h3 * num_blks_h2 * num_blks_h1 * num_blks_p6 * num_blks_p5 * num_blks_p4; 

    //
    dim3 gridsize(num_blks_kernel);
    dim3 blocksize(JK_CCSD_T_FUSED_S1_SIZE_TB_X, JK_CCSD_T_FUSED_S1_SIZE_TB_Y);

    // p4 (x) and p5 (y)
    int stride_reg_x = size_h3 * size_h2 * size_h1 * size_p6 * size_p5;
    int stride_reg_y = size_h3 * size_h2 * size_h1 * size_p6;

    // 
    dev_t3 = t3_s_d;
#ifdef DEBUG_ENALBLE_ALL_KERNEL
    //
    jk_ccsd_t_s1_9<<<gridsize, blocksize>>>(dev_t3, dev_t2, dev_v2, 
                                            size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, 
                                            num_blks_h3, num_blks_h2, num_blks_h1, num_blks_p6, num_blks_p5, num_blks_p4, 
                                            stride_reg_x, stride_reg_y);

    // cudaMemcpy(host_t3, dev_t3, size_t3, cudaMemcpyDeviceToHost);
#endif
    // cudaFree(dev_t3);
    // cudaFree(dev_t2);
    // cudaFree(dev_v2);
    freeGpuMem(dev_t2);
	freeGpuMem(dev_v2);
}

// A100 and cuda 11.1 
#if defined(USE_NV_TC)

#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline> // cuda >= 11.1

#define CUCHK(call) {	\
	cudaError_t err = call; \
	if( cudaSuccess != err) {	\
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
				__FILE__, __LINE__, cudaGetErrorString(err) );	\
		fflush(stderr); \
		exit(EXIT_FAILURE);	\
}}

#include "tensor_core_helper.cuh" // 

//
#define SIZE_TILE_P7 16
#define SIZE_TILE_H3 4
#define SIZE_TILE_P4 4
#define SIZE_TILE_H2 4
#define SIZE_TILE_H1 4
#define SIZE_TILE_P6 4
#define SIZE_TILE_P5 4
#define SIZE_UNIT_INT SIZE_TILE_P7

// 
#define PAD 4
#define STAGE_ALIGN 32
#define SINGLE_STAGE_SIZE (64 * (PAD + 16))
#define STAGE_OFFSET ((SINGLE_STAGE_SIZE + STAGE_ALIGN - 1) / STAGE_ALIGN) * STAGE_ALIGN
#define NUM_STAGE 2

#define TEST_ENABLE_RT
#define TEST_OLD_STYLE

//------------------------------------------------------------------------------ device helper fuctions
__device__ inline void zero_shared(double *smem) {
	const int t_id = threadIdx.y * blockDim.x + threadIdx.x;
	#pragma unroll
	for (int i = t_id; i < SINGLE_STAGE_SIZE; i += blockDim.x * blockDim.y) {
		smem[i] = 0;
	}
}

#include "ccsd_t_g2s_device_functions.cu"

//------------------------------------------------------------------------------ kernels and callers
//
__global__ void next_unfused_kernel_d1_1(double* dev_t3_d, const double* __restrict__ dev_d1_t2_1, const double* __restrict__ dev_d1_v2_1, 
                                        int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, 
                                        int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
                                        int stride_reg_x, int stride_reg_y, 
                                        int size_internal) 
{
    // 
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();

    // For Shared Memory,
    const int lda = 16 + PAD;
    extern __shared__ double sm_block[];
    double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
    double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

    #pragma unroll
    for (int i = 0; i < NUM_STAGE; i++) {
        zero_shared(sm_a + STAGE_OFFSET * i);
        zero_shared(sm_b + STAGE_OFFSET * i);
    }
    block.sync();

    // Allocate shared storage for a N-stage cuda::pipeline:
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = thread_id / 32; // 0:7
    WarpRegisterMapping wrm(thread_id);

    const int tile_m = warp_id % 2; // 0:1
    const int tile_n = warp_id / 2; // 0:3

    MmaOperandC op_c;

    int internal_upperbound = 0;
    int internal_offset;

    //  
    //  based on sd2_1
    //  (p6,h2), (h1,h3)
    int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

    int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h2 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h1 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h3 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4 + idx_p6) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
    if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
    if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

    // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

    #pragma unroll 1
    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
        #pragma unroll 1
        for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
        pipeline.producer_acquire();

        const int l_fetch = fetch_batch * SIZE_UNIT_INT;
        const size_t shared_idx = fetch_batch % NUM_STAGE;
        internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
        block.sync();

        if (internal_offset > 0) { 
            internal_upperbound = internal_offset;
            zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
            zero_shared(sm_b + STAGE_OFFSET * shared_idx);
            block.sync();
        }

        if ((idx_h3 < rng_h1) && (idx_h1 < rng_p4) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) { // p4,h1
            g2s_d1_t2_1<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d1_t2_1, 
            blk_idx_h1, 					idx_h3, 
            blk_idx_p5, size_p5, 	
            blk_idx_p4, size_p4,  idx_h1, 
                        size_h7, 	threadIdx.x + l_fetch, 
                        rng_p5, 	pipeline);
        }

        if ((idx_h2 < rng_h2) && (idx_p6 < rng_h3) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) { // h3,h2
            g2s_d1_v2_1<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d1_v2_1, 
            blk_idx_p6, size_p6,	
            blk_idx_h2, size_h2, 	idx_h2, 
            blk_idx_h3, size_h3, 	idx_p6, 
                                    threadIdx.y + l_fetch, 
                        rng_p6, 	pipeline);
        }
        pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync();
        const size_t shared_idx = compute_batch % NUM_STAGE;

        #pragma unroll
        for (int ll = 0; ll < 4; ll++) {
        MmaOperandA op_a;
        op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
        MmaOperandB op_b;
        op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
        mma(op_c, op_a, op_b);
        }
        pipeline.consumer_release();
    }
    block.sync(); 

    //     (p6,h2),     (h1,h3)
    // TB_X(p4,h3), TB_Y(h2,h1), REG_X,Y(p5,p6)
    dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p4 && idx_h2 < rng_h3 && idx_h1 < rng_h2 && idx_h3 < rng_h1) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p6 && idx_reg_x < rng_p5) { 
                    dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	}
}

void driver_ccsd_t_d1_1(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2) 
{
	// 
	int numTbs = CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * 
              CEIL(size_h1, SIZE_TILE_H1) * CEIL(size_p6, SIZE_TILE_P6) * 
              CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd1_1: t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h1] * v2[h3,h2,p6,h7]
	double* dev_t3; double* dev_t2; double* dev_v2;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_h7 * size_p4 * size_p5 * size_h1));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_h3 * size_h2 * size_p6 * size_h7));

  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h7 * size_p4 * size_p5 * size_h1, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h3 * size_h2 * size_p6 * size_h7, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_h7);

	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p5;
  int stride_reg_y = stride_output_p6;
      
  // 
  dev_t3 = t3_d;

	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);

  // 
  // int maxbytes = 98304; // 96 KB
  // CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d1_1, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d1_1<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7, 
  CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
  CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
  stride_reg_x, stride_reg_y,
      size_h7);
  // cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());

  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);


  // cudaFree()
	CUCHK(cudaFree(dev_t2)); CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d1_2(double* dev_t3_d, const double* __restrict__ dev_d1_t2_2, const double* __restrict__ dev_d1_v2_2, 
	// 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
  // 
  auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h2 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h3 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h1 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4 + idx_p6) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

  #pragma unroll 1
  for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
    #pragma unroll 1
    for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
      pipeline.producer_acquire();

      const int l_fetch = fetch_batch * SIZE_UNIT_INT;
      const size_t shared_idx = fetch_batch % NUM_STAGE;
      internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
      block.sync();

      if (internal_offset > 0) { 
        internal_upperbound = internal_offset;
        zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
        zero_shared(sm_b + STAGE_OFFSET * shared_idx);
        block.sync();
      }

      if ((idx_h3 < rng_h2) && (idx_h1 < rng_p4) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
        g2s_d1_t2_2<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d1_t2_2, 
          blk_idx_h2, 					idx_h3, 
          blk_idx_p5, size_p5, 	
          blk_idx_p4, size_p4,  idx_h1, 
                      size_h7, 	threadIdx.x + l_fetch, 
                      rng_p5, 	pipeline);
      }

      if ((idx_h2 < rng_h1) && (idx_p6 < rng_h3) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
        g2s_d1_v2_2<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d1_v2_2, 
          blk_idx_p6, size_p6,	
          blk_idx_h1, size_h1, 	idx_h2, 
          blk_idx_h3, size_h3, 	idx_p6, 
                                threadIdx.y + l_fetch, 
                      rng_p6, 	pipeline);
      }
      pipeline.producer_commit();
    }
    pipeline.consumer_wait();
    block.sync();
    const size_t shared_idx = compute_batch % NUM_STAGE;

    #pragma unroll
    for (int ll = 0; ll < 4; ll++) {
      MmaOperandA op_a;
      // op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
      op_a.template load_plus<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
      MmaOperandB op_b;
      // op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
      op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
      // mma_t(op_c, op_a, op_b);
      mma(op_c, op_a, op_b);
    }
    pipeline.consumer_release();
  }
  block.sync(); 

  //     (p6,h2),     (h1,h3)
  // TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6)
  dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p4 && idx_h2 < rng_h3 && idx_h1 < rng_h1 && idx_h3 < rng_h2) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p6 && idx_reg_x < rng_p5) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d1_2(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd1_2: t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p5,h2] * v2[h3,h1,p6,h7]
	double* dev_t3; double* dev_t2; double* dev_v2;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_h7 * size_p4 * size_p5 * size_h2));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_h3 * size_h1 * size_p6 * size_h7));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h7 * size_p4 * size_p5 * size_h2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h3 * size_h1 * size_p6 * size_h7, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_h7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p5;
  int stride_reg_y = stride_output_p6;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);

  dev_t3 = t3_d;
  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d1_2, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d1_2<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_h7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

// 
__global__ void next_unfused_kernel_d1_3(double* dev_t3_d, const double* __restrict__ dev_d1_t2_3, const double* __restrict__ dev_d1_v2_3, 
	
  int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
  // 
  auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h3 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h2 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h1 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4 + idx_p6) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

  #pragma unroll 1
  for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
    #pragma unroll 1
    for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
      pipeline.producer_acquire();

      const int l_fetch = fetch_batch * SIZE_UNIT_INT;
      const size_t shared_idx = fetch_batch % NUM_STAGE;
      internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
      block.sync();

      if (internal_offset > 0) { 
        internal_upperbound = internal_offset;
        zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
        zero_shared(sm_b + STAGE_OFFSET * shared_idx);
        block.sync();
      }

      if ((idx_h3 < rng_h3) && (idx_h1 < rng_p4) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
        g2s_d1_t2_3<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d1_t2_3, 
          blk_idx_h3, 					idx_h3, 
          blk_idx_p5, size_p5, 	
          blk_idx_p4, size_p4,  idx_h1, 
                      size_h7, 	threadIdx.x + l_fetch, 
                      rng_p5, 	pipeline);
      }

      if ((idx_h2 < rng_h1) && (idx_p6 < rng_h2) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
        g2s_d1_v2_3<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d1_v2_3, 
          blk_idx_p6, size_p6,	
          blk_idx_h1, size_h1, 	idx_h2, 
          blk_idx_h2, size_h2, 	idx_p6, 
                                threadIdx.y + l_fetch, 
                      rng_p6, 	pipeline);
      }
      pipeline.producer_commit();
    }
    pipeline.consumer_wait();
    block.sync();
    const size_t shared_idx = compute_batch % NUM_STAGE;

    #pragma unroll
    for (int ll = 0; ll < 4; ll++) {
      MmaOperandA op_a;
      // op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
      op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
      MmaOperandB op_b;
      // op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
      op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
      // mma_t(op_c, op_a, op_b);
      mma(op_c, op_a, op_b);
    }
    pipeline.consumer_release();
  }
  block.sync(); 

  //     (p6,h2),     (h1,h3)
	// TB_X(p4,h2), TB_X(h1,h3), REG_X,Y(p5,p6)
  dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p4 && idx_h2 < rng_h2 && idx_h1 < rng_h1 && idx_h3 < rng_h3) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p6 && idx_reg_x < rng_p5) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d1_3(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd1_3: t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p5,h3] * v2[h2,h1,p6,h7]
	double* dev_t3; double* dev_t2; double* dev_v2;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_h7 * size_p4 * size_p5 * size_h3));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_h2 * size_h1 * size_p6 * size_h7));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h7 * size_p4 * size_p5 * size_h3, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h2 * size_h1 * size_p6 * size_h7, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_h7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p5;
  int stride_reg_y = stride_output_p6;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);
  dev_t3 = t3_d;

  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d1_3, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d1_3<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_h7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

// 
__global__ void next_unfused_kernel_d1_4(double* dev_t3_d, const double* __restrict__ dev_d1_t2_4, const double* __restrict__ dev_d1_v2_4, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
  // 
  auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p5,p4)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h2 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h1 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h3 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h1) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d1_t2_4<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d1_t2_4, 
					blk_idx_h1, 					idx_h3, 
					blk_idx_p6, size_p6,  idx_h1,  	
					blk_idx_p5, size_p5,
											size_h7, 	threadIdx.x + l_fetch, 
											rng_p5, 	pipeline);
			}

			if ((idx_h2 < rng_h2) && (idx_p6 < rng_h3) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d1_v2_4<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d1_v2_4, 
					blk_idx_p4, size_p4,	
					blk_idx_h2, size_h2, 	idx_h2, 
					blk_idx_h3, size_h3, 	idx_p6, 
																threadIdx.y + l_fetch, 
											rng_p4, 	pipeline);
			}
			pipeline.producer_commit();
		}
		pipeline.consumer_wait();
		block.sync();
		const size_t shared_idx = compute_batch % NUM_STAGE;

		#pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			// op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			// op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			// mma_t(op_c, op_a, op_b);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync(); 

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p5,p4)
  dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p6 && idx_h2 < rng_h3 && idx_h1 < rng_h2 && idx_h3 < rng_h1) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p4 && idx_reg_x < rng_p5) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d1_4(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd1_4: t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h1] * v2[h3,h2,p4,h7]
	double* dev_t3; double* dev_t2; double* dev_v2;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_h7 * size_p5 * size_p6 * size_h1));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_h3 * size_h2 * size_p4 * size_h7));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h7 * size_p5 * size_p6 * size_h1, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h3 * size_h2 * size_p4 * size_h7, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_h7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p5;
  int stride_reg_y = stride_output_p4;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);

  dev_t3 = t3_d;
  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d1_4, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d1_4<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_h7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

// 
__global__ void next_unfused_kernel_d1_5(double* dev_t3_d, const double* __restrict__ dev_d1_t2_5, const double* __restrict__ dev_d1_v2_5, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
  // 
  auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h1,h2),  REG_X,Y(p5,p4)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h2 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h3 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h1 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h2) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d1_t2_5<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d1_t2_5, 
					blk_idx_h2, 					idx_h3, 
					blk_idx_p6, size_p6,  idx_h1,  	
					blk_idx_p5, size_p5,
											size_h7, 	threadIdx.x + l_fetch, 
											rng_p5, 	pipeline);
			}

			if ((idx_h2 < rng_h1) && (idx_p6 < rng_h3) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d1_v2_5<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d1_v2_5, 
					blk_idx_p4, size_p4,	
					blk_idx_h1, size_h1, 	idx_h2, 
					blk_idx_h3, size_h3, 	idx_p6, 
																threadIdx.y + l_fetch, 
											rng_p4, 	pipeline);
			}
			pipeline.producer_commit();
		}
		pipeline.consumer_wait();
		block.sync();
		const size_t shared_idx = compute_batch % NUM_STAGE;

		#pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			// op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			op_a.template load_plus<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			// op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			// mma_t(op_c, op_a, op_b);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync(); 

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h1,h2),  REG_X,Y(p5,p4)
  dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p6 && idx_h2 < rng_h3 && idx_h1 < rng_h1 && idx_h3 < rng_h2) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p4 && idx_reg_x < rng_p5) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d1_5(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd1_5: t3[h3,h2,h1,p6,p5,p4] += t2[h7,p5,p6,h2] * v2[h3,h1,p4,h7]
	double* dev_t3; double* dev_t2; double* dev_v2;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_h7 * size_p5 * size_p6 * size_h2));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_h3 * size_h1 * size_p4 * size_h7));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h7 * size_p5 * size_p6 * size_h2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h3 * size_h1 * size_p4 * size_h7, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_h7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p5;
  int stride_reg_y = stride_output_p4;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);

  dev_t3 = t3_d;
  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d1_5, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d1_5<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_h7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d1_6(double* dev_t3_d, const double* __restrict__ dev_d1_t2_6, const double* __restrict__ dev_d1_v2_6, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
  // 
  auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p5,p4)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h3 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h2 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h1 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + idx_p6 +
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h3) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d1_t2_6<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d1_t2_6, 
					blk_idx_h3, 					idx_h3, 
					blk_idx_p6, size_p6,  idx_h1,  	
					blk_idx_p5, size_p5,
											size_h7, 	threadIdx.x + l_fetch, 
											rng_p5, 	pipeline);
			}

			if ((idx_h2 < rng_h1) && (idx_p6 < rng_h2) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d1_v2_6<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d1_v2_6, 
					blk_idx_p4, size_p4,	
					blk_idx_h1, size_h1, 	idx_h2, 
					blk_idx_h2, size_h2, 	idx_p6, 
																threadIdx.y + l_fetch, 
											rng_p4, 	pipeline);
			}
			pipeline.producer_commit();
		}
		pipeline.consumer_wait();
		block.sync();
		const size_t shared_idx = compute_batch % NUM_STAGE;

		#pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync(); 

	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p5,p4)
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p6 && idx_h2 < rng_h2 && idx_h1 < rng_h1 && idx_h3 < rng_h3) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p4 && idx_reg_x < rng_p5) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d1_6(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd1_6:  	t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p5,p6,h3] * v2[h2,h1,p4,h7]
	double* dev_t3; double* dev_t2; double* dev_v2;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_h7 * size_p5 * size_p6 * size_h3));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_h2 * size_h1 * size_p4 * size_h7));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h7 * size_p5 * size_p6 * size_h3, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h2 * size_h1 * size_p4 * size_h7, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_h7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p5;
  int stride_reg_y = stride_output_p4;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);

  dev_t3 = t3_d;
  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d1_6, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d1_6<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_h7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d1_7(double* dev_t3_d, const double* __restrict__ dev_d1_t2_7, const double* __restrict__ dev_d1_v2_7, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
  // 
  auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p4,p5)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h2 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h1 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h3 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + idx_p6 +  
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
		for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
			#pragma unroll 1
			for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
				pipeline.producer_acquire();

				const int l_fetch = fetch_batch * SIZE_UNIT_INT;
				const size_t shared_idx = fetch_batch % NUM_STAGE;
				internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
				block.sync();

				if (internal_offset > 0) { 
					internal_upperbound = internal_offset;
					zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
					zero_shared(sm_b + STAGE_OFFSET * shared_idx);
					block.sync();
				}

				if ((idx_h3 < rng_h1) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
					g2s_d1_t2_7<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d1_t2_7, 
						blk_idx_h1, 					idx_h3, 
						blk_idx_p6, size_p6,  idx_h1,  	
						blk_idx_p4, size_p4,
												size_h7, 	threadIdx.x + l_fetch, 
												rng_p4, 	pipeline);
				}

				if ((idx_h2 < rng_h2) && (idx_p6 < rng_h3) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
					g2s_d1_v2_7<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d1_v2_7, 
						blk_idx_p5, size_p5,	
						blk_idx_h2, size_h2, 	idx_h2, 
						blk_idx_h3, size_h3, 	idx_p6, 
																	threadIdx.y + l_fetch, 
												rng_p5, 	pipeline);
				}
				pipeline.producer_commit();
			}
			pipeline.consumer_wait();
			block.sync();
			const size_t shared_idx = compute_batch % NUM_STAGE;

			#pragma unroll
			for (int ll = 0; ll < 4; ll++) {
				MmaOperandA op_a;
				// op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
				op_a.template load_plus<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
				MmaOperandB op_b;
				// op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
				op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
				// mma_t(op_c, op_a, op_b);
				mma(op_c, op_a, op_b);
			}
			pipeline.consumer_release();
		}
		block.sync(); 

	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p4,p5)
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p6 && idx_h2 < rng_h3 && idx_h1 < rng_h2 && idx_h3 < rng_h1) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p5 && idx_reg_x < rng_p4) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d1_7(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd1_7: t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h1] * v2[h3,h2,p5,h7]
	double* dev_t3; double* dev_t2; double* dev_v2;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_h7 * size_p4 * size_p6 * size_h1));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_h3 * size_h2 * size_p5 * size_h7));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h7 * size_p4 * size_p6 * size_h1, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h3 * size_h2 * size_p5 * size_h7, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_h7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p4;
  int stride_reg_y = stride_output_p5;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);

  dev_t3 = t3_d;
  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d1_7, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d1_7<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_h7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d1_8(double* dev_t3_d, const double* __restrict__ dev_d1_t2_8, const double* __restrict__ dev_d1_v2_8, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
  // 
  auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p4,p5)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h2 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h3 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h1 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h2) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d1_t2_8<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d1_t2_8, 
					blk_idx_h2, 					idx_h3, 
					blk_idx_p6, size_p6,  idx_h1,  	
					blk_idx_p4, size_p4,
											size_h7, 	threadIdx.x + l_fetch, 
											rng_p4, 	pipeline);
			}

			if ((idx_h2 < rng_h1) && (idx_p6 < rng_h3) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d1_v2_8<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d1_v2_8, 
					blk_idx_p5, size_p5,	
					blk_idx_h1, size_h1, 	idx_h2, 
					blk_idx_h3, size_h3, 	idx_p6, 
																threadIdx.y + l_fetch, 
											rng_p5, 	pipeline);
			}
			pipeline.producer_commit();
		}
		pipeline.consumer_wait();
		block.sync();
		const size_t shared_idx = compute_batch % NUM_STAGE;

		#pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			// op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			op_a.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			// op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			// mma_t(op_c, op_a, op_b);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync(); 

	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p4,p5)
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p6 && idx_h2 < rng_h3 && idx_h1 < rng_h1 && idx_h3 < rng_h2) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p5 && idx_reg_x < rng_p4) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d1_8(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd1_8: t3[h3,h2,h1,p6,p5,p4] -= t2[h7,p4,p6,h2] * v2[h3,h1,p5,h7]
	double* dev_t3; double* dev_t2; double* dev_v2;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_h7 * size_p4 * size_p6 * size_h2));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_h3 * size_h1 * size_p5 * size_h7));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h7 * size_p4 * size_p6 * size_h2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h3 * size_h1 * size_p5 * size_h7, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_h7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p4;
  int stride_reg_y = stride_output_p5;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);

  dev_t3 = t3_d;
  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d1_8, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d1_8<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_h7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d1_9(double* dev_t3_d, const double* __restrict__ dev_d1_t2_9, const double* __restrict__ dev_d1_v2_9, 
	// 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
  // 
  auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p4,p5)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h3 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h2 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h1 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
		for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
			#pragma unroll 1
			for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
				pipeline.producer_acquire();

				const int l_fetch = fetch_batch * SIZE_UNIT_INT;
				const size_t shared_idx = fetch_batch % NUM_STAGE;
				internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
				block.sync();

				if (internal_offset > 0) { 
					internal_upperbound = internal_offset;
					zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
					zero_shared(sm_b + STAGE_OFFSET * shared_idx);
					block.sync();
				}

				if ((idx_h3 < rng_h3) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
					g2s_d1_t2_9<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d1_t2_9, 
						blk_idx_h3, 					idx_h3, 
						blk_idx_p6, size_p6,  idx_h1,  	
						blk_idx_p4, size_p4,
												size_h7, 	threadIdx.x + l_fetch, 
												rng_p4, 	pipeline);
				}

				if ((idx_h2 < rng_h1) && (idx_p6 < rng_h2) && threadIdx.y < SIZE_UNIT_INT - internal_upperbound) {
					g2s_d1_v2_9<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d1_v2_9, 
						blk_idx_p5, size_p5,	
						blk_idx_h1, size_h1, 	idx_h2, 
						blk_idx_h2, size_h2, 	idx_p6, 
																	threadIdx.y + l_fetch, 
												rng_p5, 	pipeline);
				}
				pipeline.producer_commit();
			}
			pipeline.consumer_wait();
			block.sync();
			const size_t shared_idx = compute_batch % NUM_STAGE;

			#pragma unroll
			for (int ll = 0; ll < 4; ll++) {
				MmaOperandA op_a;
				// op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
				op_a.template load_plus<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
				MmaOperandB op_b;
				// op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
				op_b.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
				// mma_t(op_c, op_a, op_b);
				mma(op_c, op_a, op_b);
			}
			pipeline.consumer_release();
		}
		block.sync(); 

	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p4,p5)
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p6 && idx_h2 < rng_h2 && idx_h1 < rng_h1 && idx_h3 < rng_h3) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p5 && idx_reg_x < rng_p4) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d1_9(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd1_9: t3[h3,h2,h1,p6,p5,p4] += t2[h7,p4,p6,h3] * v2[h2,h1,p5,h7]
	double* dev_t3; double* dev_t2; double* dev_v2;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_h7 * size_p4 * size_p6 * size_h3));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_h2 * size_h1 * size_p5 * size_h7));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_h7 * size_p4 * size_p6 * size_h3, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_h2 * size_h1 * size_p5 * size_h7, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_h7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p4;
  int stride_reg_y = stride_output_p5;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);

  dev_t3 = t3_d;
  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d1_9, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d1_9<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_h7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_h7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * (size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4), cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

///
///
///
__global__ void next_unfused_kernel_d2_1(double* dev_t3_d, const double* __restrict__ dev_d2_t2_1, const double* __restrict__ dev_d2_v2_1, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
	// 
  auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p5,p4)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h3 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h2 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h1 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h1) && (idx_h1 < rng_h2) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_t2_1<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d2_t2_1, 
					blk_idx_h2, 					idx_h1, 
					blk_idx_h1, size_h1, 	idx_h3, 
					blk_idx_p4, size_p4, 
											size_p7, 	threadIdx.x + l_fetch, rng_p4, pipeline);
			}

			if ((idx_h3 < rng_h3) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_v2_1<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d2_v2_1, 
					blk_idx_p5, 
					blk_idx_p6, size_p6, idx_h1, 
					blk_idx_h3, size_h3, idx_h3, 
											size_p7, threadIdx.x + l_fetch, rng_p5, pipeline);
			}
			pipeline.producer_commit();
		}

		pipeline.consumer_wait();
		block.sync();
		const size_t shared_idx = compute_batch % NUM_STAGE;

		#pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync(); 
	
	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p5,p4)
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p6 && idx_h2 < rng_h2 && idx_h1 < rng_h1 && idx_h3 < rng_h3) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p4 && idx_reg_x < rng_p5) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d2_1(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd2_1: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p4,h1,h2] * v2[p7,h3,p6,p5]
	double* dev_t3; double* dev_t2; double* dev_v2;

	size_t size_t3 = size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;
	size_t size_t2 = size_p7 * size_p4 * size_h1 * size_h2;
	size_t size_v2 = size_p7 * size_h3 * size_p6 * size_p5;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_t3));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_t2));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_v2));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_t3, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_t2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_v2, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_p7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p5;
  int stride_reg_y = stride_output_p4;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);
  dev_t3 = t3_d;

  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d2_1, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d2_1<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_p7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_p7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_t3, cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d2_2(double* dev_t3_d, const double* __restrict__ dev_d2_t2_2, const double* __restrict__ dev_d2_v2_2, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
	// 
	auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p5,p4)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h2 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h1 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h3 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h2) && (idx_h1 < rng_h3) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_t2_2<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d2_t2_2, 
					blk_idx_h3, 					idx_h1, 
					blk_idx_h2, size_h2, 	idx_h3, 
					blk_idx_p4, size_p4,  
											size_p7, threadIdx.x + l_fetch, rng_p4, pipeline);
			}

			if ((idx_h3 < rng_h1) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_v2_2<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d2_v2_2, 
					blk_idx_p5, 
					blk_idx_p6, size_p6, idx_h1, 
					blk_idx_h1, size_h1, idx_h3, 
											size_p7, threadIdx.x + l_fetch, rng_p5, pipeline);
			}
			pipeline.producer_commit();
		}

		pipeline.consumer_wait();
		block.sync();
		const size_t shared_idx = compute_batch % NUM_STAGE;
		
		#pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync(); 
	
	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p5,p4)
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p6 && idx_h2 < rng_h3 && idx_h1 < rng_h2 && idx_h3 < rng_h1) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p4 && idx_reg_x < rng_p5) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d2_2(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd2_2: t3[h3,h2,h1,p6,p5,p4] -= t2[p7,p4,h2,h3] * v2[p7,h1,p6,p5]
	double* dev_t3; double* dev_t2; double* dev_v2;

	size_t size_t3 = size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;
	size_t size_t2 = size_p7 * size_p4 * size_h2 * size_h3;
	size_t size_v2 = size_p7 * size_h1 * size_p6 * size_p5;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_t3));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_t2));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_v2));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_t3, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_t2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_v2, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_p7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p5;
  int stride_reg_y = stride_output_p4;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);
  dev_t3 = t3_d;

  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d2_2, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d2_2<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_p7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_p7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_t3, cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d2_3(double* dev_t3_d, const double* __restrict__ dev_d2_t2_3, const double* __restrict__ dev_d2_v2_3, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
	// 
	auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h2 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h3 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h1 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx);
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h1) && (idx_h1 < rng_h3) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_t2_3<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d2_t2_3,  
					blk_idx_h3, 					idx_h1, 
					blk_idx_h1, size_h1, 	idx_h3, 
					blk_idx_p4, size_p4,  
											size_p7, 	threadIdx.x + l_fetch, rng_p4, pipeline);
			}

			if ((idx_h3 < rng_h2) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_v2_3<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d2_v2_3, 
					blk_idx_p5, 				 	
					blk_idx_p6, size_p6, 	idx_h1, 
					blk_idx_h2, size_h2, 	idx_h3, 
											size_p7, 	threadIdx.x + l_fetch, rng_p5, pipeline);
			}
			pipeline.producer_commit();
		}

		pipeline.consumer_wait();
		block.sync();
		const size_t shared_idx = compute_batch % NUM_STAGE;
		
		#pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			// mma_t(op_c, op_a, op_b);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync(); 
	
	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p5,p4)
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p6 && idx_h2 < rng_h3 && idx_h1 < rng_h1 && idx_h3 < rng_h2) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p4 && idx_reg_x < rng_p5) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d2_3(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd2_3: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p4,h1,h3] * v2[p7,h2,p6,p5]
	double* dev_t3; double* dev_t2; double* dev_v2;

	size_t size_t3 = size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;
	size_t size_t2 = size_p7 * size_p4 * size_h1 * size_h3;
	size_t size_v2 = size_p7 * size_h2 * size_p6 * size_p5;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_t3));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_t2));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_v2));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_t3, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_t2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_v2, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_p7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p5;
  int stride_reg_y = stride_output_p4;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);
  dev_t3 = t3_d;

  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d2_3, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d2_3<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_p7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_p7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_t3, cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d2_4(double* dev_t3_d, const double* __restrict__ dev_d2_t2_4, const double* __restrict__ dev_d2_v2_4, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
	// 
	auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p4,p5)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h3 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h2 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h1 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx);
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h1) && (idx_h1 < rng_h2) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_t2_4<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d2_t2_4,  
					blk_idx_h2, 					idx_h1, 
					blk_idx_h1, size_h1, 	idx_h3, 
					blk_idx_p5, size_p5,  // reg_y: p5
											size_p7, 	threadIdx.x + l_fetch, rng_p5, pipeline);
			}

			if ((idx_h3 < rng_h3) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_v2_4<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d2_v2_4, 
					blk_idx_p4, 				 	// reg_x: p4
					blk_idx_p6, size_p6, 	idx_h1, 
					blk_idx_h3, size_h3, 	idx_h3, 
											size_p7, 	threadIdx.x + l_fetch, rng_p4, pipeline);
			}
			pipeline.producer_commit();
		}

		pipeline.consumer_wait();
		block.sync();
		const size_t shared_idx = compute_batch % NUM_STAGE;
		
		#pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			// mma_t(op_c, op_a, op_b);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync(); 
	
	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h2), TB_Y(h1,h3), REG_X,Y(p4,p5)
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p6 && idx_h2 < rng_h2 && idx_h1 < rng_h1 && idx_h3 < rng_h3) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p5 && idx_reg_x < rng_p4) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d2_4(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd2_4: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h1,h2] * v2[p7,h3,p6,p4]
	double* dev_t3; double* dev_t2; double* dev_v2;

	size_t size_t3 = size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;
	size_t size_t2 = size_p7 * size_p5 * size_h1 * size_h2;
	size_t size_v2 = size_p7 * size_h3 * size_p6 * size_p4;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_t3));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_t2));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_v2));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_t3, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_t2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_v2, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_p7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p4;
  int stride_reg_y = stride_output_p5;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);
  dev_t3 = t3_d;

  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d2_4, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d2_4<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_p7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_p7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_t3, cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d2_5(double* dev_t3_d, const double* __restrict__ dev_d2_t2_5, const double* __restrict__ dev_d2_v2_5, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
	// 
	auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p4,p5)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h2 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h1 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h3 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h2) && (idx_h1 < rng_h3) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_t2_5<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d2_t2_5, 
					blk_idx_h3, 					idx_h1, 
					blk_idx_h2, size_h2, 	idx_h3, 
					blk_idx_p5, size_p5,  
											size_p7, 	threadIdx.x + l_fetch, rng_p5, pipeline);
			}

			if ((idx_h3 < rng_h1) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_v2_5<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d2_v2_5, 
					blk_idx_p4, 				 	
					blk_idx_p6, size_p6, 	idx_h1, 
					blk_idx_h1, size_h1, 	idx_h3, 
											size_p7, 	threadIdx.x + l_fetch, rng_p4, pipeline);
			}
			pipeline.producer_commit();
		}

		pipeline.consumer_wait();
		block.sync();
		const size_t shared_idx = compute_batch % NUM_STAGE;
		
		#pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			// mma_t(op_c, op_a, op_b);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync(); 
	
	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h2,h1), REG_X,Y(p4,p5)
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p6 && idx_h2 < rng_h3 && idx_h1 < rng_h2 && idx_h3 < rng_h1) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p5 && idx_reg_x < rng_p4) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d2_5(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd2_5: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p5,h2,h3] * v2[p7,h1,p6,p4]
	double* dev_t3; double* dev_t2; double* dev_v2;

	size_t size_t3 = size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;
	size_t size_t2 = size_p7 * size_p5 * size_h2 * size_h3;
	size_t size_v2 = size_p7 * size_h1 * size_p6 * size_p4;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_t3));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_t2));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_v2));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_t3, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_t2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_v2, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_p7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p4;
  int stride_reg_y = stride_output_p5;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);
  dev_t3 = t3_d;

  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d2_5, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d2_5<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_p7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_p7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_t3, cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d2_6(double* dev_t3_d, const double* __restrict__ dev_d2_t2_6, const double* __restrict__ dev_d2_v2_6, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
	// 
	auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p4,p5) // sd2_6
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h2 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h3 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h1 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + idx_p6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h1) && (idx_h1 < rng_h3) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_t2_6<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d2_t2_6, 
					blk_idx_h3, 					idx_h1, 
					blk_idx_h1, size_h1, 	idx_h3, 
					blk_idx_p5, size_p5,  
											size_p7, 	threadIdx.x + l_fetch, rng_p5, pipeline);
			}

			if ((idx_h3 < rng_h2) && (idx_h1 < rng_p6) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_v2_6<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d2_v2_6, 
					blk_idx_p4, 				 	
					blk_idx_p6, size_p6, 	idx_h1,
					blk_idx_h2, size_h2, 	idx_h3, 
											size_p7, 	threadIdx.x + l_fetch, rng_p4, pipeline);
			}
			pipeline.producer_commit();
		}

		pipeline.consumer_wait();
		block.sync();
		const size_t shared_idx = compute_batch % NUM_STAGE;
		
		#pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			// mma_t(op_c, op_a, op_b);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync(); 
	
	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p6,h3), TB_Y(h1,h2), REG_X,Y(p4,p5) // sd2_6
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p6 && idx_h2 < rng_h3 && idx_h1 < rng_h1 && idx_h3 < rng_h2) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p5 && idx_reg_x < rng_p4) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d2_6(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd2_6: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p5,h1,h3] * v2[p7,h2,p6,p4]
	double* dev_t3; double* dev_t2; double* dev_v2;

	size_t size_t3 = size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;
	size_t size_t2 = size_p7 * size_p5 * size_h1 * size_h3;
	size_t size_v2 = size_p7 * size_h2 * size_p6 * size_p4;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_t3));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_t2));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_v2));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_t3, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_t2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_v2, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_p7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p4;
  int stride_reg_y = stride_output_p5;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);
  dev_t3 = t3_d;

  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d2_6, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d2_6<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_p7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_p7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_t3, cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d2_7(double* dev_t3_d, const double* __restrict__ dev_d2_t2_7, const double* __restrict__ dev_d2_v2_7, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
	// 
	auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p4,h2), TB_Y(h1,h3), REG(p5,p6)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h3 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h2 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h1 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4 + idx_p6) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h1) && (idx_h1 < rng_h2) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_t2_7<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d2_t2_7, 
					blk_idx_h2, 					idx_h1, 
					blk_idx_h1, size_h1, 	idx_h3, 
					blk_idx_p6, size_p6,  
											size_p7, 	threadIdx.x + l_fetch, rng_p6, pipeline);
			}

			if ((idx_h3 < rng_h3) && (idx_h1 < rng_p4) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_v2_7<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d2_v2_7,  
					blk_idx_p4, 				 	idx_h1, 
					blk_idx_p5, size_p5, 
					blk_idx_h3, size_h3, 	idx_h3, 
											size_p7, 	threadIdx.x + l_fetch, rng_p5, pipeline);
			}
			pipeline.producer_commit();
		}

		pipeline.consumer_wait();
		block.sync();
		const size_t shared_idx = compute_batch % NUM_STAGE;
		
		#pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			// mma_t(op_c, op_a, op_b);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync();
	
	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p4,h2), TB_Y(h1,h3), REG(p5,p6)
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p4 && idx_h2 < rng_h2 && idx_h1 < rng_h1 && idx_h3 < rng_h3) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p6 && idx_reg_x < rng_p5) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d2_7(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd2_7: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h1,h2] * v2[p7,h3,p5,p4]
	double* dev_t3; double* dev_t2; double* dev_v2;

	size_t size_t3 = size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;
	size_t size_t2 = size_p7 * size_p6 * size_h1 * size_h2;
	size_t size_v2 = size_p7 * size_h3 * size_p5 * size_p4;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_t3));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_t2));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_v2));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_t3, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_t2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_v2, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_p7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p5;
  int stride_reg_y = stride_output_p6;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);
  dev_t3 = t3_d;

  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d2_7, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d2_7<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_p7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_p7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_t3, cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d2_8(double* dev_t3_d, const double* __restrict__ dev_d2_t2_8, const double* __restrict__ dev_d2_v2_8, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
	// 
	auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p4,h3), TB_Y(h2,h1), REG_X,Y(p5,p6)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h2 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h1 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h3 + 
                    (blk_idx_p6 * SIZE_TILE_P6 +  
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4 + idx_p6) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h2) && (idx_h1 < rng_h3) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_t2_8<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d2_t2_8, 
					blk_idx_h3, 					idx_h1, 
					blk_idx_h2, size_h2, 	idx_h3, 
					blk_idx_p6, size_p6,  
											size_p7, threadIdx.x + l_fetch, rng_p6, pipeline);
			}

			if ((idx_h3 < rng_h1) && (idx_h1 < rng_p4) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_v2_8<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d2_v2_8, 
					blk_idx_p4, 				 	idx_h1, 
					blk_idx_p5, size_p5, 
					blk_idx_h1, size_h1, 	idx_h3, 
											size_p7, 	threadIdx.x + l_fetch, rng_p5, pipeline);
			}
			pipeline.producer_commit();
		}
		pipeline.consumer_wait();
		block.sync();

		const size_t shared_idx = compute_batch % NUM_STAGE;
		// #pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			op_a.template load<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			// mma_t(op_c, op_a, op_b);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync(); 
	
	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p4,h3), TB_Y(h2,h1), REG_X,Y(p5,p6)
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p4 && idx_h2 < rng_h3 && idx_h1 < rng_h2 && idx_h3 < rng_h1) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p6 && idx_reg_x < rng_p5) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d2_8(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd2_8: t3[h3,h2,h1,p6,p5,p4] −= t2[p7,p6,h2,h3] * v2[p7,h1,p5,p4]
	double* dev_t3; double* dev_t2; double* dev_v2;

	size_t size_t3 = size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;
	size_t size_t2 = size_p7 * size_p6 * size_h2 * size_h3;
	size_t size_v2 = size_p7 * size_h1 * size_p5 * size_p4;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_t3));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_t2));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_v2));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_t3, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_t2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_v2, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_p7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p5;
  int stride_reg_y = stride_output_p6;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);
  dev_t3 = t3_d;

  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d2_8, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d2_8<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_p7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_p7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_t3, cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

__global__ void next_unfused_kernel_d2_9(double* dev_t3_d, const double* __restrict__ dev_d2_t2_9, const double* __restrict__ dev_d2_v2_9, 
	int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, 
	int numBlk_h3, int numBlk_h2, int numBlk_h1, int numBlk_p6, int numBlk_p5, int numBlk_p4, 
	int stride_reg_x, int stride_reg_y, 
	int size_internal) 
{
	// 
  auto grid = cooperative_groups::this_grid();
	auto block = cooperative_groups::this_thread_block();

	// For Shared Memory,
	const int lda = 16 + PAD;
	extern __shared__ double sm_block[];
	double *sm_a = reinterpret_cast<double *>(sm_block) + 0 * STAGE_OFFSET;
	double *sm_b = reinterpret_cast<double *>(sm_block) + NUM_STAGE * STAGE_OFFSET;

	#pragma unroll
	for (int i = 0; i < NUM_STAGE; i++) {
		zero_shared(sm_a + STAGE_OFFSET * i);
		zero_shared(sm_b + STAGE_OFFSET * i);
	}
	block.sync();

	// Allocate shared storage for a N-stage cuda::pipeline:
	cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

	const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32; // 0:7
	WarpRegisterMapping wrm(thread_id);

	const int tile_m = warp_id % 2; // 0:1
	const int tile_n = warp_id / 2; // 0:3

	MmaOperandC op_c;

	int internal_upperbound = 0;
	int internal_offset;

  //  
  //  based on sd2_1
  //  (p6,h2), (h1,h3)
	int idx_p6 = threadIdx.x % SIZE_TILE_P6; // this is not used for sd2. 
	int idx_h2 = threadIdx.x / SIZE_TILE_P6;
	int idx_h1 = threadIdx.y % SIZE_TILE_H1;
	int idx_h3 = threadIdx.y / SIZE_TILE_H1;

	int blk_idx_p4 = blockIdx.x / (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	int tmp_blkIdx = blockIdx.x % (numBlk_p5 * numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p5 = tmp_blkIdx / (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_p6 * numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_p6 = tmp_blkIdx / (numBlk_h1 * numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h1 * numBlk_h2 * numBlk_h3);

	int blk_idx_h1 = tmp_blkIdx / (numBlk_h2 * numBlk_h3);
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h2 * numBlk_h3);

	int blk_idx_h2 = tmp_blkIdx / numBlk_h3;
	    tmp_blkIdx = tmp_blkIdx % (numBlk_h3);

	int blk_idx_h3 = tmp_blkIdx;

	//     (p6,h2),     (h1,h3)
	// TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6)
  int base_addr_t3 = blk_idx_h3 * SIZE_TILE_H3 + idx_h2 +
                    (blk_idx_h2 * SIZE_TILE_H2 + idx_h3 + 
                    (blk_idx_h1 * SIZE_TILE_H1 + idx_h1 + 
                    (blk_idx_p6 * SIZE_TILE_P6 + 
                    (blk_idx_p5 * SIZE_TILE_P5 + 
                    (blk_idx_p4 * SIZE_TILE_P4 + idx_p6) * size_p5) * size_p6) * size_h1) * size_h2) * size_h3;

	// need to support partial tiles
	int rng_h3, rng_h2, rng_h1, rng_p6, rng_p5, rng_p4;
	if ((size_h3 - (blk_idx_h3 * SIZE_TILE_H3)) >= SIZE_TILE_H3)  { rng_h3 = SIZE_TILE_H3; }
	else                                                          { rng_h3 = size_h3 % SIZE_TILE_H3; }
	
  if ((size_h2 - (blk_idx_h2 * SIZE_TILE_H2)) >= SIZE_TILE_H2)  { rng_h2 = SIZE_TILE_H2; }
	else                                                          { rng_h2 = size_h2 % SIZE_TILE_H2; }
	
  if ((size_h1 - (blk_idx_h1 * SIZE_TILE_H1)) >= SIZE_TILE_H1)  { rng_h1 = SIZE_TILE_H1; }
	else                                                          { rng_h1 = size_h1 % SIZE_TILE_H1; }

	if ((size_p6 - (blk_idx_p6 * SIZE_TILE_P6)) >= SIZE_TILE_P6)  { rng_p6 = SIZE_TILE_P6; }
	else                                                          { rng_p6 = size_p6 % SIZE_TILE_P6; }

	if ((size_p5 - (blk_idx_p5 * SIZE_TILE_P5)) >= SIZE_TILE_P5)  { rng_p5 = SIZE_TILE_P5; }
	else                                                          { rng_p5 = size_p5 % SIZE_TILE_P5; }

	if ((size_p4 - (blk_idx_p4 * SIZE_TILE_P4)) >= SIZE_TILE_P4)  { rng_p4 = SIZE_TILE_P4; }
	else                                                          { rng_p4 = size_p4 % SIZE_TILE_P4; }

  // 
	const size_t num_batches = (size_internal + SIZE_UNIT_INT - 1) / SIZE_UNIT_INT;

	#pragma unroll 1
	for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < num_batches; ++compute_batch) {
		#pragma unroll 1
		for (; fetch_batch < num_batches && fetch_batch < (compute_batch + NUM_STAGE); ++fetch_batch) {
			pipeline.producer_acquire();

			const int l_fetch = fetch_batch * SIZE_UNIT_INT;
			const size_t shared_idx = fetch_batch % NUM_STAGE;
			internal_offset = (l_fetch + SIZE_UNIT_INT) - size_internal;
			block.sync();

			if (internal_offset > 0) { 
				internal_upperbound = internal_offset;
				zero_shared(sm_a + STAGE_OFFSET * shared_idx); // Zero out shared memory if partial tile
				zero_shared(sm_b + STAGE_OFFSET * shared_idx);
				block.sync();
			}

			if ((idx_h3 < rng_h1) && (idx_h1 < rng_h3) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_t2_9<lda, 1, 4 * lda>(sm_a + STAGE_OFFSET * shared_idx, dev_d2_t2_9, 
					blk_idx_h3, 					idx_h1, 
					blk_idx_h1, size_h1, 	idx_h3, 
					blk_idx_p6, size_p6,  
											size_p7, 	threadIdx.x + l_fetch, 
											rng_p6, pipeline);
			}

			if ((idx_h3 < rng_h2) && (idx_h1 < rng_p4) && threadIdx.x < SIZE_UNIT_INT - internal_upperbound) {
				g2s_d2_v2_9<lda, 1, 4 * lda>(sm_b + STAGE_OFFSET * shared_idx, dev_d2_v2_9, 
					blk_idx_p4, 				 	idx_h1, 
					blk_idx_p5, size_p5, 
					blk_idx_h2, size_h2, 	idx_h3, 
											size_p7, 	threadIdx.x + l_fetch, 
											rng_p5, pipeline);
			}
			pipeline.producer_commit();
		}
		pipeline.consumer_wait();
		block.sync();
		const size_t shared_idx = compute_batch % NUM_STAGE;

		#pragma unroll
		for (int ll = 0; ll < 4; ll++) {
			MmaOperandA op_a;
			op_a.template load_plus<lda>(sm_a + STAGE_OFFSET * shared_idx, ll, tile_m, wrm);
			MmaOperandB op_b;
			op_b.template load<lda>(sm_b + STAGE_OFFSET * shared_idx, ll, tile_n, wrm);
			// mma_t(op_c, op_a, op_b);
			mma(op_c, op_a, op_b);
		}
		pipeline.consumer_release();
	}
	block.sync(); 
	
	// 
	//     (p6,h2),     (h1,h3)
	// TB_X(p4,h3), TB_Y(h1,h2), REG_X,Y(p5,p6)
	dev_t3_d = dev_t3_d + base_addr_t3;
	if (idx_p6 < rng_p4 && idx_h2 < rng_h3 && idx_h1 < rng_h1 && idx_h3 < rng_h2) {
		#pragma unroll 4
		for (int idx_reg_y = 0; idx_reg_y < 4; idx_reg_y++) {
			#pragma unroll 4
			for (int idx_reg_x = 0; idx_reg_x < 4; idx_reg_x++) {
				// 
				if (idx_reg_y < rng_p6 && idx_reg_x < rng_p5) { 
          dev_t3_d[idx_reg_y * stride_reg_y + idx_reg_x * stride_reg_x] += op_c.reg[idx_reg_x + idx_reg_y * 4];
				}
			}
		}
	} 
}

void driver_ccsd_t_d2_9(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2) {
	// 
	int numTbs = 	CEIL(size_h3, SIZE_TILE_H3) * CEIL(size_h2, SIZE_TILE_H2) * CEIL(size_h1, SIZE_TILE_H1) * 
								CEIL(size_p6, SIZE_TILE_P6) * CEIL(size_p5, SIZE_TILE_P5) * CEIL(size_p4, SIZE_TILE_P4);
	
	// sd2_9: t3[h3,h2,h1,p6,p5,p4] += t2[p7,p6,h1,h3] * v2[p7,h2,p5,p4]
	double* dev_t3; double* dev_t2; double* dev_v2;

	size_t size_t3 = size_h3 * size_h2 * size_h1 * size_p6 * size_p5 * size_p4;
	size_t size_t2 = size_p7 * size_p6 * size_h1 * size_h3;
	size_t size_v2 = size_p7 * size_h2 * size_p5 * size_p4;
  
  // cudaMalloc()
  // CUCHK(cudaMalloc((void**) &dev_t3, sizeof(double) * size_t3));
  CUCHK(cudaMalloc((void**) &dev_t2, sizeof(double) * size_t2));
  CUCHK(cudaMalloc((void**) &dev_v2, sizeof(double) * size_v2));
	
  // cudaMemcpy()
  // CUCHK(cudaMemcpy(dev_t3, host_t3, sizeof(double) * size_t3, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_t2, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_v2, cudaMemcpyHostToDevice));
  
	// Related to Kernels
  // size_t numOperations = 2 * (size_t)(size_h3) * (size_t)(size_h2) * (size_t)(size_h1) * (size_t)(size_p6) * (size_t)(size_p5) * (size_t)(size_p4) * (size_t)(size_p7);
	
  // printf ("========================================= fusedKernels =============================================\n");
	// printf ("[%s] Grid Size (1D): %6d\n", __func__, numTbs);
	// printf ("[%s] Block Size (2D): %2d, %2d\n", __func__, 16, 16);
  // printf ("[%s] # of Operations: %lu\n", __func__, numOperations);
  // printf ("====================================================================================================\n");
  
	// 
	dim3 gridsize_1(numTbs);
	dim3 blocksize_1(16, 16);

  int stride_output_h3 = 1;
  int stride_output_h2 = stride_output_h3 * size_h3;
  int stride_output_h1 = stride_output_h2 * size_h2;
  int stride_output_p6 = stride_output_h1 * size_h1;
  int stride_output_p5 = stride_output_p6 * size_p6;
  int stride_output_p4 = stride_output_p5 * size_p5;
  int stride_reg_x = stride_output_p5;
  int stride_reg_y = stride_output_p6;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// cudaEvent_t start_kernel;
  // cudaEvent_t stop_kernel;
  // cudaEventCreate(&start_kernel);
  // cudaEventCreate(&stop_kernel);
  // cudaEventRecord(start_kernel);
  dev_t3 = t3_d;

  // 
  int maxbytes = 98304; // 96 KB
  CUCHK(cudaFuncSetAttribute(next_unfused_kernel_d2_9, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
  next_unfused_kernel_d2_9<<<gridsize_1, blocksize_1, 2 * NUM_STAGE * 8 * STAGE_OFFSET, 0>>>(dev_t3, dev_t2, dev_v2, size_h3, size_h2, size_h1, size_p6, size_p5, size_p4, size_p7, 
		CEIL(size_h3, SIZE_TILE_H3), CEIL(size_h2, SIZE_TILE_H2), CEIL(size_h1, SIZE_TILE_H1), 
		CEIL(size_p6, SIZE_TILE_P6), CEIL(size_p5, SIZE_TILE_P5), CEIL(size_p4, SIZE_TILE_P4), 
		stride_reg_x, stride_reg_y,
    size_p7);
  cudaDeviceSynchronize();
  CUCHK(cudaGetLastError());
  
  // cudaEventRecord(stop_kernel);
  // cudaEventSynchronize(stop_kernel);
  // float kernel_ms = 0;
  // cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
  // printf ("[%s] kernel: %f (ms)\n", __func__, kernel_ms);

  // Copy the Result from Device to Host
  // CUCHK(cudaMemcpy(host_t3, dev_t3, sizeof(double) * size_t3, cudaMemcpyDeviceToHost));

  // cudaFree()
  // CUCHK(cudaFree(dev_t3)); 
	CUCHK(cudaFree(dev_t2)); 
	CUCHK(cudaFree(dev_v2));
}

#endif // A100 and cuda 11.1 