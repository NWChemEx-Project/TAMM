//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------
#include <iostream>
#include "tensor/corf.h"
#include "tensor/equations.h"
#include "tensor/input.h"
#include "tensor/schedulers.h"
#include "tensor/t_assign.h"
#include "tensor/t_mult.h"
#include "tensor/tensor.h"
#include "tensor/tensors_and_ops.h"
#include "tensor/variables.h"

/*
 *  lambda1Mod {
 *
 *  index h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12 = O;
 *  index p1,p2,p3,p4,p5,p6,p7,p8,p9 = V;
 *
 *  array i0[O][V];
 *  array f[N][N]: irrep_f;
 *  array y_ov[O][V]: irrep_y;
 *  array t_vo[V][O]: irrep_t;
 *  array v[N,N][N,N]: irrep_v;
 *  array t_vvoo[V,V][O,O]: irrep_t;
 *  array y_oovv[O,O][V,V]: irrep_y;
 *  array lambda1Mod_9_1[O][O];
 *  array lambda1Mod_14_2_1[O,O][O,V];
 *  array lambda1Mod_8_1[O][O];
 *  array lambda1Mod_2_2_1[O][V];
 *  array lambda1Mod_12_4_1[O][O];
 *  array lambda1Mod_7_1[V][O];
 *  array lambda1Mod_12_2_1[O,O][O,O];
 *  array lambda1Mod_12_2_2_1[O,O][O,V];
 *  array lambda1Mod_5_2_1[O,O][O,O];
 *  array lambda1Mod_12_3_1[O,O][O,V];
 *  array lambda1Mod_12_1[O,V][O,O];
 *  array lambda1Mod_5_4_1[O][V];
 *  array lambda1Mod_13_1[O,O][O,O];
 *  array lambda1Mod_5_5_1[O,O][O,V];
 *  array lambda1Mod_10_1[V][V];
 *  array lambda1Mod_11_1[O,O][O,V];
 *  array lambda1Mod_3_3_1[O][V];
 *  array lambda1Mod_6_1[V,V][O,V];
 *  array lambda1Mod_5_1[O,V][O,O];
 *  array lambda1Mod_14_1[O,V][O,V];
 *  array lambda1Mod_5_3_1[O,V][O,V];
 *  array lambda1Mod_3_1[V][V];
 *  array lambda1Mod_13_2_1[O,O][O,V];
 *  array lambda1Mod_5_2_2_1[O,O][O,V];
 *  array lambda1Mod_7_4_1[O,O][O,V];
 *  array lambda1Mod_2_1[O][O];
 *  array lambda1Mod_7_3_1[O][O];
 *
 *  lambda1Mod_1:       i0[h2,p1] += 1 * f[h2,p1];
 *  lambda1Mod_2_1:     lambda1Mod_2_1[h2,h7] += 1 * f[h2,h7];
 *  lambda1Mod_2_2_1:   lambda1Mod_2_2_1[h2,p3] += 1 * f[h2,p3];
 *  lambda1Mod_2_2_2:   lambda1Mod_2_2_1[h2,p3] += 1 * t_vo[p5,h6] *
 * v[h2,h6,p3,p5];
 *  lambda1Mod_2_2:     lambda1Mod_2_1[h2,h7] += 1 * t_vo[p3,h7] *
 * lambda1Mod_2_2_1[h2,p3];
 *  lambda1Mod_2_3:     lambda1Mod_2_1[h2,h7] += 1 * t_vo[p3,h4] *
 * v[h2,h4,h7,p3];
 *  lambda1Mod_2_4:     lambda1Mod_2_1[h2,h7] += -1/2 * t_vvoo[p3,p4,h6,h7] *
 * v[h2,h6,p3,p4];
 *  lambda1Mod_2:       i0[h2,p1] += -1 * y_ov[h7,p1] * lambda1Mod_2_1[h2,h7];
 *  lambda1Mod_3_1:     lambda1Mod_3_1[p7,p1] += 1 * f[p7,p1];
 *  lambda1Mod_3_2:     lambda1Mod_3_1[p7,p1] += -1 * t_vo[p3,h4] *
 * v[h4,p7,p1,p3];
 *  lambda1Mod_3_3_1:   lambda1Mod_3_3_1[h4,p1] += 1 * t_vo[p5,h6] *
 * v[h4,h6,p1,p5];
 *  lambda1Mod_3_3:     lambda1Mod_3_1[p7,p1] += -1 * t_vo[p7,h4] *
 * lambda1Mod_3_3_1[h4,p1];
 *  lambda1Mod_3:       i0[h2,p1] += 1 * y_ov[h2,p7] * lambda1Mod_3_1[p7,p1];
 *  lambda1Mod_4:       i0[h2,p1] += -1 * y_ov[h4,p3] * v[h2,p3,h4,p1];
 *  lambda1Mod_5_1:     lambda1Mod_5_1[h2,p9,h11,h12] += 1 * v[h2,p9,h11,h12];
 *  lambda1Mod_5_2_1:   lambda1Mod_5_2_1[h2,h7,h11,h12] += 1 * v[h2,h7,h11,h12];
 *  lambda1Mod_5_2_2_1: lambda1Mod_5_2_2_1[h2,h7,h12,p3] += 1 * v[h2,h7,h12,p3];
 *  lambda1Mod_5_2_2_2: lambda1Mod_5_2_2_1[h2,h7,h12,p3] += -1/2 * t_vo[p5,h12]
 * * v[h2,h7,p3,p5];
 *  lambda1Mod_5_2_2:   lambda1Mod_5_2_1[h2,h7,h11,h12] += -2 * t_vo[p3,h11] *
 * lambda1Mod_5_2_2_1[h2,h7,h12,p3];
 *  lambda1Mod_5_2_3:   lambda1Mod_5_2_1[h2,h7,h11,h12] += 1/2 *
 * t_vvoo[p3,p4,h11,h12] * v[h2,h7,p3,p4];
 *  lambda1Mod_5_2:     lambda1Mod_5_1[h2,p9,h11,h12] += -1 * t_vo[p9,h7] *
 * lambda1Mod_5_2_1[h2,h7,h11,h12];
 *  lambda1Mod_5_3_1:   lambda1Mod_5_3_1[h2,p9,h12,p3] += 1 * v[h2,p9,h12,p3];
 *  lambda1Mod_5_3_2:   lambda1Mod_5_3_1[h2,p9,h12,p3] += -1/2 * t_vo[p5,h12] *
 * v[h2,p9,p3,p5];
 *  lambda1Mod_5_3:     lambda1Mod_5_1[h2,p9,h11,h12] += -2 * t_vo[p3,h11] *
 * lambda1Mod_5_3_1[h2,p9,h12,p3];
 *  lambda1Mod_5_4_1:   lambda1Mod_5_4_1[h2,p5] += 1 * f[h2,p5];
 *  lambda1Mod_5_4_2:   lambda1Mod_5_4_1[h2,p5] += 1 * t_vo[p7,h8] *
 * v[h2,h8,p5,p7];
 *  lambda1Mod_5_4:     lambda1Mod_5_1[h2,p9,h11,h12] += 1 *
 * t_vvoo[p5,p9,h11,h12] * lambda1Mod_5_4_1[h2,p5];
 *  lambda1Mod_5_5_1:   lambda1Mod_5_5_1[h2,h6,h12,p4] += 1 * v[h2,h6,h12,p4];
 *  lambda1Mod_5_5_2:   lambda1Mod_5_5_1[h2,h6,h12,p4] += -1 * t_vo[p7,h12] *
 * v[h2,h6,p4,p7];
 *  lambda1Mod_5_5:     lambda1Mod_5_1[h2,p9,h11,h12] += -2 *
 * t_vvoo[p4,p9,h6,h11] * lambda1Mod_5_5_1[h2,h6,h12,p4];
 *  lambda1Mod_5_6:     lambda1Mod_5_1[h2,p9,h11,h12] += 1/2 *
 * t_vvoo[p3,p4,h11,h12] * v[h2,p9,p3,p4];
 *  lambda1Mod_5:       i0[h2,p1] += -1/2 * y_oovv[h11,h12,p1,p9] *
 * lambda1Mod_5_1[h2,p9,h11,h12];
 *  lambda1Mod_6_1:     lambda1Mod_6_1[p5,p8,h7,p1] += -1 * v[p5,p8,h7,p1];
 *  lambda1Mod_6_2:     lambda1Mod_6_1[p5,p8,h7,p1] += 1 * t_vo[p3,h7] *
 * v[p5,p8,p1,p3];
 *  lambda1Mod_6:       i0[h2,p1] += 1/2 * y_oovv[h2,h7,p5,p8] *
 * lambda1Mod_6_1[p5,p8,h7,p1];
 *  lambda1Mod_7_1:     lambda1Mod_7_1[p9,h10] += 1 * t_vo[p9,h10];
 *  lambda1Mod_7_2:     lambda1Mod_7_1[p9,h10] += 1 * t_vvoo[p3,p9,h5,h10] *
 * y_ov[h5,p3];
 *  lambda1Mod_7_3_1:   lambda1Mod_7_3_1[h6,h10] += 1 * t_vo[p5,h10] *
 * y_ov[h6,p5];
 *  lambda1Mod_7_3_2:   lambda1Mod_7_3_1[h6,h10] += 1/2 * t_vvoo[p3,p4,h5,h10] *
 * y_oovv[h5,h6,p3,p4];
 *  lambda1Mod_7_3:     lambda1Mod_7_1[p9,h10] += -1 * t_vo[p9,h6] *
 * lambda1Mod_7_3_1[h6,h10];
 *  lambda1Mod_7_4_1:   lambda1Mod_7_4_1[h5,h6,h10,p3] += 1 * t_vo[p7,h10] *
 * y_oovv[h5,h6,p3,p7];
 *  lambda1Mod_7_4:     lambda1Mod_7_1[p9,h10] += -1/2 * t_vvoo[p3,p9,h5,h6] *
 * lambda1Mod_7_4_1[h5,h6,h10,p3];
 *  lambda1Mod_7:       i0[h2,p1] += 1 * lambda1Mod_7_1[p9,h10] *
 * v[h2,h10,p1,p9];
 *  lambda1Mod_8_1:     lambda1Mod_8_1[h2,h3] += 1 * t_vo[p4,h3] * y_ov[h2,p4];
 *  lambda1Mod_8_2:     lambda1Mod_8_1[h2,h3] += 1/2 * t_vvoo[p4,p5,h3,h6] *
 * y_oovv[h2,h6,p4,p5];
 *  lambda1Mod_8:       i0[h2,p1] += -1 * lambda1Mod_8_1[h2,h3] * f[h3,p1];
 *  lambda1Mod_9_1:     lambda1Mod_9_1[h6,h8] += 1 * t_vo[p3,h8] * y_ov[h6,p3];
 *  lambda1Mod_9_2:     lambda1Mod_9_1[h6,h8] += 1/2 * t_vvoo[p3,p4,h5,h8] *
 * y_oovv[h5,h6,p3,p4];
 *  lambda1Mod_9:       i0[h2,p1] += 1 * lambda1Mod_9_1[h6,h8] * v[h2,h8,h6,p1];
 *  lambda1Mod_10_1:    lambda1Mod_10_1[p7,p8] += 1 * t_vo[p7,h4] * y_ov[h4,p8];
 *  lambda1Mod_10_2:    lambda1Mod_10_1[p7,p8] += 1/2 * t_vvoo[p3,p7,h5,h6] *
 * y_oovv[h5,h6,p3,p8];
 *  lambda1Mod_10:      i0[h2,p1] += 1 * lambda1Mod_10_1[p7,p8] *
 * v[h2,p8,p1,p7];
 *  lambda1Mod_11_1:    lambda1Mod_11_1[h2,h6,h4,p5] += 1 * t_vo[p3,h4] *
 * y_oovv[h2,h6,p3,p5];
 *  lambda1Mod_11:      i0[h2,p1] += 1 * lambda1Mod_11_1[h2,h6,h4,p5] *
 * v[h4,p5,h6,p1];
 *  lambda1Mod_12_1:    lambda1Mod_12_1[h2,p9,h6,h12] += -1 *
 * t_vvoo[p3,p9,h6,h12] * y_ov[h2,p3];
 *  lambda1Mod_12_2_1:  lambda1Mod_12_2_1[h2,h10,h6,h12] += -1 *
 * t_vvoo[p3,p4,h6,h12] * y_oovv[h2,h10,p3,p4];
 *  lambda1Mod_12_2_2_1:lambda1Mod_12_2_2_1[h2,h10,h6,p5] += 1 * t_vo[p7,h6] *
 * y_oovv[h2,h10,p5,p7];
 *  lambda1Mod_12_2_2:  lambda1Mod_12_2_1[h2,h10,h6,h12] += 2 * t_vo[p5,h12] *
 * lambda1Mod_12_2_2_1[h2,h10,h6,p5];
 *  lambda1Mod_12_2:    lambda1Mod_12_1[h2,p9,h6,h12] += -1/2 * t_vo[p9,h10] *
 * lambda1Mod_12_2_1[h2,h10,h6,h12];
 *  lambda1Mod_12_3_1:  lambda1Mod_12_3_1[h2,h5,h6,p3] += 1 * t_vo[p7,h6] *
 * y_oovv[h2,h5,p3,p7];
 *  lambda1Mod_12_3:    lambda1Mod_12_1[h2,p9,h6,h12] += 2 *
 * t_vvoo[p3,p9,h5,h12] * lambda1Mod_12_3_1[h2,h5,h6,p3];
 *  lambda1Mod_12_4_1:  lambda1Mod_12_4_1[h2,h12] += 1 * t_vvoo[p3,p4,h5,h12] *
 * y_oovv[h2,h5,p3,p4];
 *  lambda1Mod_12_4:    lambda1Mod_12_1[h2,p9,h6,h12] += -1 * t_vo[p9,h6] *
 * lambda1Mod_12_4_1[h2,h12];
 *  lambda1Mod_12:      i0[h2,p1] += 1/2 * lambda1Mod_12_1[h2,p9,h6,h12] *
 * v[h6,h12,p1,p9];
 *  lambda1Mod_13_1:    lambda1Mod_13_1[h2,h7,h6,h8] += -1 * t_vvoo[p3,p4,h6,h8]
 * * y_oovv[h2,h7,p3,p4];
 *  lambda1Mod_13_2_1:  lambda1Mod_13_2_1[h2,h7,h6,p3] += 1 * t_vo[p5,h6] *
 * y_oovv[h2,h7,p3,p5];
 *  lambda1Mod_13_2:    lambda1Mod_13_1[h2,h7,h6,h8] += 2 * t_vo[p3,h8] *
 * lambda1Mod_13_2_1[h2,h7,h6,p3];
 *  lambda1Mod_13:      i0[h2,p1] += 1/4 * lambda1Mod_13_1[h2,h7,h6,h8] *
 * v[h6,h8,h7,p1];
 *  lambda1Mod_14_1:    lambda1Mod_14_1[h2,p8,h6,p7] += 1 * t_vvoo[p3,p8,h5,h6]
 * * y_oovv[h2,h5,p3,p7];
 *  lambda1Mod_14_2_1:  lambda1Mod_14_2_1[h2,h4,h6,p7] += 1 * t_vo[p5,h6] *
 * y_oovv[h2,h4,p5,p7];
 *  lambda1Mod_14_2:    lambda1Mod_14_1[h2,p8,h6,p7] += -1 * t_vo[p8,h4] *
 * lambda1Mod_14_2_1[h2,h4,h6,p7];
 *  lambda1Mod_14:      i0[h2,p1] += 1 * lambda1Mod_14_1[h2,p8,h6,p7] *
 * v[h6,p7,p1,p8];
 *
 *  }
*/

extern "C" {
void ccsd_lambda1Mod_1_(Integer *d_f, Integer *k_f_offset, Integer *d_i0,
                        Integer *k_i0_offset);
void ccsd_lambda1Mod_2_1_(Integer *d_f, Integer *k_f_offset,
                          Integer *d_lambda1Mod_2_1,
                          Integer *k_lambda1Mod_2_1_offset);
void ccsd_lambda1Mod_2_2_1_(Integer *d_f, Integer *k_f_offset,
                            Integer *d_lambda1Mod_2_2_1,
                            Integer *k_lambda1Mod_2_2_1_offset);
void ccsd_lambda1Mod_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                            Integer *d_v, Integer *k_v_offset,
                            Integer *d_lambda1Mod_2_2_1,
                            Integer *k_lambda1Mod_2_2_1_offset);
void ccsd_lambda1Mod_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_lambda1Mod_2_2_1,
                          Integer *k_lambda1Mod_2_2_1_offset,
                          Integer *d_lambda1Mod_2_1,
                          Integer *k_lambda1Mod_2_1_offset);
void ccsd_lambda1Mod_2_3_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                          Integer *k_v_offset, Integer *d_lambda1Mod_2_1,
                          Integer *k_lambda1Mod_2_1_offset);
void ccsd_lambda1Mod_2_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                          Integer *d_v, Integer *k_v_offset,
                          Integer *d_lambda1Mod_2_1,
                          Integer *k_lambda1Mod_2_1_offset);
void ccsd_lambda1Mod_2_(Integer *d_y_ov, Integer *k_y_ov_offset,
                        Integer *d_lambda1Mod_2_1,
                        Integer *k_lambda1Mod_2_1_offset, Integer *d_i0,
                        Integer *k_i0_offset);
void ccsd_lambda1Mod_3_1_(Integer *d_f, Integer *k_f_offset,
                          Integer *d_lambda1Mod_3_1,
                          Integer *k_lambda1Mod_3_1_offset);
void ccsd_lambda1Mod_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                          Integer *k_v_offset, Integer *d_lambda1Mod_3_1,
                          Integer *k_lambda1Mod_3_1_offset);
void ccsd_lambda1Mod_3_3_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                            Integer *d_v, Integer *k_v_offset,
                            Integer *d_lambda1Mod_3_3_1,
                            Integer *k_lambda1Mod_3_3_1_offset);
void ccsd_lambda1Mod_3_3_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_lambda1Mod_3_3_1,
                          Integer *k_lambda1Mod_3_3_1_offset,
                          Integer *d_lambda1Mod_3_1,
                          Integer *k_lambda1Mod_3_1_offset);
void ccsd_lambda1Mod_3_(Integer *d_y_ov, Integer *k_y_ov_offset,
                        Integer *d_lambda1Mod_3_1,
                        Integer *k_lambda1Mod_3_1_offset, Integer *d_i0,
                        Integer *k_i0_offset);
void ccsd_lambda1Mod_4_(Integer *d_y_ov, Integer *k_y_ov_offset, Integer *d_v,
                        Integer *k_v_offset, Integer *d_i0,
                        Integer *k_i0_offset);
void ccsd_lambda1Mod_5_1_(Integer *d_v, Integer *k_v_offset,
                          Integer *d_lambda1Mod_5_1,
                          Integer *k_lambda1Mod_5_1_offset);
void ccsd_lambda1Mod_5_2_1_(Integer *d_v, Integer *k_v_offset,
                            Integer *d_lambda1Mod_5_2_1,
                            Integer *k_lambda1Mod_5_2_1_offset);
void ccsd_lambda1Mod_5_2_2_1_(Integer *d_v, Integer *k_v_offset,
                              Integer *d_lambda1Mod_5_2_2_1,
                              Integer *k_lambda1Mod_5_2_2_1_offset);
void ccsd_lambda1Mod_5_2_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                              Integer *d_v, Integer *k_v_offset,
                              Integer *d_lambda1Mod_5_2_2_1,
                              Integer *k_lambda1Mod_5_2_2_1_offset);
void ccsd_lambda1Mod_5_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                            Integer *d_lambda1Mod_5_2_2_1,
                            Integer *k_lambda1Mod_5_2_2_1_offset,
                            Integer *d_lambda1Mod_5_2_1,
                            Integer *k_lambda1Mod_5_2_1_offset);
void ccsd_lambda1Mod_5_2_3_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                            Integer *d_v, Integer *k_v_offset,
                            Integer *d_lambda1Mod_5_2_1,
                            Integer *k_lambda1Mod_5_2_1_offset);
void ccsd_lambda1Mod_5_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_lambda1Mod_5_2_1,
                          Integer *k_lambda1Mod_5_2_1_offset,
                          Integer *d_lambda1Mod_5_1,
                          Integer *k_lambda1Mod_5_1_offset);
void ccsd_lambda1Mod_5_3_1_(Integer *d_v, Integer *k_v_offset,
                            Integer *d_lambda1Mod_5_3_1,
                            Integer *k_lambda1Mod_5_3_1_offset);
void ccsd_lambda1Mod_5_3_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                            Integer *d_v, Integer *k_v_offset,
                            Integer *d_lambda1Mod_5_3_1,
                            Integer *k_lambda1Mod_5_3_1_offset);
void ccsd_lambda1Mod_5_3_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_lambda1Mod_5_3_1,
                          Integer *k_lambda1Mod_5_3_1_offset,
                          Integer *d_lambda1Mod_5_1,
                          Integer *k_lambda1Mod_5_1_offset);
void ccsd_lambda1Mod_5_4_1_(Integer *d_f, Integer *k_f_offset,
                            Integer *d_lambda1Mod_5_4_1,
                            Integer *k_lambda1Mod_5_4_1_offset);
void ccsd_lambda1Mod_5_4_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                            Integer *d_v, Integer *k_v_offset,
                            Integer *d_lambda1Mod_5_4_1,
                            Integer *k_lambda1Mod_5_4_1_offset);
void ccsd_lambda1Mod_5_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                          Integer *d_lambda1Mod_5_4_1,
                          Integer *k_lambda1Mod_5_4_1_offset,
                          Integer *d_lambda1Mod_5_1,
                          Integer *k_lambda1Mod_5_1_offset);
void ccsd_lambda1Mod_5_5_1_(Integer *d_v, Integer *k_v_offset,
                            Integer *d_lambda1Mod_5_5_1,
                            Integer *k_lambda1Mod_5_5_1_offset);
void ccsd_lambda1Mod_5_5_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                            Integer *d_v, Integer *k_v_offset,
                            Integer *d_lambda1Mod_5_5_1,
                            Integer *k_lambda1Mod_5_5_1_offset);
void ccsd_lambda1Mod_5_5_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                          Integer *d_lambda1Mod_5_5_1,
                          Integer *k_lambda1Mod_5_5_1_offset,
                          Integer *d_lambda1Mod_5_1,
                          Integer *k_lambda1Mod_5_1_offset);
void ccsd_lambda1Mod_5_6_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                          Integer *d_v, Integer *k_v_offset,
                          Integer *d_lambda1Mod_5_1,
                          Integer *k_lambda1Mod_5_1_offset);
void ccsd_lambda1Mod_5_(Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda1Mod_5_1,
                        Integer *k_lambda1Mod_5_1_offset, Integer *d_i0,
                        Integer *k_i0_offset);
void ccsd_lambda1Mod_6_1_(Integer *d_v, Integer *k_v_offset,
                          Integer *d_lambda1Mod_6_1,
                          Integer *k_lambda1Mod_6_1_offset);
void ccsd_lambda1Mod_6_2_(Integer *d_t_vo, Integer *k_t_vo_offset, Integer *d_v,
                          Integer *k_v_offset, Integer *d_lambda1Mod_6_1,
                          Integer *k_lambda1Mod_6_1_offset);
void ccsd_lambda1Mod_6_(Integer *d_y_oovv, Integer *k_y_oovv_offset,
                        Integer *d_lambda1Mod_6_1,
                        Integer *k_lambda1Mod_6_1_offset, Integer *d_i0,
                        Integer *k_i0_offset);
void ccsd_lambda1Mod_7_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_lambda1Mod_7_1,
                          Integer *k_lambda1Mod_7_1_offset);
void ccsd_lambda1Mod_7_2_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                          Integer *d_y_ov, Integer *k_y_ov_offset,
                          Integer *d_lambda1Mod_7_1,
                          Integer *k_lambda1Mod_7_1_offset);
void ccsd_lambda1Mod_7_3_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                            Integer *d_y_ov, Integer *k_y_ov_offset,
                            Integer *d_lambda1Mod_7_3_1,
                            Integer *k_lambda1Mod_7_3_1_offset);
void ccsd_lambda1Mod_7_3_2_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                            Integer *d_y_oovv, Integer *k_y_oovv_offset,
                            Integer *d_lambda1Mod_7_3_1,
                            Integer *k_lambda1Mod_7_3_1_offset);
void ccsd_lambda1Mod_7_3_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_lambda1Mod_7_3_1,
                          Integer *k_lambda1Mod_7_3_1_offset,
                          Integer *d_lambda1Mod_7_1,
                          Integer *k_lambda1Mod_7_1_offset);
void ccsd_lambda1Mod_7_4_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                            Integer *d_y_oovv, Integer *k_y_oovv_offset,
                            Integer *d_lambda1Mod_7_4_1,
                            Integer *k_lambda1Mod_7_4_1_offset);
void ccsd_lambda1Mod_7_4_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                          Integer *d_lambda1Mod_7_4_1,
                          Integer *k_lambda1Mod_7_4_1_offset,
                          Integer *d_lambda1Mod_7_1,
                          Integer *k_lambda1Mod_7_1_offset);
void ccsd_lambda1Mod_7_(Integer *d_lambda1Mod_7_1,
                        Integer *k_lambda1Mod_7_1_offset, Integer *d_v,
                        Integer *k_v_offset, Integer *d_i0,
                        Integer *k_i0_offset);
void ccsd_lambda1Mod_8_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_y_ov, Integer *k_y_ov_offset,
                          Integer *d_lambda1Mod_8_1,
                          Integer *k_lambda1Mod_8_1_offset);
void ccsd_lambda1Mod_8_2_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                          Integer *d_y_oovv, Integer *k_y_oovv_offset,
                          Integer *d_lambda1Mod_8_1,
                          Integer *k_lambda1Mod_8_1_offset);
void ccsd_lambda1Mod_8_(Integer *d_lambda1Mod_8_1,
                        Integer *k_lambda1Mod_8_1_offset, Integer *d_f,
                        Integer *k_f_offset, Integer *d_i0,
                        Integer *k_i0_offset);
void ccsd_lambda1Mod_9_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                          Integer *d_y_ov, Integer *k_y_ov_offset,
                          Integer *d_lambda1Mod_9_1,
                          Integer *k_lambda1Mod_9_1_offset);
void ccsd_lambda1Mod_9_2_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                          Integer *d_y_oovv, Integer *k_y_oovv_offset,
                          Integer *d_lambda1Mod_9_1,
                          Integer *k_lambda1Mod_9_1_offset);
void ccsd_lambda1Mod_9_(Integer *d_lambda1Mod_9_1,
                        Integer *k_lambda1Mod_9_1_offset, Integer *d_v,
                        Integer *k_v_offset, Integer *d_i0,
                        Integer *k_i0_offset);
void ccsd_lambda1Mod_10_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                           Integer *d_y_ov, Integer *k_y_ov_offset,
                           Integer *d_lambda1Mod_10_1,
                           Integer *k_lambda1Mod_10_1_offset);
void ccsd_lambda1Mod_10_2_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                           Integer *d_y_oovv, Integer *k_y_oovv_offset,
                           Integer *d_lambda1Mod_10_1,
                           Integer *k_lambda1Mod_10_1_offset);
void ccsd_lambda1Mod_10_(Integer *d_lambda1Mod_10_1,
                         Integer *k_lambda1Mod_10_1_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_i0,
                         Integer *k_i0_offset);
void ccsd_lambda1Mod_11_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                           Integer *d_y_oovv, Integer *k_y_oovv_offset,
                           Integer *d_lambda1Mod_11_1,
                           Integer *k_lambda1Mod_11_1_offset);
void ccsd_lambda1Mod_11_(Integer *d_lambda1Mod_11_1,
                         Integer *k_lambda1Mod_11_1_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_i0,
                         Integer *k_i0_offset);
void ccsd_lambda1Mod_12_1_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                           Integer *d_y_ov, Integer *k_y_ov_offset,
                           Integer *d_lambda1Mod_12_1,
                           Integer *k_lambda1Mod_12_1_offset);
void ccsd_lambda1Mod_12_2_1_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                             Integer *d_y_oovv, Integer *k_y_oovv_offset,
                             Integer *d_lambda1Mod_12_2_1,
                             Integer *k_lambda1Mod_12_2_1_offset);
void ccsd_lambda1Mod_12_2_2_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                               Integer *d_y_oovv, Integer *k_y_oovv_offset,
                               Integer *d_lambda1Mod_12_2_2_1,
                               Integer *k_lambda1Mod_12_2_2_1_offset);
void ccsd_lambda1Mod_12_2_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                             Integer *d_lambda1Mod_12_2_2_1,
                             Integer *k_lambda1Mod_12_2_2_1_offset,
                             Integer *d_lambda1Mod_12_2_1,
                             Integer *k_lambda1Mod_12_2_1_offset);
void ccsd_lambda1Mod_12_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                           Integer *d_lambda1Mod_12_2_1,
                           Integer *k_lambda1Mod_12_2_1_offset,
                           Integer *d_lambda1Mod_12_1,
                           Integer *k_lambda1Mod_12_1_offset);
void ccsd_lambda1Mod_12_3_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                             Integer *d_y_oovv, Integer *k_y_oovv_offset,
                             Integer *d_lambda1Mod_12_3_1,
                             Integer *k_lambda1Mod_12_3_1_offset);
void ccsd_lambda1Mod_12_3_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                           Integer *d_lambda1Mod_12_3_1,
                           Integer *k_lambda1Mod_12_3_1_offset,
                           Integer *d_lambda1Mod_12_1,
                           Integer *k_lambda1Mod_12_1_offset);
void ccsd_lambda1Mod_12_4_1_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                             Integer *d_y_oovv, Integer *k_y_oovv_offset,
                             Integer *d_lambda1Mod_12_4_1,
                             Integer *k_lambda1Mod_12_4_1_offset);
void ccsd_lambda1Mod_12_4_(Integer *d_t_vo, Integer *k_t_vo_offset,
                           Integer *d_lambda1Mod_12_4_1,
                           Integer *k_lambda1Mod_12_4_1_offset,
                           Integer *d_lambda1Mod_12_1,
                           Integer *k_lambda1Mod_12_1_offset);
void ccsd_lambda1Mod_12_(Integer *d_lambda1Mod_12_1,
                         Integer *k_lambda1Mod_12_1_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_i0,
                         Integer *k_i0_offset);
void ccsd_lambda1Mod_13_1_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                           Integer *d_y_oovv, Integer *k_y_oovv_offset,
                           Integer *d_lambda1Mod_13_1,
                           Integer *k_lambda1Mod_13_1_offset);
void ccsd_lambda1Mod_13_2_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                             Integer *d_y_oovv, Integer *k_y_oovv_offset,
                             Integer *d_lambda1Mod_13_2_1,
                             Integer *k_lambda1Mod_13_2_1_offset);
void ccsd_lambda1Mod_13_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                           Integer *d_lambda1Mod_13_2_1,
                           Integer *k_lambda1Mod_13_2_1_offset,
                           Integer *d_lambda1Mod_13_1,
                           Integer *k_lambda1Mod_13_1_offset);
void ccsd_lambda1Mod_13_(Integer *d_lambda1Mod_13_1,
                         Integer *k_lambda1Mod_13_1_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_i0,
                         Integer *k_i0_offset);
void ccsd_lambda1Mod_14_1_(Integer *d_t_vvoo, Integer *k_t_vvoo_offset,
                           Integer *d_y_oovv, Integer *k_y_oovv_offset,
                           Integer *d_lambda1Mod_14_1,
                           Integer *k_lambda1Mod_14_1_offset);
void ccsd_lambda1Mod_14_2_1_(Integer *d_t_vo, Integer *k_t_vo_offset,
                             Integer *d_y_oovv, Integer *k_y_oovv_offset,
                             Integer *d_lambda1Mod_14_2_1,
                             Integer *k_lambda1Mod_14_2_1_offset);
void ccsd_lambda1Mod_14_2_(Integer *d_t_vo, Integer *k_t_vo_offset,
                           Integer *d_lambda1Mod_14_2_1,
                           Integer *k_lambda1Mod_14_2_1_offset,
                           Integer *d_lambda1Mod_14_1,
                           Integer *k_lambda1Mod_14_1_offset);
void ccsd_lambda1Mod_14_(Integer *d_lambda1Mod_14_1,
                         Integer *k_lambda1Mod_14_1_offset, Integer *d_v,
                         Integer *k_v_offset, Integer *d_i0,
                         Integer *k_i0_offset);

void offset_ccsd_lambda1Mod_2_1_(Integer *l_lambda1Mod_2_1_offset,
                                 Integer *k_lambda1Mod_2_1_offset,
                                 Integer *size_lambda1Mod_2_1);
void offset_ccsd_lambda1Mod_2_2_1_(Integer *l_lambda1Mod_2_2_1_offset,
                                   Integer *k_lambda1Mod_2_2_1_offset,
                                   Integer *size_lambda1Mod_2_2_1);
void offset_ccsd_lambda1Mod_3_1_(Integer *l_lambda1Mod_3_1_offset,
                                 Integer *k_lambda1Mod_3_1_offset,
                                 Integer *size_lambda1Mod_3_1);
void offset_ccsd_lambda1Mod_3_3_1_(Integer *l_lambda1Mod_3_3_1_offset,
                                   Integer *k_lambda1Mod_3_3_1_offset,
                                   Integer *size_lambda1Mod_3_3_1);
void offset_ccsd_lambda1Mod_5_1_(Integer *l_lambda1Mod_5_1_offset,
                                 Integer *k_lambda1Mod_5_1_offset,
                                 Integer *size_lambda1Mod_5_1);
void offset_ccsd_lambda1Mod_5_2_1_(Integer *l_lambda1Mod_5_2_1_offset,
                                   Integer *k_lambda1Mod_5_2_1_offset,
                                   Integer *size_lambda1Mod_5_2_1);
void offset_ccsd_lambda1Mod_5_2_2_1_(Integer *l_lambda1Mod_5_2_2_1_offset,
                                     Integer *k_lambda1Mod_5_2_2_1_offset,
                                     Integer *size_lambda1Mod_5_2_2_1);
void offset_ccsd_lambda1Mod_5_3_1_(Integer *l_lambda1Mod_5_3_1_offset,
                                   Integer *k_lambda1Mod_5_3_1_offset,
                                   Integer *size_lambda1Mod_5_3_1);
void offset_ccsd_lambda1Mod_5_4_1_(Integer *l_lambda1Mod_5_4_1_offset,
                                   Integer *k_lambda1Mod_5_4_1_offset,
                                   Integer *size_lambda1Mod_5_4_1);
void offset_ccsd_lambda1Mod_5_5_1_(Integer *l_lambda1Mod_5_5_1_offset,
                                   Integer *k_lambda1Mod_5_5_1_offset,
                                   Integer *size_lambda1Mod_5_5_1);
void offset_ccsd_lambda1Mod_6_1_(Integer *l_lambda1Mod_6_1_offset,
                                 Integer *k_lambda1Mod_6_1_offset,
                                 Integer *size_lambda1Mod_6_1);
void offset_ccsd_lambda1Mod_7_1_(Integer *l_lambda1Mod_7_1_offset,
                                 Integer *k_lambda1Mod_7_1_offset,
                                 Integer *size_lambda1Mod_7_1);
void offset_ccsd_lambda1Mod_7_3_1_(Integer *l_lambda1Mod_7_3_1_offset,
                                   Integer *k_lambda1Mod_7_3_1_offset,
                                   Integer *size_lambda1Mod_7_3_1);
void offset_ccsd_lambda1Mod_7_4_1_(Integer *l_lambda1Mod_7_4_1_offset,
                                   Integer *k_lambda1Mod_7_4_1_offset,
                                   Integer *size_lambda1Mod_7_4_1);
void offset_ccsd_lambda1Mod_8_1_(Integer *l_lambda1Mod_8_1_offset,
                                 Integer *k_lambda1Mod_8_1_offset,
                                 Integer *size_lambda1Mod_8_1);
void offset_ccsd_lambda1Mod_9_1_(Integer *l_lambda1Mod_9_1_offset,
                                 Integer *k_lambda1Mod_9_1_offset,
                                 Integer *size_lambda1Mod_9_1);
void offset_ccsd_lambda1Mod_10_1_(Integer *l_lambda1Mod_10_1_offset,
                                  Integer *k_lambda1Mod_10_1_offset,
                                  Integer *size_lambda1Mod_10_1);
void offset_ccsd_lambda1Mod_11_1_(Integer *l_lambda1Mod_11_1_offset,
                                  Integer *k_lambda1Mod_11_1_offset,
                                  Integer *size_lambda1Mod_11_1);
void offset_ccsd_lambda1Mod_12_1_(Integer *l_lambda1Mod_12_1_offset,
                                  Integer *k_lambda1Mod_12_1_offset,
                                  Integer *size_lambda1Mod_12_1);
void offset_ccsd_lambda1Mod_12_2_1_(Integer *l_lambda1Mod_12_2_1_offset,
                                    Integer *k_lambda1Mod_12_2_1_offset,
                                    Integer *size_lambda1Mod_12_2_1);
void offset_ccsd_lambda1Mod_12_2_2_1_(Integer *l_lambda1Mod_12_2_2_1_offset,
                                      Integer *k_lambda1Mod_12_2_2_1_offset,
                                      Integer *size_lambda1Mod_12_2_2_1);
void offset_ccsd_lambda1Mod_12_3_1_(Integer *l_lambda1Mod_12_3_1_offset,
                                    Integer *k_lambda1Mod_12_3_1_offset,
                                    Integer *size_lambda1Mod_12_3_1);
void offset_ccsd_lambda1Mod_12_4_1_(Integer *l_lambda1Mod_12_4_1_offset,
                                    Integer *k_lambda1Mod_12_4_1_offset,
                                    Integer *size_lambda1Mod_12_4_1);
void offset_ccsd_lambda1Mod_13_1_(Integer *l_lambda1Mod_13_1_offset,
                                  Integer *k_lambda1Mod_13_1_offset,
                                  Integer *size_lambda1Mod_13_1);
void offset_ccsd_lambda1Mod_13_2_1_(Integer *l_lambda1Mod_13_2_1_offset,
                                    Integer *k_lambda1Mod_13_2_1_offset,
                                    Integer *size_lambda1Mod_13_2_1);
void offset_ccsd_lambda1Mod_14_1_(Integer *l_lambda1Mod_14_1_offset,
                                  Integer *k_lambda1Mod_14_1_offset,
                                  Integer *size_lambda1Mod_14_1);
void offset_ccsd_lambda1Mod_14_2_1_(Integer *l_lambda1Mod_14_2_1_offset,
                                    Integer *k_lambda1Mod_14_2_1_offset,
                                    Integer *size_lambda1Mod_14_2_1);
}

namespace tamm {

extern "C" {
void ccsd_lambda1Mod_cxx_(Integer *d_t_vvoo, Integer *d_f, Integer *d_i0,
                          Integer *d_y_ov, Integer *d_y_oovv, Integer *d_t_vo,
                          Integer *d_v, Integer *k_t_vvoo_offset,
                          Integer *k_f_offset, Integer *k_i0_offset,
                          Integer *k_y_ov_offset, Integer *k_y_oovv_offset,
                          Integer *k_t_vo_offset, Integer *k_v_offset) {
  static bool set_lambda1Mod = true;

  Assignment op_lambda1Mod_1;
  Assignment op_lambda1Mod_2_1;
  Assignment op_lambda1Mod_2_2_1;
  Assignment op_lambda1Mod_3_1;
  Assignment op_lambda1Mod_5_1;
  Assignment op_lambda1Mod_5_2_1;
  Assignment op_lambda1Mod_5_2_2_1;
  Assignment op_lambda1Mod_5_3_1;
  Assignment op_lambda1Mod_5_4_1;
  Assignment op_lambda1Mod_5_5_1;
  Assignment op_lambda1Mod_6_1;
  Assignment op_lambda1Mod_7_1;
  Multiplication op_lambda1Mod_2_2_2;
  Multiplication op_lambda1Mod_2_2;
  Multiplication op_lambda1Mod_2_3;
  Multiplication op_lambda1Mod_2_4;
  Multiplication op_lambda1Mod_2;
  Multiplication op_lambda1Mod_3_2;
  Multiplication op_lambda1Mod_3_3_1;
  Multiplication op_lambda1Mod_3_3;
  Multiplication op_lambda1Mod_3;
  Multiplication op_lambda1Mod_4;
  Multiplication op_lambda1Mod_5_2_2_2;
  Multiplication op_lambda1Mod_5_2_2;
  Multiplication op_lambda1Mod_5_2_3;
  Multiplication op_lambda1Mod_5_2;
  Multiplication op_lambda1Mod_5_3_2;
  Multiplication op_lambda1Mod_5_3;
  Multiplication op_lambda1Mod_5_4_2;
  Multiplication op_lambda1Mod_5_4;
  Multiplication op_lambda1Mod_5_5_2;
  Multiplication op_lambda1Mod_5_5;
  Multiplication op_lambda1Mod_5_6;
  Multiplication op_lambda1Mod_5;
  Multiplication op_lambda1Mod_6_2;
  Multiplication op_lambda1Mod_6;
  Multiplication op_lambda1Mod_7_2;
  Multiplication op_lambda1Mod_7_3_1;
  Multiplication op_lambda1Mod_7_3_2;
  Multiplication op_lambda1Mod_7_3;
  Multiplication op_lambda1Mod_7_4_1;
  Multiplication op_lambda1Mod_7_4;
  Multiplication op_lambda1Mod_7;
  Multiplication op_lambda1Mod_8_1;
  Multiplication op_lambda1Mod_8_2;
  Multiplication op_lambda1Mod_8;
  Multiplication op_lambda1Mod_9_1;
  Multiplication op_lambda1Mod_9_2;
  Multiplication op_lambda1Mod_9;
  Multiplication op_lambda1Mod_10_1;
  Multiplication op_lambda1Mod_10_2;
  Multiplication op_lambda1Mod_10;
  Multiplication op_lambda1Mod_11_1;
  Multiplication op_lambda1Mod_11;
  Multiplication op_lambda1Mod_12_1;
  Multiplication op_lambda1Mod_12_2_1;
  Multiplication op_lambda1Mod_12_2_2_1;
  Multiplication op_lambda1Mod_12_2_2;
  Multiplication op_lambda1Mod_12_2;
  Multiplication op_lambda1Mod_12_3_1;
  Multiplication op_lambda1Mod_12_3;
  Multiplication op_lambda1Mod_12_4_1;
  Multiplication op_lambda1Mod_12_4;
  Multiplication op_lambda1Mod_12;
  Multiplication op_lambda1Mod_13_1;
  Multiplication op_lambda1Mod_13_2_1;
  Multiplication op_lambda1Mod_13_2;
  Multiplication op_lambda1Mod_13;
  Multiplication op_lambda1Mod_14_1;
  Multiplication op_lambda1Mod_14_2_1;
  Multiplication op_lambda1Mod_14_2;
  Multiplication op_lambda1Mod_14;

  DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
  static Equations eqs;

  if (set_lambda1Mod) {
    ccsd_lambda1Mod_equations(&eqs);
    set_lambda1Mod = false;
  }

  std::map<std::string, tamm::Tensor> tensors;
  std::vector<Operation> ops;
  tensors_and_ops(&eqs, &tensors, &ops);

  Tensor *i0 = &tensors["i0"];
  Tensor *f = &tensors["f"];
  Tensor *y_ov = &tensors["y_ov"];
  Tensor *t_vo = &tensors["t_vo"];
  Tensor *v = &tensors["v"];
  Tensor *t_vvoo = &tensors["t_vvoo"];
  Tensor *y_oovv = &tensors["y_oovv"];
  Tensor *lambda1Mod_9_1 = &tensors["lambda1Mod_9_1"];
  Tensor *lambda1Mod_14_2_1 = &tensors["lambda1Mod_14_2_1"];
  Tensor *lambda1Mod_8_1 = &tensors["lambda1Mod_8_1"];
  Tensor *lambda1Mod_2_2_1 = &tensors["lambda1Mod_2_2_1"];
  Tensor *lambda1Mod_12_4_1 = &tensors["lambda1Mod_12_4_1"];
  Tensor *lambda1Mod_7_1 = &tensors["lambda1Mod_7_1"];
  Tensor *lambda1Mod_12_2_1 = &tensors["lambda1Mod_12_2_1"];
  Tensor *lambda1Mod_12_2_2_1 = &tensors["lambda1Mod_12_2_2_1"];
  Tensor *lambda1Mod_5_2_1 = &tensors["lambda1Mod_5_2_1"];
  Tensor *lambda1Mod_12_3_1 = &tensors["lambda1Mod_12_3_1"];
  Tensor *lambda1Mod_12_1 = &tensors["lambda1Mod_12_1"];
  Tensor *lambda1Mod_5_4_1 = &tensors["lambda1Mod_5_4_1"];
  Tensor *lambda1Mod_13_1 = &tensors["lambda1Mod_13_1"];
  Tensor *lambda1Mod_5_5_1 = &tensors["lambda1Mod_5_5_1"];
  Tensor *lambda1Mod_10_1 = &tensors["lambda1Mod_10_1"];
  Tensor *lambda1Mod_11_1 = &tensors["lambda1Mod_11_1"];
  Tensor *lambda1Mod_3_3_1 = &tensors["lambda1Mod_3_3_1"];
  Tensor *lambda1Mod_6_1 = &tensors["lambda1Mod_6_1"];
  Tensor *lambda1Mod_5_1 = &tensors["lambda1Mod_5_1"];
  Tensor *lambda1Mod_14_1 = &tensors["lambda1Mod_14_1"];
  Tensor *lambda1Mod_5_3_1 = &tensors["lambda1Mod_5_3_1"];
  Tensor *lambda1Mod_3_1 = &tensors["lambda1Mod_3_1"];
  Tensor *lambda1Mod_13_2_1 = &tensors["lambda1Mod_13_2_1"];
  Tensor *lambda1Mod_5_2_2_1 = &tensors["lambda1Mod_5_2_2_1"];
  Tensor *lambda1Mod_7_4_1 = &tensors["lambda1Mod_7_4_1"];
  Tensor *lambda1Mod_2_1 = &tensors["lambda1Mod_2_1"];
  Tensor *lambda1Mod_7_3_1 = &tensors["lambda1Mod_7_3_1"];

  /* ----- Insert attach code ------ */
  v->set_dist(idist);
  i0->attach(*k_i0_offset, 0, *d_i0);
  f->attach(*k_f_offset, 0, *d_f);
  v->attach(*k_v_offset, 0, *d_v);

#if 1
  schedule_levels(&tensors, &ops);
#else
  op_lambda1Mod_1 = ops[0].add;
  op_lambda1Mod_2_1 = ops[1].add;
  op_lambda1Mod_2_2_1 = ops[2].add;
  op_lambda1Mod_2_2_2 = ops[3].mult;
  op_lambda1Mod_2_2 = ops[4].mult;
  op_lambda1Mod_2_3 = ops[5].mult;
  op_lambda1Mod_2_4 = ops[6].mult;
  op_lambda1Mod_2 = ops[7].mult;
  op_lambda1Mod_3_1 = ops[8].add;
  op_lambda1Mod_3_2 = ops[9].mult;
  op_lambda1Mod_3_3_1 = ops[10].mult;
  op_lambda1Mod_3_3 = ops[11].mult;
  op_lambda1Mod_3 = ops[12].mult;
  op_lambda1Mod_4 = ops[13].mult;
  op_lambda1Mod_5_1 = ops[14].add;
  op_lambda1Mod_5_2_1 = ops[15].add;
  op_lambda1Mod_5_2_2_1 = ops[16].add;
  op_lambda1Mod_5_2_2_2 = ops[17].mult;
  op_lambda1Mod_5_2_2 = ops[18].mult;
  op_lambda1Mod_5_2_3 = ops[19].mult;
  op_lambda1Mod_5_2 = ops[20].mult;
  op_lambda1Mod_5_3_1 = ops[21].add;
  op_lambda1Mod_5_3_2 = ops[22].mult;
  op_lambda1Mod_5_3 = ops[23].mult;
  op_lambda1Mod_5_4_1 = ops[24].add;
  op_lambda1Mod_5_4_2 = ops[25].mult;
  op_lambda1Mod_5_4 = ops[26].mult;
  op_lambda1Mod_5_5_1 = ops[27].add;
  op_lambda1Mod_5_5_2 = ops[28].mult;
  op_lambda1Mod_5_5 = ops[29].mult;
  op_lambda1Mod_5_6 = ops[30].mult;
  op_lambda1Mod_5 = ops[31].mult;
  op_lambda1Mod_6_1 = ops[32].add;
  op_lambda1Mod_6_2 = ops[33].mult;
  op_lambda1Mod_6 = ops[34].mult;
  op_lambda1Mod_7_1 = ops[35].add;
  op_lambda1Mod_7_2 = ops[36].mult;
  op_lambda1Mod_7_3_1 = ops[37].mult;
  op_lambda1Mod_7_3_2 = ops[38].mult;
  op_lambda1Mod_7_3 = ops[39].mult;
  op_lambda1Mod_7_4_1 = ops[40].mult;
  op_lambda1Mod_7_4 = ops[41].mult;
  op_lambda1Mod_7 = ops[42].mult;
  op_lambda1Mod_8_1 = ops[43].mult;
  op_lambda1Mod_8_2 = ops[44].mult;
  op_lambda1Mod_8 = ops[45].mult;
  op_lambda1Mod_9_1 = ops[46].mult;
  op_lambda1Mod_9_2 = ops[47].mult;
  op_lambda1Mod_9 = ops[48].mult;
  op_lambda1Mod_10_1 = ops[49].mult;
  op_lambda1Mod_10_2 = ops[50].mult;
  op_lambda1Mod_10 = ops[51].mult;
  op_lambda1Mod_11_1 = ops[52].mult;
  op_lambda1Mod_11 = ops[53].mult;
  op_lambda1Mod_12_1 = ops[54].mult;
  op_lambda1Mod_12_2_1 = ops[55].mult;
  op_lambda1Mod_12_2_2_1 = ops[56].mult;
  op_lambda1Mod_12_2_2 = ops[57].mult;
  op_lambda1Mod_12_2 = ops[58].mult;
  op_lambda1Mod_12_3_1 = ops[59].mult;
  op_lambda1Mod_12_3 = ops[60].mult;
  op_lambda1Mod_12_4_1 = ops[61].mult;
  op_lambda1Mod_12_4 = ops[62].mult;
  op_lambda1Mod_12 = ops[63].mult;
  op_lambda1Mod_13_1 = ops[64].mult;
  op_lambda1Mod_13_2_1 = ops[65].mult;
  op_lambda1Mod_13_2 = ops[66].mult;
  op_lambda1Mod_13 = ops[67].mult;
  op_lambda1Mod_14_1 = ops[68].mult;
  op_lambda1Mod_14_2_1 = ops[69].mult;
  op_lambda1Mod_14_2 = ops[70].mult;
  op_lambda1Mod_14 = ops[71].mult;

  CorFortran(1, &op_lambda1Mod_1, ccsd_lambda1Mod_1_);
  CorFortran(1, lambda1Mod_2_1, offset_ccsd_lambda1Mod_2_1_);
  CorFortran(1, &op_lambda1Mod_2_1, ccsd_lambda1Mod_2_1_);
  CorFortran(1, lambda1Mod_2_2_1, offset_ccsd_lambda1Mod_2_2_1_);
  CorFortran(1, &op_lambda1Mod_2_2_1, ccsd_lambda1Mod_2_2_1_);
  CorFortran(1, &op_lambda1Mod_2_2_2, ccsd_lambda1Mod_2_2_2_);
  CorFortran(1, &op_lambda1Mod_2_2, ccsd_lambda1Mod_2_2_);
  destroy(lambda1Mod_2_2_1);
  CorFortran(1, &op_lambda1Mod_2_3, ccsd_lambda1Mod_2_3_);
  CorFortran(1, &op_lambda1Mod_2_4, ccsd_lambda1Mod_2_4_);
  CorFortran(1, &op_lambda1Mod_2, ccsd_lambda1Mod_2_);
  destroy(lambda1Mod_2_1);
  CorFortran(1, lambda1Mod_3_1, offset_ccsd_lambda1Mod_3_1_);
  CorFortran(1, &op_lambda1Mod_3_1, ccsd_lambda1Mod_3_1_);
  CorFortran(1, &op_lambda1Mod_3_2, ccsd_lambda1Mod_3_2_);
  CorFortran(1, lambda1Mod_3_3_1, offset_ccsd_lambda1Mod_3_3_1_);
  CorFortran(1, &op_lambda1Mod_3_3_1, ccsd_lambda1Mod_3_3_1_);
  CorFortran(1, &op_lambda1Mod_3_3, ccsd_lambda1Mod_3_3_);
  destroy(lambda1Mod_3_3_1);
  CorFortran(1, &op_lambda1Mod_3, ccsd_lambda1Mod_3_);
  destroy(lambda1Mod_3_1);
  CorFortran(1, &op_lambda1Mod_4, ccsd_lambda1Mod_4_);
  CorFortran(1, lambda1Mod_5_1, offset_ccsd_lambda1Mod_5_1_);
  CorFortran(1, &op_lambda1Mod_5_1, ccsd_lambda1Mod_5_1_);
  CorFortran(1, lambda1Mod_5_2_1, offset_ccsd_lambda1Mod_5_2_1_);
  CorFortran(1, &op_lambda1Mod_5_2_1, ccsd_lambda1Mod_5_2_1_);
  CorFortran(1, lambda1Mod_5_2_2_1, offset_ccsd_lambda1Mod_5_2_2_1_);
  CorFortran(1, &op_lambda1Mod_5_2_2_1, ccsd_lambda1Mod_5_2_2_1_);
  CorFortran(1, &op_lambda1Mod_5_2_2_2, ccsd_lambda1Mod_5_2_2_2_);
  CorFortran(1, &op_lambda1Mod_5_2_2, ccsd_lambda1Mod_5_2_2_);
  destroy(lambda1Mod_5_2_2_1);
  CorFortran(1, &op_lambda1Mod_5_2_3, ccsd_lambda1Mod_5_2_3_);
  CorFortran(1, &op_lambda1Mod_5_2, ccsd_lambda1Mod_5_2_);
  destroy(lambda1Mod_5_2_1);
  CorFortran(1, lambda1Mod_5_3_1, offset_ccsd_lambda1Mod_5_3_1_);
  CorFortran(1, &op_lambda1Mod_5_3_1, ccsd_lambda1Mod_5_3_1_);
  CorFortran(1, &op_lambda1Mod_5_3_2, ccsd_lambda1Mod_5_3_2_);
  CorFortran(1, &op_lambda1Mod_5_3, ccsd_lambda1Mod_5_3_);
  destroy(lambda1Mod_5_3_1);
  CorFortran(1, lambda1Mod_5_4_1, offset_ccsd_lambda1Mod_5_4_1_);
  CorFortran(1, &op_lambda1Mod_5_4_1, ccsd_lambda1Mod_5_4_1_);
  CorFortran(1, &op_lambda1Mod_5_4_2, ccsd_lambda1Mod_5_4_2_);
  CorFortran(1, &op_lambda1Mod_5_4, ccsd_lambda1Mod_5_4_);
  destroy(lambda1Mod_5_4_1);
  CorFortran(1, lambda1Mod_5_5_1, offset_ccsd_lambda1Mod_5_5_1_);
  CorFortran(1, &op_lambda1Mod_5_5_1, ccsd_lambda1Mod_5_5_1_);
  CorFortran(1, &op_lambda1Mod_5_5_2, ccsd_lambda1Mod_5_5_2_);
  CorFortran(1, &op_lambda1Mod_5_5, ccsd_lambda1Mod_5_5_);
  destroy(lambda1Mod_5_5_1);
  CorFortran(1, &op_lambda1Mod_5_6, ccsd_lambda1Mod_5_6_);
  CorFortran(1, &op_lambda1Mod_5, ccsd_lambda1Mod_5_);
  destroy(lambda1Mod_5_1);
  CorFortran(1, lambda1Mod_6_1, offset_ccsd_lambda1Mod_6_1_);
  CorFortran(1, &op_lambda1Mod_6_1, ccsd_lambda1Mod_6_1_);
  CorFortran(1, &op_lambda1Mod_6_2, ccsd_lambda1Mod_6_2_);
  CorFortran(1, &op_lambda1Mod_6, ccsd_lambda1Mod_6_);
  destroy(lambda1Mod_6_1);
  CorFortran(1, lambda1Mod_7_1, offset_ccsd_lambda1Mod_7_1_);
  CorFortran(1, &op_lambda1Mod_7_1, ccsd_lambda1Mod_7_1_);
  CorFortran(1, &op_lambda1Mod_7_2, ccsd_lambda1Mod_7_2_);
  CorFortran(1, lambda1Mod_7_3_1, offset_ccsd_lambda1Mod_7_3_1_);
  CorFortran(1, &op_lambda1Mod_7_3_1, ccsd_lambda1Mod_7_3_1_);
  CorFortran(1, &op_lambda1Mod_7_3_2, ccsd_lambda1Mod_7_3_2_);
  CorFortran(1, &op_lambda1Mod_7_3, ccsd_lambda1Mod_7_3_);
  destroy(lambda1Mod_7_3_1);
  CorFortran(1, lambda1Mod_7_4_1, offset_ccsd_lambda1Mod_7_4_1_);
  CorFortran(1, &op_lambda1Mod_7_4_1, ccsd_lambda1Mod_7_4_1_);
  CorFortran(1, &op_lambda1Mod_7_4, ccsd_lambda1Mod_7_4_);
  destroy(lambda1Mod_7_4_1);
  CorFortran(1, &op_lambda1Mod_7, ccsd_lambda1Mod_7_);
  destroy(lambda1Mod_7_1);
  CorFortran(1, lambda1Mod_8_1, offset_ccsd_lambda1Mod_8_1_);
  CorFortran(1, &op_lambda1Mod_8_1, ccsd_lambda1Mod_8_1_);
  CorFortran(1, &op_lambda1Mod_8_2, ccsd_lambda1Mod_8_2_);
  CorFortran(1, &op_lambda1Mod_8, ccsd_lambda1Mod_8_);
  destroy(lambda1Mod_8_1);
  CorFortran(1, lambda1Mod_9_1, offset_ccsd_lambda1Mod_9_1_);
  CorFortran(1, &op_lambda1Mod_9_1, ccsd_lambda1Mod_9_1_);
  CorFortran(1, &op_lambda1Mod_9_2, ccsd_lambda1Mod_9_2_);
  CorFortran(1, &op_lambda1Mod_9, ccsd_lambda1Mod_9_);
  destroy(lambda1Mod_9_1);
  CorFortran(1, lambda1Mod_10_1, offset_ccsd_lambda1Mod_10_1_);
  CorFortran(1, &op_lambda1Mod_10_1, ccsd_lambda1Mod_10_1_);
  CorFortran(1, &op_lambda1Mod_10_2, ccsd_lambda1Mod_10_2_);
  CorFortran(1, &op_lambda1Mod_10, ccsd_lambda1Mod_10_);
  destroy(lambda1Mod_10_1);
  CorFortran(1, lambda1Mod_11_1, offset_ccsd_lambda1Mod_11_1_);
  CorFortran(1, &op_lambda1Mod_11_1, ccsd_lambda1Mod_11_1_);
  CorFortran(1, &op_lambda1Mod_11, ccsd_lambda1Mod_11_);
  destroy(lambda1Mod_11_1);
  CorFortran(1, lambda1Mod_12_1, offset_ccsd_lambda1Mod_12_1_);
  CorFortran(1, &op_lambda1Mod_12_1, ccsd_lambda1Mod_12_1_);
  CorFortran(1, lambda1Mod_12_2_1, offset_ccsd_lambda1Mod_12_2_1_);
  CorFortran(1, &op_lambda1Mod_12_2_1, ccsd_lambda1Mod_12_2_1_);
  CorFortran(1, lambda1Mod_12_2_2_1, offset_ccsd_lambda1Mod_12_2_2_1_);
  CorFortran(1, &op_lambda1Mod_12_2_2_1, ccsd_lambda1Mod_12_2_2_1_);
  CorFortran(1, &op_lambda1Mod_12_2_2, ccsd_lambda1Mod_12_2_2_);
  destroy(lambda1Mod_12_2_2_1);
  CorFortran(1, &op_lambda1Mod_12_2, ccsd_lambda1Mod_12_2_);
  destroy(lambda1Mod_12_2_1);
  CorFortran(1, lambda1Mod_12_3_1, offset_ccsd_lambda1Mod_12_3_1_);
  CorFortran(1, &op_lambda1Mod_12_3_1, ccsd_lambda1Mod_12_3_1_);
  CorFortran(1, &op_lambda1Mod_12_3, ccsd_lambda1Mod_12_3_);
  destroy(lambda1Mod_12_3_1);
  CorFortran(1, lambda1Mod_12_4_1, offset_ccsd_lambda1Mod_12_4_1_);
  CorFortran(1, &op_lambda1Mod_12_4_1, ccsd_lambda1Mod_12_4_1_);
  CorFortran(1, &op_lambda1Mod_12_4, ccsd_lambda1Mod_12_4_);
  destroy(lambda1Mod_12_4_1);
  CorFortran(1, &op_lambda1Mod_12, ccsd_lambda1Mod_12_);
  destroy(lambda1Mod_12_1);
  CorFortran(1, lambda1Mod_13_1, offset_ccsd_lambda1Mod_13_1_);
  CorFortran(1, &op_lambda1Mod_13_1, ccsd_lambda1Mod_13_1_);
  CorFortran(1, lambda1Mod_13_2_1, offset_ccsd_lambda1Mod_13_2_1_);
  CorFortran(1, &op_lambda1Mod_13_2_1, ccsd_lambda1Mod_13_2_1_);
  CorFortran(1, &op_lambda1Mod_13_2, ccsd_lambda1Mod_13_2_);
  destroy(lambda1Mod_13_2_1);
  CorFortran(1, &op_lambda1Mod_13, ccsd_lambda1Mod_13_);
  destroy(lambda1Mod_13_1);
  CorFortran(1, lambda1Mod_14_1, offset_ccsd_lambda1Mod_14_1_);
  CorFortran(1, &op_lambda1Mod_14_1, ccsd_lambda1Mod_14_1_);
  CorFortran(1, lambda1Mod_14_2_1, offset_ccsd_lambda1Mod_14_2_1_);
  CorFortran(1, &op_lambda1Mod_14_2_1, ccsd_lambda1Mod_14_2_1_);
  CorFortran(1, &op_lambda1Mod_14_2, ccsd_lambda1Mod_14_2_);
  destroy(lambda1Mod_14_2_1);
  CorFortran(1, &op_lambda1Mod_14, ccsd_lambda1Mod_14_);
  destroy(lambda1Mod_14_1);
#endif  // if 1

  /* ----- Insert detach code ------ */
  f->detach();
  i0->detach();
  v->detach();
}
}  // extern C
};  // namespace tamm
