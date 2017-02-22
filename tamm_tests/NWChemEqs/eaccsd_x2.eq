x2 {

index h1,h2,h3,h4,h5,h6,h7,h8,h9,h10 = O;
index p1,p2,p3,p4,p5,p6,p7,p8,p9,p10 = V;

array i0[V,V][O];
array x_v[V][]: irrep_x;
array v[N,N][N,N]: irrep_v;
array x_vvo[V,V][O]: irrep_x;
array f[N][N]: irrep_f;
array t_vo[V][O]: irrep_t;
array t_vvoo[V,V][O,O]: irrep_t;
array x2_10_1[V,O][V];
array x2_2_1[O][O];
array x2_6_1[O,V][O];
array x2_6_5_1[O,O][O];
array x2_3_1[V][V];
array x2_6_7_1[O,O][V];
array x2_8_1[O][];
array x2_7_1[V,V][V];
array x2_6_2_1[O][V];
array x2_8_1_1[O][V];
array x2_9_1[O,O][O];
array x2_6_6_1[V,O][V];
array x2_9_3_1[O,O][V];
array x2_4_1[O,V][O,V];
array x2_6_5_3_1[O,O][V];
array x2_6_3_1[O,O][O,V];
array x2_2_2_1[O][V];

x2_1:       i0[p3,p4,h1] += 1 * x_v[p5] * v[p3,p4,h1,p5];
x2_2_1:     x2_2_1[h8,h1] += 1 * f[h8,h1];
x2_2_2_1:   x2_2_2_1[h8,p9] += 1 * f[h8,p9];
x2_2_2_2:   x2_2_2_1[h8,p9] += 1 * t_vo[p6,h7] * v[h7,h8,p6,p9];
x2_2_2:     x2_2_1[h8,h1] += 1 * t_vo[p9,h1] * x2_2_2_1[h8,p9];
x2_2_3:     x2_2_1[h8,h1] += -1 * t_vo[p5,h6] * v[h6,h8,h1,p5];
x2_2_4:     x2_2_1[h8,h1] += -1/2 * t_vvoo[p5,p6,h1,h7] * v[h7,h8,p5,p6];
x2_2:       i0[p3,p4,h1] += -1 * x_vvo[p3,p4,h8] * x2_2_1[h8,h1];
x2_3_1:     x2_3_1[p3,p8] += 1 * f[p3,p8];
x2_3_2:     x2_3_1[p3,p8] += 1 * t_vo[p5,h6] * v[h6,p3,p5,p8];
x2_3_3:     x2_3_1[p3,p8] += 1/2 * t_vvoo[p3,p5,h6,h7] * v[h6,h7,p5,p8];
x2_3:       i0[p3,p4,h1] += 1 * x_vvo[p3,p8,h1] * x2_3_1[p4,p8];
x2_4_1:     x2_4_1[h7,p3,h1,p8] += 1 * v[h7,p3,h1,p8];
x2_4_2:     x2_4_1[h7,p3,h1,p8] += 1 * t_vo[p5,h1] * v[h7,p3,p5,p8];
x2_4:       i0[p3,p4,h1] += -1 * x_vvo[p3,p8,h7] * x2_4_1[h7,p4,h1,p8];
x2_5:       i0[p3,p4,h1] += 1/2 * x_vvo[p5,p6,h1] * v[p3,p4,p5,p6];
x2_6_1:     x2_6_1[h9,p3,h2] += 1/2 * x_v[p6] * v[h9,p3,h2,p6];
x2_6_2_1:   x2_6_2_1[h9,p5] += 1 * f[h9,p5];
x2_6_2_2:   x2_6_2_1[h9,p5] += -1 * t_vo[p6,h7] * v[h7,h9,p5,p6];
x2_6_2:     x2_6_1[h9,p3,h1] += -1/2 * x_vvo[p3,p5,h1] * x2_6_2_1[h9,p5];
x2_6_3_1:   x2_6_3_1[h8,h9,h1,p10] += 1 * v[h8,h9,h1,p10];
x2_6_3_2:   x2_6_3_1[h8,h9,h1,p10] += 1 * t_vo[p5,h1] * v[h8,h9,p5,p10];
x2_6_3:     x2_6_1[h9,p3,h2] += 1/2 * x_vvo[p3,p10,h8] * x2_6_3_1[h8,h9,h2,p10];
x2_6_4:     x2_6_1[h9,p3,h1] += 1/4 * x_vvo[p6,p7,h1] * v[h9,p3,p6,p7];
x2_6_5_1:   x2_6_5_1[h9,h10,h2] += 1/2 * x_v[p7] * v[h9,h10,h2,p7];
x2_6_5_2:   x2_6_5_1[h9,h10,h1] += 1/4 * x_vvo[p7,p8,h1] * v[h9,h10,p7,p8];
x2_6_5_3_1: x2_6_5_3_1[h9,h10,p5] += 1 * x_v[p8] * v[h9,h10,p5,p8];
x2_6_5_3:   x2_6_5_1[h9,h10,h1] += 1/2 * t_vo[p5,h1] * x2_6_5_3_1[h9,h10,p5];
x2_6_5:     x2_6_1[h9,p3,h1] += -1/2 * t_vo[p3,h10] * x2_6_5_1[h9,h10,h1];
x2_6_6_1:   x2_6_6_1[p3,h9,p5] += 1 * x_v[p7] * v[h9,p3,p5,p7];
x2_6_6:     x2_6_1[h9,p3,h1] += 1/2 * t_vo[p5,h1] * x2_6_6_1[p3,h9,p5];
x2_6_7_1:   x2_6_7_1[h6,h9,p5] += 1 * x_v[p8] * v[h6,h9,p5,p8];
x2_6_7:     x2_6_1[h9,p3,h1] += -1/2 * t_vvoo[p3,p5,h1,h6] * x2_6_7_1[h6,h9,p5];
x2_6:       i0[p3,p4,h1] += -2 * t_vo[p3,h9] * x2_6_1[h9,p4,h1];
x2_7_1:     x2_7_1[p3,p4,p5] += 1 * x_v[p6] * v[p3,p4,p5,p6];
x2_7:       i0[p3,p4,h1] += 1 * t_vo[p5,h1] * x2_7_1[p3,p4,p5];
x2_8_1_1:   x2_8_1_1[h5,p9] += 1 * f[h5,p9];
x2_8_1_2:   x2_8_1_1[h5,p9] += -1 * t_vo[p6,h7] * v[h5,h7,p6,p9];
x2_8_1:     x2_8_1[h5] += 1 * x_v[p9] * x2_8_1_1[h5,p9];
x2_8_2:     x2_8_1[h5] += -1/2 * x_vvo[p7,p8,h6] * v[h5,h6,p7,p8];
x2_8:       i0[p3,p4,h1] += -1 * t_vvoo[p3,p4,h1,h5] * x2_8_1[h5];
x2_9_1:     x2_9_1[h5,h6,h2] += 1/2 * x_v[p7] * v[h5,h6,h2,p7];
x2_9_2:     x2_9_1[h5,h6,h1] += 1/4 * x_vvo[p7,p8,h1] * v[h5,h6,p7,p8];
x2_9_3_1:   x2_9_3_1[h5,h6,p7] += 1 * x_v[p8] * v[h5,h6,p7,p8];
x2_9_3:     x2_9_1[h5,h6,h1] += 1/2 * t_vo[p7,h1] * x2_9_3_1[h5,h6,p7];
x2_9:       i0[p3,p4,h1] += 1 * t_vvoo[p3,p4,h5,h6] * x2_9_1[h5,h6,h1];
x2_10_1:    x2_10_1[p3,h6,p5] += 1 * x_v[p7] * v[h6,p3,p5,p7];
x2_10_2:    x2_10_1[p3,h6,p5] += -1 * x_vvo[p3,p8,h7] * v[h6,h7,p5,p8];
x2_10:      i0[p3,p4,h1] += 1 * t_vvoo[p3,p5,h1,h6] * x2_10_1[p4,h6,p5];

}
