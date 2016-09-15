x1 {

index h1,h2,h3,h4,h5,h6,h7,h8 = O;
index p1,p2,p3,p4,p5,p6,p7 = V;

array i0[][O];
array x_o[][O]: irrep_x;
array f[N][N]: irrep_f;
array t_vo[V][O]: irrep_t;
array v[N,N][N,N]: irrep_v;
array t_vvoo[V,V][O,O]: irrep_t;
array x_voo[V][O,O]: irrep_x;
array x1_2_1[O][V];
array x1_1_2_1[O][V];
array x1_3_1[O,O][O,V];
array x1_1_1[O][O];

x1_1_1:     x1_1_1[h6,h1] += 1 * f[h6,h1];
x1_1_2_1:   x1_1_2_1[h6,p7] += 1 * f[h6,p7];
x1_1_2_2:   x1_1_2_1[h6,p7] += 1 * t_vo[p4,h5] * v[h5,h6,p4,p7];
x1_1_2:     x1_1_1[h6,h1] += 1 * t_vo[p7,h1] * x1_1_2_1[h6,p7];
x1_1_3:     x1_1_1[h6,h1] += -1 * t_vo[p3,h4] * v[h4,h6,h1,p3];
x1_1_4:     x1_1_1[h6,h1] += -1/2 * t_vvoo[p3,p4,h1,h5] * v[h5,h6,p3,p4];
x1_1:       i0[h1] += -1 * x_o[h6] * x1_1_1[h6,h1];
x1_2_1:     x1_2_1[h6,p7] += 1 * f[h6,p7];
x1_2_2:     x1_2_1[h6,p7] += 1 * t_vo[p3,h4] * v[h4,h6,p3,p7];
x1_2:       i0[h1] += -1 * x_voo[p7,h1,h6] * x1_2_1[h6,p7];
x1_3_1:     x1_3_1[h6,h8,h1,p7] += 1 * v[h6,h8,h1,p7];
x1_3_2:     x1_3_1[h6,h8,h1,p7] += 1 * t_vo[p3,h1] * v[h6,h8,p3,p7];
x1_3:       i0[h1] += 1/2 * x_voo[p7,h6,h8] * x1_3_1[h6,h8,h1,p7];

}
