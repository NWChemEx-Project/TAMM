#include "ccse_tensors.hpp"
#include <tamm/tamm_git.hpp>

using CCEType = double;
TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;

Tensor<CCEType> _a01V, _a02V, _a007V;
Tensor<CCEType> _a01V_sp, _a02V_sp, _a007V_sp;

Tensor<CCEType> _a01_sp, _a02_sp, _a03_sp, _a04_sp, _a05_sp, _a06_sp, _a001_sp, _a004_sp, _a006_sp,
  _a008_sp, _a009_sp, _a017_sp, _a019_sp, _a020_sp, _a021_sp, _a022_sp;

Tensor<CCEType> i0_temp, t2_aaaa_temp; // CS only

Char2TISMap index_to_sub_string{{{'I', "occ_alpha"},
                                 {'J', "occ_alpha"},
                                 {'K', "occ_alpha"},
                                 {'L', "occ_alpha"},
                                 {'i', "occ_beta"},
                                 {'j', "occ_beta"},
                                 {'k', "occ_beta"},
                                 {'l', "occ_beta"},
                                 {'A', "virt_alpha"},
                                 {'B', "virt_alpha"},
                                 {'C', "virt_alpha"},
                                 {'D', "virt_alpha"},
                                 {'a', "virt_beta"},
                                 {'b', "virt_beta"},
                                 {'c', "virt_beta"},
                                 {'d', "virt_beta"},
                                 {'X', "all"}}};

auto generate_spin_check(const TiledIndexSpaceVec&        t_spaces,
                         const std::vector<SpinPosition>& spin_mask) {
  EXPECTS(t_spaces.size() == spin_mask.size());

  auto is_non_zero_spin = [t_spaces, spin_mask](const IndexVector& blockid) -> bool {
    Spin upper_total = 0, lower_total = 0, other_total = 0;
    for(size_t i = 0; i < blockid.size(); i++) {
      const auto& tis = t_spaces[i];
      if(spin_mask[i] == SpinPosition::upper) { upper_total += tis.spin(blockid[i]); }
      else if(spin_mask[i] == SpinPosition::lower) { lower_total += tis.spin(blockid[i]); }
      else { other_total += tis.spin(blockid[i]); }
    }

    return (upper_total == lower_total);
  };

  return is_non_zero_spin;
}

template<typename T>
std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>>
setupTensors_cs(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1) {
  auto rank = ec.pg().rank();

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();

  TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;
  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(oatiles, otiles)};
  v_beta  = {MO("virt"), range(vatiles, vtiles)};

  std::vector<T> p_evl_sorted = tamm::diagonal(d_f1);

  // Tensor<T> d_r1{{v_alpha, o_alpha}, {1, 1}};
  // Tensor<T> d_r2{{v_alpha, v_beta, o_alpha, o_beta}, {2, 2}};

  // non_zero check
  auto non_zero_check_Va_Oa =
    generate_spin_check({v_alpha, o_alpha}, {SpinPosition::lower, SpinPosition::upper});
  auto non_zero_check_Va_Vb_Oa_Ob = generate_spin_check(
    {v_alpha, v_beta, o_alpha, o_beta},
    {SpinPosition::lower, SpinPosition::lower, SpinPosition::upper, SpinPosition::upper});

  TensorInfo block_info_Va_Oa{{v_alpha, o_alpha}, non_zero_check_Va_Oa};
  TensorInfo block_info_Va_Vb_Oa_Ob{{v_alpha, v_beta, o_alpha, o_beta}, non_zero_check_Va_Vb_Oa_Ob};

  // Block Sparse Tensor
  Tensor<T> d_r1{{v_alpha, o_alpha}, block_info_Va_Oa};
  Tensor<T> d_r2{{v_alpha, v_beta, o_alpha, o_beta}, block_info_Va_Vb_Oa_Ob};

  Tensor<T>::allocate(&ec, d_r1, d_r2);

  // Tensor<T> d_t1{{v_alpha, o_alpha}, {1, 1}};
  // Tensor<T> d_t2{{v_alpha, v_beta, o_alpha, o_beta}, {2, 2}};

  // Block Sparse Tensor
  Tensor<T> d_t1{{v_alpha, o_alpha}, block_info_Va_Oa};
  Tensor<T> d_t2{{v_alpha, v_beta, o_alpha, o_beta}, block_info_Va_Vb_Oa_Ob};

  Tensor<T>::allocate(&ec, d_t1, d_t2);

  return std::make_tuple(p_evl_sorted, d_t1, d_t2, d_r1, d_r2);
}

template<typename T>
void ccsd_e_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& de,
               const Tensor<T>& t1_aa, const Tensor<T>& t2_abab, const Tensor<T>& t2_aaaa,
               const Tensor<T>& f1, const Tensor<T>& chol3d) {
  auto [cind] = CI.labels<1>("all");

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob]        = o_beta.labels<1>("all");

  // clang-format off
  sch
    (t2_aaaa_temp()=0)
    .exact_copy(t2_aaaa(p1_va, p2_va, h1_oa, h2_oa), t2_abab(p1_va, p1_vb, h1_oa, h1_ob), true)
    (t2_aaaa_temp() = t2_aaaa(),
    "t2_aaaa_temp() = t2_aaaa()")
    (t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa),
    "t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa)")
    (t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa),
    "t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa)")

    (_a01V(cind) = t1_aa(p1_va, h1_oa) * chol3d(h1_oa, p1_va, cind),
    "_a01V(cind) = t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (_a02_sp(h1_oa, h2_oa, cind)    = t1_aa(p1_va, h1_oa) * chol3d(h2_oa, p1_va, cind),
    "_a02_sp(h1_oa, h2_oa, cind)    = t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")
    (_a03_sp(h2_oa, p2_va, cind) = t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa) * chol3d(h1_oa, p1_va, cind),
    "_a03_sp(h2_oa, p2_va, cind) = t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (de()  =  2.0 * _a01V() * _a01V(),
    "de()  =  2.0 * _a01V() * _a01V()")
    (de() += -1.0 * _a02_sp(h1_oa, h2_oa, cind) * _a02_sp(h2_oa, h1_oa, cind),
    "de() += -1.0 * _a02_sp(h1_oa, h2_oa, cind) * _a02_sp(h2_oa, h1_oa, cind)")
    (de() +=  1.0 * _a03_sp(h1_oa, p1_va, cind) * chol3d(h1_oa, p1_va, cind),
    "de() +=  1.0 * _a03_sp(h1_oa, p1_va, cind) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (de() +=  2.0 * t1_aa(p1_va, h1_oa) * f1(h1_oa, p1_va),
    "de() +=  2.0 * t1_aa(p1_va, h1_oa) * f1(h1_oa, p1_va)") // NEW TERM
    ;
  // clang-format on
}

template<typename T>
void ccsd_t1_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                Tensor<T>& i0_aa, const Tensor<T>& t1_aa, const Tensor<T>& t2_abab,
                const Tensor<T>& f1, const Tensor<T>& chol3d) {
  auto [cind] = CI.labels<1>("all");
  auto [p2]   = MO.labels<1>("virt");
  auto [h1]   = MO.labels<1>("occ");

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob]        = o_beta.labels<1>("all");

  // clang-format off
  sch
    (i0_aa(p2_va, h1_oa)             =  1.0 * f1(h1_oa, p2_va),
    "i0_aa(p2_va, h1_oa)             =  1.0 * f1(h1_oa, p2_va)")
    (_a01_sp(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d(h2_oa, p1_va, cind),
    "_a01_sp(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d(h2_oa, p1_va, cind)")                 // ovm
    (_a02V(cind)                     =  2.0 * t1_aa(p1_va, h1_oa) * chol3d(h1_oa, p1_va, cind),
    "_a02V(cind)                     =  2.0 * t1_aa(p1_va, h1_oa) * chol3d(h1_oa, p1_va, cind)")                 // ovm
    // (_a02V(cind)                  =  2.0 * _a01_sp(h1_oa, h1_oa, cind))
    (_a05_sp(h2_oa, p1_va)        = -1.0 * chol3d(h1_oa, p1_va, cind) * _a01_sp(h2_oa, h1_oa, cind),
    "_a05_sp(h2_oa, p1_va)        = -1.0 * chol3d(h1_oa, p1_va, cind) * _a01_sp(h2_oa, h1_oa, cind)")      // o2vm
    (_a05_sp(h2_oa, p1_va)       +=  1.0 * f1(h2_oa, p1_va),
    "_a05_sp(h2_oa, p1_va)       +=  1.0 * f1(h2_oa, p1_va)") // NEW TERM
    // .exact_copy(_a05_bb(h1_ob,p1_vb),_a05_aa(h1_ob,p1_vb))

    (_a06_sp(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d(h2_oa, p2_va, cind),
    "_a06_sp(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d(h2_oa, p2_va, cind)") // o2v2m
    (_a04_sp(h2_oa, h1_oa)        = -1.0 * f1(h2_oa, h1_oa),
    "_a04_sp(h2_oa, h1_oa)        = -1.0 * f1(h2_oa, h1_oa)") // MOVED TERM
    (_a04_sp(h2_oa, h1_oa)       +=  1.0 * chol3d(h2_oa, p1_va, cind) * _a06_sp(p1_va, h1_oa, cind),
    "_a04_sp(h2_oa, h1_oa)       +=  1.0 * chol3d(h2_oa, p1_va, cind) * _a06_sp(p1_va, h1_oa, cind)")   // o2vm
    (_a04_sp(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1(h2_oa, p1_va),
    "_a04_sp(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1(h2_oa, p1_va)") // NEW TERM
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04_sp(h2_oa, h1_oa),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04_sp(h2_oa, h1_oa)")                         // o2v
    (i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d(h2_oa, p1_va, cind) * _a02V(cind),
    "i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d(h2_oa, p1_va, cind) * _a02V(cind)")                      // ovm
    (i0_aa(p1_va, h2_oa)            +=  1.0 * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * _a05_sp(h1_oa, p2_va),
    "i0_aa(p1_va, h2_oa)            +=  1.0 * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * _a05_sp(h1_oa, p2_va)")
    (i0_aa(p2_va, h1_oa)            += -1.0 * chol3d(p2_va, p1_va, cind) * _a06_sp(p1_va, h1_oa, cind),
    "i0_aa(p2_va, h1_oa)            += -1.0 * chol3d(p2_va, p1_va, cind) * _a06_sp(p1_va, h1_oa, cind)")   // ov2m
    (_a06_sp(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d(p2_va, p1_va, cind),
    "_a06_sp(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d(p2_va, p1_va, cind)")              // ov2m
    (i0_aa(p1_va, h2_oa)            += -1.0 * _a06_sp(p1_va, h2_oa, cind) * _a02V(cind),
    "i0_aa(p1_va, h2_oa)            += -1.0 * _a06_sp(p1_va, h2_oa, cind) * _a02V(cind)")                           // ovm
    (_a06_sp(p2_va, h1_oa, cind) += -1.0 * t1_aa(p2_va, h1_oa) * _a02V(cind),
    "_a06_sp(p2_va, h1_oa, cind) += -1.0 * t1_aa(p2_va, h1_oa) * _a02V(cind)")                                      // ovm
    (_a06_sp(p2_va, h1_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01_sp(h2_oa, h1_oa, cind),
    "_a06_sp(p2_va, h1_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01_sp(h2_oa, h1_oa, cind)")                   // o2vm
    (_a01_sp(h2_oa, h1_oa, cind) +=  1.0 * chol3d(h2_oa, h1_oa, cind),
    "_a01_sp(h2_oa, h1_oa, cind) +=  1.0 * chol3d(h2_oa, h1_oa, cind)")                                    // o2m
    (i0_aa(p2_va, h1_oa)            +=  1.0 * _a01_sp(h2_oa, h1_oa, cind) * _a06_sp(p2_va, h2_oa, cind),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * _a01_sp(h2_oa, h1_oa, cind) * _a06_sp(p2_va, h2_oa, cind)")        // o2vm
    // (i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h2_oa) * f1(h2_oa, h1_oa), // MOVED ABOVE
    // "i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h2_oa) * f1(h2_oa, h1_oa)")                        // o2v
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p1_va, h1_oa) * f1(p2_va, p1_va),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p1_va, h1_oa) * f1(p2_va, p1_va)")                        // ov2
    ;
  // clang-format on
}

template<typename T>
void ccsd_t2_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                Tensor<T>& i0_abab, const Tensor<T>& t1_aa, Tensor<T>& t2_abab, Tensor<T>& t2_aaaa,
                const Tensor<T>& f1, const Tensor<T>& chol3d) {
  auto [cind]   = CI.labels<1>("all");
  auto [p3, p4] = MO.labels<2>("virt");
  auto [h1, h2] = MO.labels<2>("occ");

  auto [p1_va, p2_va, p3_va] = v_alpha.labels<3>("all");
  auto [p1_vb, p2_vb]        = v_beta.labels<2>("all");
  auto [h1_oa, h2_oa, h3_oa] = o_alpha.labels<3>("all");
  auto [h1_ob, h2_ob]        = o_beta.labels<2>("all");

  // clang-format off
  sch
    (_a017_sp(p1_va, h2_oa, cind)         = -1.0  * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * chol3d(h1_oa, p2_va, cind),
    "_a017_sp(p1_va, h2_oa, cind)         = -1.0  * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * chol3d(h1_oa, p2_va, cind)")
    (_a006_sp(h2_oa, h1_oa)               = -1.0  * chol3d(h2_oa, p2_va, cind) * _a017_sp(p2_va, h1_oa, cind),
    "_a006_sp(h2_oa, h1_oa)               = -1.0  * chol3d(h2_oa, p2_va, cind) * _a017_sp(p2_va, h1_oa, cind)")
    (_a007V(cind)                            =  2.0  * chol3d(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa),
    "_a007V(cind)                            =  2.0  * chol3d(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa)")
    (_a009_sp(h1_oa, h2_oa, cind)         =  1.0  * chol3d(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa),
    "_a009_sp(h1_oa, h2_oa, cind)         =  1.0  * chol3d(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa)")
    (_a021_sp(p2_va, p1_va, cind)         = -0.5  * chol3d(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa),
    "_a021_sp(p2_va, p1_va, cind)         = -0.5  * chol3d(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa)")
    (_a021_sp(p2_va, p1_va, cind)        +=  0.5  * chol3d(p2_va, p1_va, cind),
    "_a021_sp(p2_va, p1_va, cind)        +=  0.5  * chol3d(p2_va, p1_va, cind)")
    (_a017_sp(p1_va, h2_oa, cind)        += -2.0  * t1_aa(p2_va, h2_oa) * _a021_sp(p1_va, p2_va, cind),
    "_a017_sp(p1_va, h2_oa, cind)        += -2.0  * t1_aa(p2_va, h2_oa) * _a021_sp(p1_va, p2_va, cind)")
    (_a008_sp(h2_oa, h1_oa, cind)         =  1.0  * _a009_sp(h2_oa, h1_oa, cind),
    "_a008_sp(h2_oa, h1_oa, cind)         =  1.0  * _a009_sp(h2_oa, h1_oa, cind)")
    (_a009_sp(h2_oa, h1_oa, cind)        +=  1.0  * chol3d(h2_oa, h1_oa, cind),
    "_a009_sp(h2_oa, h1_oa, cind)        +=  1.0  * chol3d(h2_oa, h1_oa, cind)")
    // .exact_copy(_a009_sp(h2_ob,h1_ob,cind),_a009_sp(h2_ob,h1_ob,cind))
    // .exact_copy(_a021_sp(p2_vb,p1_vb,cind),_a021_sp(p2_vb,p1_vb,cind))
    (_a001_sp(p1_va, p2_va)               = -2.0  * _a021_sp(p1_va, p2_va, cind) * _a007V(cind),
    "_a001_sp(p1_va, p2_va)               = -2.0  * _a021_sp(p1_va, p2_va, cind) * _a007V(cind)")
    (_a001_sp(p1_va, p2_va)              += -1.0  * _a017_sp(p1_va, h2_oa, cind) * chol3d(h2_oa, p2_va, cind),
    "_a001_sp(p1_va, p2_va)              += -1.0  * _a017_sp(p1_va, h2_oa, cind) * chol3d(h2_oa, p2_va, cind)")
    (_a006_sp(h2_oa, h1_oa)              +=  1.0  * _a009_sp(h2_oa, h1_oa, cind) * _a007V(cind),
    "_a006_sp(h2_oa, h1_oa)              +=  1.0  * _a009_sp(h2_oa, h1_oa, cind) * _a007V(cind)")
    (_a006_sp(h3_oa, h1_oa)              += -1.0  * _a009_sp(h2_oa, h1_oa, cind) * _a008_sp(h3_oa, h2_oa, cind),
    "_a006_sp(h3_oa, h1_oa)              += -1.0  * _a009_sp(h2_oa, h1_oa, cind) * _a008_sp(h3_oa, h2_oa, cind)")
    (_a019_sp(h2_oa, h1_ob, h1_oa, h2_ob)  =  0.25 * _a009_sp(h2_oa, h1_oa, cind) * _a009_sp(h1_ob, h2_ob, cind),
    "_a019_sp(h2_oa, h1_ob, h1_oa, h2_ob)  =  0.25 * _a009_sp(h2_oa, h1_oa, cind) * _a009_sp(h1_ob, h2_ob, cind)")
    (_a020_sp(p2_va, h2_oa, p1_va, h1_oa)  = -2.0  * _a009_sp(h2_oa, h1_oa, cind) * _a021_sp(p2_va, p1_va, cind),
    "_a020_sp(p2_va, h2_oa, p1_va, h1_oa)  = -2.0  * _a009_sp(h2_oa, h1_oa, cind) * _a021_sp(p2_va, p1_va, cind)")
    // .exact_copy(_a020_sp(p2_vb, h2_oa, p1_vb, h1_oa),_a020_sp(p2_vb, h2_oa, p1_vb, h1_oa))
    (_a020_sp(p1_va, h3_oa, p3_va, h2_oa) +=  0.5  * _a004_sp(p2_va, p3_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa),
    "_a020_sp(p1_va, h3_oa, p3_va, h2_oa) +=  0.5  * _a004_sp(p2_va, p3_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)")
    (_a020_sp(p1_vb, h2_oa, p1_va, h2_ob)  = -0.5  * _a004_sp(p2_va, p1_va, h2_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
    "_a020_sp(p1_vb, h2_oa, p1_va, h2_ob)  = -0.5  * _a004_sp(p2_va, p1_va, h2_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)")
    (_a020_sp(p1_vb, h1_oa, p2_vb, h2_oa) +=  0.5  * _a004_sp(p1_va, p2_vb, h1_oa, h1_ob) * t2_abab(p1_va,p1_vb,h2_oa,h1_ob),
    "_a020_sp(p1_vb, h1_oa, p2_vb, h2_oa) +=  0.5  * _a004_sp(p1_va, p2_vb, h1_oa, h1_ob) * t2_abab(p1_va,p1_vb,h2_oa,h1_ob)")
    (_a017_sp(p1_va, h2_oa, cind)           +=  1.0  * t1_aa(p1_va, h1_oa) * chol3d(h1_oa, h2_oa, cind),
    "_a017_sp(p1_va, h2_oa, cind)           +=  1.0  * t1_aa(p1_va, h1_oa) * chol3d(h1_oa, h2_oa, cind)")
    (_a017_sp(p1_va, h2_oa, cind)           += -1.0  * chol3d(h2_oa, p1_va, cind),
    "_a017_sp(p1_va, h2_oa, cind)           += -1.0  * chol3d(h2_oa, p1_va, cind)")
    (_a001_sp(p2_va, p1_va)                 += -1.0  * f1(p2_va, p1_va),
    "_a001_sp(p2_va, p1_va)                 += -1.0  * f1(p2_va, p1_va)")
    (_a001_sp(p2_va, p1_va)                 +=  1.0  * t1_aa(p2_va, h1_oa) * f1(h1_oa, p1_va),
    "_a001_sp(p2_va, p1_va)                 +=  1.0  * t1_aa(p2_va, h1_oa) * f1(h1_oa, p1_va)") // NEW TERM
    (_a006_sp(h2_oa, h1_oa)                 +=  1.0  * f1(h2_oa, h1_oa),
    "_a006_sp(h2_oa, h1_oa)                 +=  1.0  * f1(h2_oa, h1_oa)")
    (_a006_sp(h2_oa, h1_oa)                 +=  1.0  * t1_aa(p1_va, h1_oa) * f1(h2_oa, p1_va),
    "_a006_sp(h2_oa, h1_oa)                 +=  1.0  * t1_aa(p1_va, h1_oa) * f1(h2_oa, p1_va)")
    // .exact_copy(_a017_sp(p1_vb, h1_ob, cind), _a017_sp(p1_vb, h1_ob, cind))
    // .exact_copy(_a006_sp(h1_ob, h2_ob), _a006_sp(h1_ob, h2_ob))
    // .exact_copy(_a001_sp(p1_vb, p2_vb), _a001_sp(p1_vb, p2_vb))
    // .exact_copy(_a021_sp(p1_vb, p2_vb, cind), _a021_sp(p1_vb, p2_vb, cind))
    // .exact_copy(_a020_sp(p1_vb, h1_ob, p2_vb, h2_ob), _a020_sp(p1_vb, h1_ob, p2_vb, h2_ob))

    (i0_abab(p1_va, p2_vb, h2_oa, h1_ob)          =  1.0  * _a020_sp(p2_vb, h2_ob, p1_vb, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h2_ob),
    "i0_abab(p1_va, p2_vb, h2_oa, h1_ob)          =  1.0  * _a020_sp2_vb, h2_ob, p1_vb, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h2_ob)")
    (i0_abab(p2_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020_sp(p1_vb, h1_oa, p1_va, h1_ob) * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa),
    "i0_abab(p2_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020_sp1_vb, h1_oa, p1_va, h1_ob) * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa)")
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020_sp(p1_vb, h1_oa, p2_vb, h2_oa) * t2_abab(p1_va, p2_vb, h1_oa, h1_ob),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020_sp1_vb, h1_oa, p2_vb, h2_oa) * t2_abab(p1_va, p2_vb, h1_oa, h1_ob)")
    // .exact_copy(i0_temp(p1_vb,p1_va,h2_ob,h1_oa),i0_abab(p1_vb,p1_va,h2_ob,h1_oa))
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * i0_temp(p1_vb, p1_va, h1_ob, h2_oa),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * i0_temp(p1_vb, p1_va, h1_ob, h2_oa)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  1.0  * _a017_sp(p1_va, h1_oa, cind) * _a017_sp(p1_vb, h2_ob, cind),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  1.0  * _a017_sp(p1_va, h1_oa, cind) * _a017_sp(p1_vb, h2_ob, cind)")
    (_a022_sp(p1_va,p2_vb,p2_va,p1_vb)       =  1.0  * _a021_sp(p1_va,p2_va,cind) * _a021_sp(p2_vb,p1_vb,cind),
    "_a022_sp(p1_va,p2_vb,p2_va,p1_vb)       =  1.0  * _a021_sp(p1_va,p2_va,cind) * _a021_sp(p2_vb,p1_vb,cind)")
    (i0_abab(p1_va, p2_vb, h1_oa, h2_ob)         +=  4.0  * _a022_sp(p1_va, p2_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
    "i0_abab(p1_va, p2_vb, h1_oa, h2_ob)         +=  4.0  * _a022_sp(p1_va, p2_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)")
    (_a019_sp(h2_oa, h1_ob, h1_oa, h2_ob)   +=  0.25 * _a004_sp(p1_va, p2_vb, h2_oa, h1_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob),
    "_a019_sp(h2_oa, h1_ob, h1_oa, h2_ob)   +=  0.25 * _a004_sp(p1_va, p2_vb, h2_oa, h1_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  4.0  * _a019_sp(h2_oa, h1_ob, h1_oa, h2_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  4.0  * _a019_sp(h2_oa, h1_ob, h1_oa, h2_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001_sp(p1_vb, p2_vb),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001_sp(p1_vb, p2_vb)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001_sp(p1_va, p2_va),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001_sp(p1_va, p2_va)")
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006_sp(h1_oa, h2_oa),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006_sp(h1_oa, h2_oa)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006_sp(h1_ob, h2_ob),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006_sp(h1_ob, h2_ob)")
    ;
  // clang-format on
}

int main(int argc, char* argv[]) {
  using T = double;

  tamm::initialize(argc, argv);

  if(argc < 5) {
    tamm_terminate("Please provide occ_alpha, virt_alpha, cholesky-count and tile size");
  }

  size_t n_occ_alpha = atoi(argv[1]);
  size_t n_vir_alpha = atoi(argv[2]);
  size_t chol_count  = atoi(argv[3]);
  Tile   tile_size   = atoi(argv[4]);

  const auto nbf = n_occ_alpha + n_vir_alpha;

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  ExecutionHW exhw = ec.exhw();

  Scheduler sch{ec};

  bool profile = true;

  if(ec.print()) {
    std::cout << tamm_git_info() << std::endl;
    auto current_time   = std::chrono::system_clock::now();
    auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
    auto cur_local_time = localtime(&current_time_t);
    std::cout << std::endl << "date: " << std::put_time(cur_local_time, "%c") << std::endl;
    std::cout << "nnodes: " << ec.nnodes() << ", ";
    std::cout << "nproc_per_node: " << ec.ppn() << ", ";
    std::cout << "nproc_total: " << ec.nnodes() * ec.ppn() << ", ";
    std::cout << "ngpus_per_node: " << ec.gpn() << ", ";
    std::cout << "ngpus_total: " << ec.nnodes() * ec.gpn() << std::endl;
    ec.print_mem_info();
    std::cout << std::endl;
    std::cout << "basis functions: " << nbf << ", occ_alpha: " << n_occ_alpha
              << ", virt_alpha: " << n_vir_alpha << ", chol-count: " << chol_count
              << ", tilesize: " << tile_size << std::endl;
  }

  //-----------------------------------

  TAMM_SIZE n_occ_beta = n_occ_alpha;
  Tile      tce_tile   = tile_size;

  TAMM_SIZE nmo        = 2 * nbf;
  TAMM_SIZE n_vir_beta = n_vir_alpha;
  TAMM_SIZE nocc       = 2 * n_occ_alpha;

  const TAMM_SIZE total_orbitals = nmo;

  // Construction of tiled index space MO
  IndexSpace MO_IS{
    range(0, total_orbitals),
    {{"occ", {range(0, nocc)}},
     {"occ_alpha", {range(0, n_occ_alpha)}},
     {"occ_beta", {range(n_occ_alpha, nocc)}},
     {"virt", {range(nocc, total_orbitals)}},
     {"virt_alpha", {range(nocc, nocc + n_vir_alpha)}},
     {"virt_beta", {range(nocc + n_vir_alpha, total_orbitals)}}},
    {{Spin{1}, {range(0, n_occ_alpha), range(nocc, nocc + n_vir_alpha)}},
     {Spin{2}, {range(n_occ_alpha, nocc), range(nocc + n_vir_alpha, total_orbitals)}}}};

  std::vector<Tile> mo_tiles;

  tamm::Tile est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_alpha / tce_tile));
  for(tamm::Tile x = 0; x < est_nt; x++)
    mo_tiles.push_back(n_occ_alpha / est_nt + (x < (n_occ_alpha % est_nt)));

  est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_beta / tce_tile));
  for(tamm::Tile x = 0; x < est_nt; x++)
    mo_tiles.push_back(n_occ_beta / est_nt + (x < (n_occ_beta % est_nt)));

  est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_alpha / tce_tile));
  for(tamm::Tile x = 0; x < est_nt; x++)
    mo_tiles.push_back(n_vir_alpha / est_nt + (x < (n_vir_alpha % est_nt)));

  est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_beta / tce_tile));
  for(tamm::Tile x = 0; x < est_nt; x++)
    mo_tiles.push_back(n_vir_beta / est_nt + (x < (n_vir_beta % est_nt)));

  TiledIndexSpace MO{MO_IS, mo_tiles};

  //----------------------------------------------------

  TiledIndexSpace N = MO("all");

  // add the new lambda based spin based block sparsity
  auto is_non_zero_2D_spin =
    generate_spin_check({N, N}, {SpinPosition::lower, SpinPosition::upper});

  TensorInfo tensor_info_N_N{{N, N}, is_non_zero_2D_spin};

  // Tensor<T> d_f1{{N, N}, {1, 1}};
  Tensor<T> d_f1{{N, N}, tensor_info_N_N};
  Tensor<T>::allocate(&ec, d_f1);

  tamm::random_ip(d_f1);

  std::vector<T> p_evl_sorted;
  Tensor<T>      t1_aa, t2_abab, r1_aa, r2_abab;

  std::tie(p_evl_sorted, t1_aa, t2_abab, r1_aa, r2_abab) = setupTensors_cs(ec, MO, d_f1);

  IndexSpace      chol_is{range(0, chol_count)};
  TiledIndexSpace CI{chol_is, 1000};

  // cholVpr = {{N, N, CI}, {SpinPosition::upper, SpinPosition::lower, SpinPosition::ignore}};
  // Tensor<TensorType>::allocate(&ec, cholVpr);

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  auto [cind]              = CI.labels<1>("all");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(oatiles, otiles)};
  v_beta  = {MO("virt"), range(vatiles, vtiles)};

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb, p2_vb] = v_beta.labels<2>("all");
  auto [h3_oa, h4_oa] = o_alpha.labels<2>("all");
  auto [h3_ob, h4_ob] = o_beta.labels<2>("all");

  Tensor<T> d_e{};

  auto is_nonzero_Va_Va_Oa_Oa = generate_spin_check(
    {v_alpha, v_alpha, o_alpha, o_alpha},
    {SpinPosition::lower, SpinPosition::lower, SpinPosition::upper, SpinPosition::upper});

  // Tensor<T> t2_aaaa = {{v_alpha, v_alpha, o_alpha, o_alpha}, {2, 2}};

  TensorInfo tensor_info_Va_Va_Oa_Oa{{v_alpha, v_alpha, o_alpha, o_alpha}, is_nonzero_Va_Va_Oa_Oa};

  // Block Sparse Tensor
  Tensor<T> t2_aaaa = {{v_alpha, v_alpha, o_alpha, o_alpha}, tensor_info_Va_Va_Oa_Oa};

  // Block Sparse Tensor f1
  TensorInfo tensor_info_f1{
    {MO, MO},           // Tensor dims
    {"IJ", "IA", "AB"}, // Allowed blocks
    index_to_sub_string // Char to named sub-space string
  };
  Tensor<T> f1{{MO, MO}, tensor_info_f1};
  // Constructor without TensorInfo struct
  // Tensor<T> f1{{MO, MO}, {"IJ", "IA", "AB"}, index_to_sub_string};

  // Block Sparse Tensor chol3d
  TensorInfo tensor_info_chol3d{
    {MO, MO, CI},          // Tensor dims
    {"IJX", "IAX", "ABX"}, // Allowed blocks
    index_to_sub_string    // Char to named sub-space string
  };
  Tensor<T> chol3d{{MO, MO, CI}, tensor_info_chol3d};
  // Constructor without TensorInfo struct
  // Tensor<T> chol3d{{MO, MO, CI}, {"IJX", "IAX", "ABX"}, index_to_sub_string};

  _a01V = {CI};
  // _a02_sp  = Tensor<T>{{MO, MO, CI}, {"IJX"}, index_to_sub_string};
  // _a03_sp  = Tensor<T>{{MO, MO, CI}, {"IAX"}, index_to_sub_string};
  // _a004_sp = Tensor<T>{{MO, MO, MO, MO}, {"ABIJ", "AbIj"}, index_to_sub_string};

  _a02_sp  = Tensor<T>{{MO, MO, CI}, {{h3_oa, h4_oa, cind}}};
  _a03_sp  = Tensor<T>{{MO, MO, CI}, {{h3_oa, p1_va, cind}}};
  _a004_sp = Tensor<T>{{MO, MO, MO, MO},
                       {{"virt_alpha, virt_alpha, occ_alpha, occ_alpha"},
                        {"virt_alpha, virt_beta, occ_alpha, occ_beta"}}};

  t2_aaaa_temp = {v_alpha, v_alpha, o_alpha, o_alpha};
  i0_temp      = {v_beta, v_alpha, o_beta, o_alpha};

  // Intermediates
  // T1
  _a02V = {CI};
  // _a01_sp = Tensor<T>{{MO, MO, CI}, {"IJX"}, index_to_sub_string};
  // _a04_sp = Tensor<T>{{MO, MO}, {"IJ"}, index_to_sub_string};
  // _a05_sp = Tensor<T>{{MO, MO}, {"IA"}, index_to_sub_string}; // bb
  // _a06_sp = Tensor<T>{{MO, MO, CI}, {"AIX"}, index_to_sub_string};

  _a01_sp = Tensor<T>{{MO, MO, CI}, {{h3_oa, h4_oa, cind}}};
  _a04_sp = Tensor<T>{{MO, MO}, {{h3_oa, h4_oa}}};
  _a05_sp = Tensor<T>{{MO, MO}, {{h3_oa, p1_va}}}; // bb
  _a06_sp = Tensor<T>{{MO, MO, CI}, {{p1_va, h3_oa, cind}}};

  // T2
  _a007V = {CI};
  // _a001_sp = Tensor<T>{{MO, MO}, {"AB", "ab"}, index_to_sub_string};
  // _a006_sp = Tensor<T>{{MO, MO}, {"IJ", "ij"}, index_to_sub_string};

  // _a008_sp = Tensor<T>{{MO, MO, CI}, {"IJX"}, index_to_sub_string};
  // _a009_sp = Tensor<T>{{MO, MO, CI}, {"IJX", "ijX"}, index_to_sub_string};
  // _a017_sp = Tensor<T>{{MO, MO, CI}, {"AIX", "aiX"}, index_to_sub_string};
  // _a021_sp = Tensor<T>{{MO, MO, CI}, {"ABX", "abX"}, index_to_sub_string};

  // _a019_sp = Tensor<T>{{MO, MO, MO, MO}, {"IjKl"}, index_to_sub_string};
  // _a022_sp = Tensor<T>{{MO, MO, MO, MO}, {"AbCd"}, index_to_sub_string};
  // _a020_sp = Tensor<T>{{MO, MO, MO, MO}, {"AIBJ", "aIbJ", "aIBj", "aibj"}, index_to_sub_string};

  _a001_sp = Tensor<T>{{MO, MO}, {{p1_va, p2_va}, {p1_vb, p2_vb}}};
  _a006_sp = Tensor<T>{{MO, MO}, {{h3_oa, h4_oa}, {h3_ob, h4_ob}}};

  _a008_sp = Tensor<T>{{MO, MO, CI}, {{h3_oa, h4_oa, cind}}};
  _a009_sp = Tensor<T>{{MO, MO, CI}, {{h3_oa, h4_oa, cind}, {h3_ob, h4_ob, cind}}};
  _a017_sp = Tensor<T>{{MO, MO, CI}, {{p1_va, h3_oa, cind}, {p1_vb, h3_ob, cind}}};
  _a021_sp = Tensor<T>{{MO, MO, CI}, {{p1_va, p2_va, cind}, {p1_vb, p2_vb, cind}}};

  _a019_sp = Tensor<T>{{MO, MO, MO, MO}, {{h3_oa, h3_ob, h4_oa, h4_ob}}};
  _a022_sp = Tensor<T>{{MO, MO, MO, MO}, {{p1_va, p1_vb, p2_va, p2_vb}}};
  _a020_sp = Tensor<T>{{MO, MO, MO, MO},
                       {{p1_va, h3_oa, p2_va, h4_oa},
                        {p1_vb, h3_oa, p2_vb, h4_oa},
                        {p1_vb, h3_oa, p2_va, h4_ob},
                        {p1_vb, h3_ob, p2_vb, h4_ob}}};

  sch.allocate(t2_aaaa);
  sch.allocate(d_e, i0_temp, t2_aaaa_temp, _a01V);
  sch.allocate(f1, chol3d);
  sch.allocate(_a02_sp, _a03_sp);

  // allocate all intermediates
  sch.allocate(_a02V, _a007V);
  sch.allocate(_a004_sp, _a01_sp, _a04_sp, _a05_sp, _a06_sp, _a001_sp, _a006_sp, _a008_sp, _a009_sp,
               _a017_sp, _a019_sp, _a020_sp, _a021_sp, _a022_sp);
  sch.execute();

  tamm::random_ip(f1(h3_oa, h4_oa));
  tamm::random_ip(f1(h3_oa, p2_va));
  tamm::random_ip(f1(p1_va, p2_va));
  tamm::random_ip(chol3d(h3_oa, h4_oa, cind));
  tamm::random_ip(chol3d(h3_oa, p2_va, cind));
  tamm::random_ip(chol3d(p1_va, p2_va, cind));

  // clang-format off
  sch
  (_a004_sp(p1_va, p2_va, h4_oa, h3_oa) = 1.0 * chol3d(h4_oa, p1_va, cind) * chol3d(h3_oa, p2_va, cind)) 
   .exact_copy(_a004_sp(p1_va, p1_vb, h3_oa, h3_ob),
      _a004_sp(p1_va, p2_va, h3_oa, h4_oa), true)
  ;
  // clang-format on

  sch.execute(exhw);

  const auto timer_start = std::chrono::high_resolution_clock::now();

  ccsd_e_cs(sch, MO, CI, d_e, t1_aa, t2_abab, t2_aaaa, f1, chol3d);
  ccsd_t1_cs(sch, MO, CI, r1_aa, t1_aa, t2_abab, f1, chol3d);
  ccsd_t2_cs(sch, MO, CI, r2_abab, t1_aa, t2_abab, t2_aaaa, f1, chol3d);

  sch.execute(exhw, profile);

  const auto timer_end = std::chrono::high_resolution_clock::now();
  auto       iter_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

  if(ec.print()) std::cout << "Time taken for closed-shell CD-CCSD: " << iter_time << std::endl;

  if(profile && ec.print()) {
    std::string profile_csv = "ccsd_profile_" + std::to_string(nbf) + "bf_" +
                              std::to_string(n_occ_alpha) + "oa_" + std::to_string(n_vir_alpha) +
                              "va_" + std::to_string(chol_count) + "cv_" +
                              std::to_string(tile_size) + "TS.csv";
    std::ofstream pds(profile_csv, std::ios::out);
    if(!pds) std::cerr << "Error opening file " << profile_csv << std::endl;
    pds << ec.get_profile_header() << std::endl;
    pds << ec.get_profile_data().str() << std::endl;
    pds.close();
  }

  // if(false) {
  //   ExecutionContext ec_dense{ec.pg(), DistributionKind::dense, MemoryManagerKind::ga};
  //   Tensor<T>        c3dvv_dense = tamm::to_dense_tensor(ec_dense, chol3d_vv("aa"));
  //   print_dense_tensor(c3dvv_dense, "c3d_vv_aa_dense");
  // }

  // deallocate all intermediates
  sch.deallocate(_a02V, _a007V);
  sch.deallocate(_a004_sp, _a01_sp, _a04_sp, _a05_sp, _a06_sp, _a001_sp, _a006_sp, _a008_sp,
                 _a009_sp, _a017_sp, _a019_sp, _a020_sp, _a021_sp, _a022_sp);

  sch.deallocate(d_e, i0_temp, t2_aaaa_temp, _a01V);
  sch.deallocate(_a02_sp, _a03_sp);
  sch.deallocate(f1, chol3d);

  sch.execute();

  sch.deallocate(t1_aa, t2_abab, r1_aa, r2_abab, d_f1, t2_aaaa).execute();

  tamm::finalize();

  return 0;
}
