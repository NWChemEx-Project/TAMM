#include "ccse_tensors.hpp"
#include <tamm/tamm_git.hpp>

using CCEType = double;
TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;

Tensor<CCEType>       _a01V, _a02V, _a007V;
CCSE_Tensors<CCEType> _a01, _a02, _a03, _a04, _a05, _a06, _a001, _a004, _a006, _a008, _a009, _a017,
  _a019, _a020, _a021, _a022;

Tensor<CCEType> i0_temp, t2_aaaa_temp; // CS only

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

  Tensor<T> d_r1{{v_alpha, o_alpha}, {1, 1}};
  Tensor<T> d_r2{{v_alpha, v_beta, o_alpha, o_beta}, {2, 2}};

  Tensor<T>::allocate(&ec, d_r1, d_r2);

  Tensor<T> d_t1{{v_alpha, o_alpha}, {1, 1}};
  Tensor<T> d_t2{{v_alpha, v_beta, o_alpha, o_beta}, {2, 2}};

  Tensor<T>::allocate(&ec, d_t1, d_t2);

  return std::make_tuple(p_evl_sorted, d_t1, d_t2, d_r1, d_r2);
}

template<typename T>
void ccsd_e_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& de,
               const Tensor<T>& t1_aa, const Tensor<T>& t2_abab, const Tensor<T>& t2_aaaa,
               std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind] = CI.labels<1>("all");

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob]        = o_beta.labels<1>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_ov     = f1_se[1];
  auto chol3d_ov = chol3d_se[1];

  // clang-format off
  sch
    (t2_aaaa_temp()=0)
    .exact_copy(t2_aaaa(p1_va, p2_va, h1_oa, h2_oa), t2_abab(p1_va, p2_va, h1_oa, h2_oa))
    (t2_aaaa_temp() = t2_aaaa(),
    "t2_aaaa_temp() = t2_aaaa()")
    (t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa),
    "t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa)")
    (t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa),
    "t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa)")

    (_a01V(cind) = t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "_a01V(cind) = t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (_a02("aa")(h1_oa, h2_oa, cind)    = t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind),
    "_a02( aa )(h1_oa, h2_oa, cind)    = t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")
    (_a03("aa")(h2_oa, p2_va, cind) = t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "_a03( aa )(h2_oa, p2_va, cind) = t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (de()  =  2.0 * _a01V() * _a01V(),
    "de()  =  2.0 * _a01V() * _a01V()")
    (de() += -1.0 * _a02("aa")(h1_oa, h2_oa, cind) * _a02("aa")(h2_oa, h1_oa, cind),
    "de() += -1.0 * _a02( aa )(h1_oa, h2_oa, cind) * _a02( aa )(h2_oa, h1_oa, cind)")
    (de() +=  1.0 * _a03("aa")(h1_oa, p1_va, cind) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "de() +=  1.0 * _a03( aa )(h1_oa, p1_va, cind) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (de() +=  2.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va),
    "de() +=  2.0 * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h1_oa, p1_va)") // NEW TERM
    ;
  // clang-format on
}

template<typename T>
void ccsd_t1_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                Tensor<T>& i0_aa, const Tensor<T>& t1_aa, const Tensor<T>& t2_abab,
                std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind] = CI.labels<1>("all");
  auto [p2]   = MO.labels<1>("virt");
  auto [h1]   = MO.labels<1>("occ");

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob]        = o_beta.labels<1>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_oo     = f1_se[0];
  auto f1_ov     = f1_se[1];
  auto f1_vv     = f1_se[2];
  auto chol3d_oo = chol3d_se[0];
  auto chol3d_ov = chol3d_se[1];
  auto chol3d_vv = chol3d_se[2];

  // clang-format off
  sch
    (i0_aa(p2_va, h1_oa)             =  1.0 * f1_ov("aa")(h1_oa, p2_va),
    "i0_aa(p2_va, h1_oa)             =  1.0 * f1_ov( aa )(h1_oa, p2_va)")
    (_a01("aa")(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind),
    "_a01( aa )(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")                 // ovm
    (_a02V(cind)                     =  2.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "_a02V(cind)                     =  2.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")                 // ovm
    // (_a02V(cind)                  =  2.0 * _a01("aa")(h1_oa, h1_oa, cind))
    (_a05("aa")(h2_oa, p1_va)        = -1.0 * chol3d_ov("aa")(h1_oa, p1_va, cind) * _a01("aa")(h2_oa, h1_oa, cind),
    "_a05( aa )(h2_oa, p1_va)        = -1.0 * chol3d_ov( aa )(h1_oa, p1_va, cind) * _a01( aa )(h2_oa, h1_oa, cind)")      // o2vm
    (_a05("aa")(h2_oa, p1_va)       +=  1.0 * f1_ov("aa")(h2_oa, p1_va),
    "_a05( aa )(h2_oa, p1_va)       +=  1.0 * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
    // .exact_copy(_a05_bb(h1_ob,p1_vb),_a05_aa(h1_ob,p1_vb))

    (_a06("aa")(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d_ov("aa")(h2_oa, p2_va, cind),
    "_a06( aa )(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d_ov( aa )(h2_oa, p2_va, cind)") // o2v2m
    (_a04("aa")(h2_oa, h1_oa)        = -1.0 * f1_oo("aa")(h2_oa, h1_oa),
    "_a04( aa )(h2_oa, h1_oa)        = -1.0 * f1_oo( aa )(h2_oa, h1_oa)") // MOVED TERM
    (_a04("aa")(h2_oa, h1_oa)       +=  1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind),
    "_a04( aa )(h2_oa, h1_oa)       +=  1.0 * chol3d_ov( aa )(h2_oa, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // o2vm
    (_a04("aa")(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va),
    "_a04( aa )(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04("aa")(h2_oa, h1_oa),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04( aa )(h2_oa, h1_oa)")                         // o2v
    (i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a02V(cind),
    "i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_ov( aa )(h2_oa, p1_va, cind) * _a02V(cind)")                      // ovm
    (i0_aa(p1_va, h2_oa)            +=  1.0 * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * _a05("aa")(h1_oa, p2_va),
    "i0_aa(p1_va, h2_oa)            +=  1.0 * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * _a05( aa )(h1_oa, p2_va)")
    (i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv("aa")(p2_va, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind),
    "i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv( aa )(p2_va, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // ov2m
    (_a06("aa")(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv("aa")(p2_va, p1_va, cind),
    "_a06( aa )(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv( aa )(p2_va, p1_va, cind)")              // ov2m
    (i0_aa(p1_va, h2_oa)            += -1.0 * _a06("aa")(p1_va, h2_oa, cind) * _a02V(cind),
    "i0_aa(p1_va, h2_oa)            += -1.0 * _a06( aa )(p1_va, h2_oa, cind) * _a02V(cind)")                           // ovm
    (_a06("aa")(p2_va, h1_oa, cind) += -1.0 * t1_aa(p2_va, h1_oa) * _a02V(cind),
    "_a06( aa )(p2_va, h1_oa, cind) += -1.0 * t1_aa(p2_va, h1_oa) * _a02V(cind)")                                      // ovm
    (_a06("aa")(p2_va, h1_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01("aa")(h2_oa, h1_oa, cind),
    "_a06( aa )(p2_va, h1_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01( aa )(h2_oa, h1_oa, cind)")                   // o2vm
    (_a01("aa")(h2_oa, h1_oa, cind) +=  1.0 * chol3d_oo("aa")(h2_oa, h1_oa, cind),
    "_a01( aa )(h2_oa, h1_oa, cind) +=  1.0 * chol3d_oo( aa )(h2_oa, h1_oa, cind)")                                    // o2m
    (i0_aa(p2_va, h1_oa)            +=  1.0 * _a01("aa")(h2_oa, h1_oa, cind) * _a06("aa")(p2_va, h2_oa, cind),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * _a01( aa )(h2_oa, h1_oa, cind) * _a06( aa )(p2_va, h2_oa, cind)")        // o2vm
    // (i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h2_oa) * f1_oo("aa")(h2_oa, h1_oa), // MOVED ABOVE
    // "i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h2_oa) * f1_oo( aa )(h2_oa, h1_oa)")                        // o2v
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p1_va, h1_oa) * f1_vv("aa")(p2_va, p1_va),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p1_va, h1_oa) * f1_vv( aa )(p2_va, p1_va)")                        // ov2
    ;
  // clang-format on
}

template<typename T>
void ccsd_t2_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                Tensor<T>& i0_abab, const Tensor<T>& t1_aa, Tensor<T>& t2_abab, Tensor<T>& t2_aaaa,
                std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind]   = CI.labels<1>("all");
  auto [p3, p4] = MO.labels<2>("virt");
  auto [h1, h2] = MO.labels<2>("occ");

  auto [p1_va, p2_va, p3_va] = v_alpha.labels<3>("all");
  auto [p1_vb, p2_vb]        = v_beta.labels<2>("all");
  auto [h1_oa, h2_oa, h3_oa] = o_alpha.labels<3>("all");
  auto [h1_ob, h2_ob]        = o_beta.labels<2>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_oo     = f1_se[0];
  auto f1_ov     = f1_se[1];
  auto f1_vv     = f1_se[2];
  auto chol3d_oo = chol3d_se[0];
  auto chol3d_ov = chol3d_se[1];
  auto chol3d_vv = chol3d_se[2];

  // clang-format off
  sch
    (_a017("aa")(p1_va, h2_oa, cind)         = -1.0  * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * chol3d_ov("aa")(h1_oa, p2_va, cind),
    "_a017( aa )(p1_va, h2_oa, cind)         = -1.0  * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * chol3d_ov( aa )(h1_oa, p2_va, cind)")
    (_a006("aa")(h2_oa, h1_oa)               = -1.0  * chol3d_ov("aa")(h2_oa, p2_va, cind) * _a017("aa")(p2_va, h1_oa, cind),
    "_a006( aa )(h2_oa, h1_oa)               = -1.0  * chol3d_ov( aa )(h2_oa, p2_va, cind) * _a017( aa )(p2_va, h1_oa, cind)")
    (_a007V(cind)                            =  2.0  * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa),
    "_a007V(cind)                            =  2.0  * chol3d_ov( aa )(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa)")
    (_a009("aa")(h1_oa, h2_oa, cind)         =  1.0  * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa),
    "_a009( aa )(h1_oa, h2_oa, cind)         =  1.0  * chol3d_ov( aa )(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa)")
    (_a021("aa")(p2_va, p1_va, cind)         = -0.5  * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa),
    "_a021( aa )(p2_va, p1_va, cind)         = -0.5  * chol3d_ov( aa )(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa)")
    (_a021("aa")(p2_va, p1_va, cind)        +=  0.5  * chol3d_vv("aa")(p2_va, p1_va, cind),
    "_a021( aa )(p2_va, p1_va, cind)        +=  0.5  * chol3d_vv( aa )(p2_va, p1_va, cind)")
    (_a017("aa")(p1_va, h2_oa, cind)        += -2.0  * t1_aa(p2_va, h2_oa) * _a021("aa")(p1_va, p2_va, cind),
    "_a017( aa )(p1_va, h2_oa, cind)        += -2.0  * t1_aa(p2_va, h2_oa) * _a021( aa )(p1_va, p2_va, cind)")
    (_a008("aa")(h2_oa, h1_oa, cind)         =  1.0  * _a009("aa")(h2_oa, h1_oa, cind),
    "_a008( aa )(h2_oa, h1_oa, cind)         =  1.0  * _a009( aa )(h2_oa, h1_oa, cind)")
    (_a009("aa")(h2_oa, h1_oa, cind)        +=  1.0  * chol3d_oo("aa")(h2_oa, h1_oa, cind),
    "_a009( aa )(h2_oa, h1_oa, cind)        +=  1.0  * chol3d_oo( aa )(h2_oa, h1_oa, cind)")
    // .exact_copy(_a009("bb")(h2_ob,h1_ob,cind),_a009("aa")(h2_ob,h1_ob,cind))
    // .exact_copy(_a021("bb")(p2_vb,p1_vb,cind),_a021("aa")(p2_vb,p1_vb,cind))
    (_a001("aa")(p1_va, p2_va)               = -2.0  * _a021("aa")(p1_va, p2_va, cind) * _a007V(cind),
    "_a001( aa )(p1_va, p2_va)               = -2.0  * _a021( aa )(p1_va, p2_va, cind) * _a007V(cind)")
    (_a001("aa")(p1_va, p2_va)              += -1.0  * _a017("aa")(p1_va, h2_oa, cind) * chol3d_ov("aa")(h2_oa, p2_va, cind),
    "_a001( aa )(p1_va, p2_va)              += -1.0  * _a017( aa )(p1_va, h2_oa, cind) * chol3d_ov( aa )(h2_oa, p2_va, cind)")
    (_a006("aa")(h2_oa, h1_oa)              +=  1.0  * _a009("aa")(h2_oa, h1_oa, cind) * _a007V(cind),
    "_a006( aa )(h2_oa, h1_oa)              +=  1.0  * _a009( aa )(h2_oa, h1_oa, cind) * _a007V(cind)")
    (_a006("aa")(h3_oa, h1_oa)              += -1.0  * _a009("aa")(h2_oa, h1_oa, cind) * _a008("aa")(h3_oa, h2_oa, cind),
    "_a006( aa )(h3_oa, h1_oa)              += -1.0  * _a009( aa )(h2_oa, h1_oa, cind) * _a008( aa )(h3_oa, h2_oa, cind)")
    (_a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob)  =  0.25 * _a009("aa")(h2_oa, h1_oa, cind) * _a009("bb")(h1_ob, h2_ob, cind),
    "_a019( abab )(h2_oa, h1_ob, h1_oa, h2_ob)  =  0.25 * _a009( aa )(h2_oa, h1_oa, cind) * _a009( bb )(h1_ob, h2_ob, cind)")
    (_a020("aaaa")(p2_va, h2_oa, p1_va, h1_oa)  = -2.0  * _a009("aa")(h2_oa, h1_oa, cind) * _a021("aa")(p2_va, p1_va, cind),
    "_a020( aaaa )(p2_va, h2_oa, p1_va, h1_oa)  = -2.0  * _a009( aa )(h2_oa, h1_oa, cind) * _a021( aa )(p2_va, p1_va, cind)")
    // .exact_copy(_a020("baba")(p2_vb, h2_oa, p1_vb, h1_oa),_a020("aaaa")(p2_vb, h2_oa, p1_vb, h1_oa))
    (_a020("aaaa")(p1_va, h3_oa, p3_va, h2_oa) +=  0.5  * _a004("aaaa")(p2_va, p3_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa),
    "_a020( aaaa )(p1_va, h3_oa, p3_va, h2_oa) +=  0.5  * _a004( aaaa )(p2_va, p3_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)")
    (_a020("baab")(p1_vb, h2_oa, p1_va, h2_ob)  = -0.5  * _a004("aaaa")(p2_va, p1_va, h2_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
    "_a020( baab )(p1_vb, h2_oa, p1_va, h2_ob)  = -0.5  * _a004( aaaa )(p2_va, p1_va, h2_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)")
    (_a020("baba")(p1_vb, h1_oa, p2_vb, h2_oa) +=  0.5  * _a004("abab")(p1_va, p2_vb, h1_oa, h1_ob) * t2_abab(p1_va,p1_vb,h2_oa,h1_ob),
    "_a020( baba )(p1_vb, h1_oa, p2_vb, h2_oa) +=  0.5  * _a004( abab )(p1_va, p2_vb, h1_oa, h1_ob) * t2_abab(p1_va,p1_vb,h2_oa,h1_ob)")
    (_a017("aa")(p1_va, h2_oa, cind)           +=  1.0  * t1_aa(p1_va, h1_oa) * chol3d_oo("aa")(h1_oa, h2_oa, cind),
    "_a017( aa )(p1_va, h2_oa, cind)           +=  1.0  * t1_aa(p1_va, h1_oa) * chol3d_oo( aa )(h1_oa, h2_oa, cind)")
    (_a017("aa")(p1_va, h2_oa, cind)           += -1.0  * chol3d_ov("aa")(h2_oa, p1_va, cind),
    "_a017( aa )(p1_va, h2_oa, cind)           += -1.0  * chol3d_ov( aa )(h2_oa, p1_va, cind)")
    (_a001("aa")(p2_va, p1_va)                 += -1.0  * f1_vv("aa")(p2_va, p1_va),
    "_a001( aa )(p2_va, p1_va)                 += -1.0  * f1_vv( aa )(p2_va, p1_va)")
    (_a001("aa")(p2_va, p1_va)                 +=  1.0  * t1_aa(p2_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va),
    "_a001( aa )(p2_va, p1_va)                 +=  1.0  * t1_aa(p2_va, h1_oa) * f1_ov( aa )(h1_oa, p1_va)") // NEW TERM
    (_a006("aa")(h2_oa, h1_oa)                 +=  1.0  * f1_oo("aa")(h2_oa, h1_oa),
    "_a006( aa )(h2_oa, h1_oa)                 +=  1.0  * f1_oo( aa )(h2_oa, h1_oa)")
    (_a006("aa")(h2_oa, h1_oa)                 +=  1.0  * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va),
    "_a006( aa )(h2_oa, h1_oa)                 +=  1.0  * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h2_oa, p1_va)")
    // .exact_copy(_a017("bb")(p1_vb, h1_ob, cind), _a017("aa")(p1_vb, h1_ob, cind))
    // .exact_copy(_a006("bb")(h1_ob, h2_ob), _a006("aa")(h1_ob, h2_ob))
    // .exact_copy(_a001("bb")(p1_vb, p2_vb), _a001("aa")(p1_vb, p2_vb))
    // .exact_copy(_a021("bb")(p1_vb, p2_vb, cind), _a021("aa")(p1_vb, p2_vb, cind))
    // .exact_copy(_a020("bbbb")(p1_vb, h1_ob, p2_vb, h2_ob), _a020("aaaa")(p1_vb, h1_ob, p2_vb, h2_ob))

    (i0_abab(p1_va, p2_vb, h2_oa, h1_ob)          =  1.0  * _a020("bbbb")(p2_vb, h2_ob, p1_vb, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h2_ob),
    "i0_abab(p1_va, p2_vb, h2_oa, h1_ob)          =  1.0  * _a020(bbbb)(p2_vb, h2_ob, p1_vb, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h2_ob)")
    (i0_abab(p2_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020("baab")(p1_vb, h1_oa, p1_va, h1_ob) * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa),
    "i0_abab(p2_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020(baab)(p1_vb, h1_oa, p1_va, h1_ob) * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa)")
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020("baba")(p1_vb, h1_oa, p2_vb, h2_oa) * t2_abab(p1_va, p2_vb, h1_oa, h1_ob),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020(baba)(p1_vb, h1_oa, p2_vb, h2_oa) * t2_abab(p1_va, p2_vb, h1_oa, h1_ob)")
    // .exact_copy(i0_temp(p1_vb,p1_va,h2_ob,h1_oa),i0_abab(p1_vb,p1_va,h2_ob,h1_oa))
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * i0_temp(p1_vb, p1_va, h1_ob, h2_oa),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * i0_temp(p1_vb, p1_va, h1_ob, h2_oa)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  1.0  * _a017("aa")(p1_va, h1_oa, cind) * _a017("bb")(p1_vb, h2_ob, cind),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  1.0  * _a017( aa )(p1_va, h1_oa, cind) * _a017( bb )(p1_vb, h2_ob, cind)")
    (_a022("abab")(p1_va,p2_vb,p2_va,p1_vb)       =  1.0  * _a021("aa")(p1_va,p2_va,cind) * _a021("bb")(p2_vb,p1_vb,cind),
    "_a022( abab )(p1_va,p2_vb,p2_va,p1_vb)       =  1.0  * _a021( aa )(p1_va,p2_va,cind) * _a021( bb )(p2_vb,p1_vb,cind)")
    (i0_abab(p1_va, p2_vb, h1_oa, h2_ob)         +=  4.0  * _a022("abab")(p1_va, p2_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
    "i0_abab(p1_va, p2_vb, h1_oa, h2_ob)         +=  4.0  * _a022( abab )(p1_va, p2_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)")
    (_a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob)   +=  0.25 * _a004("abab")(p1_va, p2_vb, h2_oa, h1_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob),
    "_a019( abab )(h2_oa, h1_ob, h1_oa, h2_ob)   +=  0.25 * _a004( abab )(p1_va, p2_vb, h2_oa, h1_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  4.0  * _a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  4.0  * _a019( abab )(h2_oa, h1_ob, h1_oa, h2_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001("bb")(p1_vb, p2_vb),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001( bb )(p1_vb, p2_vb)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001("aa")(p1_va, p2_va),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001( aa )(p1_va, p2_va)")
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006("aa")(h1_oa, h2_oa),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006( aa )(h1_oa, h2_oa)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006("bb")(h1_ob, h2_ob),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006( bb )(h1_ob, h2_ob)")
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

  Tensor<T> d_f1{{N, N}, {1, 1}};
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

  Tensor<T> t2_aaaa = {{v_alpha, v_alpha, o_alpha, o_alpha}, {2, 2}};

  CCSE_Tensors<T> f1_oo{MO, {O, O}, "f1_oo", {"aa"}}; // bb
  CCSE_Tensors<T> f1_ov{MO, {O, V}, "f1_ov", {"aa"}}; // bb
  CCSE_Tensors<T> f1_vv{MO, {V, V}, "f1_vv", {"aa"}}; // bb

  CCSE_Tensors<T> chol3d_oo{MO, {O, O, CI}, "chol3d_oo", {"aa"}}; // bb
  CCSE_Tensors<T> chol3d_ov{MO, {O, V, CI}, "chol3d_ov", {"aa"}}; // bb
  CCSE_Tensors<T> chol3d_vv{MO, {V, V, CI}, "chol3d_vv", {"aa"}}; // bb

  std::vector<CCSE_Tensors<T>> f1_se{f1_oo, f1_ov, f1_vv};
  std::vector<CCSE_Tensors<T>> chol3d_se{chol3d_oo, chol3d_ov, chol3d_vv};

  _a01V = {CI};
  _a02  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a02", {"aa"}};
  _a03  = CCSE_Tensors<T>{MO, {O, V, CI}, "_a03", {"aa"}};
  _a004 = CCSE_Tensors<T>{MO, {V, V, O, O}, "_a004", {"aaaa", "abab"}};

  t2_aaaa_temp = {v_alpha, v_alpha, o_alpha, o_alpha};
  i0_temp      = {v_beta, v_alpha, o_beta, o_alpha};

  // Intermediates
  // T1
  _a02V = {CI};
  _a01  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a01", {"aa"}};
  _a04  = CCSE_Tensors<T>{MO, {O, O}, "_a04", {"aa"}};
  _a05  = CCSE_Tensors<T>{MO, {O, V}, "_a05", {"aa"}}; // bb
  _a06  = CCSE_Tensors<T>{MO, {V, O, CI}, "_a06", {"aa"}};

  // T2
  _a007V = {CI};
  _a001  = CCSE_Tensors<T>{MO, {V, V}, "_a001", {"aa", "bb"}};
  _a006  = CCSE_Tensors<T>{MO, {O, O}, "_a006", {"aa", "bb"}};

  _a008 = CCSE_Tensors<T>{MO, {O, O, CI}, "_a008", {"aa"}};
  _a009 = CCSE_Tensors<T>{MO, {O, O, CI}, "_a009", {"aa", "bb"}};
  _a017 = CCSE_Tensors<T>{MO, {V, O, CI}, "_a017", {"aa", "bb"}};
  _a021 = CCSE_Tensors<T>{MO, {V, V, CI}, "_a021", {"aa", "bb"}};

  _a019 = CCSE_Tensors<T>{MO, {O, O, O, O}, "_a019", {"abab"}};
  _a022 = CCSE_Tensors<T>{MO, {V, V, V, V}, "_a022", {"abab"}};
  _a020 = CCSE_Tensors<T>{MO, {V, O, V, O}, "_a020", {"aaaa", "baba", "baab", "bbbb"}};

  sch.allocate(t2_aaaa);
  sch.allocate(d_e, i0_temp, t2_aaaa_temp, _a01V);
  CCSE_Tensors<T>::allocate_list(sch, f1_oo, f1_ov, f1_vv, chol3d_oo, chol3d_ov, chol3d_vv);
  CCSE_Tensors<T>::allocate_list(sch, _a02, _a03);

  // allocate all intermediates
  sch.allocate(_a02V, _a007V);
  CCSE_Tensors<T>::allocate_list(sch, _a004, _a01, _a04, _a05, _a06, _a001, _a006, _a008, _a009,
                                 _a017, _a019, _a020, _a021, _a022);
  sch.execute();

  tamm::random_ip(f1_oo("aa"));
  tamm::random_ip(f1_ov("aa"));
  tamm::random_ip(f1_vv("aa"));
  tamm::random_ip(chol3d_oo("aa"));
  tamm::random_ip(chol3d_ov("aa"));
  tamm::random_ip(chol3d_vv("aa"));

  // clang-format off
  sch
      (_a004("aaaa")(p1_va, p2_va, h4_oa, h3_oa) = 1.0 * chol3d_ov("aa")(h4_oa, p1_va, cind) *
      chol3d_ov("aa")(h3_oa, p2_va, cind)) .exact_copy(_a004("abab")(p1_va, p1_vb, h3_oa, h3_ob),
      _a004("aaaa")(p1_va, p1_vb, h3_oa, h3_ob))
      ;
  // clang-format on

  sch.execute(exhw);

  const auto timer_start = std::chrono::high_resolution_clock::now();

  ccsd_e_cs(sch, MO, CI, d_e, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se);
  ccsd_t1_cs(sch, MO, CI, r1_aa, t1_aa, t2_abab, f1_se, chol3d_se);
  ccsd_t2_cs(sch, MO, CI, r2_abab, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se);

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

  if(false) {
    ExecutionContext ec_dense{ec.pg(), DistributionKind::dense, MemoryManagerKind::ga};
    Tensor<T>        c3dvv_dense = tamm::to_dense_tensor(ec_dense, chol3d_vv("aa"));
    print_dense_tensor(c3dvv_dense, "c3d_vv_aa_dense");
  }

  // deallocate all intermediates
  sch.deallocate(_a02V, _a007V);
  CCSE_Tensors<T>::deallocate_list(sch, _a004, _a01, _a04, _a05, _a06, _a001, _a006, _a008, _a009,
                                   _a017, _a019, _a020, _a021, _a022);

  sch.deallocate(d_e, i0_temp, t2_aaaa_temp, _a01V);
  CCSE_Tensors<T>::deallocate_list(sch, _a02, _a03);
  CCSE_Tensors<T>::deallocate_list(sch, f1_oo, f1_ov, f1_vv, chol3d_oo, chol3d_ov, chol3d_vv);

  sch.execute();

  sch.deallocate(t1_aa, t2_abab, r1_aa, r2_abab, d_f1, t2_aaaa).execute();

  tamm::finalize();

  return 0;
}