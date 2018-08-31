#define CATCH_CONFIG_RUNNER

#include "toyHF/hartree_fock.hpp"
#include "toyHF/diis.hpp"
#include "toyHF/4index_transform.hpp"
#include "toyHF/NK.hpp"
#include "catch/catch.hpp"
#include "tamm/tamm.hpp"
#include "macdecls.h"
#include "ga-mpi.h"


using namespace tamm;

template<typename T>
std::ostream& operator << (std::ostream &os, std::vector<T>& vec){
    os << "[";
    for(auto &x: vec)
        os << x << ",";
    os << "]\n";
    return os;
}

template<typename T>
void print_tensor(Tensor<T> &t){
    for (auto it: t.loop_nest())
    {
        TAMM_SIZE size = t.block_size(it);
        std::vector<T> buf(size);
        t.get(it, buf);
        std::cout << "block" << it;
        for (TAMM_SIZE i = 0; i < size;i++)
      if (buf[i]>0.0000000000001||buf[i]<-0.0000000000001) {
      //   std::cout << buf[i] << " ";
         std::cout << buf[i] << endl;
//        std::cout << std::endl;
      }
    }

}

template<typename T>
void ccsd_e(ExecutionContext &ec,
            const TiledIndexSpace& MO, Tensor<T>& de, const Tensor<T>& t1,
            const Tensor<T>& t2, const Tensor<T>& f1, const Tensor<T>& v2) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    Tensor<T> i1{O, V};

    TiledIndexLabel p1, p2, p3, p4, p5;
    TiledIndexLabel h3, h4, h5, h6;

    std::tie(p1, p2, p3, p4, p5) = MO.labels<5>("virt");
    std::tie(h3, h4, h5, h6)     = MO.labels<4>("occ");

    Scheduler{&ec}.allocate(i1)
        (i1(h6, p5) = f1(h6, p5))
        (i1(h6, p5) += 0.5 * t1(p3, h4) * v2(h4, h6, p3, p5))
        (de() = 0)
        (de() += t1(p5, h6) * i1(h6, p5))
        (de() += 0.25 * t2(p1, p2, h3, h4) * v2(h3, h4, p1, p2))
        .deallocate(i1)
        .execute();
}

template<typename T>
void ccsd_t1(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, const Tensor<T>& t2, const Tensor<T>& f1,
             const Tensor<T>& v2) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    Tensor<T> t1_2_1{O, O};
    Tensor<T> t1_2_2_1{O, V};
    Tensor<T> t1_3_1{V, V};
    Tensor<T> t1_5_1{O, V};
    Tensor<T> t1_6_1{O, O, O, V};

    TiledIndexLabel p2, p3, p4, p5, p6, p7;
    TiledIndexLabel h1, h4, h5, h6, h7, h8;

    std::tie(p2, p3, p4, p5, p6, p7) = MO.labels<6>("virt");
    std::tie(h1, h4, h5, h6, h7, h8) = MO.labels<6>("occ");

    Scheduler sch{&ec};
    sch
      .allocate(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1)
      (t1_2_1(h7, h1) = 0)
      (t1_3_1(p2, p3)  = 0)
      ( i0(p2,h1)            =        f1(p2,h1))
      ( t1_2_1(h7,h1)        =        f1(h7,h1))
      ( t1_2_2_1(h7,p3)      =        f1(h7,p3))
      ( t1_2_2_1(h7,p3)     += -1   * t1(p5,h6)       * v2(h6,h7,p3,p5))
      ( t1_2_1(h7,h1)       +=        t1(p3,h1)       * t1_2_2_1(h7,p3))
      ( t1_2_1(h7,h1)       += -1   * t1(p4,h5)       * v2(h5,h7,h1,p4))
      ( t1_2_1(h7,h1)       += -0.5 * t2(p3,p4,h1,h5) * v2(h5,h7,p3,p4))
      ( i0(p2,h1)           += -1   * t1(p2,h7)       * t1_2_1(h7,h1))
      ( t1_3_1(p2,p3)        =        f1(p2,p3))
      ( t1_3_1(p2,p3)       += -1   * t1(p4,h5)       * v2(h5,p2,p3,p4))
      ( i0(p2,h1)           +=        t1(p3,h1)       * t1_3_1(p2,p3))
      ( i0(p2,h1)           += -1   * t1(p3,h4)       * v2(h4,p2,h1,p3))
      ( t1_5_1(h8,p7)        =        f1(h8,p7))
      ( t1_5_1(h8,p7)       +=        t1(p5,h6)       * v2(h6,h8,p5,p7))
      ( i0(p2,h1)           +=        t2(p2,p7,h1,h8) * t1_5_1(h8,p7))
      ( t1_6_1(h4,h5,h1,p3)  =        v2(h4,h5,h1,p3))
      ( t1_6_1(h4,h5,h1,p3) += -1   * t1(p6,h1)       * v2(h4,h5,p3,p6))
      ( i0(p2,h1)           += -0.5 * t2(p2,p3,h4,h5) * t1_6_1(h4,h5,h1,p3))
      ( i0(p2,h1)           += -0.5 * t2(p3,p4,h1,h5) * v2(h5,p2,p3,p4))
    .deallocate(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1)
    .execute();

}

template<typename T>
void ccsd_t2(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, Tensor<T>& t2, const Tensor<T>& f1,
             const Tensor<T>& v2) {
    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");

    Tensor<T> i0_temp{V, V, O, O};
    Tensor<T> t2_temp{V, V, O, O};
    Tensor<T> t2_2_1{O, V, O, O};
    Tensor<T> t2_2_1_temp{O, V, O, O};
    Tensor<T> t2_2_2_1{O, O, O, O};
    Tensor<T> t2_2_2_1_temp{O, O, O, O};
    Tensor<T> t2_2_2_2_1{O, O, O, V};
    Tensor<T> t2_2_4_1{O, V};
    Tensor<T> t2_2_5_1{O, O, O, V};
    Tensor<T> t2_4_1{O, O};
    Tensor<T> t2_4_2_1{O, V};
    Tensor<T> t2_5_1{V, V};
    Tensor<T> t2_6_1{O, O, O, O};
    Tensor<T> t2_6_1_temp{O, O, O, O};
    Tensor<T> t2_6_2_1{O, O, O, V};
    Tensor<T> t2_7_1{O, V, O, V};
    Tensor<T> vt1t1_1{O, V, O, O};
    Tensor<T> vt1t1_1_temp{O, V, O, O};

    TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8, p9;
    TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11;

    std::tie(p1, p2, p3, p4, p5, p6, p7, p8, p9) = MO.labels<9>("virt");
    std::tie(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11) = MO.labels<11>("occ");

    Scheduler sch{&ec};
    sch.allocate(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
             t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1,vt1t1_1_temp,t2_2_2_1_temp,
             t2_2_1_temp,i0_temp,t2_temp,t2_6_1_temp)
    (i0(p3, p4, h1, h2) = v2(p3, p4, h1, h2))
    (t2_4_1(h9, h1) = 0)
    (t2_5_1(p3, p5) = 0)
    (t2_2_1(h10, p3, h1, h2) = v2(h10, p3, h1, h2))

    (t2_2_2_1(h10, h11, h1, h2) = -1 * v2(h10, h11, h1, h2))
    (t2_2_2_2_1(h10, h11, h1, p5) = v2(h10, h11, h1, p5))
    (t2_2_2_2_1(h10, h11, h1, p5) += -0.5 * t1(p6, h1) * v2(h10, h11, p5, p6))

//    (t2_2_2_1(h10, h11, h1, h2) += t1(p5, h1) * t2_2_2_2_1(h10, h11, h2, p5))
//    (t2_2_2_1(h10, h11, h2, h1) += -1 * t1(p5, h1) * t2_2_2_2_1(h10, h11, h2, p5)) //perm symm
    (t2_2_2_1_temp(h10, h11, h1, h2) = 0)
    (t2_2_2_1_temp(h10, h11, h1, h2) += t1(p5, h1) * t2_2_2_2_1(h10, h11, h2, p5))
    (t2_2_2_1(h10, h11, h1, h2) += t2_2_2_1_temp(h10, h11, h1, h2))
    (t2_2_2_1(h10, h11, h2, h1) += -1 * t2_2_2_1_temp(h10, h11, h1, h2)) //perm symm

    (t2_2_2_1(h10, h11, h1, h2) += -0.5 * t2(p7, p8, h1, h2) * v2(h10, h11, p7, p8))
    (t2_2_1(h10, p3, h1, h2) += 0.5 * t1(p3, h11) * t2_2_2_1(h10, h11, h1, h2))
    
    (t2_2_4_1(h10, p5) = f1(h10, p5))
    (t2_2_4_1(h10, p5) += -1 * t1(p6, h7) * v2(h7, h10, p5, p6))
    (t2_2_1(h10, p3, h1, h2) += -1 * t2(p3, p5, h1, h2) * t2_2_4_1(h10, p5))
    (t2_2_5_1(h7, h10, h1, p9) = v2(h7, h10, h1, p9))
    (t2_2_5_1(h7, h10, h1, p9) += t1(p5, h1) * v2(h7, h10, p5, p9))

    // (t2_2_1(h10, p3, h1, h2) += t2(p3, p9, h1, h7) * t2_2_5_1(h7, h10, h2, p9))
    // (t2_2_1(h10, p3, h2, h1) += -1 * t2(p3, p9, h1, h7) * t2_2_5_1(h7, h10, h2, p9)) //perm symm
    (t2_2_1_temp(h10, p3, h1, h2) = 0)
    (t2_2_1_temp(h10, p3, h1, h2) += t2(p3, p9, h1, h7) * t2_2_5_1(h7, h10, h2, p9))
    (t2_2_1(h10, p3, h1, h2) += t2_2_1_temp(h10, p3, h1, h2))
    (t2_2_1(h10, p3, h2, h1) += -1 * t2_2_1_temp(h10, p3, h1, h2)) //perm symm

    // (t2(p1, p2, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4))
    // (t2(p1, p2, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
    // (t2(p2, p1, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    // (t2(p2, p1, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    (t2_temp(p1, p2, h3, h4) = 0)
    (t2_temp(p1, p2, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4))
    (t2(p1, p2, h3, h4) += t2_temp(p1, p2, h3, h4))
    (t2(p1, p2, h4, h3) += -1 * t2_temp(p1, p2, h3, h4)) //4 perms
    (t2(p2, p1, h3, h4) += -1 * t2_temp(p1, p2, h3, h4)) //perm
    (t2(p2, p1, h4, h3) += t2_temp(p1, p2, h3, h4)) //perm

    (t2_2_1(h10, p3, h1, h2) += 0.5 * t2(p5, p6, h1, h2) * v2(h10, p3, p5, p6))
    // (t2(p1, p2, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4))
    // (t2(p1, p2, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
    // (t2(p2, p1, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    // (t2(p2, p1, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    (t2(p1, p2, h3, h4) += -1 * t2_temp(p1, p2, h3, h4))
    (t2(p1, p2, h4, h3) += t2_temp(p1, p2, h3, h4)) //4 perms
    (t2(p2, p1, h3, h4) += t2_temp(p1, p2, h3, h4)) //perm
    (t2(p2, p1, h4, h3) += -1 * t2_temp(p1, p2, h3, h4)) //perm
    

//    (i0(p3, p4, h1, h2) += -1 * t1(p3, h10) * t2_2_1(h10, p4, h1, h2))
//    (i0(p4, p3, h1, h2) += 1 * t1(p3, h10) * t2_2_1(h10, p4, h1, h2)) //perm sym
    (i0_temp(p3, p4, h1, h2) = 0)
    (i0_temp(p3, p4, h1, h2) += t1(p3, h10) * t2_2_1(h10, p4, h1, h2))
    (i0(p3, p4, h1, h2) += -1 * i0_temp(p3, p4, h1, h2))
    (i0(p4, p3, h1, h2) += i0_temp(p3, p4, h1, h2)) //perm sym


    //  (i0(p3, p4, h1, h2) += -1 * t1(p5, h1) * v2(p3, p4, h2, p5))
    //  (i0(p3, p4, h2, h1) += 1 * t1(p5, h1) * v2(p3, p4, h2, p5)) //perm sym
    (i0_temp(p3, p4, h1, h2) = 0)
    (i0_temp(p3, p4, h1, h2) += t1(p5, h1) * v2(p3, p4, h2, p5))
    (i0(p3, p4, h1, h2) += -1 * i0_temp(p3, p4, h1, h2))
    (i0(p3, p4, h2, h1) += i0_temp(p3, p4, h1, h2)) //perm sym

    (t2_4_1(h9, h1) = f1(h9, h1))
    (t2_4_2_1(h9, p8) = f1(h9, p8))
    (t2_4_2_1(h9, p8) += t1(p6, h7) * v2(h7, h9, p6, p8))
    (t2_4_1(h9, h1) += t1(p8, h1) * t2_4_2_1(h9, p8))
    (t2_4_1(h9, h1) += -1 * t1(p6, h7) * v2(h7, h9, h1, p6))
    (t2_4_1(h9, h1) += -0.5 * t2(p6, p7, h1, h8) * v2(h8, h9, p6, p7))

    // (i0(p3, p4, h1, h2) += -1 * t2(p3, p4, h1, h9) * t2_4_1(h9, h2))
    // (i0(p3, p4, h2, h1) += 1 * t2(p3, p4, h1, h9) * t2_4_1(h9, h2)) //perm sym
    (i0_temp(p3, p4, h1, h2) = 0)
    (i0_temp(p3, p4, h1, h2) += t2(p3, p4, h1, h9) * t2_4_1(h9, h2))
    (i0(p3, p4, h1, h2) += -1 * i0_temp(p3, p4, h1, h2))
    (i0(p3, p4, h2, h1) += i0_temp(p3, p4, h1, h2)) //perm sym


    (t2_5_1(p3, p5) = f1(p3, p5))
    (t2_5_1(p3, p5) += -1 * t1(p6, h7) * v2(h7, p3, p5, p6))
    (t2_5_1(p3, p5) += -0.5 * t2(p3, p6, h7, h8) * v2(h7, h8, p5, p6))

//  (i0(p3, p4, h1, h2) += 1 * t2(p3, p5, h1, h2) * t2_5_1(p4, p5))
//  (i0(p4, p3, h1, h2) += -1 * t2(p3, p5, h1, h2) * t2_5_1(p4, p5)) //perm sym
    (i0_temp(p3, p4, h1, h2) = 0)
    (i0_temp(p3, p4, h1, h2) += t2(p3, p5, h1, h2) * t2_5_1(p4, p5))
    (i0(p3, p4, h1, h2) += i0_temp(p3, p4, h1, h2))
    (i0(p4, p3, h1, h2) += -1 * i0_temp(p3, p4, h1, h2)) //perm sym

    (t2_6_1(h9, h11, h1, h2) = -1 * v2(h9, h11, h1, h2))
    (t2_6_2_1(h9, h11, h1, p8) = v2(h9, h11, h1, p8))
    (t2_6_2_1(h9, h11, h1, p8) += 0.5 * t1(p6, h1) * v2(h9, h11, p6, p8))
    
//    (t2_6_1(h9, h11, h1, h2) += t1(p8, h1) * t2_6_2_1(h9, h11, h2, p8))
//    (t2_6_1(h9, h11, h2, h1) += -1 * t1(p8, h1) * t2_6_2_1(h9, h11, h2, p8)) //perm symm
    (t2_6_1_temp(h9, h11, h1, h2) = 0)
    (t2_6_1_temp(h9, h11, h1, h2) += t1(p8, h1) * t2_6_2_1(h9, h11, h2, p8))
    (t2_6_1(h9, h11, h1, h2) += t2_6_1_temp(h9, h11, h1, h2))
    (t2_6_1(h9, h11, h2, h1) += -1 * t2_6_1_temp(h9, h11, h1, h2)) //perm symm

    (t2_6_1(h9, h11, h1, h2) += -0.5 * t2(p5, p6, h1, h2) * v2(h9, h11, p5, p6))
    (i0(p3, p4, h1, h2) += -0.5 * t2(p3, p4, h9, h11) * t2_6_1(h9, h11, h1, h2))

    (t2_7_1(h6, p3, h1, p5) = v2(h6, p3, h1, p5))
    (t2_7_1(h6, p3, h1, p5) += -1 * t1(p7, h1) * v2(h6, p3, p5, p7))
    (t2_7_1(h6, p3, h1, p5) += -0.5 * t2(p3, p7, h1, h8) * v2(h6, h8, p5, p7))

    // (i0(p3, p4, h1, h2) += -1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5))
    // (i0(p3, p4, h2, h1) += 1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5)) //4 perms
    // (i0(p4, p3, h1, h2) += 1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5)) //perm
    // (i0(p4, p3, h2, h1) += -1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5)) //perm

    (i0_temp(p3, p4, h1, h2) = 0)
    (i0_temp(p3, p4, h1, h2) += t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5))
    (i0(p3, p4, h1, h2) += -1 * i0_temp(p3, p4, h1, h2))
    (i0(p3, p4, h2, h1) +=  1 * i0_temp(p3, p4, h1, h2)) //4 perms
    (i0(p4, p3, h1, h2) +=  1 * i0_temp(p3, p4, h1, h2)) //perm
    (i0(p4, p3, h2, h1) += -1 * i0_temp(p3, p4, h1, h2)) //perm

    //(vt1t1_1(h5, p3, h1, h2) = 0)
    //(vt1t1_1(h5, p3, h1, h2) += -2 * t1(p6, h1) * v2(h5, p3, h2, p6))
    //(vt1t1_1(h5, p3, h2, h1) += 2 * t1(p6, h1) * v2(h5, p3, h2, p6)) //perm symm
    (vt1t1_1_temp()=0)
    (vt1t1_1_temp(h5, p3, h1, h2) += t1(p6, h1) * v2(h5, p3, h2, p6))
    (vt1t1_1(h5, p3, h1, h2) = -2 * vt1t1_1_temp(h5, p3, h1, h2))
    (vt1t1_1(h5, p3, h2, h1) += 2 * vt1t1_1_temp(h5, p3, h1, h2)) //perm symm

    // (i0(p3, p4, h1, h2) += -0.5 * t1(p3, h5) * vt1t1_1(h5, p4, h1, h2))
    // (i0(p4, p3, h1, h2) += 0.5 * t1(p3, h5) * vt1t1_1(h5, p4, h1, h2)) //perm symm
    (i0_temp(p3, p4, h1, h2) = 0)
    (i0_temp(p3, p4, h1, h2) += -0.5 * t1(p3, h5) * vt1t1_1(h5, p4, h1, h2))
    (i0(p3, p4, h1, h2) += i0_temp(p3, p4, h1, h2))
    (i0(p4, p3, h1, h2) += -1 * i0_temp(p3, p4, h1, h2)) //perm symm

    // (t2(p1, p2, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4))
    // (t2(p1, p2, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
    // (t2(p2, p1, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    // (t2(p2, p1, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    (t2(p1, p2, h3, h4) += t2_temp(p1, p2, h3, h4))
    (t2(p1, p2, h4, h3) += -1 * t2_temp(p1, p2, h3, h4)) //4 perms
    (t2(p2, p1, h3, h4) += -1 * t2_temp(p1, p2, h3, h4)) //perm
    (t2(p2, p1, h4, h3) += t2_temp(p1, p2, h3, h4)) //perm

    (i0(p3, p4, h1, h2) += 0.5 * t2(p5, p6, h1, h2) * v2(p3, p4, p5, p6))
    
    // (t2(p1, p2, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4))
    // (t2(p1, p2, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
    // (t2(p2, p1, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perms
    // (t2(p2, p1, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perms
    (t2(p1, p2, h3, h4) += -1 * t2_temp(p1, p2, h3, h4))
    (t2(p1, p2, h4, h3) += t2_temp(p1, p2, h3, h4)) //4 perms
    (t2(p2, p1, h3, h4) += t2_temp(p1, p2, h3, h4)) //perms
    (t2(p2, p1, h4, h3) += -1 * t2_temp(p1, p2, h3, h4)) //perms

    .deallocate(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
              t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1,vt1t1_1_temp,t2_2_2_1_temp,
              t2_2_1_temp,i0_temp,t2_temp,t2_6_1_temp);
    sch.execute();

}


/**
 *
 * @tparam T
 * @param MO
 * @param p_evl_sorted
 * @return pair of residual and energy
 */
template<typename T>
std::pair<double,double> rest(ExecutionContext& ec,
                              const TiledIndexSpace& MO,
                               Tensor<T>& d_r1,
                               Tensor<T>& d_r2,
                               Tensor<T>& d_t1,
                               Tensor<T>& d_t2,
                              const Tensor<T>& de,
                              std::vector<T>& p_evl_sorted, T zshiftl, 
                              const TAMM_SIZE& noab, bool transpose=false) {

    T residual, energy;
    Scheduler sch{&ec};
    Tensor<T> d_r1_residual{}, d_r2_residual{};
    Tensor<T>::allocate(&ec,d_r1_residual, d_r2_residual);
    sch
      (d_r1_residual() = 0)
      (d_r2_residual() = 0)
      (d_r1_residual() += d_r1()  * d_r1())
      (d_r2_residual() += d_r2()  * d_r2())
      .execute();

      auto l0 = [&]() {
        T r1, r2;
        d_r1_residual.get({}, {&r1, 1});
        d_r2_residual.get({}, {&r2, 1});
        r1 = 0.5*std::sqrt(r1);
        r2 = 0.5*std::sqrt(r2);
        de.get({}, {&energy, 1});
        residual = std::max(r1,r2);
      };

      auto l1 =  [&]() {
        jacobi(ec, d_r1, d_t1, -1.0 * zshiftl, transpose, p_evl_sorted,noab);
      };
      auto l2 = [&]() {
        jacobi(ec, d_r2, d_t2, -2.0 * zshiftl, transpose, p_evl_sorted,noab);
      };

      l0();
      l1();
      l2();

      Tensor<T>::deallocate(d_r1_residual, d_r2_residual);
      
    return {residual, energy};
}


void iteration_print(const ProcGroup& pg, int iter, double residual, double energy) {
  if(pg.rank() == 0) {
    std::cout.width(6); std::cout << std::right << iter+1 << "  ";
    std::cout << std::setprecision(13) << residual << "  ";
    std::cout << std::fixed << std::setprecision(13) << energy << " ";
    std::cout << std::string(4, ' ') << "0.0";
    std::cout << std::string(5, ' ') << "0.0";
    std::cout << std::string(5, ' ') << "0.0" << std::endl;
  }
}

void iteration_print_lambda(const ProcGroup& pg, int iter, double residual) {
  if(pg.rank() == 0) {
    std::cout.width(6); std::cout << std::right << iter+1 << "  ";
    std::cout << std::setprecision(13) << residual << "  ";
    std::cout << std::string(8, ' ') << "0.0";
    std::cout << std::string(5, ' ') << "0.0" << std::endl;
  }
}


template<typename T>
void ccsd_driver(ExecutionContext* ec, const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_f1, Tensor<T>& d_v2,
                   int maxiter, double thresh,
                   double zshiftl,
                   int ndiis, double hf_energy,
                   long int total_orbitals, const TAMM_SIZE& noab) {

    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    const TiledIndexSpace& N = MO("all");

    std::cout.precision(15);

    Scheduler sch{ec};
  /// @todo: make it a tamm tensor
  std::cout << "Total orbitals = " << total_orbitals << std::endl;
  //std::vector<double> p_evl_sorted(total_orbitals);

    // Tensor<T> d_evl{N};
    // Tensor<T>::allocate(ec, d_evl);
    // TiledIndexLabel n1;
    // std::tie(n1) = MO.labels<1>("all");

    // sch(d_evl(n1) = 0.0)
    // .execute();

    std::vector<double> p_evl_sorted = d_f1.diagonal();
//   {
//       for(const auto& blockid : d_f1.loop_nest()) {
//           if(blockid[0] == blockid[1]) {
//               const TAMM_SIZE size = d_f1.block_size(blockid);
//               std::vector<T> buf(size);
//               d_f1.get(blockid, buf);
//               auto block_dims   = d_f1.block_dims(blockid);
//               auto block_offset = d_f1.block_offsets(blockid);
//               auto dim          = block_dims[0];
//               auto offset       = block_offset[0];
//               size_t i          = 0;
//               for(auto p = offset; p < offset + dim; p++, i++) {
//                   p_evl_sorted[p] = buf[i * dim + i];
//               }
//           }
//       }
//   }
//   ec->pg().barrier();

  if(ec->pg().rank() == 0) {
    std::cout << "p_evl_sorted:" << '\n';
    for(size_t p = 0; p < p_evl_sorted.size(); p++)
      std::cout << p_evl_sorted[p] << '\n';
  }

  if(ec->pg().rank() == 0) {
    std::cout << "\n\n";
    std::cout << " CCSD iterations" << std::endl;
    std::cout << std::string(66, '-') << std::endl;
    std::cout <<
        " Iter          Residuum       Correlation     Cpu    Wall    V2*C2"
              << std::endl;
    std::cout << std::string(66, '-') << std::endl;
  }
   
  std::vector<Tensor<T>*> d_r1s, d_r2s, d_t1s, d_t2s;

  for(int i=0; i<ndiis; i++) {
    d_r1s.push_back(new Tensor<T>{V,O});
    d_r2s.push_back(new Tensor<T>{V,V,O,O});
    d_t1s.push_back(new Tensor<T>{V,O});
    d_t2s.push_back(new Tensor<T>{V,V,O,O});
    Tensor<T>::allocate(ec,*d_r1s[i], *d_r2s[i], *d_t1s[i], *d_t2s[i]);
  }
 
  Tensor<T> d_r1{V,O};
  Tensor<T> d_r2{V,V,O,O};
  Tensor<T>::allocate(ec,d_r1, d_r2);

  Scheduler{ec}   
  (d_r1() = 0)
  (d_r2() = 0)
  .execute();

  double corr = 0;
  double residual = 0.0;
  double energy = 0.0;

  {
      auto lambda2 = [&](const IndexVector& blockid) {
          if(blockid[0] != blockid[1]) {
              Tensor<T> tensor     = d_f1;
              const TAMM_SIZE size = tensor.block_size(blockid);

              std::vector<T> buf(size);
              tensor.get(blockid, buf);

              auto block_dims   = tensor.block_dims(blockid);
              auto block_offset = tensor.block_offsets(blockid);

              TAMM_SIZE c = 0;
              for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0];
                  i++) {
                  for(auto j = block_offset[1];
                      j < block_offset[1] + block_dims[1]; j++, c++) {
                      buf[c] = 0;
                  }
              }
              d_f1.put(blockid, buf);
          }
      };
      block_for(ec->pg(), d_f1(), lambda2);
  }

  for(int titer = 0; titer < maxiter; titer += ndiis) {
      for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
          int off = iter - titer;

          Tensor<T> d_e{};
          Tensor<T> d_r1_residual{};
          Tensor<T> d_r2_residual{};

          Tensor<T>::allocate(ec, d_e, d_r1_residual, d_r2_residual);

          Scheduler{ec}(d_e() = 0)(d_r1_residual() = 0)(d_r2_residual() = 0)
            .execute();

          Scheduler{ec}((*d_t1s[off])() = d_t1())((*d_t2s[off])() = d_t2())
            .execute();

          ccsd_e(*ec, MO, d_e, d_t1, d_t2, d_f1, d_v2);
          ccsd_t1(*ec, MO, d_r1, d_t1, d_t2, d_f1, d_v2);
          ccsd_t2(*ec, MO, d_r2, d_t1, d_t2, d_f1, d_v2);

          std::tie(residual, energy) = rest(*ec, MO, d_r1, d_r2, d_t1, d_t2,
                                            d_e, p_evl_sorted, zshiftl, noab);

          {
              auto lambdar2 = [&](const IndexVector& blockid) {
                  if((blockid[0] > blockid[1]) || (blockid[2] > blockid[3])) {
                      Tensor<T> tensor     = d_r2;
                      const TAMM_SIZE size = tensor.block_size(blockid);

                      std::vector<T> buf(size);
                      tensor.get(blockid, buf);

                      auto block_dims   = tensor.block_dims(blockid);
                      auto block_offset = tensor.block_offsets(blockid);

                      TAMM_SIZE c = 0;
                      for(auto i = block_offset[0];
                          i < block_offset[0] + block_dims[0]; i++) {
                          for(auto j = block_offset[1];
                              j < block_offset[1] + block_dims[1]; j++) {
                              for(auto k = block_offset[2];
                                  k < block_offset[2] + block_dims[2]; k++) {
                                  for(auto l = block_offset[3];
                                      l < block_offset[3] + block_dims[3];
                                      l++, c++) {
                                      buf[c] = 0;
                                  }
                              }
                          }
                      }
                      d_r2.put(blockid, buf);
                  }
              };
              block_for(ec->pg(), d_r2(), lambdar2);
          }

          Scheduler{ec}((*d_r1s[off])() = d_r1())((*d_r2s[off])() = d_r2())
            .execute();

          iteration_print(ec->pg(), iter, residual, energy);
          Tensor<T>::deallocate(d_e, d_r1_residual, d_r2_residual);

          if(residual < thresh) { break; }
      }

      if(residual < thresh || titer + ndiis >= maxiter) { break; }
      if(ec->pg().rank() == 0) {
          std::cout << " MICROCYCLE DIIS UPDATE:";
          std::cout.width(21);
          std::cout << std::right << std::min(titer + ndiis, maxiter) + 1;
          std::cout.width(21);
          std::cout << std::right << "5" << std::endl;
      }

      std::vector<std::vector<Tensor<T>*>*> rs{&d_r1s, &d_r2s};
      std::vector<std::vector<Tensor<T>*>*> ts{&d_t1s, &d_t2s};
      std::vector<Tensor<T>*> next_t{&d_t1, &d_t2};
      diis<T>(*ec, rs, ts, next_t);
  }

  if(ec->pg().rank() == 0) {
    std::cout << std::string(66, '-') << std::endl;
    if(residual < thresh) {
        std::cout << " Iterations converged" << std::endl;
        std::cout.precision(15);
        std::cout << " CCSD correlation energy / hartree ="
                  << std::setw(26) << std::right << energy
                  << std::endl;
        std::cout << " CCSD total energy / hartree       ="
                  << std::setw(26) << std::right
                  << energy + hf_energy << std::endl;
    }
  }

  for(size_t i=0; i<ndiis; i++) {
    Tensor<T>::deallocate(*d_r1s[i], *d_r2s[i], *d_t1s[i], *d_t2s[i]);
  }
  d_r1s.clear();
  d_r2s.clear();
  Tensor<T>::deallocate(d_r1, d_r2);

}

template<typename T>
void lambda_ccsd_y1(ExecutionContext& ec, const TiledIndexSpace& MO,
                    Tensor<T>& i0, const Tensor<T>& t1, const Tensor<T>& t2,
                    const Tensor<T>& y1, const Tensor<T>& y2,
                    const Tensor<T>& f1, const Tensor<T>& v2) {

    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");

    TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;
    TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12;

    std::tie(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11) = MO.labels<11>("virt");
    std::tie(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12) = MO.labels<12>("occ");

    Tensor<T> i_1     {O , O};
    Tensor<T> i_1_1   {O , V};
    Tensor<T> i_2     {V , V};
    Tensor<T> i_2_1   {O , V};
    Tensor<T> i_3     {V , O};
    Tensor<T> i_3_1   {O , O};
    Tensor<T> i_3_1_1 {O , V};
    Tensor<T> i_3_2   {V , V};
    Tensor<T> i_3_3   {O , V};
    Tensor<T> i_3_4   {O,O , O,V};
    Tensor<T> i_4     {O,V , O,O};
    Tensor<T> i_4_1   {O,O , O,O};
    Tensor<T> i_4_1_1 {O,O , O,V};
    Tensor<T> i_4_2   {O,V , O,V};
    Tensor<T> i_4_3   {O , V};
    Tensor<T> i_4_4   {O,O , O,V};
    Tensor<T> i_5     {V,V , O,V};
    Tensor<T> i_6     {V , O};
    Tensor<T> i_6_1   {O , O};
    Tensor<T> i_6_2   {O,O , O,V};
    Tensor<T> i_7     {O , O};
    Tensor<T> i_8     {O , O};
    Tensor<T> i_9     {V , V};
    Tensor<T> i_10    {O,O , O,V};
    Tensor<T> i_11    {O,V , O,O};
    Tensor<T> i_11_1  {O,O , O,O};
    Tensor<T> i_11_1_1{O,O , O,V};
    Tensor<T> i_11_2  {O,O , O,V};
    Tensor<T> i_11_3  {O , O};
    Tensor<T> i_12    {O,O , O,O};
    Tensor<T> i_12_1  {O,O , O,V};
    Tensor<T> i_13    {O,V , O,V};
    Tensor<T> i_13_1  {O,O , O,V};

    Scheduler sch{&ec};

    sch
      .allocate(i_1, i_1_1, i_2, i_2_1, i_3, i_3_1, i_3_1_1, i_3_2, i_3_3, i_3_4,
             i_4, i_4_1, i_4_1_1, i_4_2, i_4_3, i_4_4, i_5, i_6, i_6_1, i_6_2,
             i_7, i_8, i_9, i_10, i_11, i_11_1, i_11_1_1, i_11_2, i_11_3, i_12,
             i_12_1, i_13, i_13_1)
      ( i0(h2,p1)                     =         f1(h2,p1)                                   )
      (   i_1(h2,h7)                  =         f1(h2,h7)                                   )
      (     i_1_1(h2,p3)              =         f1(h2,p3)                                   )
      (     i_1_1(h2,p3)             +=         t1(p5,h6)          * v2(h2,h6,p3,p5)        )
      (   i_1(h2,h7)                 +=         t1(p3,h7)          * i_1_1(h2,p3)           )
      (   i_1(h2,h7)                 +=         t1(p3,h4)          * v2(h2,h4,h7,p3)        )
      (   i_1(h2,h7)                 += -0.5  * t2(p3,p4,h6,h7)    * v2(h2,h6,p3,p4)        )
      ( i0(h2,p1)                    += -1    * y1(h7,p1)          * i_1(h2,h7)             )
      (   i_2(p7,p1)                  =         f1(p7,p1)                                   )
      (   i_2(p7,p1)                 += -1    * t1(p3,h4)          * v2(h4,p7,p1,p3)        )
      (     i_2_1(h4,p1)              =  0                                                  )
      (     i_2_1(h4,p1)             +=         t1(p5,h6)          * v2(h4,h6,p1,p5)        )
      (   i_2(p7,p1)                 += -1    * t1(p7,h4)          * i_2_1(h4,p1)           )
      ( i0(h2,p1)                    +=         y1(h2,p7)          * i_2(p7,p1)             )
      ( i0(h2,p1)                    += -1    * y1(h4,p3)          * v2(h2,p3,h4,p1)        )
      (   i_3(p9,h11)                 =         f1(p9,h11)                                  )
      (     i_3_1(h10,h11)            =         f1(h10,h11)                                 )
      (       i_3_1_1(h10,p3)         =         f1(h10,p3)                                  )
      (       i_3_1_1(h10,p3)        += -1    * t1(p7,h8)          * v2(h8,h10,p3,p7)       ) 
      (     i_3_1(h10,h11)           +=         t1(p3,h11)         * i_3_1_1(h10,p3)        )
      (     i_3_1(h10,h11)           += -1    * t1(p5,h6)          * v2(h6,h10,h11,p5)      )
      (     i_3_1(h10,h11)           +=  0.5  * t2(p3,p4,h6,h11)   * v2(h6,h10,p3,p4)       )
      (   i_3(p9,h11)                += -1    * t1(p9,h10)         * i_3_1(h10,h11)         )
      (     i_3_2(p9,p7)              =         f1(p9,p7)                                   )
      (     i_3_2(p9,p7)             +=         t1(p5,h6)          * v2(h6,p9,p5,p7)        )
      (   i_3(p9,h11)                +=         t1(p7,h11)         * i_3_2(p9,p7)           )
      (   i_3(p9,h11)                += -1    * t1(p3,h4)          * v2(h4,p9,h11,p3)       )
      (     i_3_3(h5,p4)              =         f1(h5,p4)                                   )
      (     i_3_3(h5,p4)             +=         t1(p7,h8)          * v2(h5,h8,p4,p7)        )
      (   i_3(p9,h11)                +=         t2(p4,p9,h5,h11)   * i_3_3(h5,p4)           )
      (     i_3_4(h5,h6,h11,p4)       =         v2(h5,h6,h11,p4)                            )
      (     i_3_4(h5,h6,h11,p4)      += -1    * t1(p7,h11)         * v2(h5,h6,p4,p7)        )
      (   i_3(p9,h11)                +=  0.5  * t2(p4,p9,h5,h6)    * i_3_4(h5,h6,h11,p4)    )
      (   i_3(p9,h11)                +=  0.5  * t2(p3,p4,h6,h11)   * v2(h6,p9,p3,p4)        )
      ( i0(h2,p1)                    +=         y2(h2,h11,p1,p9)   * i_3(p9,h11)            )      
      (   i_4(h2,p9,h11,h12)          =         v2(h2,p9,h11,h12)                           )
      (     i_4_1(h2,h7,h11,h12)      =         v2(h2,h7,h11,h12)                           )
      (       i_4_1_1(h2,h7,h12,p3)   =         v2(h2,h7,h12,p3)                            )
      (       i_4_1_1(h2,h7,h12,p3)  += -0.5  * t1(p5,h12)         * v2(h2,h7,p3,p5)        )
      (     i_4_1(h2,h7,h11,h12)     += -2    * t1(p3,h11)         * i_4_1_1(h2,h7,h12,p3)  )
      (     i_4_1(h2,h7,h11,h12)     +=  0.5  * t2(p3,p4,h11,h12)  * v2(h2,h7,p3,p4)        )
      (   i_4(h2,p9,h11,h12)         += -1    * t1(p9,h7)          * i_4_1(h2,h7,h11,h12)   )
      (     i_4_2(h2,p9,h12,p3)       =         v2(h2,p9,h12,p3)                            )
      (     i_4_2(h2,p9,h12,p3)      += -0.5  * t1(p5,h12)         * v2(h2,p9,p3,p5)        )
      (   i_4(h2,p9,h11,h12)         += -2    * t1(p3,h11)         * i_4_2(h2,p9,h12,p3)    )
      (     i_4_3(h2,p5)              =         f1(h2,p5)                                   )
      (     i_4_3(h2,p5)             +=         t1(p7,h8)          * v2(h2,h8,p5,p7)        )
      (   i_4(h2,p9,h11,h12)         +=         t2(p5,p9,h11,h12)  * i_4_3(h2,p5)           )
      (     i_4_4(h2,h6,h12,p4)       =         v2(h2,h6,h12,p4)                            )
      (     i_4_4(h2,h6,h12,p4)      += -1    * t1(p7,h12)         * v2(h2,h6,p4,p7)        )
      (   i_4(h2,p9,h11,h12)         += -2    * t2(p4,p9,h6,h11)   * i_4_4(h2,h6,h12,p4)    )
      (   i_4(h2,p9,h11,h12)         +=  0.5  * t2(p3,p4,h11,h12)  * v2(h2,p9,p3,p4)        )
      ( i0(h2,p1)                    += -0.5  * y2(h11,h12,p1,p9)  * i_4(h2,p9,h11,h12)     )
      (   i_5(p5,p8,h7,p1)            = -1    * v2(p5,p8,h7,p1)                             )
      (   i_5(p5,p8,h7,p1)           +=         t1(p3,h7)          * v2(p5,p8,p1,p3)        )
      ( i0(h2,p1)                    +=  0.5  * y2(h2,h7,p5,p8)    * i_5(p5,p8,h7,p1)       )
      (   i_6(p9,h10)                 =         t1(p9,h10)                                  )
      (   i_6(p9,h10)                +=         t2(p3,p9,h5,h10)   * y1(h5,p3)              )
      (     i_6_1(h6,h10)             =  0                                                  )
      (     i_6_1(h6,h10)            +=         t1(p5,h10)         * y1(h6,p5)              )
      (     i_6_1(h6,h10)            +=  0.5  * t2(p3,p4,h5,h10)   * y2(h5,h6,p3,p4)        )
      (   i_6(p9,h10)                += -1    * t1(p9,h6)          * i_6_1(h6,h10)          )
      (     i_6_2(h5,h6,h10,p3)       =  0                                                  )
      (     i_6_2(h5,h6,h10,p3)      +=         t1(p7,h10)         * y2(h5,h6,p3,p7)        )
      (   i_6(p9,h10)                += -0.5  * t2(p3,p9,h5,h6)    * i_6_2(h5,h6,h10,p3)    )
      ( i0(h2,p1)                    +=         i_6(p9,h10)        * v2(h2,h10,p1,p9)       )
      (   i_7(h2,h3)                  =  0                                                  )
      (   i_7(h2,h3)                 +=         t1(p4,h3)          * y1(h2,p4)              )
      (   i_7(h2,h3)                 +=  0.5  * t2(p4,p5,h3,h6)    * y2(h2,h6,p4,p5)        )
      ( i0(h2,p1)                    += -1    * i_7(h2,h3)         * f1(h3,p1)              )
      (   i_8(h6,h8)                  =  0                                                  )
      (   i_8(h6,h8)                 +=         t1(p3,h8)          * y1(h6,p3)              )
      (   i_8(h6,h8)                 +=  0.5  * t2(p3,p4,h5,h8)    * y2(h5,h6,p3,p4)        )
      ( i0(h2,p1)                    +=         i_8(h6,h8)         * v2(h2,h8,h6,p1)        )
      (   i_9(p7,p8)                  =  0                                                  )
      (   i_9(p7,p8)                 +=         t1(p7,h4)          * y1(h4,p8)              )
      (   i_9(p7,p8)                 +=  0.5  * t2(p3,p7,h5,h6)    * y2(h5,h6,p3,p8)        )
      ( i0(h2,p1)                    +=         i_9(p7,p8)         * v2(h2,p8,p1,p7)        )
      (   i_10(h2,h6,h4,p5)           =  0                                                  )
      (   i_10(h2,h6,h4,p5)          +=         t1(p3,h4)          * y2(h2,h6,p3,p5)        )
      ( i0(h2,p1)                    +=         i_10(h2,h6,h4,p5)  * v2(h4,p5,h6,p1)        )
      (   i_11(h2,p9,h6,h12)          =  0                                                  )
      (   i_11(h2,p9,h6,h12)         += -1    * t2(p3,p9,h6,h12)   * y1(h2,p3)              )
      (     i_11_1(h2,h10,h6,h12)     =  0                                                  )
      (     i_11_1(h2,h10,h6,h12)    += -1    * t2(p3,p4,h6,h12)   * y2(h2,h10,p3,p4)       )
      (       i_11_1_1(h2,h10,h6,p5)  =  0                                                  )
      (       i_11_1_1(h2,h10,h6,p5) +=         t1(p7,h6)          * y2(h2,h10,p5,p7)       )
      (     i_11_1(h2,h10,h6,h12)    +=  2    * t1(p5,h12)         * i_11_1_1(h2,h10,h6,p5) )
      (   i_11(h2,p9,h6,h12)         += -0.5  * t1(p9,h10)         * i_11_1(h2,h10,h6,h12)  )
      (     i_11_2(h2,h5,h6,p3)       =  0                                                  )
      (     i_11_2(h2,h5,h6,p3)      +=         t1(p7,h6)          * y2(h2,h5,p3,p7)        )
      (   i_11(h2,p9,h6,h12)         +=  2    * t2(p3,p9,h5,h12)   * i_11_2(h2,h5,h6,p3)    )
      (     i_11_3(h2,h12)            =  0                                                  )
      (     i_11_3(h2,h12)           +=         t2(p3,p4,h5,h12)   * y2(h2,h5,p3,p4)        )
      (   i_11(h2,p9,h6,h12)         += -1    * t1(p9,h6)          * i_11_3(h2,h12)         )
      ( i0(h2,p1)                    +=  0.5  * i_11(h2,p9,h6,h12) * v2(h6,h12,p1,p9)       )
      (   i_12(h2,h7,h6,h8)           =  0                                                  )
      (   i_12(h2,h7,h6,h8)          += -1    * t2(p3,p4,h6,h8)    * y2(h2,h7,p3,p4)        )
      (     i_12_1(h2,h7,h6,p3)       =  0                                                  )
      (     i_12_1(h2,h7,h6,p3)      +=         t1(p5,h6)          * y2(h2,h7,p3,p5)        )
      (   i_12(h2,h7,h6,h8)          +=  2    * t1(p3,h8)          * i_12_1(h2,h7,h6,p3)    )
      ( i0(h2,p1)                    +=  0.25 * i_12(h2,h7,h6,h8)  * v2(h6,h8,h7,p1)        )
      (   i_13(h2,p8,h6,p7)           =  0                                                  )
      (   i_13(h2,p8,h6,p7)          +=         t2(p3,p8,h5,h6)    * y2(h2,h5,p3,p7)        )
      (     i_13_1(h2,h4,h6,p7)       =  0                                                  )
      (     i_13_1(h2,h4,h6,p7)      +=         t1(p5,h6)          * y2(h2,h4,p5,p7)        )
      (   i_13(h2,p8,h6,p7)          += -1    * t1(p8,h4)          * i_13_1(h2,h4,h6,p7)    )
      ( i0(h2,p1)                    +=         i_13(h2,p8,h6,p7)  * v2(h6,p7,p1,p8)        )      
   .deallocate(i_1, i_1_1, i_2, i_2_1, i_3, i_3_1, i_3_1_1, i_3_2, i_3_3,
            i_3_4, i_4, i_4_1, i_4_1_1, i_4_2, i_4_3, i_4_4, i_5, i_6,
            i_6_1, i_6_2, i_7, i_8, i_9, i_10, i_11, i_11_1, i_11_1_1,
            i_11_2, i_11_3, i_12, i_12_1, i_13, i_13_1).execute();
}

template<typename T>
void lambda_ccsd_y2(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, Tensor<T>& t2, const Tensor<T>& y1, Tensor<T>& y2,
             const Tensor<T>& f1,  const Tensor<T>& v2) {

    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");

    TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;
    TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12;

    std::tie(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11) = MO.labels<11>("virt");
    std::tie(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12) = MO.labels<12>("occ");

  Tensor<T> i_1    {O,V};
  Tensor<T> i_2    {O,O,O,V};
  Tensor<T> i_3    {O,O};
  Tensor<T> i_3_1  {O,V};
  Tensor<T> i_4    {V,V};
  Tensor<T> i_4_1  {O,V};
  Tensor<T> i_5    {O,O,O,O};
  Tensor<T> i_5_1  {O,O,O,V};
  Tensor<T> i_6    {O,V,O,V};
  Tensor<T> i_7    {O,O};
  Tensor<T> i_8    {O,O,O,V};
  Tensor<T> i_9    {O,O,O,V};
  Tensor<T> i_10   {O,O,O,V};
  Tensor<T> i_11   {V,V};
  Tensor<T> i_12   {O,O,O,O};
  Tensor<T> i_12_1 {O,O,O,V};
  Tensor<T> i_13   {O,V,O,V};
  Tensor<T> i_13_1 {O,O,O,V};

  Scheduler sch{&ec};

  sch.allocate(i_1, i_2, i_3, i_3_1, i_4, i_4_1, i_5,
            i_5_1, i_6, i_7, i_8, i_9, i_10, i_11,
            i_12, i_12_1, i_13, i_13_1)
      ( i0(h3,h4,p1,p2)          =         v2(h3,h4,p1,p2)                         )
      (   i_1(h3,p1)             =         f1(h3,p1)                               )
      (   i_1(h3,p1)            +=         t1(p5,h6)         * v2(h3,h6,p1,p5)     )
      ( i0(h3,h4,p1,p2)         +=         y1(h3,p1)         * i_1(h4,p2)          )
      ( i0(h3,h4,p2,p1)         += -1.0  * y1(h3,p1)         * i_1(h4,p2)          ) //P(p1/p2)
      ( i0(h4,h3,p1,p2)         += -1.0  * y1(h3,p1)         * i_1(h4,p2)          ) //P(h3/h4)
      ( i0(h4,h3,p2,p1)         +=         y1(h3,p1)         * i_1(h4,p2)          ) //P(p1/p2,h3/h4)
      (   i_2(h3,h4,h7,p1)       =         v2(h3,h4,h7,p1)                         )
      (   i_2(h3,h4,h7,p1)      += -1    * t1(p5,h7)         * v2(h3,h4,p1,p5)     )
      ( i0(h3,h4,p1,p2)         += -1    * y1(h7,p1)         * i_2(h3,h4,h7,p2)    )
      ( i0(h3,h4,p2,p1)         +=         y1(h7,p1)         * i_2(h3,h4,h7,p2)    ) //P(p1/p2)
      ( i0(h3,h4,p1,p2)         += -1    * y1(h3,p5)         * v2(h4,p5,p1,p2)     )
      ( i0(h4,h3,p1,p2)         +=         y1(h3,p5)         * v2(h4,p5,p1,p2)     ) //P(h3/h4)
      (   i_3(h3,h9)             =         f1(h3,h9)                               )
      (     i_3_1(h3,p5)         =         f1(h3,p5)                               )
      (     i_3_1(h3,p5)        +=         t1(p7,h8)         * v2(h3,h8,p5,p7)     )
      (   i_3(h3,h9)            +=         t1(p5,h9)         * i_3_1(h3,p5)        )
      (   i_3(h3,h9)            +=         t1(p5,h6)         * v2(h3,h6,h9,p5)     )
      (   i_3(h3,h9)            += -0.5  * t2(p5,p6,h8,h9)   * v2(h3,h8,p5,p6)     )
      ( i0(h3,h4,p1,p2)         += -1    * y2(h3,h9,p1,p2)   * i_3(h4,h9)          )
      ( i0(h4,h3,p1,p2)         +=         y2(h3,h9,p1,p2)   * i_3(h4,h9)          ) //P(h3/h4)
      (   i_4(p10,p1)            =         f1(p10,p1)                              )
      (   i_4(p10,p1)           += -1    * t1(p5,h6)         * v2(h6,p10,p1,p5)    )
      (   i_4(p10,p1)           +=  0.5  * t2(p6,p10,h7,h8)  * v2(h7,h8,p1,p6)     )
      (     i_4_1(h6,p1)         =  0                                              )
      (     i_4_1(h6,p1)        +=         t1(p7,h8)         * v2(h6,h8,p1,p7)     )
      (   i_4(p10,p1)           += -1    * t1(p10,h6)        * i_4_1(h6,p1)        )
      ( i0(h3,h4,p1,p2)         +=         y2(h3,h4,p1,p10)  * i_4(p10,p2)         )
      ( i0(h3,h4,p2,p1)         += -1    * y2(h3,h4,p1,p10)  * i_4(p10,p2)         ) //P(p1/p2)
      (   i_5(h3,h4,h9,h10)      =         v2(h3,h4,h9,h10)                        )
      (     i_5_1(h3,h4,h10,p5)  =         v2(h3,h4,h10,p5)                        )
      (     i_5_1(h3,h4,h10,p5) += -0.5  * t1(p7,h10)        * v2(h3,h4,p5,p7)     )
      (   i_5(h3,h4,h9,h10)     += -2    * t1(p5,h9)         * i_5_1(h3,h4,h10,p5) )
      (   i_5(h3,h4,h9,h10)     +=  0.5  * t2(p5,p6,h9,h10)  * v2(h3,h4,p5,p6)     )
      ( i0(h3,h4,p1,p2)         +=  0.5  * y2(h9,h10,p1,p2)  * i_5(h3,h4,h9,h10)   )
      (   i_6(h3,p7,h9,p1)       =         v2(h3,p7,h9,p1)                         )
      (   i_6(h3,p7,h9,p1)      += -1    * t1(p5,h9)         * v2(h3,p7,p1,p5)     )
      (   i_6(h3,p7,h9,p1)      += -1    * t2(p6,p7,h8,h9)   * v2(h3,h8,p1,p6)     )
      ( i0(h3,h4,p1,p2)         += -1    * y2(h3,h9,p1,p7)   * i_6(h4,p7,h9,p2)    )
      ( i0(h3,h4,p2,p1)         +=         y2(h3,h9,p1,p7)   * i_6(h4,p7,h9,p2)    ) //P(p1/p2)
      ( i0(h4,h3,p1,p2)         +=         y2(h3,h9,p1,p7)   * i_6(h4,p7,h9,p2)    ) //P(h3/h4)
      ( i0(h4,h3,p2,p1)         += -1    * y2(h3,h9,p1,p7)   * i_6(h4,p7,h9,p2)    ) //P(p1/p2,h3/h3)
      ( i0(h3,h4,p1,p2)         +=  0.5  * y2(h3,h4,p5,p6)   * v2(p5,p6,p1,p2)     )
      (   i_7(h3,h9)             =  0                                              )
      (   i_7(h3,h9)            +=         t1(p5,h9)         * y1(h3,p5)           )
      (   i_7(h3,h9)            += -0.5  * t2(p5,p6,h7,h9)   * y2(h3,h7,p5,p6)     )
      ( i0(h3,h4,p1,p2)         +=         i_7(h3,h9)        * v2(h4,h9,p1,p2)     )
      ( i0(h4,h3,p1,p2)         += -1    * i_7(h3,h9)        * v2(h4,h9,p1,p2)     ) //P(h3/h4)
      (   i_8(h3,h4,h5,p1)       =  0                                              )
      (   i_8(h3,h4,h5,p1)      += -1    * t1(p6,h5)         * y2(h3,h4,p1,p6)     )
      ( i0(h3,h4,p1,p2)         +=         i_8(h3,h4,h5,p1)  * f1(h5,p2)           )
      ( i0(h3,h4,p1,p2)         += -1    * i_8(h3,h4,h5,p1)  * f1(h5,p2)           ) //P(p1/p2)
      (   i_9(h3,h7,h6,p1)       =  0                                              )
      (   i_9(h3,h7,h6,p1)      +=         t1(p5,h6)         * y2(h3,h7,p1,p5)     )
      ( i0(h3,h4,p1,p2)         +=         i_9(h3,h7,h6,p1)  * v2(h4,h6,h7,p2)     )
      ( i0(h3,h4,p2,p1)         += -1    * i_9(h3,h7,h6,p1)  * v2(h4,h6,h7,p2)     ) //P(p1/p2)
      ( i0(h4,h3,p1,p2)         += -1    * i_9(h3,h7,h6,p1)  * v2(h4,h6,h7,p2)     ) //P(h3/h4)
      ( i0(h4,h3,p2,p1)         +=         i_9(h3,h7,h6,p1)  * v2(h4,h6,h7,p2)     ) //P(p1/p2,h3/h3)
      (   i_10(h3,h4,h6,p7)      =  0                                              )
      (   i_10(h3,h4,h6,p7)     += -1    * t1(p5,h6)         * y2(h3,h4,p5,p7)     )
      ( i0(h3,h4,p1,p2)         +=         i_10(h3,h4,h6,p7) * v2(h6,p7,p1,p2)     )
      (   i_11(p6,p1)            =  0                                              )
      (   i_11(p6,p1)           +=         t2(p5,p6,h7,h8)   * y2(h7,h8,p1,p5)     )
      ( i0(h3,h4,p1,p2)         += -0.5  * i_11(p6,p1)       * v2(h3,h4,p2,p6)     )
      ( i0(h3,h4,p2,p1)         +=  0.5  * i_11(p6,p1)       * v2(h3,h4,p2,p6)     ) //P(p1/p2)
      (   i_12(h3,h4,h8,h9)      =  0                                              )
      (   i_12(h3,h4,h8,h9)     +=         t2(p5,p6,h8,h9)   * y2(h3,h4,p5,p6)     )
      (     i_12_1(h3,h4,h8,p5)  =  0                                              )
      (     i_12_1(h3,h4,h8,p5) += -1    * t1(p7,h8)         * y2(h3,h4,p5,p7)     )
      (   i_12(h3,h4,h8,h9)     +=  2    * t1(p5,h9)         * i_12_1(h3,h4,h8,p5) )
      ( i0(h3,h4,p1,p2)         +=  0.25 * i_12(h3,h4,h8,h9) * v2(h8,h9,p1,p2)     )
      (     i_13_1(h3,h6,h8,p1)  =  0                                              )
      (     i_13_1(h3,h6,h8,p1) +=         t1(p7,h8)         * y2(h3,h6,p1,p7)     )
      (   i_13(h3,p5,h8,p1)      =  0                                              )
      (   i_13(h3,p5,h8,p1)     +=         t1(p5,h6)         * i_13_1(h3,h6,h8,p1) )
      ( i0(h3,h4,p1,p2)         += -1    * i_13(h3,p5,h8,p1) * v2(h4,h8,p2,p5)     )
      ( i0(h3,h4,p2,p1)         +=         i_13(h3,p5,h8,p1) * v2(h4,h8,p2,p5)     ) //P(p1/p2)
      ( i0(h4,h3,p1,p2)         +=         i_13(h3,p5,h8,p1) * v2(h4,h8,p2,p5)     ) //P(h3/h4)
      ( i0(h4,h3,p2,p1)         += -1    * i_13(h3,p5,h8,p1) * v2(h4,h8,p2,p5)     ) //P(p1/p2,h3/h4)
   .deallocate(i_1, i_2, i_3, i_3_1, i_4, i_4_1, i_5, 
            i_5_1, i_6, i_7, i_8, i_9, i_10, i_11, 
            i_12, i_12_1, i_13, i_13_1).execute();                                           
}

template<typename T>
void lambda_ccsd_driver(ExecutionContext* ec, const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_y1, Tensor<T>& d_y2,
                   Tensor<T>& d_f1, Tensor<T>& d_v2,
                   int maxiter, double thresh,
                   double zshiftl,
                   int ndiis, double hf_energy,
                   long int total_orbitals, const TAMM_SIZE& noab) {

    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    const TiledIndexSpace& N = MO("all");

    std::cout.precision(15);

    Scheduler sch{ec};
  /// @todo: make it a tamm tensor
  std::cout << "Total orbitals = " << total_orbitals << std::endl;
  //std::vector<double> p_evl_sorted(total_orbitals);

    // Tensor<T> d_evl{N};
    // Tensor<T>::allocate(ec, d_evl);
    // TiledIndexLabel n1;
    // std::tie(n1) = MO.labels<1>("all");

    // sch(d_evl(n1) = 0.0)
    // .execute();

      std::vector<double> p_evl_sorted = d_f1.diagonal();

//   {
//       for(const auto& blockid : d_f1.loop_nest()) {
//           if(blockid[0] == blockid[1]) {
//               const TAMM_SIZE size = d_f1.block_size(blockid);
//               std::vector<T> buf(size);
//               d_f1.get(blockid, buf);
//               auto block_dims   = d_f1.block_dims(blockid);
//               auto block_offset = d_f1.block_offsets(blockid);
//               auto dim          = block_dims[0];
//               auto offset       = block_offset[0];
//               size_t i          = 0;
//               for(auto p = offset; p < offset + dim; p++, i++) {
//                   p_evl_sorted[p] = buf[i * dim + i];
//               }
//           }
//       }
//   }
//   ec->pg().barrier();

  if(ec->pg().rank() == 0) {
    std::cout << "p_evl_sorted:" << '\n';
    for(size_t p = 0; p < p_evl_sorted.size(); p++)
      std::cout << p_evl_sorted[p] << '\n';
  }

  if(ec->pg().rank() == 0) {
    std::cout << "\n\n";
    std::cout << " Lambda CCSD iterations" << std::endl;
    std::cout << std::string(44, '-') << std::endl;
    std::cout <<
        " Iter          Residuum          Cpu    Wall"
              << std::endl;
    std::cout << std::string(44, '-') << std::endl;
  }
   
  std::vector<Tensor<T>*> d_r1s, d_r2s, d_y1s, d_y2s;

  for(int i=0; i<ndiis; i++) {
    d_r1s.push_back(new Tensor<T>{O,V});
    d_r2s.push_back(new Tensor<T>{O,O,V,V});
    d_y1s.push_back(new Tensor<T>{O,V});
    d_y2s.push_back(new Tensor<T>{O,O,V,V});
    Tensor<T>::allocate(ec,*d_r1s[i], *d_r2s[i], *d_y1s[i], *d_y2s[i]);
  }
 
  Tensor<T> d_r1{O,V};
  Tensor<T> d_r2{O,O,V,V};
  Tensor<T>::allocate(ec,d_r1, d_r2);

  Scheduler{ec}   
  (d_r1() = 0)
  (d_r2() = 0)
  .execute();

  double corr = 0;
  double residual = 0.0;
  double energy = 0.0;

  {
      auto lambda2 = [&](const IndexVector& blockid) {
          if(blockid[0] != blockid[1]) {
              Tensor<T> tensor     = d_f1;
              const TAMM_SIZE size = tensor.block_size(blockid);

              std::vector<T> buf(size);
              tensor.get(blockid, buf);

              auto block_dims   = tensor.block_dims(blockid);
              auto block_offset = tensor.block_offsets(blockid);

              TAMM_SIZE c = 0;
              for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0];
                  i++) {
                  for(auto j = block_offset[1];
                      j < block_offset[1] + block_dims[1]; j++, c++) {
                      buf[c] = 0;
                  }
              }
              d_f1.put(blockid, buf);
          }
      };
      block_for(ec->pg(), d_f1(), lambda2);
  }

  for(int titer = 0; titer < maxiter; titer += ndiis) {
      for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
          int off = iter - titer;

          Tensor<T> d_e{};
          Tensor<T> d_r1_residual{};
          Tensor<T> d_r2_residual{};

          Tensor<T>::allocate(ec, d_e, d_r1_residual, d_r2_residual);

          Scheduler{ec}(d_e() = 0)(d_r1_residual() = 0)(d_r2_residual() = 0)
            .execute();

          Scheduler{ec}
          ((*d_y1s[off])() = d_y1())
          ((*d_y2s[off])() = d_y2())
            .execute();

          lambda_ccsd_y1(*ec, MO, d_r1, d_t1, d_t2, d_y1, d_y2, d_f1, d_v2);
          lambda_ccsd_y2(*ec, MO, d_r2, d_t1, d_t2, d_y1, d_y2, d_f1, d_v2);

          std::tie(residual, energy) = rest(*ec, MO, d_r1, d_r2, d_y1, d_y2,
                                            d_e, p_evl_sorted, zshiftl, noab,true);

          {
              auto lambdar2 = [&](const IndexVector& blockid) {
                  if((blockid[0] > blockid[1]) || (blockid[2] > blockid[3])) {
                      Tensor<T> tensor     = d_r2;
                      const TAMM_SIZE size = tensor.block_size(blockid);

                      std::vector<T> buf(size);
                      tensor.get(blockid, buf);

                      auto block_dims   = tensor.block_dims(blockid);
                      auto block_offset = tensor.block_offsets(blockid);

                      TAMM_SIZE c = 0;
                      for(auto i = block_offset[0];
                          i < block_offset[0] + block_dims[0]; i++) {
                          for(auto j = block_offset[1];
                              j < block_offset[1] + block_dims[1]; j++) {
                              for(auto k = block_offset[2];
                                  k < block_offset[2] + block_dims[2]; k++) {
                                  for(auto l = block_offset[3];
                                      l < block_offset[3] + block_dims[3];
                                      l++, c++) {
                                      buf[c] = 0;
                                  }
                              }
                          }
                      }
                      d_r2.put(blockid, buf);
                  }
              };
              block_for(ec->pg(), d_r2(), lambdar2);
          }

          Scheduler{ec}
          ((*d_r1s[off])() = d_r1())
          ((*d_r2s[off])() = d_r2())
            .execute();

          iteration_print_lambda(ec->pg(), iter, residual);
          Tensor<T>::deallocate(d_e, d_r1_residual, d_r2_residual);

          if(residual < thresh) { break; }
      }

      if(residual < thresh || titer + ndiis >= maxiter) { break; }
      if(ec->pg().rank() == 0) {
          std::cout << " MICROCYCLE DIIS UPDATE:";
          std::cout.width(21);
          std::cout << std::right << std::min(titer + ndiis, maxiter) + 1;
          std::cout.width(21);
          std::cout << std::right << "5" << std::endl;
      }

      std::vector<std::vector<Tensor<T>*>*> rs{&d_r1s, &d_r2s};
      std::vector<std::vector<Tensor<T>*>*> ys{&d_y1s, &d_y2s};
      std::vector<Tensor<T>*> next_y{&d_y1, &d_y2};
      diis<T>(*ec, rs, ys, next_y);
  }

  if(ec->pg().rank() == 0) {
    std::cout << std::string(66, '-') << std::endl;
    if(residual < thresh) {
        std::cout << " Iterations converged" << std::endl;
    }
  }

  for(size_t i=0; i<ndiis; i++) {
    Tensor<T>::deallocate(*d_r1s[i], *d_r2s[i], *d_y1s[i], *d_y2s[i]);
  }
  d_r1s.clear();
  d_r2s.clear();
  Tensor<T>::deallocate(d_r1, d_r2);

}

std::string filename; //bad, but no choice
int main( int argc, char* argv[] )
{
    if(argc<2){
        std::cout << "Please provide an input file!\n";
        return 1;
    }

    filename = std::string(argv[1]);
    std::ifstream testinput(filename); 
    if(!testinput){
        std::cout << "Input file provided [" << filename << "] does not exist!\n";
        return 1;
    }

    MPI_Init(&argc,&argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);
    
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int res = Catch::Session().run();
    
    GA_Terminate();
    MPI_Finalize();

    return res;
}


TEST_CASE("CCSD Driver") {

    std::cout << "Input file provided = " << filename << std::endl;

    using T = double;

    Matrix C;
    Matrix F;
    Tensor4D V2;
    TAMM_SIZE ov_alpha{0};
    TAMM_SIZE freeze_core    = 0;
    TAMM_SIZE freeze_virtual = 0;

    double hf_energy{0.0};
    libint2::BasisSet shells;
    TAMM_SIZE nao{0};

    std::vector<TAMM_SIZE> sizes;

    auto hf_t1 = std::chrono::high_resolution_clock::now();
    std::tie(ov_alpha, nao, hf_energy, shells) = hartree_fock(filename, C, F);
    auto hf_t2 = std::chrono::high_resolution_clock::now();

   double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    std::cout << "\nTime taken for Hartree-Fock: " << hf_time << " secs\n";

    hf_t1        = std::chrono::high_resolution_clock::now();
    std::tie(V2) = four_index_transform(ov_alpha, nao, freeze_core,
                                        freeze_virtual, C, F, shells);
    hf_t2        = std::chrono::high_resolution_clock::now();
    double two_4index_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    std::cout << "\nTime taken for 4-index transform: " << two_4index_time
              << " secs\n";

    TAMM_SIZE ov_beta{nao - ov_alpha};

    std::cout << "ov_alpha,nao === " << ov_alpha << ":" << nao << std::endl;
    sizes = {ov_alpha - freeze_core, ov_alpha - freeze_core,
             ov_beta - freeze_virtual, ov_beta - freeze_virtual};

    std::cout << "sizes vector -- \n";
    for(const auto& x : sizes) std::cout << x << ", ";
    std::cout << "\n";

    const long int total_orbitals = 2*ov_alpha+2*ov_beta;
    
    // Construction of tiled index space MO

    IndexSpace MO_IS{range(0, total_orbitals),
                    {{"occ", {range(0, 2*ov_alpha)}},
                     {"virt", {range(2*ov_alpha, total_orbitals)}}}};

    // IndexSpace MO_IS{range(0, total_orbitals),
    //                 {{"occ", {range(0, ov_alpha+ov_beta)}}, //0-7
    //                  {"virt", {range(total_orbitals/2, total_orbitals)}}, //7-14
    //                  {"alpha", {range(0, ov_alpha),range(ov_alpha+ov_beta,2*ov_alpha+ov_beta)}}, //0-5,7-12
    //                  {"beta", {range(ov_alpha,ov_alpha+ov_beta), range(2*ov_alpha+ov_beta,total_orbitals)}} //5-7,12-14   
    //                  }};
    const unsigned int ova = static_cast<unsigned int>(ov_alpha);
    const unsigned int ovb = static_cast<unsigned int>(ov_beta);
    TiledIndexSpace MO{MO_IS, {ova,ova,ovb,ovb}};

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext *ec = new ExecutionContext{pg,&distribution,mgr};

    TiledIndexSpace O = MO("occ");
    TiledIndexSpace V = MO("virt");
    TiledIndexSpace N = MO("all");

    Tensor<T> d_t1{V, O};
    Tensor<T> d_t2{V, V, O, O};
    Tensor<T> d_y1{O,V};
    Tensor<T> d_y2{O,O,V,V};
    Tensor<T> d_f1{N, N};
    Tensor<T> d_v2{N, N, N, N};
    
    int maxiter    = 50;
    double thresh  = 1.0e-10;
    double zshiftl = 0.0;
    size_t ndiis      = 5;

  Tensor<double>::allocate(ec,d_t1,d_t2,d_y1,d_y2,d_f1,d_v2);

  Scheduler{ec}
      (d_t1() = 0)
      (d_t2() = 0)
      (d_y1() = 0)
      (d_y2() = 0)      
      (d_f1() = 0)
      (d_v2() = 0)
    .execute();


  //Tensor Map 
  block_for(ec->pg(), d_f1(), [&](IndexVector it) {
    Tensor<T> tensor = d_f1().tensor();
    const TAMM_SIZE size = tensor.block_size(it);
    
    std::vector<T> buf(size);

    auto block_offset = tensor.block_offsets(it);
    auto block_dims = tensor.block_dims(it);

    TAMM_SIZE c=0;
    for (auto i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for (auto j = block_offset[1]; j < block_offset[1] + block_dims[1];
           j++, c++) {
        buf[c] = F(i, j);
      }
    }
    d_f1.put(it,buf);
  });

  block_for(ec->pg(), d_v2(), [&](IndexVector it) {
      Tensor<T> tensor     = d_v2().tensor();
      const TAMM_SIZE size = tensor.block_size(it);

      std::vector<T> buf(size);

      auto block_dims = tensor.block_dims(it);
      auto block_offset = tensor.block_offsets(it);

      TAMM_SIZE c = 0;
      for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
          for(auto j = block_offset[1]; j < block_offset[1] + block_dims[1];
              j++) {
              for(auto k = block_offset[2]; k < block_offset[2] + block_dims[2];
                  k++) {
                  for(auto l = block_offset[3];
                      l < block_offset[3] + block_dims[3]; l++, c++) {
                      buf[c] = V2(i,j,k,l);
                  }
              }
          }
      }
      d_v2.put(it, buf);
  });

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  CHECK_NOTHROW(ccsd_driver<T>(ec, MO, d_t1, d_t2, d_f1, d_v2, maxiter, thresh,
                               zshiftl, ndiis, hf_energy, total_orbitals,
                               2 * ov_alpha));

  auto cc_t2 = std::chrono::high_resolution_clock::now();

  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  std::cout << "\nTime taken for CCSD: " << ccsd_time << " secs\n";                               

  cc_t1 = std::chrono::high_resolution_clock::now();
  CHECK_NOTHROW(lambda_ccsd_driver<T>(ec, MO, d_t1, d_t2, d_y1, d_y2, d_f1,
                                      d_v2, maxiter, thresh, zshiftl, ndiis,
                                      hf_energy, total_orbitals, 2 * ov_alpha));
  cc_t2 = std::chrono::high_resolution_clock::now();

  ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  std::cout << "\nTime taken for Lambda CCSD: " << ccsd_time << " secs\n";              

  Tensor<T>::deallocate(d_t1, d_t2, d_y1, d_y2, d_f1, d_v2);
  MemoryManagerGA::destroy_coll(mgr);

}
