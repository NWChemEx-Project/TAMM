#ifndef TAMMX_LABELED_BLOCK_H_
#define TAMMX_LABELED_BLOCK_H_

#include "tammx/types.h"
#include "tammx/block.h"

#if defined(__ICC) || defined(__INTEL_COMPILER)
  #include "mkl_cblas.h"
#else
  #include "cblas.h"
#endif

namespace tammx {

template<typename T>
struct LabeledBlock;

template<typename T1,
         typename T2>
inline std::tuple<T1, LabeledBlock<T2>>
operator * (T1 alpha, LabeledBlock<T2> block) {
  return {alpha, block};
}

template<typename T1,
         typename T2>
inline std::tuple<T1, LabeledBlock<T2>>
operator * (LabeledBlock<T2> block, T1 alpha) {
  return {alpha, block};
}

template<typename T1, typename T2>
inline std::tuple<T1, LabeledBlock<T2>, LabeledBlock<T2>>
operator * (const std::tuple<T1, LabeledBlock<T2>>& rhs1, LabeledBlock<T2> rhs2)  {
  return std::tuple_cat(rhs1, std::make_tuple(rhs2));
}

template<typename T>
inline std::tuple<LabeledBlock<T>, LabeledBlock<T>>
operator * (LabeledBlock<T> rhs1, LabeledBlock<T> rhs2)  {
  return std::make_tuple(rhs1, rhs2);
}

template<typename T1,
         typename T2>
inline std::tuple<T1, LabeledBlock<T2>, LabeledBlock<T2>>
operator * (T1 alpha, std::tuple<LabeledBlock<T2>, LabeledBlock<T2>> rhs) {
  return std::tuple_cat(std::make_tuple(alpha), rhs);
}


template<typename T>
struct LabeledBlock {
  Block<T> *block_;
  IndexLabelVec label_;

  template<typename T1,
           typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
  void operator = (T1 value) {
    auto buf = block_->buf();
    auto rval = static_cast<T>(value);
    for(int i=0; i<block_->size(); i++) {
      buf[i] = rval;
    }
  }

  void operator = (LabeledBlock<T> rhs) {
    *this = 1 * rhs;
  }

  template<typename T1>
  void operator = (std::tuple<T1, LabeledBlock<T>> rhs);

  template<typename T1>
  void operator = (std::tuple<T1, LabeledBlock<T>, LabeledBlock<T>> rhs);

  void operator = (std::tuple<LabeledBlock<T>, LabeledBlock<T>> rhs) {
    *this = 1 * std::get<0>(rhs) * std::get<1>(rhs);
  }

  void operator += (LabeledBlock<T> rhs) {
    *this += 1 * rhs;
  }

  template<typename T1>
  void operator += (std::tuple<T1, LabeledBlock<T>> rhs);

  template<typename T1>
  void operator += (std::tuple<T1, LabeledBlock<T>, LabeledBlock<T>> rhs);

  void operator += (std::tuple<LabeledBlock<T>, LabeledBlock<T>> rhs) {
    *this += 1 * std::get<0>(rhs) * std::get<1>(rhs);
  }
};


namespace impl {
/**
 * performs: cbuf[dims] = scale *abuf[perm(dims)]
 *
 * @todo unsafe. passing 1 instead of 1.0 might lead to unexpected results.
 */
template<typename T1, typename T2>
inline void
index_permute_acc(T1* dbuf, const T1* sbuf, const PermVec& perm, const BlockDimVec& ddims, T2 scale) {
  static_assert(std::is_same<T1, double>(), "index_permute_acc only works with doubles");
  static_assert(std::is_convertible<T2, double>(), "index_permute_acc only works with scale convertible to double");
  EXPECTS(dbuf!=nullptr && sbuf!=nullptr);
  EXPECTS(perm.size() == ddims.size());

  auto inv_perm = perm_invert(perm);
  auto inv_sizes = perm_apply(ddims, inv_perm);
  TensorVec<size_t> sizes;
  TensorVec<int> iperm;
  for(unsigned i=0; i<ddims.size(); i++) {
    sizes.push_back(inv_sizes[i].value());
    iperm.push_back(perm[i]+1);
  }
  index_sortacc(sbuf, dbuf,
                sizes.size(), &sizes[0], &iperm[0], scale);
}

/**
 *  @todo unsafe. passing 1 instead of 1.0 might lead to unexpected results.
 */
template<typename T1, typename T2>
inline void
index_permute(T1* dbuf, const T1* sbuf, const PermVec& perm, const BlockDimVec& ddims, T2 scale) {
  static_assert(std::is_same<T1, double>(), "index_permute only works with doubles");
  static_assert(std::is_convertible<T2, double>(), "index_permute only works with scale convertible to double");
  EXPECTS(dbuf!=nullptr && sbuf!=nullptr);
  EXPECTS(perm.size() == ddims.size());

  auto inv_perm = perm_invert(perm);
  auto inv_sizes = perm_apply(ddims, inv_perm);
  TensorVec<size_t> sizes;
  TensorVec<int> iperm;
  for(unsigned i=0; i<ddims.size(); i++) {
    sizes.push_back(inv_sizes[i].value());
    iperm.push_back(perm[i]+1);
  }
  index_sort(sbuf, dbuf,
             sizes.size(), &sizes[0], &iperm[0], scale);
}



template<typename T>
// C storage order: A[m,k], B[k,n], C[m,n]
inline void
matmul(int m, int n, int k, T *A, int lda, T *B, int ldb, T *C, int ldc, T alpha, T beta) {
  EXPECTS(m>0 && n>0 && k>0);
  EXPECTS(A!=nullptr && B!=nullptr && C!=nullptr);

//  for(int x=0; x<m; x++) {
//    for(int y=0; y<n; y++) {
//      T value = 0;
//      for(int z=0; z<k; z++) {
//        value += A[x*lda + z] * B[z*ldb + y];
//      }
//      C[x*ldc + y] = beta * C[x*ldc + y] + alpha * value;
//    }
//  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<typename T>
inline PermVec
perm_compute(const LabeledBlock<T>& lblock_from, const IndexLabelVec& label_to) {
  auto store = perm_apply(lblock_from.label_,
                          perm_invert(lblock_from.block_->layout()));
  return perm_compute(store, label_to);
}

//@todo optimize. Eliminate copies. Memoize the split of the indices
//and the permutation required
template<typename T, typename T1>
void multiply(LabeledBlock<T>& clb, std::tuple<T1, LabeledBlock<T>, LabeledBlock<T>> rhs, T beta) {
  const LabeledBlock<T>& alb = std::get<1>(rhs);
  const LabeledBlock<T>& blb = std::get<2>(rhs);

  auto &ablock = *alb.block_;
  auto &bblock = *blb.block_;
  auto &cblock = *clb.block_;

  auto &alabel = alb.label_;
  auto &blabel = blb.label_;
  auto &clabel = clb.label_;

  auto aext_labels = intersect(alabel, clabel);
  auto bext_labels = intersect(blabel, clabel);
  auto sum_labels = intersect(alabel, blabel);

  auto alabel_sort = aext_labels;
  alabel_sort.insert_back(sum_labels.begin(), sum_labels.end());
  auto blabel_sort = sum_labels;
  blabel_sort.insert_back(bext_labels.begin(), bext_labels.end());
  auto clabel_sort = aext_labels;
  clabel_sort.insert_back(bext_labels.begin(), bext_labels.end());

  //TTGT
  //TT
  auto abuf_sort = std::make_unique<T[]>(ablock.size());
  auto bbuf_sort = std::make_unique<T[]>(bblock.size());
  auto cbuf_sort = std::make_unique<T[]>(cblock.size());

  auto aperm = perm_compute(ablock(alabel), alabel_sort);
  auto bperm = perm_compute(bblock(blabel), blabel_sort);

  index_permute(abuf_sort.get(), ablock.buf(), aperm,
                perm_apply(ablock.block_dims(), aperm), 1);
  index_permute(bbuf_sort.get(), bblock.buf(), bperm,
                perm_apply(bblock.block_dims(), bperm), 1);
  for(size_t i=0; i<cblock.size(); i++) {
    cbuf_sort[i] = cblock.buf()[i];
  }

  // G
  auto alpha = std::get<0>(rhs) * ablock.sign() * bblock.sign();
  auto lmap = LabelMap<BlockIndex>()
      .update(alabel, ablock.block_dims())
      .update(blabel, bblock.block_dims());
  auto aext_dims = lmap.get_blockid(aext_labels);
  auto bext_dims = lmap.get_blockid(bext_labels);
  auto sum_dims = lmap.get_blockid(sum_labels);
  int m = std::accumulate(aext_dims.begin(), aext_dims.end(), BlockIndex{1}, std::multiplies<>()).value();
  int n = std::accumulate(bext_dims.begin(), bext_dims.end(), BlockIndex{1}, std::multiplies<>()).value();
  int k = std::accumulate(sum_dims.begin(), sum_dims.end(), BlockIndex{1}, std::multiplies<>()).value();

  matmul<T>(m, n, k, abuf_sort.get(), k,
            bbuf_sort.get(), n,
            cbuf_sort.get(), n,
            static_cast<T>(alpha), static_cast<T>(beta));
  auto cperm = perm_invert(perm_compute(cblock(clabel), clabel_sort));
  //T
  index_permute(cblock.buf(), cbuf_sort.get(),
                cperm, cblock.block_dims(), 1);
}

template<typename T, typename T1>
inline void
block_add (LabeledBlock<T>& clb, std::tuple<T1, LabeledBlock<T>> rhs, bool update) {
  const LabeledBlock<T>& alb = std::get<1>(rhs);

  auto &ablock = *alb.block_;
  auto &cblock = *clb.block_;

  auto &clabel = clb.label_;
  auto &alabel = alb.label_;

  auto label_perm = perm_compute(alabel, clabel);
  for(unsigned i=0; i<label_perm.size(); i++) {
    EXPECTS(cblock.block_dims()[i] == ablock.block_dims()[label_perm[i]]);
  }

  auto &alayout = ablock.layout();
  auto &clayout = cblock.layout();

  EXPECTS(clayout.size() == cblock.tensor().rank());
  EXPECTS(clabel.size() == perm_invert(clayout).size());
  EXPECTS(alabel.size() == perm_invert(alayout).size());
  auto cstore = perm_apply(clabel, perm_invert(clayout));
  auto astore = perm_apply(alabel, perm_invert(alayout));

  auto store_perm = perm_compute(astore, cstore);
  auto alpha = std::get<0>(rhs) * ablock.sign();
  if(!update) {
    index_permute(cblock.buf(), ablock.buf(), store_perm, cblock.block_dims(), static_cast<T>(alpha));
  } else {
    index_permute_acc(cblock.buf(), ablock.buf(), store_perm, cblock.block_dims(), static_cast<T>(alpha));
  }
}

} // namespace tammx::impl

template<typename T>
template<typename T1>
inline void
LabeledBlock<T>::operator = (std::tuple<T1, LabeledBlock<T>> rhs) {
  impl::block_add(*this, rhs, false);
}

template<typename T>
template<typename T1>
inline void
LabeledBlock<T>::operator += (std::tuple<T1, LabeledBlock<T>> rhs) {
  impl::block_add(*this, rhs, true);
}

template<typename T>
template<typename T1>
inline void
LabeledBlock<T>::operator += (std::tuple<T1, LabeledBlock<T>, LabeledBlock<T>> rhs) {
  impl::multiply(*this, rhs, T(1));
}


template<typename T>
template<typename T1>
inline void
LabeledBlock<T>::operator = (std::tuple<T1, LabeledBlock<T>, LabeledBlock<T>> rhs) {
  impl::multiply(*this, rhs, T(0));
}

} // namespace tammx

#endif  // TAMMX_LABELED_BLOCK_H_
