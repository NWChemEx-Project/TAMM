#pragma once

#include "ga/ga.h"
#include "tamm/blockops_cpu.hpp"
#include "tamm/distribution.hpp"
#include "tamm/execution_context.hpp"
#include "tamm/index_loop_nest.hpp"
#include "tamm/index_space.hpp"
#include "tamm/mem_profiler.hpp"
#include "tamm/memory_manager_local.hpp"
#include "tamm/tensor_base.hpp"
#include <functional>
#include <gsl/span>
#include <type_traits>
#if defined(USE_UPCXX)
#include <upcxx/upcxx.hpp>
#endif

namespace tamm {

using gsl::span;

#if defined(USE_UPCXX)
class TensorTile {
public:
  int64_t lo[4];
  int64_t dim[4];
  int64_t rank;
  int64_t offset;

  TensorTile(int64_t lo_0, int64_t lo_1, int64_t lo_2, int64_t lo_3, int64_t dim_0, int64_t dim_1,
             int64_t dim_2, int64_t dim_3, int64_t _rank, int64_t _offset) {
    lo[0]  = lo_0;
    lo[1]  = lo_1;
    lo[2]  = lo_2;
    lo[3]  = lo_3;
    dim[0] = dim_0;
    dim[1] = dim_1;
    dim[2] = dim_2;
    dim[3] = dim_3;
    rank   = _rank;
    offset = _offset;
  }

  bool contains(int64_t i, int64_t j, int64_t k, int64_t l) const {
    return i >= lo[0] && j >= lo[1] && i < lo[0] + dim[0] && j < lo[1] + dim[1] && k >= lo[2] &&
           l >= lo[3] && k < lo[2] + dim[2] && l < lo[3] + dim[3];
  }
};
#endif

/**
 * @ingroup tensors
 * @brief Implementation class for TensorBase class
 *
 * @tparam T Element type of Tensor
 */
template<typename T>
class TensorImpl: public TensorBase {
public:
  using TensorBase::TensorBase;
  // Ctors
  TensorImpl() = default;

  /**
   * @brief Construct a new TensorImpl object using a vector of
   * TiledIndexSpace objects for each mode of the tensor
   *
   * @param [in] tis vector of TiledIndexSpace objects for each
   * mode used to construct the tensor
   */
  TensorImpl(const TiledIndexSpaceVec& tis): TensorBase{tis} {
    has_spin_symmetry_    = false;
    has_spatial_symmetry_ = false;
    kind_                 = TensorBase::TensorKind::normal;
  }

  /**
   * @brief Construct a new TensorImpl object using a vector of
   * TiledIndexSpace objects for each mode of the tensor
   *
   * @param [in] lbls vector of tiled index labels used for extracting
   * corresponding TiledIndexSpace objects for each mode used to construct
   * the tensor
   */
  TensorImpl(const std::vector<TiledIndexLabel>& lbls): TensorBase{lbls} {
    has_spin_symmetry_    = false;
    has_spatial_symmetry_ = false;
    kind_                 = TensorBase::TensorKind::normal;
  }

  /**
   * @brief Construct a new TensorBase object recursively with a set of
   * TiledIndexSpace objects followed by a lambda expression
   *
   * @tparam Ts variadic template for rest of the arguments
   * @param [in] tis TiledIndexSpace object used as a mode
   * @param [in] rest remaining part of the arguments
   */
  template<class... Ts>
  TensorImpl(const TiledIndexSpace& tis, Ts... rest): TensorBase{tis, rest...} {
    num_modes_ = block_indices_.size();
    construct_dep_map();
    has_spin_symmetry_    = false;
    has_spatial_symmetry_ = false;
    kind_                 = TensorBase::TensorKind::normal;
  }

  // SpinTensor related constructors
  /**
   * @brief Construct a new SpinTensorImpl object using set of TiledIndexSpace
   * objects and Spin attribute mask
   *
   * @param [in] t_spaces
   * @param [in] spin_mask
   */
  TensorImpl(TiledIndexSpaceVec t_spaces, SpinMask spin_mask): TensorBase(t_spaces) {
    EXPECTS(t_spaces.size() == spin_mask.size());

    // for(const auto& tis : t_spaces) { EXPECTS(tis.has_spin()); }

    spin_mask_            = spin_mask;
    has_spin_symmetry_    = true;
    has_spatial_symmetry_ = false;
    // spin_total_        = calculate_spin();
    kind_ = TensorBase::TensorKind::spin;
  }

  /**
   * @brief Construct a new SpinTensorImpl object using set of TiledIndexLabel
   * objects and Spin attribute mask
   *
   * @param [in] t_labels
   * @param [in] spin_mask
   */
  TensorImpl(IndexLabelVec t_labels, SpinMask spin_mask): TensorBase(t_labels) {
    EXPECTS(t_labels.size() == spin_mask.size());
    // for(const auto& tlbl : t_labels) {
    //     EXPECTS(tlbl.tiled_index_space().has_spin());
    // }
    spin_mask_            = spin_mask;
    has_spin_symmetry_    = true;
    has_spatial_symmetry_ = false;
    // spin_total_        = calculate_spin();
    kind_ = TensorBase::TensorKind::spin;
  }

  /**
   * @brief Construct a new SpinTensorImpl object using set of TiledIndexSpace
   * objects and Spin attribute mask
   *
   * @param [in] t_spaces
   * @param [in] spin_mask
   */
  TensorImpl(TiledIndexSpaceVec t_spaces, std::vector<size_t> spin_sizes): TensorBase(t_spaces) {
    // EXPECTS(t_spaces.size() == spin_mask.size());
    int spin_size_sum = std::accumulate(spin_sizes.begin(), spin_sizes.end(), 0);
    EXPECTS(t_spaces.size() >= spin_size_sum);

    EXPECTS(spin_sizes.size() > 0);
    // for(const auto& tis : t_spaces) { EXPECTS(tis.has_spin()); }
    SpinMask spin_mask;
    size_t   upper  = spin_sizes[0];
    size_t   lower  = spin_sizes.size() > 1 ? spin_sizes[1] : t_spaces.size() - upper;
    size_t   ignore = spin_sizes.size() > 2 ? spin_sizes[1] : t_spaces.size() - (upper + lower);

    for(size_t i = 0; i < upper; i++) { spin_mask.push_back(SpinPosition::upper); }

    for(size_t i = 0; i < lower; i++) { spin_mask.push_back(SpinPosition::lower); }

    for(size_t i = 0; i < ignore; i++) { spin_mask.push_back(SpinPosition::ignore); }

    spin_mask_            = spin_mask;
    has_spin_symmetry_    = true;
    has_spatial_symmetry_ = false;
    // spin_total_        = calculate_spin();
    kind_ = TensorBase::TensorKind::spin;
  }

  /**
   * @brief Construct a new TensorImpl object using the specialized non-zero check function
   *
   * @param t_spaces
   * @param zero_check
   */
  TensorImpl(const TiledIndexSpaceVec& t_spaces, const NonZeroCheck& zero_check):
    TensorBase(t_spaces, zero_check) {
    kind_ = TensorBase::TensorKind::block_sparse;
  }

  /**
   * @brief Construct a new SpinTensorImpl object using set of TiledIndexLabel
   * objects and Spin attribute mask
   *
   * @param [in] t_labels
   * @param [in] spin_mask
   */
  TensorImpl(IndexLabelVec t_labels, std::vector<size_t> spin_sizes): TensorBase(t_labels) {
    // EXPECTS(t_labels.size() == spin_mask.size());
    int spin_size_sum = std::accumulate(spin_sizes.begin(), spin_sizes.end(), 0);
    EXPECTS(t_labels.size() >= spin_size_sum);
    EXPECTS(spin_sizes.size() > 0);
    // for(const auto& tlbl : t_labels) {
    //     EXPECTS(tlbl.tiled_index_space().has_spin());
    // }

    SpinMask spin_mask;
    size_t   upper  = spin_sizes[0];
    size_t   lower  = spin_sizes.size() > 1 ? spin_sizes[1] : t_labels.size() - upper;
    size_t   ignore = spin_sizes.size() > 2 ? spin_sizes[1] : t_labels.size() - (upper + lower);

    for(size_t i = 0; i < upper; i++) { spin_mask.push_back(SpinPosition::upper); }

    for(size_t i = 0; i < lower; i++) { spin_mask.push_back(SpinPosition::lower); }

    for(size_t i = 0; i < ignore; i++) { spin_mask.push_back(SpinPosition::ignore); }

    spin_mask_            = spin_mask;
    has_spin_symmetry_    = true;
    has_spatial_symmetry_ = false;
    // spin_total_        = calculate_spin();
    kind_ = TensorBase::TensorKind::spin;
  }

  // Copy/Move Ctors and Assignment Operators
  TensorImpl(TensorImpl&&)                 = default;
  TensorImpl(const TensorImpl&)            = delete;
  TensorImpl& operator=(TensorImpl&&)      = default;
  TensorImpl& operator=(const TensorImpl&) = delete;

  // Dtor
  ~TensorImpl() {
    if(mpb_ != nullptr) { mpb_->allocation_status_ = AllocationStatus::orphaned; }
  }

  /**
   * @brief Virtual method for deallocating a Tensor
   *
   */
  virtual void deallocate() {
    EXPECTS(allocation_status_ == AllocationStatus::created);
    EXPECTS(mpb_);
    // get memory profiler instance
    auto& memprof = MemProfiler::instance();

    ec_->unregister_for_dealloc(mpb_);
    mpb_->dealloc_coll();
    MemoryManager* memory_manager = &(mpb_->mgr());
    delete memory_manager;
    delete mpb_;
    mpb_ = nullptr;
    update_status(AllocationStatus::deallocated);
    // update memory profiler instance
    memprof.mem_deallocated += size();
    memprof.dealloc_counter++;
  }

  /**
   * @brief Virtual method for allocating a Tensor using an ExecutionContext
   *
   * @param [in] ec ExecutionContext to be used for allocation
   */
  virtual void allocate(ExecutionContext* ec) {
    {
      EXPECTS(allocation_status_ == AllocationStatus::invalid);
      // get memory profiler instance
      auto& memprof = MemProfiler::instance();

      auto          defd = ec->get_default_distribution();
      Distribution* distribution =
        ec->distribution(defd->get_tensor_base(), defd->get_dist_proc()); // defd->kind());
      // Distribution* distribution    =
      // ec->distribution(defd.tensor_base(), nproc );
      MemoryManager* memory_manager = ec->memory_manager();
      EXPECTS(distribution != nullptr);
      EXPECTS(memory_manager != nullptr);
      ec_ = ec;
      // distribution_ =
      // DistributionFactory::make_distribution(*distribution, this,
      // pg.size());
      {
        distribution_ =
          std::shared_ptr<Distribution>(distribution->clone(this, memory_manager->pg().size()));
      }

      // Delete unused pointers
      delete defd;
      delete distribution;
#if 0
        auto rank = memory_manager->pg().rank();
        auto buf_size = distribution_->buf_size(rank);
        auto eltype = tensor_element_type<T>();
        EXPECTS(buf_size >= 0);
        mpb_ = memory_manager->alloc_coll(eltype, buf_size);
#else
      auto eltype = tensor_element_type<T>();
      if(proc_list_.size() > 0)
        mpb_ = memory_manager->alloc_coll_balanced(eltype, distribution_->max_proc_buf_size(),
                                                   proc_list_);
      else mpb_ = memory_manager->alloc_coll_balanced(eltype, distribution_->max_proc_buf_size());

#endif
      EXPECTS(mpb_ != nullptr);
      ec_->register_for_dealloc(mpb_);
      update_status(AllocationStatus::created);

      // update memory profiler instance
      memprof.alloc_counter++;
      const auto tsize = size();
      memprof.mem_allocated += tsize;
      memprof.max_in_single_allocate =
        tsize > memprof.max_in_single_allocate ? tsize : memprof.max_in_single_allocate;
      memprof.max_total_allocated = (memprof.mem_allocated - memprof.mem_deallocated) >
                                        memprof.max_total_allocated
                                      ? (memprof.mem_allocated - memprof.mem_deallocated)
                                      : memprof.max_total_allocated;
    }
  }

  virtual const Distribution& distribution() const { return *distribution_.get(); }

  // Tensor Accessors
  /**
   * @brief Tensor accessor method for getting values from a set of
   * indices to specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to fetch the values
   * @param [in] buff_span memory span where to put the fetched values
   */

  virtual void get(const IndexVector& idx_vec, span<T> buff_span) const {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);

    if(!is_non_zero(idx_vec)) {
      Size size = block_size(idx_vec);
      EXPECTS(size <= buff_span.size());
      for(size_t i = 0; i < size; i++) { buff_span[i] = (T) 0; }
      return;
    }

    Proc   proc;
    Offset offset;
    std::tie(proc, offset) = distribution_->locate(idx_vec);
    Size size              = block_size(idx_vec);
    EXPECTS(size <= buff_span.size());
    mpb_->mgr().get(*mpb_, proc, offset, Size{size}, buff_span.data());
  }

  /**
   * @brief Tensor accessor method for getting values in nonblocking fashion
   * from a set of indices to specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to fetch the values
   * @param [in] buff_span memory span where to put the fetched values
   */

  virtual void nb_get(const IndexVector& idx_vec, span<T> buff_span,
                      DataCommunicationHandlePtr data_comm_handle) const {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);

    if(!is_non_zero(idx_vec)) {
      Size size = block_size(idx_vec);
      EXPECTS(size <= buff_span.size());
      for(size_t i = 0; i < size; i++) { buff_span[i] = (T) 0; }
      return;
    }

    Proc   proc;
    Offset offset;
    std::tie(proc, offset) = distribution_->locate(idx_vec);
    Size size              = block_size(idx_vec);
    EXPECTS(size <= buff_span.size());
    mpb_->mgr().nb_get(*mpb_, proc, offset, Size{size}, buff_span.data(), data_comm_handle);
  }

  /**
   * @brief Tensor accessor method for putting values to a set of indices
   * with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  virtual void put(const IndexVector& idx_vec, span<T> buff_span) {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);

    if(!is_non_zero(idx_vec)) { return; }

    Proc   proc;
    Offset offset;
    std::tie(proc, offset) = distribution_->locate(idx_vec);
    Size size              = block_size(idx_vec);
    EXPECTS(size <= buff_span.size());
    mpb_->mgr().put(*mpb_, proc, offset, Size{size}, buff_span.data());
  }

  /**
   * @brief Tensor accessor method for putting values in nonblocking fashion
   * to a set of indices with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  virtual void nb_put(const IndexVector& idx_vec, span<T> buff_span,
                      DataCommunicationHandlePtr data_comm_handle) {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);

    if(!is_non_zero(idx_vec)) { return; }

    Proc   proc;
    Offset offset;
    std::tie(proc, offset) = distribution_->locate(idx_vec);
    Size size              = block_size(idx_vec);
    EXPECTS(size <= buff_span.size());
    mpb_->mgr().nb_put(*mpb_, proc, offset, Size{size}, buff_span.data(), data_comm_handle);
  }

  /**
   * @brief Tensor accessor method for adding svalues to a set of indices
   * with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  virtual void add(const IndexVector& idx_vec, span<T> buff_span) {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);

    if(!is_non_zero(idx_vec)) { return; }

    Proc   proc;
    Offset offset;
    std::tie(proc, offset) = distribution_->locate(idx_vec);
    Size size              = block_size(idx_vec);
    EXPECTS(size <= buff_span.size());
    mpb_->mgr().add(*mpb_, proc, offset, Size{size}, buff_span.data());
  }

  /**
   * @brief Tensor accessor method for adding svalues in nonblocking fashion
   * to a set of indices with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  virtual void nb_add(const IndexVector& idx_vec, span<T> buff_span,
                      DataCommunicationHandlePtr data_comm_handle) {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);

    if(!is_non_zero(idx_vec)) { return; }

    Proc   proc;
    Offset offset;
    std::tie(proc, offset) = distribution_->locate(idx_vec);
    Size size              = block_size(idx_vec);
    EXPECTS(size <= buff_span.size());
    mpb_->mgr().nb_add(*mpb_, proc, offset, Size{size}, buff_span.data(), data_comm_handle);
  }

  /**
   * @brief Virtual method for getting the diagonal values in a Tensor
   *
   * @returns a vector with the diagonal values
   * @warning available for tensors with 2 modes
   */
  // virtual std::vector<T> diagonal() {
  //     EXPECTS(num_modes() == 2);
  //     std::vector<T> dest;
  //     for(const IndexVector& blockid : loop_nest()) {
  //         if(blockid[0] == blockid[1]) {
  //             const TAMM_SIZE size = block_size(blockid);
  //             std::vector<T> buf(size);
  //             get(blockid, buf);
  //             auto block_dims1  = block_dims(blockid);
  //             auto block_offset = block_offsets(blockid);
  //             auto dim          = block_dims1[0];
  //             auto offset       = block_offset[0];
  //             size_t i          = 0;
  //             for(auto p = offset; p < offset + dim; p++, i++) {
  //                 dest.push_back(buf[i * dim + i]);
  //             }
  //         }
  //     }
  //     return dest;
  // }

#if !defined(USE_UPCXX)
  virtual int ga_handle() {
    const MemoryRegionGA& mr = static_cast<const MemoryRegionGA&>(*mpb_);
    return mr.ga();
  }
#endif

  virtual std::vector<int64_t> local_buf_dims() const { abort(); }

  virtual bool is_local_element(int64_t i, int64_t j, int64_t k, int64_t l) const { abort(); }

  virtual std::vector<int64_t> local_tiles_offsets() const { abort(); }

  virtual std::pair<int64_t, int64_t> local_element_offsets(int64_t i, int64_t j, int64_t k,
                                                            int64_t l) const {
    abort();
  }

#ifdef USE_UPCXX
  virtual std::vector<TensorTile>::const_iterator local_tiles_begin() const { abort(); }

  virtual std::vector<TensorTile>::const_iterator local_tiles_end() const { abort(); }
#endif

  virtual void put_raw_contig(int64_t* lo, int64_t* hi, void* buf) const { abort(); }

  virtual void put_raw(int64_t* lo, int64_t* hi, void* buf) const { abort(); }

  virtual void get_raw_contig(int64_t* lo, int64_t* hi, void* buf) const { abort(); }

  virtual void get_raw(int64_t* lo, int64_t* hi, void* buf) const { abort(); }

  virtual T* access_local_buf() {
    EXPECTS(mpb_);
    return static_cast<T*>(mpb_->mgr().access(*mpb_, Offset{0}));
  }

  virtual const T* access_local_buf() const {
    EXPECTS(mpb_);
    return static_cast<T*>(mpb_->mgr().access(*mpb_, Offset{0}));
  }

  virtual size_t local_buf_size() const {
    EXPECTS(mpb_);
    return mpb_->local_nelements().value();
  }

  virtual size_t total_buf_size(Proc proc) const {
    EXPECTS(distribution_);
    return distribution_->buf_size(proc).value();
  }

  MemoryRegion* memory_region() const {
    EXPECTS(mpb_);
    return mpb_;
  }

  virtual int64_t size() const {
    EXPECTS(distribution_ != nullptr);
    return distribution_->total_size().value();
  }

  bool is_allocated() const { return (allocation_status_ == AllocationStatus::created); }

  void set_proc_list(const ProcList& proc_list) { proc_list_ = proc_list; }

  virtual bool is_block_cyclic() { return false; }

protected:
  std::shared_ptr<Distribution> distribution_;  /**< shared pointer to associated Distribution */
  MemoryRegion*                 mpb_ = nullptr; /**< Raw pointer memory region (default null) */
  ProcList                      proc_list_ = {};

}; // TensorImpl

/**
 * @ingroup tensors
 * @brief Implementation class for TensorBase with Lambda function construction
 *
 * @tparam T Element type of Tensor
 */
template<typename T>
class LambdaTensorImpl: public TensorImpl<T> {
public:
  using TensorBase::block_indices_;
  using TensorBase::setKind;

  /// @brief Function signature for the Lambda method
  using Func = std::function<void(const IndexVector&, span<T>)>;
  // Ctors
  LambdaTensorImpl() = default;

  // Copy/Move Ctors and Assignment Operators
  LambdaTensorImpl(LambdaTensorImpl&&)                 = default;
  LambdaTensorImpl(const LambdaTensorImpl&)            = delete;
  LambdaTensorImpl& operator=(LambdaTensorImpl&&)      = default;
  LambdaTensorImpl& operator=(const LambdaTensorImpl&) = delete;

  /**
   * @brief Construct a new LambdaTensorImpl object using a Lambda function
   *
   * @param [in] tis_vec vector of TiledIndexSpace objects for each mode of
   * the Tensor
   * @param [in] lambda a function for constructing the Tensor
   */
  LambdaTensorImpl(const TiledIndexSpaceVec& tis_vec, Func lambda):
    TensorImpl<T>(tis_vec), lambda_{lambda} {
    setKind(TensorBase::TensorKind::lambda);
  }

  LambdaTensorImpl(const IndexLabelVec& til_vec, Func lambda):
    TensorImpl<T>(til_vec), lambda_{lambda} {
    setKind(TensorBase::TensorKind::lambda);
  }

  LambdaTensorImpl(const IndexLabelVec& til_vec, Tensor<T> ref_tensor,
                   const IndexLabelVec& ref_labels, const IndexLabelVec& use_labels,
                   const std::vector<TranslateFunc>& translate_func_vec):
    TensorImpl<T>(til_vec),
    lambda_{construct_lambda(til_vec, ref_tensor, ref_labels, use_labels, translate_func_vec)} {}

  // Dtor
  ~LambdaTensorImpl() = default;

  void deallocate() override {
    // NO_OP();
  }

  void allocate(ExecutionContext* ec) override {
    // NO_OP();
  }

  void get(const IndexVector& idx_vec, span<T> buff_span) const override {
    lambda_(idx_vec, buff_span);
  }

  void put(const IndexVector& idx_vec, span<T> buff_span) override { NOT_ALLOWED(); }

  void add(const IndexVector& idx_vec, span<T> buff_span) override { NOT_ALLOWED(); }

  T* access_local_buf() override { NOT_ALLOWED(); }

  const T* access_local_buf() const override { NOT_ALLOWED(); }

  size_t local_buf_size() const override { NOT_ALLOWED(); }

  int64_t size() const override {
    int64_t res = 1;
    for(const auto& tis: block_indices_) { res *= tis.max_num_indices(); }
    return res;
  }

protected:
  Func lambda_; /**< Lambda function for the Tensor */

  Func construct_lambda(const IndexLabelVec& til_vec, Tensor<T> ref_tensor,
                        const IndexLabelVec& ref_labels, const IndexLabelVec& use_labels,
                        const std::vector<TranslateFunc>& translate_func_vec) {
    auto ip_gen_loop_builder_ptr = std::make_shared<blockops::cpu::IpGenLoopBuilder<2>>(
      std::array<IndexLabelVec, 2>{use_labels, ref_labels});

    auto perm_dup_map_compute = [](const IndexLabelVec& from, const IndexLabelVec& to) {
      std::vector<int> ret(from.size(), -1);
      for(size_t i = 0; i < from.size(); i++) {
        auto it = std::find(to.begin(), to.end(), from[i]);
        if(it != to.end()) { ret[i] = it - to.begin(); }
      }
      return ret;
    };

    auto perm_map = perm_dup_map_compute(use_labels, ref_labels);
    EXPECTS(perm_map.size() == use_labels.size());

    auto lambda = [til_vec, ref_tensor, ip_gen_loop_builder_ptr, translate_func_vec,
                   perm_map](const IndexVector& blockid, span<T> buff) {
      EXPECTS(blockid.size() == til_vec.size());
      EXPECTS(translate_func_vec.size() == blockid.size());

      IndexVector ref_blockid;
      for(size_t i = 0; i < blockid.size(); i++) {
        auto translated_id = translate_func_vec[i](blockid[i]);
        if(translated_id != -1) ref_blockid.push_back(translated_id);
      }

      auto           ref_buff_size  = ref_tensor.block_size(ref_blockid);
      auto           ref_block_dims = ref_tensor.block_dims(ref_blockid);
      std::vector<T> ref_buf(ref_buff_size);
      BlockSpan      ref_span(ref_buf.data(), ref_block_dims);

      std::vector<size_t> use_block_dims;
      for(size_t i = 0; i < perm_map.size(); i++) {
        use_block_dims.push_back(ref_block_dims[perm_map[i]]);
      }

      BlockSpan lhs_span(buff.data(), use_block_dims);
      EXPECTS(lhs_span.num_elements() == buff.size());

      ref_tensor.get(ref_blockid, ref_buf);

      blockops::cpu::flat_set(lhs_span, 0);

      ip_gen_loop_builder_ptr->update_plan(lhs_span, ref_span);
      blockops::cpu::ipgen_loop_assign(lhs_span.buf(), ip_gen_loop_builder_ptr->u2ald()[0],
                                       ref_span.buf(), ip_gen_loop_builder_ptr->u2ald()[1],
                                       ip_gen_loop_builder_ptr->unique_label_dims());
    };

    return lambda;
  }
}; // class LambdaTensorImpl

/**
 * @ingroup tensors
 * @brief Implementation class for TensorBase with dense multidimensional GA
 *
 * @tparam T Element type of Tensor
 */
template<typename T>
class DenseTensorImpl: public TensorImpl<T> {
public:
  using TensorImpl<T>::TensorBase::ec_;
  using TensorImpl<T>::TensorBase::setKind;
  using TensorImpl<T>::TensorBase::block_indices_;
  using TensorImpl<T>::TensorBase::allocation_status_;

  using TensorImpl<T>::block_size;
  using TensorImpl<T>::block_dims;
  using TensorImpl<T>::block_offsets;
  using TensorImpl<T>::proc_list_;
  using TensorImpl<T>::distribution_;
  using TensorImpl<T>::TensorBase::tindices;
  using TensorImpl<T>::TensorBase::num_modes;
  using TensorImpl<T>::TensorBase::update_status;

  // Ctors
  DenseTensorImpl() = default;

  // Copy/Move Ctors and Assignment Operators
  DenseTensorImpl(DenseTensorImpl&&)                 = default;
  DenseTensorImpl(const DenseTensorImpl&)            = delete;
  DenseTensorImpl& operator=(DenseTensorImpl&&)      = default;
  DenseTensorImpl& operator=(const DenseTensorImpl&) = delete;

  /**
   * @brief Construct a new LambdaTensorImpl object using a Lambda function
   *
   * @param [in] tis_vec vector of TiledIndexSpace objects for each mode of
   * the Tensor
   * @param [in] lambda a function for constructing the Tensor
   */
  DenseTensorImpl(const TiledIndexSpaceVec& tis_vec, const ProcGrid proc_grid,
                  const bool is_block_cyclic = false):
    TensorImpl<T>{tis_vec}, proc_grid_{proc_grid}, is_block_cyclic_{is_block_cyclic} {
    // check no dependences
    // for(auto& tis : tis_vec) { EXPECTS(!tis.is_dependent()); }

    // check index spaces are dense
    setKind(TensorBase::TensorKind::dense);
  }

  DenseTensorImpl(const IndexLabelVec& til_vec, const ProcGrid proc_grid,
                  const bool is_block_cyclic = false):
    TensorImpl<T>{til_vec}, proc_grid_{proc_grid}, is_block_cyclic_{is_block_cyclic} {
    // check no dependences
    // check index spaces are dense
    // for(auto& til : til_vec) { EXPECTS(!til.is_dependent()); }
    setKind(TensorBase::TensorKind::dense);
  }

  // Dtor
  ~DenseTensorImpl() {
    // EXPECTS(allocation_status_ == AllocationStatus::deallocated ||
    //         allocation_status_ == AllocationStatus::invalid);
  }

  const Distribution& distribution() const override {
    // return ref_tensor_.distribution();
    return *distribution_.get();
  }

  void deallocate() {
    EXPECTS(allocation_status_ == AllocationStatus::created);
#if defined(USE_UPCXX)
    local_nelems_ = 0;
    gptrs_.clear();
    tensor_dims_.clear();
    local_buf_dims_.clear();
    tiles_.clear();
    local_tiles_.clear();
    ec_->pg().barrier();
    upcxx::delete_array(local_gptr_);
    ec_->pg().barrier();
#else
    NGA_Destroy(ga_);
    ga_ = -1;
#endif
    update_status(AllocationStatus::deallocated);
    // get memory profiler instance
    auto& memprof = MemProfiler::instance();
    // update memory profiler instance
    memprof.mem_deallocated += size();
    memprof.dealloc_counter++;
  }

  void allocate(ExecutionContext* ec) {
    EXPECTS(allocation_status_ == AllocationStatus::invalid);

#if defined(USE_UPCXX)
    EXPECTS(ec->pg().rank() >= 0 && ec->pg().rank() < ec->pg().size());
#else
    ga_ = NGA_Create_handle();
#endif

    ec_             = ec;
    const int ndims = num_modes();

    auto          defd         = ec->get_default_distribution();
    Distribution* distribution = ec->distribution(defd->get_tensor_base(), defd->get_dist_proc());

    EXPECTS(distribution != nullptr);

    if(!proc_grid_.empty()) distribution->set_proc_grid(proc_grid_);

    distribution_ = std::shared_ptr<Distribution>(distribution->clone(this, ec->pg().size()));
    proc_grid_    = distribution_->proc_grid();

    delete defd;
    delete distribution;

    auto tis_dims = tindices();

    std::vector<std::vector<Tile>> new_tiles(ndims);
    std::vector<std::vector<Tile>> tiles_for_fixed_ts_dim(ndims);
    std::vector<bool>              is_irreg_tis(ndims);

    for(int i = 0; i < ndims; i++) {
      is_irreg_tis[i] = !tis_dims[i].input_tile_sizes().empty();

      if(is_irreg_tis[i]) continue;

      auto toff = tis_dims[i].tile_offsets();

      for(size_t j = 0; j < toff.size() - 1; ++j)
        tiles_for_fixed_ts_dim[i].push_back(toff[j + 1] - toff[j]);
    }

    for(int i = 0; i < ndims; i++)
      new_tiles[i] = is_irreg_tis[i] ? tis_dims[i].input_tile_sizes() : tiles_for_fixed_ts_dim[i];

#if defined(USE_UPCXX)
    int my_rank = ec->pg().rank().value();
    int nranks  = ec->pg().size().value();

    eltype_               = tensor_element_type<T>();
    size_t  element_size  = MemoryManagerGA::get_element_size(eltype_);
    int64_t total_n_tiles = 1;

    for(int i = 0; i < ndims; ++i) {
      tensor_dims_.push_back(tis_dims[i].index_space().num_indices());
      total_n_tiles *= new_tiles[i].size();
    }

    // Pad
    for(int i = ndims; i < 4; ++i) {
      is_irreg_tis.insert(is_irreg_tis.begin(), false);
      new_tiles.insert(new_tiles.begin(), {1});
      tensor_dims_.insert(tensor_dims_.begin(), 1);
    }

#else

    std::vector<int64_t> dims;
    for(auto tis: tis_dims) dims.push_back(tis.index_space().num_indices());

    NGA_Set_data64(ga_, ndims, dims.data(), ga_eltype_);

    if(proc_list_.size() > 0) {
      int nproc = proc_list_.size();
      int proclist_c[nproc];
      std::copy(proc_list_.begin(), proc_list_.end(), proclist_c);
      GA_Set_restricted(ga_, proclist_c, nproc);
    }

#endif

    if(is_block_cyclic_) {
      /*
       * Implement a block-cyclic distribution of tiles with size
       * (tis_dims[0].input_tile_size(), tis_dims[1].input_tile_size()) over a
       * processor grid with size (proc_grid_[0], proc_grid_[1]).
       */

      EXPECTS(ndims == 2);

      std::vector<int64_t> pgrid(2);
      pgrid[0] = proc_grid_[0].value();
      pgrid[1] = proc_grid_[1].value();

#if defined(USE_UPCXX)

      int64_t total_n_procs  = pgrid[0] * pgrid[1];
      int64_t tiles_per_proc = (total_n_tiles + total_n_procs - 1) / total_n_procs;

      size_t tile_size_in_bytes = 1;
      for(int i = 0; i < new_tiles.size(); ++i) tile_size_in_bytes *= new_tiles[i][0];
      tile_size_in_bytes *= element_size;

      local_gptr_ = upcxx::new_array<uint8_t>(tiles_per_proc * tile_size_in_bytes);
      memset(local_gptr_.local(), 0x00, tiles_per_proc * tile_size_in_bytes);

      int64_t  owning_rank  = 0;
      int64_t* tile_offsets = new int64_t[nranks]();

      for(int64_t i = 0; i < tensor_dims_[2]; i += new_tiles[2][0])
        for(int64_t j = 0; j < tensor_dims_[3]; j += new_tiles[3][0]) {
          TensorTile new_tile(0, 0, i, j, new_tiles[0][0], new_tiles[1][0], new_tiles[2][0],
                              new_tiles[3][0], owning_rank, tile_offsets[owning_rank]);
          tiles_.push_back(new_tile);

          if(owning_rank == my_rank) local_tiles_.push_back(new_tile);

          tile_offsets[owning_rank] += tile_size_in_bytes / element_size;
          owning_rank = (owning_rank + 1) % nranks;
        }

      local_nelems_ = tile_offsets[my_rank];

      delete[] tile_offsets;
#else
      // blocks_ = block sizes for scalapack distribution
      NGA_Set_block_cyclic_proc_grid64(
        ga_, std::vector<int64_t>{new_tiles[0][0], new_tiles[1][0]}.data(), pgrid.data());
#endif
    }
    else {
#if defined(USE_UPCXX)

      int64_t  tile_index     = 0;
      int64_t* tile_offsets   = new int64_t[nranks]();
      int64_t  tiles_per_proc = (total_n_tiles + nranks - 1) / nranks;

      for(int64_t i = 0, ii = 0; i < tensor_dims_[0]; i += new_tiles[0][ii], ++ii)
        for(int64_t j = 0, jj = 0; j < tensor_dims_[1]; j += new_tiles[1][jj], ++jj)
          for(int64_t k = 0, kk = 0; k < tensor_dims_[2]; k += new_tiles[2][kk], ++kk)
            for(int64_t l = 0, ll = 0; l < tensor_dims_[3]; l += new_tiles[3][ll], ++ll) {
              const int  owning_rank = tile_index / tiles_per_proc;
              TensorTile new_tile(i, j, k, l, new_tiles[0][ii], new_tiles[1][jj], new_tiles[2][kk],
                                  new_tiles[3][ll], owning_rank, tile_offsets[owning_rank]);
              tiles_.push_back(new_tile);

              if(owning_rank == my_rank) local_tiles_.push_back(new_tile);

              tile_offsets[owning_rank] +=
                new_tiles[0][ii] * new_tiles[1][jj] * new_tiles[2][kk] * new_tiles[3][ll];
              tile_index++;
            }

      if(local_nelems_ = tile_offsets[my_rank])
        for(int i = 4 - ndims; i < 4; ++i)
          local_buf_dims_.push_back(local_tiles_.back().lo[i] + local_tiles_.back().dim[i] -
                                    local_tiles_.front().lo[i]);

      local_gptr_ = upcxx::new_array<uint8_t>(local_nelems_ * element_size);
      memset(local_gptr_.local(), 0x00, local_nelems_ * element_size);

      delete[] tile_offsets;
#else

      // Only needed when irreg tile sizes are provided
      if(std::any_of(is_irreg_tis.begin(), is_irreg_tis.end(), [](bool v) { return v; })) {
        int64_t nblocks;
        int64_t pgrid[ndims];
        int64_t nblock[ndims];

        for(int i = 0; i < ndims; i++) pgrid[i] = proc_grid_[i].value();

#if 0 // tiled-irreg
        for(int i = 0; i < ndims; i++) { nblock[i] = new_tiles[i].size(); }

        nblocks = std::accumulate(nblock, nblock + ndims, (int) 1, std::multiplies<int>());
        if(nblocks <= nranks) {
          int proclist_c[nblocks];
          std::iota(proclist_c, proclist_c + nblocks, 0);
          GA_Set_restricted(ga_, proclist_c, nblocks);
          // for(int i = 0; i < ndims; i++) {
          //   pgrid[i] = nblock[i];
          //   proc_grid_[i] = nblock[i];
          // }
        }
#else
        for(int i = 0; i < ndims; i++) nblock[i] = pgrid[i];

        // if the number of blocks along dimension i > dims[i],
        // reset the number of processors along that dimension to dims[i]
        // and restrict the GA to the new proc grid.
        bool is_bgd{false};
        for(int i = 0; i < ndims; i++) {
          if(nblock[i] > dims[i]) {
            nblock[i] = dims[i];
            is_bgd    = true;
          }
        }

        if(is_bgd) {
          nblocks = std::accumulate(nblock, nblock + ndims, (int) 1, std::multiplies<int>());
          int proclist_c[nblocks];
          std::iota(proclist_c, proclist_c + nblocks, 0);
          GA_Set_restricted(ga_, proclist_c, nblocks);
        }
#endif

        // distribution->set_proc_grid(proc_grid_);

        auto map_size = std::accumulate(nblock, nblock + ndims, (int64_t) 0);
        std::vector<int64_t> k_map(map_size);
        {
          auto mi = 0;
#if 0 // tiled-irreg
          for(auto idim = 0; idim < ndims; idim++) {
            int64_t boff = 0;
            k_map[mi] = boff;
            mi++;
            for(auto i = 0; i < new_tiles[idim].size() - 1; i++) {
              boff += new_tiles[idim][i];
              k_map[mi] = boff;
              mi++;
            }
          }
#else
          for(auto count_dim = 0; count_dim < ndims; count_dim++) {
            auto size_blk = dims[count_dim] / nblock[count_dim];
            for(auto i = 0; i < nblock[count_dim]; i++) {
              k_map[mi] = size_blk * i;
              mi++;
            }
          }
#endif
        }
        // if(nblocks <= nranks)
        NGA_Set_irreg_distr64(ga_, &k_map[0], nblock);
        // else NGA_Set_tiled_irreg_proc_grid64(ga_, &k_map[0], nblock, pgrid);
      }
      else {
        // Fixed tilesize for all dims
        int64_t chunk[ndims];
        for(int i = 0; i < ndims; i++) chunk[i] = tis_dims[i].input_tile_size();

        GA_Set_chunk64(ga_, chunk);
      }

#endif
    }

#if defined(USE_UPCXX)

    gptrs_.resize(nranks);
    upcxx::promise<> p(nranks);
    for(int r = 0; r < nranks; r++)
      upcxx::broadcast(local_gptr_, r, *ec->pg().comm())
        .then([this, &p, r](upcxx::global_ptr<uint8_t> result) {
          gptrs_[r] = result;
          p.fulfill_anonymous(1);
        });
    p.get_future().wait();

#else

    NGA_Set_pgroup(ga_, ec->pg().ga_pg());
    NGA_Allocate(ga_);
    distribution_->set_ga_handle(ga_);

    int64_t lo[ndims];
    int64_t hi[ndims];
    NGA_Distribution64(ga_, ec->pg().rank().value(), lo, hi);

    int64_t pbs{1};
    for(auto idim = 0; idim < ndims; idim++) {
      auto val_ = hi[idim] - lo[idim] + 1;
      if(val_ > 0) pbs *= val_;
    }
    if(pbs > 0) distribution_->set_proc_buf_size((Size) pbs);

    int64_t lmax_pbs{pbs};
    auto gmax_pbs = ec->pg().allreduce(&lmax_pbs, ReduceOp::max);
    if(gmax_pbs > 0) distribution_->set_max_proc_buf_size((Size) gmax_pbs);

#endif

    update_status(AllocationStatus::created);
    // get memory profiler instance
    auto& memprof = MemProfiler::instance();
    // update memory profiler instance
    memprof.alloc_counter++;
    const auto tsize = size();
    memprof.mem_allocated += tsize;
    memprof.max_in_single_allocate =
      tsize > memprof.max_in_single_allocate ? tsize : memprof.max_in_single_allocate;
    memprof.max_total_allocated = (memprof.mem_allocated - memprof.mem_deallocated) >
                                      memprof.max_total_allocated
                                    ? (memprof.mem_allocated - memprof.mem_deallocated)
                                    : memprof.max_total_allocated;
  }

#ifdef USE_UPCXX
  const TensorTile& find_tile(int64_t i, int64_t j, int64_t k, int64_t l) const {
    auto t = std::find_if(tiles_.cbegin(), tiles_.cend(), [i, j, k, l](const TensorTile& tile) {
      return tile.contains(i, j, k, l);
    });

    if(t != tiles_.cend()) return *t;
    abort();
  }

  std::optional<TensorTile> find_local_tile(int64_t i, int64_t j, int64_t k, int64_t l) const {
    auto t =
      std::find_if(local_tiles_.cbegin(), local_tiles_.cend(),
                   [i, j, k, l](const TensorTile& tile) { return tile.contains(i, j, k, l); });

    if(t != local_tiles_.cend()) return *t;
    return std::nullopt;
  }
#endif

  void get(const IndexVector& blockid, span<T> buff_span) const {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    EXPECTS(block_size(blockid) <= buff_span.size());

    std::vector<int64_t> lo = compute_lo(blockid);
    std::vector<int64_t> hi = compute_hi(blockid);

#if defined(USE_UPCXX)
    // Pad
    for(int i = lo.size(); i < 4; ++i) {
      lo.insert(lo.begin(), 0);
      hi.insert(hi.begin(), 0);
    }

    get_raw_contig(&lo[0], &hi[0], &buff_span[0]);
#else
    std::vector<int64_t> ld = compute_ld(blockid);
    NGA_Get64(ga_, &lo[0], &hi[0], &buff_span[0], &ld[0]);
#endif
  }

  void put(const IndexVector& blockid, span<T> buff_span) {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    EXPECTS(block_size(blockid) <= buff_span.size());

    std::vector<int64_t> lo = compute_lo(blockid);
    std::vector<int64_t> hi = compute_hi(blockid);

#if defined(USE_UPCXX)
    // Pad
    for(int i = lo.size(); i < 4; ++i) {
      lo.insert(lo.begin(), 0);
      hi.insert(hi.begin(), 0);
    }

    put_raw_contig(&lo[0], &hi[0], &buff_span[0]);
#else
    std::vector<int64_t> ld = compute_ld(blockid);
    NGA_Put64(ga_, &lo[0], &hi[0], &buff_span[0], &ld[0]);
#endif
  }

  void add(const IndexVector& blockid, span<T> buff_span) {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    EXPECTS(block_size(blockid) <= buff_span.size());

    std::vector<int64_t> lo = compute_lo(blockid);

#if defined(USE_UPCXX)
    for(size_t i = lo.size(); i < 4; ++i) lo.insert(lo.begin(), 0);

    TensorTile t = find_tile(lo[0], lo[1], lo[2], lo[3]);

    upcxx::rpc(
      *ec_->pg().comm(), t.rank,
      [](const upcxx::global_ptr<T>& dst_buf, const upcxx::view<T>& src_buf) {
        T*     dst = dst_buf.local();
        size_t n   = src_buf.size();

        for(size_t i = 0; i < n; ++i) dst[i] += src_buf[i];
      },
      upcxx::reinterpret_pointer_cast<T>(gptrs_[t.rank]) + t.offset,
      upcxx::make_view((T*) buff_span.data(), (T*) buff_span.data() + block_size(blockid)))
      .wait();

#else
    void* alpha;
    switch(from_ga_eltype(ga_eltype_)) {
      case ElementType::single_precision: alpha = reinterpret_cast<void*>(&sp_alpha); break;
      case ElementType::double_precision: alpha = reinterpret_cast<void*>(&dp_alpha); break;
      case ElementType::single_complex: alpha = reinterpret_cast<void*>(&scp_alpha); break;
      case ElementType::double_complex: alpha = reinterpret_cast<void*>(&dcp_alpha); break;
      case ElementType::invalid:
      default: UNREACHABLE();
    }

    std::vector<int64_t> hi = compute_hi(blockid);
    std::vector<int64_t> ld = compute_ld(blockid);

    NGA_Acc64(ga_, &lo[0], &hi[0], reinterpret_cast<void*>(buff_span.data()), &ld[0], alpha);
#endif
  }

#ifndef USE_UPCXX
  int ga_handle() override { return ga_; }
#endif

  bool is_block_cyclic() override { return is_block_cyclic_; }

  /// @todo Should this be GA_Nodeid() or GA_Proup_nodeid(GA_Get_pgroup(ga_))
  T* access_local_buf() override {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    T* ptr;
#if defined(USE_UPCXX)
    ptr = reinterpret_cast<T*>(local_gptr_.local());
#else
    int64_t len;
    NGA_Access_block_segment64(ga_, GA_Pgroup_nodeid(GA_Get_pgroup(ga_)),
                               reinterpret_cast<void*>(&ptr), &len);
#endif
    return ptr;
  }

  /// @todo Should this be GA_Nodeid() or GA_Proup_nodeid(GA_Get_pgroup(ga_))
  const T* access_local_buf() const override {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    T* ptr;
#if defined(USE_UPCXX)
    ptr = reinterpret_cast<T*>(local_gptr_.local());
#else
    int64_t len;
    NGA_Access_block_segment64(ga_, GA_Pgroup_nodeid(GA_Get_pgroup(ga_)),
                               reinterpret_cast<void*>(&ptr), &len);
#endif
    return ptr;
  }

  std::vector<int64_t> local_buf_dims() const {
#ifdef USE_UPCXX
    assert(!is_block_cyclic_);
    return local_buf_dims_.size() > 0 ? local_buf_dims_ : std::vector<int64_t>(num_modes());
#else
    throw std::runtime_error("Function local_buf_dims not defined for GA backend.");
#endif
  }

  bool is_local_element(int64_t i, int64_t j, int64_t k, int64_t l) const {
#ifdef USE_UPCXX
    auto t = find_local_tile(i, j, k, l);
    if(t.has_value()) return true;
    return false;
#else
    throw std::runtime_error("Function is_local_element not defined for GA backend.");
#endif
  }

  std::vector<int64_t> local_tiles_offsets() const {
#ifdef USE_UPCXX
    std::vector<int64_t> offsets(local_tiles_.size());
    std::transform(local_tiles_.cbegin(), local_tiles_.cend(), offsets.begin(),
                   [](const auto& tile) { return tile.offset; });
    return offsets;
#else
    throw std::runtime_error("Function local_tiles_offsets not defined for GA backend.");
#endif
  }

  std::pair<int64_t, int64_t> local_element_offsets(int64_t i, int64_t j, int64_t k,
                                                    int64_t l) const {
#ifdef USE_UPCXX
    auto t = find_local_tile(i, j, k, l);

    if(t.has_value()) {
      int64_t i_offset = i - t->lo[0];
      int64_t j_offset = j - t->lo[1];
      int64_t k_offset = k - t->lo[2];
      int64_t l_offset = l - t->lo[3];
      int64_t tile_offset =
        l_offset + t->dim[3] * (k_offset + t->dim[2] * (j_offset + t->dim[1] * i_offset));

      //        outer       inner
      return {t->offset, tile_offset};
    }

    return {-1, -1};
#else
    throw std::runtime_error("Function local_element_offsets not defined for GA backend.");
#endif
  }

#ifdef USE_UPCXX
  std::vector<TensorTile>::const_iterator local_tiles_begin() const {
    return local_tiles_.cbegin();
  }

  std::vector<TensorTile>::const_iterator local_tiles_end() const { return local_tiles_.cend(); }

  void put_raw_contig(int64_t* lo, int64_t* hi, void* buf) const {
    const auto elem_sz  = MemoryManagerGA::get_element_size(eltype_);
    TensorTile t        = find_tile(lo[0], lo[1], lo[2], lo[3]);
    int64_t    i_offset = lo[0] - t.lo[0];
    int64_t    j_offset = lo[1] - t.lo[1];
    int64_t    k_offset = lo[2] - t.lo[2];
    int64_t    l_offset = lo[3] - t.lo[3];
    int64_t    tile_offset =
      l_offset + t.dim[3] * (k_offset + t.dim[2] * (j_offset + t.dim[1] * i_offset));
    upcxx::global_ptr<uint8_t> remote_addr =
      gptrs_[t.rank] + (t.offset * elem_sz) + (tile_offset * elem_sz);
    auto a = (hi[3] - lo[3] + 1);
    auto b = (hi[2] - lo[2] + 1);
    auto c = (hi[1] - lo[1] + 1);
    auto d = (hi[0] - lo[0] + 1);
    upcxx::rput((uint8_t*) buf, remote_addr, elem_sz * a * b * c * d).wait();
  }

  void put_raw(int64_t* lo, int64_t* hi, void* buf) const {
    const auto       elem_sz = MemoryManagerGA::get_element_size(eltype_);
    int              next    = 0;
    upcxx::promise<> p;
    std::unordered_map<upcxx::intrank_t,
                       std::pair<std::vector<uint8_t*>, std::vector<upcxx::global_ptr<uint8_t>>>>
      all_puts;

    for(int64_t i = lo[0]; i <= hi[0]; ++i)
      for(int64_t j = lo[1]; j <= hi[1]; ++j)
        for(int64_t k = lo[2]; k <= hi[2]; ++k)
          for(int64_t l = lo[3]; l <= hi[3]; ++l) {
            TensorTile t        = find_tile(i, j, k, l);
            int64_t    i_offset = i - t.lo[0];
            int64_t    j_offset = j - t.lo[1];
            int64_t    k_offset = k - t.lo[2];
            int64_t    l_offset = l - t.lo[3];
            int64_t    tile_offset =
              l_offset + t.dim[3] * (k_offset + t.dim[2] * (j_offset + t.dim[1] * i_offset));
            upcxx::global_ptr<uint8_t> remote_addr =
              gptrs_[t.rank] + (t.offset * elem_sz) + (tile_offset * elem_sz);
            uint8_t* local_addr = ((uint8_t*) buf) + (next++ * elem_sz);

            if(remote_addr.is_local()) { *((T*) remote_addr.local()) = *(T*) local_addr; }
            else {
              all_puts[t.rank].first.push_back(local_addr);
              all_puts[t.rank].second.push_back(remote_addr);
            }
          }

    for(const auto& x: all_puts)
      upcxx::rput_regular(x.second.first.begin(), x.second.first.end(), elem_sz,
                          x.second.second.begin(), x.second.second.end(), elem_sz,
                          upcxx::operation_cx::as_promise(p));

    p.finalize().wait();
  }

  void get_raw_contig(int64_t* lo, int64_t* hi, void* buf) const {
    const auto elem_sz  = MemoryManagerGA::get_element_size(eltype_);
    TensorTile t        = find_tile(lo[0], lo[1], lo[2], lo[3]);
    int64_t    i_offset = lo[0] - t.lo[0];
    int64_t    j_offset = lo[1] - t.lo[1];
    int64_t    k_offset = lo[2] - t.lo[2];
    int64_t    l_offset = lo[3] - t.lo[3];
    int64_t    tile_offset =
      l_offset + t.dim[3] * (k_offset + t.dim[2] * (j_offset + t.dim[1] * i_offset));
    upcxx::global_ptr<uint8_t> remote_addr =
      gptrs_[t.rank] + (t.offset * elem_sz) + (tile_offset * elem_sz);
    auto a = (hi[3] - lo[3] + 1);
    auto b = (hi[2] - lo[2] + 1);
    auto c = (hi[1] - lo[1] + 1);
    auto d = (hi[0] - lo[0] + 1);
    upcxx::rget(remote_addr, (uint8_t*) buf, elem_sz * a * b * c * d).wait();
  }

  void get_raw(int64_t* lo, int64_t* hi, void* buf) const {
    const auto       elem_sz = MemoryManagerGA::get_element_size(eltype_);
    int              next    = 0;
    upcxx::promise<> p;
    std::unordered_map<upcxx::intrank_t,
                       std::pair<std::vector<upcxx::global_ptr<uint8_t>>, std::vector<uint8_t*>>>
      all_gets;

    for(int64_t i = lo[0]; i <= hi[0]; ++i)
      for(int64_t j = lo[1]; j <= hi[1]; ++j)
        for(int64_t k = lo[2]; k <= hi[2]; ++k)
          for(int64_t l = lo[3]; l <= hi[3]; ++l) {
            TensorTile t        = find_tile(i, j, k, l);
            int64_t    i_offset = i - t.lo[0];
            int64_t    j_offset = j - t.lo[1];
            int64_t    k_offset = k - t.lo[2];
            int64_t    l_offset = l - t.lo[3];
            int64_t    tile_offset =
              l_offset + t.dim[3] * (k_offset + t.dim[2] * (j_offset + t.dim[1] * i_offset));
            upcxx::global_ptr<uint8_t> remote_addr =
              gptrs_[t.rank] + (t.offset * elem_sz) + (tile_offset * elem_sz);
            uint8_t* local_addr = ((uint8_t*) buf) + (next++ * elem_sz);

            if(remote_addr.is_local()) { *(T*) local_addr = *((T*) remote_addr.local()); }
            else {
              all_gets[t.rank].first.push_back(remote_addr);
              all_gets[t.rank].second.push_back(local_addr);
            }
          }

    for(const auto& x: all_gets)
      upcxx::rget_regular(x.second.first.begin(), x.second.first.end(), elem_sz,
                          x.second.second.begin(), x.second.second.end(), elem_sz,
                          upcxx::operation_cx::as_promise(p));

    p.finalize().wait();
  }
#endif

  /// @todo Check for a GA method to get the local buf size?
  size_t local_buf_size() const override {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    size_t res;
#if defined(USE_UPCXX)
    res = (size_t) local_nelems_;
#else
    T* ptr;
    int64_t len;
    NGA_Access_block_segment64(ga_, GA_Pgroup_nodeid(GA_Get_pgroup(ga_)),
                               reinterpret_cast<void*>(&ptr), &len);
    res = (size_t) len;
#endif
    return res;
  }

  /// @todo implement accordingly
  int64_t size() const override {
    EXPECTS(allocation_status_ != AllocationStatus::invalid);
    int64_t res = 1;
    for(const auto& tis: block_indices_) { res *= tis.max_num_indices(); }
    return res;
  }

protected:
  std::vector<int64_t> compute_lo(const IndexVector& blockid) const {
    std::vector<int64_t> retv;
    std::vector<size_t>  off = block_offsets(blockid);
    for(const auto& i: off) { retv.push_back(static_cast<int64_t>(i)); }
    return retv;
  }

  std::vector<int64_t> compute_hi(const IndexVector& blockid) const {
    std::vector<int64_t> retv;
    std::vector<size_t>  boff  = block_offsets(blockid);
    std::vector<size_t>  bdims = block_dims(blockid);
    for(size_t i = 0; i < boff.size(); i++) {
      retv.push_back(static_cast<int64_t>(boff[i] + bdims[i] - 1));
    }
    return retv;
  }

  std::vector<int64_t> compute_ld(const IndexVector& blockid) const {
    std::vector<size_t>  bdims = block_dims(blockid);
    std::vector<int64_t> retv(bdims.size() - 1, 1);
    for(size_t i = 1; i < bdims.size(); i++) retv[i - 1] = (int64_t) (bdims[i]);
    return retv;
  }

#if defined(USE_UPCXX)
  ElementType                             eltype_;
  int64_t                                 local_nelems_;
  ProcGrid                                proc_grid_;
  upcxx::global_ptr<uint8_t>              local_gptr_;
  std::vector<upcxx::global_ptr<uint8_t>> gptrs_;
  std::vector<int64_t>                    tensor_dims_;
  std::vector<int64_t>                    local_buf_dims_;
  std::vector<TensorTile>                 tiles_;
  std::vector<TensorTile>                 local_tiles_;
#else
  int ga_;
  ProcGrid proc_grid_;
#endif

  bool is_block_cyclic_ = false;

  // constants for NGA_Acc call
  float         sp_alpha  = 1.0;
  double        dp_alpha  = 1.0;
  SingleComplex scp_alpha = {1, 0};
  DoubleComplex dcp_alpha = {1, 0};

  int ga_eltype_ = to_ga_eltype(tensor_element_type<T>());

}; // class DenseTensorImpl

template<typename T>
class ViewTensorImpl: public TensorImpl<T> {
public:
  using TensorImpl<T>::distribution_;
  using TensorImpl<T>::TensorBase::update_status;
  using TensorImpl<T>::TensorBase::setKind;
  using TensorImpl<T>::TensorBase::block_size;
  using TensorImpl<T>::TensorBase::block_dims;
  using TensorImpl<T>::TensorBase::ec_;
  using Func     = std::function<IndexVector(const IndexVector&)>;
  using CopyFunc = std::function<void(const BlockSpan<T>&, BlockSpan<T>&, const IndexVector&)>;

  // Ctors
  ViewTensorImpl() = default;
  ViewTensorImpl(Tensor<T> ref_tensor, const TiledIndexSpaceVec& tis_vec, Func ref_map_func):
    TensorImpl<T>(tis_vec), ref_tensor_{ref_tensor}, ref_map_func_{ref_map_func} {
    setKind(TensorBase::TensorKind::view);
    if(ref_tensor_.is_allocated()) {
      distribution_ =
        std::make_shared<ViewDistribution>(&ref_tensor_.distribution(), ref_map_func_);
      update_status(AllocationStatus::created);
    }
    EXPECTS(ref_tensor_.is_allocated());
    ec_ = ref_tensor.execution_context();
  }

  ViewTensorImpl(Tensor<T> ref_tensor, const IndexLabelVec& labels, Func ref_map_func):
    TensorImpl<T>(labels), ref_tensor_{ref_tensor}, ref_map_func_{ref_map_func} {
    setKind(TensorBase::TensorKind::view);

    if(ref_tensor_.is_allocated()) {
      distribution_ =
        std::make_shared<ViewDistribution>(&ref_tensor_.distribution(), ref_map_func_);
      update_status(AllocationStatus::created);
    }
    EXPECTS(ref_tensor_.is_allocated());
    ec_ = ref_tensor.execution_context();
  }

  ViewTensorImpl(Tensor<T> ref_tensor, const IndexLabelVec& labels, Func ref_map_func,
                 CopyFunc get_func, CopyFunc put_func):
    TensorImpl<T>(labels),
    ref_tensor_{ref_tensor},
    ref_map_func_{ref_map_func},
    get_func_{get_func},
    put_func_{put_func} {
    setKind(TensorBase::TensorKind::view);

    if(ref_tensor_.is_allocated()) {
      distribution_ =
        std::make_shared<ViewDistribution>(&ref_tensor_.distribution(), ref_map_func_);
      update_status(AllocationStatus::created);
    }
    EXPECTS(ref_tensor_.is_allocated());
    ec_ = ref_tensor.execution_context();
  }

  // Copy/Move Ctors and Assignment Operators
  ViewTensorImpl(ViewTensorImpl&&)                 = default;
  ViewTensorImpl(const ViewTensorImpl&)            = delete;
  ViewTensorImpl& operator=(ViewTensorImpl&&)      = default;
  ViewTensorImpl& operator=(const ViewTensorImpl&) = delete;

  // Dtor
  ~ViewTensorImpl() = default;

  /**
   * @brief Virtual method for deallocating a Tensor
   *
   */
  void deallocate() override {
    // No-op
  }

  /**
   * @brief Virtual method for allocating a Tensor using an ExecutionContext
   *
   * @param [in] ec ExecutionContext to be used for allocation
   */
  void allocate(ExecutionContext* ec) override {
    // No-op
    // EXPECTS(ref_tensor_.is_allocated());
    // distribution_ = std::make_shared<ViewDistribution>(
    //     &ref_tensor_.distribution(), ref_map_func_);
    // update_status(AllocationStatus::created);
  }

  const Distribution& distribution() const override {
    // return ref_tensor_.distribution();
    return *distribution_.get();
  }

  // Tensor Accessors
  /**
   * @brief Tensor accessor method for getting values from a set of
   * indices to specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to fetch the values
   * @param [in] buff_span memory span where to put the fetched values
   */

  void get(const IndexVector& view_vec, span<T> buff_span) const override {
    // Call Reference tensors get
    const auto&     idx_vec       = ref_map_func_(view_vec);
    auto            ref_buf_size  = ref_tensor_.block_size(idx_vec);
    auto            ref_blck_dims = ref_tensor_.block_dims(idx_vec);
    std::vector<T>  ref_vec(ref_buf_size, 0);
    const BlockSpan ref_span{ref_vec.data(), ref_blck_dims};
    // @todo we might not need to do a get depending on if the tile has non-zero values
    ref_tensor_.get(idx_vec, ref_vec);

    auto      blck_dims = block_dims(view_vec);
    BlockSpan blck_span{buff_span.data(), blck_dims};

    get_func_(ref_span, blck_span, view_vec);
  }

  /**
   * @brief Tensor accessor method for getting values in nonblocking fashion
   * from a set of indices to specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to fetch the values
   * @param [in] buff_span memory span where to put the fetched values
   *
   * @bug should fix handle return
   */

  void nb_get(const IndexVector& view_vec, span<T> buff_span,
              DataCommunicationHandlePtr data_comm_handle) const override {
    // Call Reference tensors nb_get
    const auto&     idx_vec       = ref_map_func_(view_vec);
    auto            ref_buf_size  = ref_tensor_.block_size(idx_vec);
    auto            ref_blck_dims = ref_tensor_.block_dims(idx_vec);
    std::vector<T>  ref_vec(ref_buf_size, 0);
    const BlockSpan ref_span{ref_vec.data(), ref_blck_dims};

    ref_tensor_.nb_get(idx_vec, ref_vec, data_comm_handle);

    auto      blck_dims = block_dims(view_vec);
    BlockSpan blck_span{buff_span.data(), blck_dims};

    get_func_(ref_span, blck_span, view_vec);
  }

  /**
   * @brief Tensor accessor method for putting values to a set of indices
   * with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  void put(const IndexVector& view_vec, span<T> buff_span) override {
    NOT_ALLOWED();
    // Call Reference tensors put
    const auto&    idx_vec       = ref_map_func_(view_vec);
    auto           ref_buf_size  = ref_tensor_.block_size(idx_vec);
    auto           ref_blck_dims = ref_tensor_.block_dims(idx_vec);
    std::vector<T> ref_vec(ref_buf_size, 0);
    BlockSpan      ref_span{ref_vec.data(), ref_blck_dims};

    auto            blck_dims = block_dims(view_vec);
    const BlockSpan blck_span{buff_span.data(), blck_dims};

    put_func_(blck_span, ref_span, view_vec);

    ref_tensor_.put(idx_vec, ref_vec);
  }

  /**
   * @brief Tensor accessor method for putting values in nonblocking fashion
   * to a set of indices with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  void nb_put(const IndexVector& view_vec, span<T> buff_span,
              DataCommunicationHandlePtr data_comm_handle) override {
    NOT_ALLOWED();
    // Call Reference tensors nb_put
    const auto&    idx_vec       = ref_map_func_(view_vec);
    auto           ref_buf_size  = ref_tensor_.block_size(idx_vec);
    auto           ref_blck_dims = ref_tensor_.block_dims(idx_vec);
    std::vector<T> ref_vec(ref_buf_size, 0);
    BlockSpan      ref_span{ref_vec.data(), ref_blck_dims};

    auto            blck_dims = block_dims(view_vec);
    const BlockSpan blck_span{buff_span.data(), blck_dims};

    put_func_(blck_span, ref_span, view_vec);
    ref_tensor_.nb_put(idx_vec, ref_vec, data_comm_handle);
  }

  /**
   * @brief Tensor accessor method for adding svalues to a set of indices
   * with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  void add(const IndexVector& view_vec, span<T> buff_span) override {
    // Call Reference tensors nb_put
    const auto& idx_vec = ref_map_func_(view_vec);
    ref_tensor_.add(idx_vec, buff_span);
  }

  /**
   * @brief Tensor accessor method for adding svalues in nonblocking fashion
   * to a set of indices with the specified memory span
   *
   * @tparam T type of the values hold on the tensor object
   * @param [in] idx_vec a vector of indices to put the values
   * @param [in] buff_span buff_span memory span for the values to put
   */

  void nb_add(const IndexVector& view_vec, span<T> buff_span,
              DataCommunicationHandlePtr data_comm_handle) override {
    // Call Reference tensors nb_put
    const auto& idx_vec = ref_map_func_(view_vec);
    ref_tensor_.nb_add(idx_vec, buff_span, data_comm_handle);
  }

#if !defined(USE_UPCXX)
  int ga_handle() override {
    // Call Reference tensor
    return ref_tensor_.ga_handle();
  }
#endif

  T* access_local_buf() override { NOT_ALLOWED(); }

  const T* access_local_buf() const override { NOT_ALLOWED(); }

protected:
  Func      ref_map_func_;
  CopyFunc  get_func_;
  CopyFunc  put_func_;
  Tensor<T> ref_tensor_;
}; // class ViewTensorImpl

template<typename T>
class TensorUnitTiled: public TensorImpl<T> {
public:
  using TensorImpl<T>::mpb_;
  using TensorImpl<T>::distribution_;
  using TensorImpl<T>::update_status;
  using TensorImpl<T>::is_allocated;

  using TensorImpl<T>::TensorBase::ec_;
  using TensorImpl<T>::TensorBase::setKind;
  using TensorImpl<T>::TensorBase::allocation_status_;
  using TensorImpl<T>::TensorBase::is_non_zero;
  using TensorImpl<T>::TensorBase::block_size;

  // Ctors
  TensorUnitTiled() = default;

  TensorUnitTiled(const Tensor<T>& opt_tensor, size_t unit_tis_count):
    TensorImpl<T>{construct_new_tis(opt_tensor, unit_tis_count)}, tensor_opt_{opt_tensor} {
    setKind(TensorBase::TensorKind::unit_view);
    if(tensor_opt_.is_allocated()) { allocate(tensor_opt_.execution_context()); }
  }

  TensorUnitTiled(const Tensor<T>& opt_tensor, size_t unit_tis_count,
                  const std::vector<size_t>& spin_sizes):
    TensorImpl<T>{construct_new_tis(opt_tensor, unit_tis_count), spin_sizes},
    tensor_opt_{opt_tensor} {
    setKind(TensorBase::TensorKind::unit_view);
    if(tensor_opt_.is_allocated()) { allocate(tensor_opt_.execution_context()); }
  }

  // Copy/Move Ctors and Assignment Operators
  TensorUnitTiled(TensorUnitTiled&&)                 = default;
  TensorUnitTiled(const TensorUnitTiled&)            = delete;
  TensorUnitTiled& operator=(TensorUnitTiled&&)      = default;
  TensorUnitTiled& operator=(const TensorUnitTiled&) = delete;

  // Dtor
  ~TensorUnitTiled() = default;

  /**
   * @brief Virtual method implementation for deallocating a unit tiled view tensor
   * @todo Decide on the actual behavior - no action is done for now
   */
  void deallocate() override {
    EXPECTS(allocation_status_ == AllocationStatus::created);
    EXPECTS(mpb_);
    // ec_->unregister_for_dealloc(mpb_);
    // mpb_->dealloc_coll();
    // delete mpb_;
    // mpb_ = nullptr;
    // update_status(AllocationStatus::deallocated);
  }

  /**
   * @brief Virtual method implementation for allocating a unit tiled view tensor using an
   * ExecutionContext
   *
   * @param [in] ec ExecutionContext to be used for allocation
   *
   */
  void allocate(ExecutionContext* ec) override {
    EXPECTS(tensor_opt_.is_allocated());

    if(!is_allocated()) {
      auto          defd         = ec->get_default_distribution();
      Distribution* distribution = ec->distribution(defd->get_tensor_base(), defd->get_dist_proc());
      MemoryManager* memory_manager = ec->memory_manager();
      EXPECTS(distribution != nullptr);
      EXPECTS(memory_manager != nullptr);
      ec_ = ec;

      distribution_ =
        std::shared_ptr<Distribution>(new UnitTileDistribution(this, &tensor_opt_.distribution()));

      EXPECTS(distribution_ != nullptr);

      delete defd;
      delete distribution;
      delete memory_manager;

      auto eltype = tensor_element_type<T>();
      mpb_        = tensor_opt_.memory_region();
      EXPECTS(mpb_ != nullptr);
      update_status(AllocationStatus::created);
    }
  }

  const Distribution& distribution() const override { return *distribution_.get(); }

  // // Tensor Accessors
  // /**
  //  * @brief Tensor accessor method for getting values from a set of
  //  * indices to specified memory span
  //  *
  //  * @tparam T type of the values hold on the tensor object
  //  * @param [in] idx_vec a vector of indices to fetch the values
  //  * @param [in] buff_span memory span where to put the fetched values
  //  */

  // void get(const IndexVector &idx_vec, span<T> buff_span) const override {
  //   EXPECTS(allocation_status_ != AllocationStatus::invalid);

  //   if(!is_non_zero(idx_vec)) {
  //       Size size = block_size(idx_vec);
  //       EXPECTS(size <= buff_span.size());
  //       for(size_t i = 0; i < size; i++) { buff_span[i] = (T)0; }
  //       return;
  //   }

  //   Proc proc;
  //   Offset offset;
  //   std::tie(proc, offset) = distribution_->locate(idx_vec);
  //   Size size              = block_size(idx_vec);
  //   EXPECTS(size <= buff_span.size());
  //   mpb_->mgr().get(*mpb_, proc, offset, Size{size}, buff_span.data());
  // }

  //   /**
  //    * @brief Tensor accessor method for getting values in nonblocking fashion
  //    * from a set of indices to specified memory span
  //    *
  //    * @tparam T type of the values hold on the tensor object
  //    * @param [in] idx_vec a vector of indices to fetch the values
  //    * @param [in] buff_span memory span where to put the fetched values
  //    */

  //   virtual void nb_get(const IndexVector& idx_vec, span<T> buff_span,
  //                       DataCommunicationHandlePtr data_comm_handle) const {
  //       EXPECTS(allocation_status_ != AllocationStatus::invalid);

  //       if(!is_non_zero(idx_vec)) {
  //           Size size = block_size(idx_vec);
  //           EXPECTS(size <= buff_span.size());
  //           for(size_t i = 0; i < size; i++) { buff_span[i] = (T)0; }
  //           return;
  //       }

  //       Proc proc;
  //       Offset offset;
  //       std::tie(proc, offset) = distribution_->locate(idx_vec);
  //       Size size              = block_size(idx_vec);
  //       EXPECTS(size <= buff_span.size());
  //       mpb_->mgr().nb_get(*mpb_, proc, offset, Size{size}, buff_span.data(),
  //                          data_comm_handle);
  //   }

  //   /**
  //    * @brief Tensor accessor method for putting values to a set of indices
  //    * with the specified memory span
  //    *
  //    * @tparam T type of the values hold on the tensor object
  //    * @param [in] idx_vec a vector of indices to put the values
  //    * @param [in] buff_span buff_span memory span for the values to put
  //    */

  //   virtual void put(const IndexVector& idx_vec, span<T> buff_span) {
  //       EXPECTS(allocation_status_ != AllocationStatus::invalid);

  //       if(!is_non_zero(idx_vec)) { return; }

  //       Proc proc;
  //       Offset offset;
  //       std::tie(proc, offset) = distribution_->locate(idx_vec);
  //       Size size              = block_size(idx_vec);
  //       EXPECTS(size <= buff_span.size());
  //       mpb_->mgr().put(*mpb_, proc, offset, Size{size}, buff_span.data());
  //   }

  //   /**
  //    * @brief Tensor accessor method for putting values in nonblocking fashion
  //    * to a set of indices with the specified memory span
  //    *
  //    * @tparam T type of the values hold on the tensor object
  //    * @param [in] idx_vec a vector of indices to put the values
  //    * @param [in] buff_span buff_span memory span for the values to put
  //    */

  //   virtual void nb_put(const IndexVector& idx_vec, span<T> buff_span,
  //                       DataCommunicationHandlePtr data_comm_handle) {
  //       EXPECTS(allocation_status_ != AllocationStatus::invalid);

  //       if(!is_non_zero(idx_vec)) { return; }

  //       Proc proc;
  //       Offset offset;
  //       std::tie(proc, offset) = distribution_->locate(idx_vec);
  //       Size size              = block_size(idx_vec);
  //       EXPECTS(size <= buff_span.size());
  //       mpb_->mgr().nb_put(*mpb_, proc, offset, Size{size}, buff_span.data(),
  //                          data_comm_handle);
  //   }

  //   /**
  //    * @brief Tensor accessor method for adding svalues to a set of indices
  //    * with the specified memory span
  //    *
  //    * @tparam T type of the values hold on the tensor object
  //    * @param [in] idx_vec a vector of indices to put the values
  //    * @param [in] buff_span buff_span memory span for the values to put
  //    */

  //   virtual void add(const IndexVector& idx_vec, span<T> buff_span) {
  //       EXPECTS(allocation_status_ != AllocationStatus::invalid);

  //       if(!is_non_zero(idx_vec)) { return; }

  //       Proc proc;
  //       Offset offset;
  //       std::tie(proc, offset) = distribution_->locate(idx_vec);
  //       Size size              = block_size(idx_vec);
  //       EXPECTS(size <= buff_span.size());
  //       mpb_->mgr().add(*mpb_, proc, offset, Size{size}, buff_span.data());
  //   }

  //   /**
  //    * @brief Tensor accessor method for adding svalues in nonblocking fashion
  //    * to a set of indices with the specified memory span
  //    *
  //    * @tparam T type of the values hold on the tensor object
  //    * @param [in] idx_vec a vector of indices to put the values
  //    * @param [in] buff_span buff_span memory span for the values to put
  //    */

  //   virtual void nb_add(const IndexVector& idx_vec, span<T> buff_span,
  //                       DataCommunicationHandlePtr data_comm_handle) {
  //       EXPECTS(allocation_status_ != AllocationStatus::invalid);

  //       if(!is_non_zero(idx_vec)) { return; }

  //       Proc proc;
  //       Offset offset;
  //       std::tie(proc, offset) = distribution_->locate(idx_vec);
  //       Size size              = block_size(idx_vec);
  //       EXPECTS(size <= buff_span.size());
  //       mpb_->mgr().nb_add(*mpb_, proc, offset, Size{size}, buff_span.data(),
  //                          data_comm_handle);
  //   }

private:
  Tensor<T> tensor_opt_;

  TiledIndexSpaceVec construct_new_tis(const Tensor<T>& opt_tensor, size_t unit_tis_count) const {
    TiledIndexSpaceVec result_tis_list = opt_tensor.tiled_index_spaces();

    for(size_t i = 0; i < unit_tis_count; i++) {
      // get opt tiled index space
      TiledIndexSpace orig_tis = result_tis_list[i];

      // construct unit tiled index space
      TiledIndexSpace unit_tis{orig_tis.index_space()};

      // update resulting tis list
      result_tis_list[i] = unit_tis;
    }

    return result_tis_list;
  }

  bool is_unit_tiled(const TiledIndexSpace& tis) {
    return (tis.num_tiles() == tis.index_space().num_indices());
  }
}; // class TensorUnitTiled
} // namespace tamm
