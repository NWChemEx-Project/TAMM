#include "tamm/tensor_base.hpp"
#include "tamm/op_dag.hpp"

namespace tamm {
TensorUpdate::TensorUpdate(const IndexLabelVec& ilv, const std::unique_ptr<new_ops::Op>& op,
                           bool is_update) {
  ilv_ = ilv;
  op_  = op->clone();
  op_->set_attribute<new_ops::NeededLabelsAttribute>(ilv_);
  is_update_ = is_update;
}

TensorUpdate::TensorUpdate(const TensorUpdate& other) {
  ilv_       = other.ilv_;
  op_        = other.op_->clone();
  is_update_ = other.is_update_;
}

TensorUpdate& TensorUpdate::operator=(const TensorUpdate& other) {
  ilv_       = other.ilv_;
  op_        = other.op_->clone();
  is_update_ = other.is_update_;
  return *this;
}

/**
 * @brief Construct a new TensorBase object using a vector of
 * TiledIndexSpace objects for each mode of the tensor
 *
 * @param [in] block_indices vector of TiledIndexSpace objects for each mode
 * used to construct the tensor
 */
TensorBase::TensorBase(const std::vector<TiledIndexSpace>& block_indices) {
  block_indices_     = block_indices;
  allocation_status_ = AllocationStatus::invalid;
  num_modes_         = block_indices.size();
  updates_           = {};
  for(const auto& tis: block_indices_) { EXPECTS(!tis.is_dependent()); }
  fillin_tlabels();
  construct_dep_map();
}

/**
 * @brief
 */
TensorBase::TensorBase(const std::vector<TiledIndexSpace>& block_indices,
                       const NonZeroCheck&                 zero_check) {
  block_indices_        = block_indices;
  allocation_status_    = AllocationStatus::invalid;
  num_modes_            = block_indices.size();
  updates_              = {};
  is_non_zero_func_     = zero_check;
  has_user_is_non_zero_ = true;
  for(const auto& tis: block_indices_) { EXPECTS(!tis.is_dependent()); }
  fillin_tlabels();
  construct_dep_map();
}

/**
 * @brief Construct a new TensorBase object using a vector of
 * TiledIndexSpace objects for each mode of the tensor
 *
 * @param [in] lbls vector of tiled index labels used for extracting
 * corresponding TiledIndexSpace objects for each mode used to construct the
 * tensor
 */
TensorBase::TensorBase(const std::vector<TiledIndexLabel>& lbls) {
  allocation_status_ = AllocationStatus::invalid;
  num_modes_         = lbls.size();
  updates_           = {};
#if 0
        for(const auto& lbl : lbls) {
            auto tis = lbl.tiled_index_space();
            if(tis.is_dependent()){
                if(lbl.secondary_labels().size() == 0){
                    auto new_tis = tis.parent_tis();
                    // negative lbl id is used to avoid overlap
                    auto new_lbl = new_tis.label(-1);
                    block_indices_.push_back(new_tis);
                    tlabels_.push_back(new_lbl);
                }
                else {
                    EXPECTS(lbl.secondary_labels().size() ==
                        tis.num_key_tiled_index_spaces());
                    block_indices_.push_back(tis);
                    tlabels_.push_back(lbl);
                }
            } else {
                block_indices_.push_back(tis);
                tlabels_.push_back(lbl);
            }
        }
#else
  tlabels_ = lbls;
  for(auto& lbl: tlabels_) { block_indices_.push_back(lbl.tiled_index_space()); }
#endif
  construct_dep_map();
  update_labels();
}

/**
 * @brief Construct a new TensorBase object recursively with a set of
 * TiledIndexSpace/TiledIndexLabel objects followed by a lambda expression
 *
 * @tparam Ts variadic template for rest of the arguments
 * @param [in] tis TiledIndexSpace object used as a mode
 * @param [in] rest remaining part of the arguments
 */
template<class... Ts>
TensorBase::TensorBase(const TiledIndexSpace& tis, Ts... rest) {
  *this = TensorBase{rest...};
  EXPECTS(!tis.is_dependent());
  block_indices_.insert(block_indices_.begin(), tis);
  fillin_tlabels();
  construct_dep_map();
  // tlabels_.insert(tlabels_.begin(), block_indices_[0].label(-1 -
  // block_indices_.size()));
}

// Dtor
TensorBase::~TensorBase(){
  // EXPECTS(allocation_status_ == AllocationStatus::invalid);
};

void TensorBase::add_update(const TensorUpdate& new_update) { updates_.push_back(new_update); }

std::vector<TensorUpdate> TensorBase::get_updates() const { return updates_; }

void TensorBase::clear_updates() {
  updates_.clear();
  version_ = 0;
}

} // namespace tamm
