#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/labeled_tensor.hpp"
#include "tamm/op_base.hpp"
#include "tamm/runtime_engine.hpp"
#include "tamm/tensor.hpp"
#include "tamm/types.hpp"
#include "tamm/utils.hpp"
#include "tamm/work.hpp"

namespace tamm {
/**
 * @ingroup operations
 * @brief Map operation. Invoke a function on each block of a tensor to set it.
 * @tparam LabeledTensorType
 * @tparam Func
 * @tparam N
 */
template<typename LabeledTensorT, typename Func, int N>
class MapOp: public Op {
public:
  using RHS = std::array<LabeledTensorT, N>;
  using T   = typename LabeledTensorT::element_type;

  MapOp(LabeledTensorT& lhs, Func func, RHS& rhs, ResultMode mode = ResultMode::set,
        bool do_translate = true, bool is_exact_copy = false):
    lhs_{lhs}, func_{func}, rhs_{rhs}, do_translate_{do_translate}, is_exact_copy_{is_exact_copy} {
    fillin_labels();
    validate();
  }

  OpList canonicalize() const override { return OpList{(*this)}; }

  OpType op_type() const override { return OpType::map; }

  std::shared_ptr<Op> clone() const override {
    return std::shared_ptr<Op>(new MapOp<LabeledTensorT, Func, N>{*this});
  }

  void execute(ExecutionContext& ec, ExecutionHW hw = ExecutionHW::CPU) override {
    using TensorElType = typename LabeledTensorT::element_type;

    IndexLabelVec merged_labels{lhs_.labels()};
    if(is_exact_copy_ && do_translate_) {
      merged_labels.insert(merged_labels.end(), lhs_.labels().begin(), lhs_.labels().end());
    }
    else {
      for(const auto& rlt: rhs_) {
        merged_labels.insert(merged_labels.end(), rlt.labels().begin(), rlt.labels().end());
      }
    }

    LabelLoopNest loop_nest{merged_labels};
    auto          lambda_no_translate = [&](const IndexVector& itval) {
      auto        ltensor = lhs_.tensor();
      IndexVector lblockid, rblockid[N];
      auto        it = itval.begin();
      lblockid.insert(lblockid.end(), it, it + lhs_.labels().size());
      it += lhs_.labels().size();
      for(size_t i = 0; i < N; i++) {
        rblockid[i].insert(rblockid[i].end(), it, it + rhs_[i].labels().size());
        it += rhs_[i].labels().size();
      }

      const size_t              lsize = ltensor.block_size(lblockid);
      std::vector<TensorElType> lbuf(lsize);
      std::vector<TensorElType> rbuf[N];
      for(size_t i = 0; i < N; i++) {
        const auto&  rtensor_i = rhs_[i].tensor();
        const size_t isz       = rtensor_i.block_size(rblockid[i]);
        rbuf[i].resize(isz);
        rtensor_i.get(rblockid[i], rbuf[i]);
      }
      func_(ltensor, lblockid, lbuf, rblockid, rbuf);
      ltensor.put(lblockid, lbuf);
    };

    auto lambda = [&](const IndexVector& itval) {
      auto        ltensor = lhs_.tensor();
      IndexVector lblockid, rblockid[N];
      auto        it = itval.begin();
      lblockid.insert(lblockid.end(), it, it + lhs_.labels().size());
      it += lhs_.labels().size();
      for(size_t i = 0; i < N; i++) {
        rblockid[i].insert(rblockid[i].end(), it, it + rhs_[i].labels().size());
        it += rhs_[i].labels().size();
        // Translate each rhs blockid
        rblockid[i] = internal::translate_blockid(rblockid[i], rhs_[i]);
      }
      // Translate lhs blockid
      lblockid = internal::translate_blockid(lblockid, lhs_);

      const size_t              lsize = ltensor.block_size(lblockid);
      std::vector<TensorElType> lbuf(lsize);
      std::vector<TensorElType> rbuf[N];
      for(size_t i = 0; i < N; i++) {
        const auto&  rtensor_i = rhs_[i].tensor();
        const size_t isz       = rtensor_i.block_size(rblockid[i]);
        rbuf[i].resize(isz);
        rtensor_i.get(rblockid[i], rbuf[i]);
      }
      func_(ltensor, lblockid, lbuf, rblockid, rbuf);
      ltensor.put(lblockid, lbuf);
    };
    //@todo use a scheduler
    if(do_translate_) do_work(ec, loop_nest, lambda);
    else do_work(ec, loop_nest, lambda_no_translate);
  }

  TensorBase* writes() const { return lhs_.base_ptr(); }

  TensorBase* accumulates() const { return nullptr; }

  std::vector<TensorBase*> reads() const {
    std::vector<TensorBase*> res;
    for(const auto& lt: rhs_) { res.push_back(lt.base_ptr()); }
    return res;
  }

  bool is_memory_barrier() const { return false; }

protected:
  // Check if two list of blockids have overlap
  bool have_overlap(const std::vector<IndexVector>& vec1, const std::vector<IndexVector>& vec2) {
    // Create an unordered_set with custom hash and equality
    std::unordered_set<IndexVector, IndexVectorHash, IndexVectorEqual> set(vec1.begin(),
                                                                           vec1.end());
    for(const auto& value: vec2) {
      if(set.count(value)) { // Check if value exists in the set
        return true;
      }
    }
    return false;
  }

  std::vector<IndexVector> construct_blockIds(const LabeledTensorT& lt) {
    std::vector<IndexVector> result;

    auto          lt_lbls = lt.labels();
    LabelLoopNest loop_nest{lt_lbls};

    for(const auto& blockid: loop_nest) {
      result.push_back(internal::translate_blockid(blockid, lt));
    }

    return result;
  }

  bool have_overlapping_blocks(const LabeledTensorT& lhs, const LabeledTensorT& rhs) {
    EXPECTS(lhs.tensor().base_ptr() == rhs.tensor().base_ptr());

    return have_overlap(construct_blockIds(lhs), construct_blockIds(rhs));
  }

  void fillin_labels() {
    using internal::fillin_tensor_label_from_map;
    using internal::update_fillin_map;

    std::map<std::string, Label> str_to_labels;
    update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
    size_t off = lhs_.str_labels().size();
    update_fillin_map(str_to_labels, lhs_.str_map(), lhs_.str_labels(), 0);
    for(size_t i = 0; i < N; i++) {
      update_fillin_map(str_to_labels, rhs_[i].str_map(), rhs_[i].str_labels(), off);
      off += rhs_[i].str_labels().size();
    }
    fillin_tensor_label_from_map(lhs_, str_to_labels);
    for(size_t i = 0; i < N; i++) { fillin_tensor_label_from_map(rhs_[i], str_to_labels); }
  }

  void validate() {
    for(auto& rhs: rhs_) {
      if(lhs_.tensor().base_ptr() == rhs.tensor().base_ptr()) {
        if(have_overlapping_blocks(lhs_, rhs)) {
          std::ostringstream os;
          os << "[TAMM ERROR] Self assignment is not supported in tensor operations!\n"
             << __FILE__ << ":L" << __LINE__;
          tamm_terminate(os.str());
        }
      }
    }

    IndexLabelVec ilv{lhs_.labels()};
    for(size_t i = 0; i < N; i++) {
      ilv.insert(ilv.end(), rhs_[i].labels().begin(), rhs_[i].labels().end());
    }

    for(size_t i = 0; i < ilv.size(); i++) {
      for(const auto& dl: ilv[i].secondary_labels()) {
        size_t j;
        for(j = 0; j < ilv.size(); j++) {
          if(dl.tiled_index_space() == ilv[j].tiled_index_space() && dl.label() == ilv[j].label()) {
            break;
          }
        }
        EXPECTS(j < ilv.size());
      }
    }

    for(size_t i = 0; i < ilv.size(); i++) {
      const auto& ilbl = ilv[i];
      for(size_t j = i + 1; j < ilv.size(); j++) {
        const auto& jlbl = ilv[j];
        if(ilbl.tiled_index_space() == jlbl.tiled_index_space() && ilbl.label() == jlbl.label() &&
           ilbl.label_str() == jlbl.label_str()) {
          EXPECTS(ilbl == jlbl);
        }
      }
    }
  }

  LabeledTensorT                lhs_;
  Func                          func_;
  std::array<LabeledTensorT, N> rhs_;
  bool                          do_translate_;
  bool                          is_exact_copy_;

public:
  std::string opstr_;
};
} // namespace tamm
