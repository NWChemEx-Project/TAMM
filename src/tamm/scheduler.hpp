#pragma once

#include <set>

#include "ga/ga-mpi.h"
#include "tamm/dag_impl.hpp"
#include "tamm/execution_context.hpp"
#include "tamm/ip_hptt.hpp"
#include "tamm/ops.hpp"
#include "tamm/tensor.hpp"

namespace tamm {

using internal::DAGImpl;

/**
 * @brief Scheduler to execute a list of operations.
 * @ingroup operations
 */
class Scheduler {
public:
  /**
   * @brief Allocation status of tensor.
   *
   * This is used to perform some correctness checks on the list of
   * operations.
   * @todo Can this be replaced by AllocationStatus
   */
  // enum class TensorStatus { invalid, allocated, deallocated, initialized };

  // @to-do: what is the default scheduler?
  Scheduler()                            = default;
  Scheduler(const Scheduler&)            = default;
  Scheduler(Scheduler&&)                 = default;
  Scheduler& operator=(const Scheduler&) = default;
  Scheduler& operator=(Scheduler&&)      = default;

  Scheduler(ExecutionContext& ec): ec_{ec} {}

  template<typename OpType>
  Scheduler& operator()(const OpType& op, std::string opstr = "",
                        ExecutionHW exhw = ExecutionHW::DEFAULT) {
    OpList t_ops = op.canonicalize();

    for(auto& op: t_ops) {
      op->opstr_ = opstr;
      op->exhw_  = exhw;
      ops_.push_back(op);
    }
    return (*this);
  }

  Scheduler& allocate() { return *this; }

  ExecutionContext& ec() { return ec_; }

  template<typename TensorType, typename... Args>
  Scheduler& allocate(TensorType tensor, Args&... tensors) {
    ops_.push_back(std::make_shared<AllocOp<TensorType>>(tensor, ec()));
    return allocate(tensors...);
  }

  Scheduler& deallocate() { return *this; }

  template<typename TensorType, typename... Args>
  Scheduler& deallocate(TensorType tensor, Args&... tensors) {
    ops_.push_back(std::make_shared<DeallocOp<TensorType>>(tensor));
    return deallocate(tensors...);
  }

  template<typename T>
  bool has_intersect(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    for(const auto& l_item: lhs) {
      for(const auto& r_item: rhs) {
        if(l_item == r_item) return true;
      }
    }
    return false;
  }
  bool has_dependence(const std::vector<TensorBase*>& R1, const std::vector<TensorBase*>& W1,
                      const std::vector<TensorBase*>& R2, const std::vector<TensorBase*>& W2) {
    return (has_intersect(R1, W2) || has_intersect(W1, R2) || has_intersect(W1, W2));
  }

  bool has_dependence(const std::vector<TensorBase*>& R1, const std::vector<TensorBase*>& W1,
                      const std::vector<TensorBase*>& A1, const std::vector<TensorBase*>& R2,
                      const std::vector<TensorBase*>& W2, const std::vector<TensorBase*>& A2) {
    return (has_intersect(R1, W2) || has_intersect(W1, R2) || has_intersect(W1, W2) ||
            has_intersect(R1, A2) || has_intersect(W1, A2) || has_intersect(R2, A1) ||
            has_intersect(W2, A1));
  }

  std::vector<size_t> levelize(const std::vector<std::shared_ptr<Op>>& ops, size_t start_id,
                               size_t end_id) {
    EXPECTS(start_id >= 0 && start_id <= ops.size());
    EXPECTS(end_id >= start_id && end_id <= ops.size());

    std::vector<size_t> groups;

    size_t                   group_start = start_id;
    std::vector<TensorBase*> group_reads, group_writes, group_accums;
    for(size_t i = start_id; i < end_id; i++) {
      std::vector<TensorBase*> reads, writes, accums;
      reads = std::vector<TensorBase*>(ops[i]->reads());
      if(auto wr = ops[i]->writes(); wr != nullptr) { writes = std::vector<TensorBase*>{wr}; }
      if(auto ac = ops[i]->accumulates(); ac != nullptr) { accums = std::vector<TensorBase*>{ac}; }

      if(ops[i]->is_memory_barrier() ||
         has_dependence(group_reads, group_writes, group_accums, reads, writes, accums)) {
        groups.push_back(i - group_start);
        group_start  = i;
        group_reads  = reads;
        group_writes = writes;
        group_accums = accums;
      }
      else {
        group_reads.insert(group_reads.end(), reads.begin(), reads.end());
        group_writes.insert(group_writes.end(), writes.begin(), writes.end());
        group_accums.insert(group_accums.end(), accums.begin(), accums.end());
      }
    }

    if(group_start < end_id) { groups.push_back(end_id - group_start); }

    return groups;
  }

  bool op_has_dependence(const Op* op1, const Op* op2) {
    std::vector<TensorBase*> R1, W1, A1;
    R1 = op1->reads();
    if(auto wr = op1->writes(); wr != nullptr) { W1 = std::vector<TensorBase*>{wr}; }
    if(auto ac = op1->accumulates(); ac != nullptr) { A1 = std::vector<TensorBase*>{ac}; }
    std::vector<TensorBase*> R2, W2, A2;
    R2 = op2->reads();
    if(auto wr = op2->writes(); wr != nullptr) { W2 = std::vector<TensorBase*>{wr}; }
    if(auto ac = op2->accumulates(); ac != nullptr) { A2 = std::vector<TensorBase*>{ac}; }
    return has_dependence(R1, W1, A1, R2, W2, A2);
  }

  std::vector<std::pair<size_t, size_t>>
  levelize_and_order(const std::vector<std::shared_ptr<Op>>& ops, size_t start_id, size_t end_id) {
    EXPECTS(start_id >= 0 && start_id <= ops.size());
    EXPECTS(end_id >= start_id && end_id <= ops.size());

    std::vector<std::pair<size_t, size_t>> order;

    for(size_t i = start_id; i < end_id; i++) { order.push_back(std::make_pair(0, i)); }
    for(size_t i = 0; i < end_id - start_id - 1; i++) {
      for(size_t j = i + 1; j < end_id - start_id; j++) {
        if(op_has_dependence(ops[start_id + i].get(), ops[start_id + j].get())) {
          order[j].first = std::max(order[i].first + 1, order[j].first);
        }
      }
    }
    std::sort(order.begin(), order.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    return order;
  }

  void execute(ExecutionHW execute_on = ExecutionHW::CPU, bool profile = false) {
    if(start_idx_ == ops_.size()) return;
    auto& oprof = tamm::OpProfiler::instance();
#if 0
        auto order = levelize_and_order(ops_, start_idx_, ops_.size());
        EXPECTS(order.size() == ops_.size() - start_idx_);
        size_t lvl           = 0;
        for(size_t i = 0; i < order.size(); i++) {
            if(order[i].first != lvl) {
                EXPECTS(order[i].first == lvl + 1);
                auto bt1 = std::chrono::high_resolution_clock::now();
                ec().pg().barrier();
                auto bt2 = std::chrono::high_resolution_clock::now();
                oprof.tbarrierTime += std::chrono::duration_cast<std::chrono::duration<double>>((bt2 - bt1)).count(); 
                lvl += 1;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            ops_[order[i].second]->execute(ec(), execute_on);
            auto t2 = std::chrono::high_resolution_clock::now();
            double mop_time = 
                std::chrono::duration_cast<std::chrono::duration<double>>((t2 - t1)).count();  
            if(ops_[order[i].second]->op_type() == OpType::mult) oprof.multOpTime += mop_time;
            if(ops_[order[i].second]->op_type() == OpType::add) oprof.addOpTime += mop_time;       
            if(ops_[order[i].second]->op_type() == OpType::set) oprof.setOpTime += mop_time;       
            if(ops_[order[i].second]->op_type() == OpType::alloc) oprof.allocOpTime += mop_time;       
            if(ops_[order[i].second]->op_type() == OpType::dealloc) oprof.deallocOpTime += mop_time;  
            oprof.taddTime += oprof.multOpAddTime;
            oprof.tgetTime += oprof.multOpGetTime;
            oprof.twaitTime += oprof.multOpWaitTime; 
            oprof.tBCTime += oprof.multOpBCTime;
            oprof.tcopyTime += oprof.multOpCopyTime;
            oprof.multOpGetTime = 0;
            oprof.multOpWaitTime = 0;  
            oprof.multOpBCTime = 0;
            oprof.multOpAddTime = 0;
            oprof.multOpCopyTime = 0;
        }

        auto bt1 = std::chrono::high_resolution_clock::now();
        ec().pg().barrier();
        auto bt2 = std::chrono::high_resolution_clock::now();
        oprof.tbarrierTime += std::chrono::duration_cast<std::chrono::duration<double>>((bt2 - bt1)).count(); 
        start_idx_ = ops_.size();
#elif 1
    auto misc_start = std::chrono::high_resolution_clock::now();
    auto order      = levelize_and_order(ops_, start_idx_, ops_.size());
    EXPECTS(order.size() == ops_.size() - start_idx_);
    size_t         lvl = 0;
    AtomicCounter* ac  = new AtomicCounterGA(ec().pg(), order.size());
    ac->allocate(0);
    auto   misc_end = std::chrono::high_resolution_clock::now();
    double misc_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((misc_end - misc_start)).count();
    // auto t1 = misc_end;

    // double nranks = 1.0 * ec_.pg().size().value();
    oprof.multOpGetTime  = 0;
    oprof.multOpBCTime   = 0;
    oprof.multOpAddTime  = 0;
    oprof.multOpCopyTime = 0;

    std::vector<double> load_imbalance_times;
    std::vector<double> op_times;
    std::vector<double> multop_get_times;
    std::vector<double> multop_dgemm_times;
    std::vector<double> multop_add_times;
    std::vector<double> multop_copy_times;
    const int           nops = order.size();

    assert(order.size() == 0 || order[0].first == 0); // level 0 sanity check
    for(int i = 0; i < nops; i++) {
      if(order[i].first != lvl) {
        assert(order[i].first == lvl + 1);
        // auto t2 = std::chrono::high_resolution_clock::now();
        ec().pg().barrier();
        lvl += 1;
        // auto t3 = std::chrono::high_resolution_clock::now();
        // load_imbalance_times.push_back(std::chrono::duration_cast<std::chrono::duration<double>>((t3
        // - t2)).count());
        // level_times.push_back(std::chrono::duration_cast<std::chrono::duration<double>>((t3 -
        // t1)).count()); multop_get_times.push_back(oprof.multOpGetTime);
        // multop_dgemm_times.push_back(oprof.multOpBCTime);
        // multop_add_times.push_back(oprof.multOpAddTime);
        // oprof.multOpGetTime = 0;
        // oprof.multOpBCTime = 0;
        // oprof.multOpAddTime = 0;
        // t1 = t3;
      }
      ec().set_ac(IndexedAC(ac, i));
      if(ops_[order[i].second]->exhw_ != ExecutionHW::DEFAULT)
        execute_on = ops_[order[i].second]->exhw_;
      auto t2 = std::chrono::high_resolution_clock::now();
      ops_[order[i].second]->execute(ec(), execute_on);
      auto t3 = std::chrono::high_resolution_clock::now();
      op_times.push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>((t3 - t2)).count());
      multop_get_times.push_back(oprof.multOpGetTime);
      multop_dgemm_times.push_back(oprof.multOpBCTime);
      multop_add_times.push_back(oprof.multOpAddTime);
      multop_copy_times.push_back(oprof.multOpCopyTime);
      oprof.multOpGetTime  = 0;
      oprof.multOpBCTime   = 0;
      oprof.multOpAddTime  = 0;
      oprof.multOpCopyTime = 0;
    }
    // auto t2 = std::chrono::high_resolution_clock::now();
    ec().pg().barrier();
    lvl += 1;
    auto t3 = std::chrono::high_resolution_clock::now();
    // load_imbalance_times.push_back(std::chrono::duration_cast<std::chrono::duration<double>>((t3
    // - t2)).count());
    // level_times.push_back(std::chrono::duration_cast<std::chrono::duration<double>>((t3 -
    // t1)).count()); multop_get_times.push_back(oprof.multOpGetTime);
    // multop_dgemm_times.push_back(oprof.multOpBCTime);
    // multop_add_times.push_back(oprof.multOpAddTime);
    // oprof.multOpGetTime = 0;
    // oprof.multOpBCTime = 0;
    // oprof.multOpAddTime = 0;
    start_idx_ = ops_.size();
    ec().set_ac(IndexedAC(nullptr, 0));
    misc_start = t3;
    ac->deallocate();
    delete ac;
    misc_end = std::chrono::high_resolution_clock::now();
    misc_time +=
      std::chrono::duration_cast<std::chrono::duration<double>>((misc_end - misc_start)).count();

    if(profile) {
      assert(op_times.size() == order.size()); // all vectors are of the same size

      // std::vector<double> global_load_imbalance_times_min(nops);
      std::vector<double> global_op_times_min(nops);
      std::vector<double> global_multop_get_times_min(nops);
      std::vector<double> global_multop_dgemm_times_min(nops);
      std::vector<double> global_multop_add_times_min(nops);
      std::vector<double> global_multop_copy_times_min(nops);

      // std::vector<double> global_load_imbalance_times_max(nops);
      std::vector<double> global_op_times_max(nops);
      std::vector<double> global_multop_get_times_max(nops);
      std::vector<double> global_multop_dgemm_times_max(nops);
      std::vector<double> global_multop_add_times_max(nops);
      std::vector<double> global_multop_copy_times_max(nops);

      // std::vector<double> global_load_imbalance_times_sum(nops);
      std::vector<double> global_op_times_sum(nops);
      std::vector<double> global_multop_get_times_sum(nops);
      std::vector<double> global_multop_dgemm_times_sum(nops);
      std::vector<double> global_multop_add_times_sum(nops);
      std::vector<double> global_multop_copy_times_sum(nops);

      // ec_.pg().reduce(load_imbalance_times.data(), global_load_imbalance_times_min.data(), lvl,
      // ReduceOp::min, 0);
      ec_.pg().reduce(op_times.data(), global_op_times_min.data(), nops, ReduceOp::min, 0);
      ec_.pg().reduce(multop_get_times.data(), global_multop_get_times_min.data(), nops,
                      ReduceOp::min, 0);
      ec_.pg().reduce(multop_dgemm_times.data(), global_multop_dgemm_times_min.data(), nops,
                      ReduceOp::min, 0);
      ec_.pg().reduce(multop_add_times.data(), global_multop_add_times_min.data(), nops,
                      ReduceOp::min, 0);
      ec_.pg().reduce(multop_copy_times.data(), global_multop_copy_times_min.data(), nops,
                      ReduceOp::min, 0);

      // ec_.pg().reduce(load_imbalance_times.data(), global_load_imbalance_times_max.data(), lvl,
      // ReduceOp::max, 0);
      ec_.pg().reduce(op_times.data(), global_op_times_max.data(), nops, ReduceOp::max, 0);
      ec_.pg().reduce(multop_get_times.data(), global_multop_get_times_max.data(), nops,
                      ReduceOp::max, 0);
      ec_.pg().reduce(multop_dgemm_times.data(), global_multop_dgemm_times_max.data(), nops,
                      ReduceOp::max, 0);
      ec_.pg().reduce(multop_add_times.data(), global_multop_add_times_max.data(), nops,
                      ReduceOp::max, 0);
      ec_.pg().reduce(multop_copy_times.data(), global_multop_copy_times_max.data(), nops,
                      ReduceOp::max, 0);

      // ec_.pg().reduce(load_imbalance_times.data(), global_load_imbalance_times_sum.data(), lvl,
      // ReduceOp::sum, 0);
      ec_.pg().reduce(op_times.data(), global_op_times_sum.data(), nops, ReduceOp::sum, 0);
      ec_.pg().reduce(multop_get_times.data(), global_multop_get_times_sum.data(), nops,
                      ReduceOp::sum, 0);
      ec_.pg().reduce(multop_dgemm_times.data(), global_multop_dgemm_times_sum.data(), nops,
                      ReduceOp::sum, 0);
      ec_.pg().reduce(multop_add_times.data(), global_multop_add_times_sum.data(), nops,
                      ReduceOp::sum, 0);
      ec_.pg().reduce(multop_copy_times.data(), global_multop_copy_times_sum.data(), nops,
                      ReduceOp::sum, 0);

      int   np    = ec_.pg().size().value();
      auto& pdata = ec_.get_profile_data();
      if(ec_.pg().rank() == 0) {
        for(int i = 0; i < nops; i++) {
          pdata << i << ";" << order[i].first << ";"
                << ops_[order[i].second]->opstr_
                // << "," << global_load_imbalance_times_min[i]
                // << "," << global_load_imbalance_times_max[i]
                // << "," << global_load_imbalance_times_sum[i]/np
                << ";" << global_op_times_min[i] << ";" << global_op_times_max[i] << ";"
                << global_op_times_sum[i] / np << ";" << global_multop_get_times_min[i] << ";"
                << global_multop_get_times_max[i] << ";" << global_multop_get_times_sum[i] / np
                << ";" << global_multop_dgemm_times_min[i] << ";"
                << global_multop_dgemm_times_max[i] << ";" << global_multop_dgemm_times_sum[i] / np
                << ";" << global_multop_copy_times_min[i] << ";" << global_multop_copy_times_max[i]
                << ";" << global_multop_copy_times_sum[i] / np << ";"
                << global_multop_add_times_min[i] << ";" << global_multop_add_times_max[i] << ";"
                << global_multop_add_times_sum[i] / np << std::endl;
        }
        pdata << ";"
              << "SUM"
              << ";;;;"
              << (std::accumulate(global_op_times_sum.begin(), global_op_times_sum.end(),
                                  decltype(global_op_times_sum)::value_type(0))) /
                   np
              << ";;;"
              << (std::accumulate(global_multop_get_times_sum.begin(),
                                  global_multop_get_times_sum.end(),
                                  decltype(global_multop_get_times_sum)::value_type(0))) /
                   np
              << ";;;"
              << (std::accumulate(global_multop_dgemm_times_sum.begin(),
                                  global_multop_dgemm_times_sum.end(),
                                  decltype(global_multop_dgemm_times_sum)::value_type(0))) /
                   np
              << ";;;"
              << (std::accumulate(global_multop_copy_times_sum.begin(),
                                  global_multop_copy_times_sum.end(),
                                  decltype(global_multop_copy_times_sum)::value_type(0))) /
                   np
              << ";;;"
              << (std::accumulate(global_multop_add_times_sum.begin(),
                                  global_multop_add_times_sum.end(),
                                  decltype(global_multop_add_times_sum)::value_type(0))) /
                   np
              << std::endl;
        // << "," << globalgettime/nranks << ","
        // << globaladdtime/nranks << "," << globalgemmtime/nranks << std::endl;
      }
    }

#else
    auto groups = levelize(ops_, start_idx_, ops_.size());
    // std::cerr << "Groups: [ ";
    // for(const auto& sz : groups) {
    //     std::cerr << sz << " ";
    // }
    // std::cerr << "]" << std::endl;

    // AtomicCounter* ac = new AtomicCounterGA(ec().pg(), ops_.size() -
    // start_idx_); ac->allocate(0);

    size_t off = start_idx_;
    for(size_t g: groups) {
      EXPECTS(g > 0);
      AtomicCounter* ac = new AtomicCounterGA(ec().pg(), g);
      ac->allocate(0);

      for(size_t i = off; i < off + g; i++, start_idx_++) {
        ec().set_ac(IndexedAC(ac, i - off));
        ops_[i]->execute(ec());
      }

      ec().set_ac(IndexedAC(nullptr, 0));
      ac->deallocate();
      delete ac;

      // memory fence. for now GA_Sync()
      // GA_Sync();
      // pg.barrier()
      ec().pg().barrier();
      off += g;
    }
    // ac->deallocate();
    // delete ac;
    // // for(auto& op : ops_) { op->execute(ec()->pg()); }
    // for(size_t i = start_idx_; i < ops_.size(); i++) {
    //     ops_[i]->execute(ec());
    //     start_idx_++;
    // }
#endif
  }

  template<typename Func, typename... Args>
  static void execute(DAGImpl<Func, Args...> dag) {}

  ~Scheduler() {
    // delete ops
  }

  template<typename LabeledTensorType, typename Func>
  Scheduler& gop(LabeledTensorType lhs, Func func) {
    ops_.push_back(std::make_shared<ScanOp<LabeledTensorType, Func>>(lhs, func));
    return *this;
  }

  template<typename LabeledTensorType, typename Func, size_t N>
  Scheduler& gop(LabeledTensorType lhs, std::array<LabeledTensorType, N> rhs, Func func,
                 ResultMode mode = ResultMode::set, bool do_translate = true) {
    ops_.push_back(
      std::make_shared<MapOp<LabeledTensorType, Func, N>>(lhs, func, rhs, mode, do_translate));
    return *this;
  }

  template<typename T>
  Scheduler& exact_copy(LabeledTensor<T> lhs, LabeledTensor<T> rhs, bool do_translate = false) {
    auto copy_buf = [](const Tensor<T>& t, const IndexVector& lhs_iv, std::vector<T>& lhs_buf,
                       const IndexVector rhs_iv[], std::vector<T> rhs_buf[]) {
      std::copy(rhs_buf[0].begin(), rhs_buf[0].end(), lhs_buf.begin());
    };

    auto rhs_arr = std::array<LabeledTensor<T>, 1>{rhs};

    ops_.push_back(std::make_shared<MapOp<LabeledTensor<T>, decltype(copy_buf), 1>>(
      lhs, copy_buf, rhs_arr, ResultMode::set, do_translate, true));

    return *this;
  }

  // template<typename TensorType>
  // inline void exact_copy(Tensor<TensorType>& dst, const Tensor<TensorType>& src,
  //                        bool is_assign = false, TensorType scale = TensorType{1},
  //                        const IndexVector& perm = {}) {

  //     // auto lambda = [&](const IndexVector& itval) {
  //     //     IndexVector src_id = itval;
  //     //     for(size_t i = 0; i < perm.size(); i++) { src_id[i] = itval[perm[i]]; }
  //     //     size_t size = dst.block_size(itval);
  //     //     std::vector<TensorType> buf(size);
  //     //     src.get(src_id, buf);
  //     //     TensorType {1};
  //     //     if(scale != TensorType{1}) {
  //     //         for(size_t i = 0; i < size; i++) {
  //     //             buf[i] *= scale;
  //     //         }
  //     //     }
  //     //     if(is_assign)
  //     //         dst.put(itval, buf);
  //     //     else
  //     //         dst.add(itval, buf);
  //     // };

  //     PermVector perm_to_dest{perm.begin(), perm.end()};
  //     TensorType lscale = is_assign ? TensorType{0} : TensorType{1};

  //     auto lambda_hptt = [&](const IndexVector& itval) {
  //         IndexVector src_id = itval;
  //         size_t dst_size = dst.block_size(itval);
  //         std::vector<TensorType> src_buf(dst_size);
  //         src.get(src_id, src_buf);
  //         std::vector<TensorType> dest_buf(dst_size);

  //         blockops::hptt::index_permute_hptt(lscale, dest_buf.data(), scale,
  //                                            src_buf.data(), perm_to_dest,
  //                                            src.block_dims(src_id));

  //         if(is_assign)
  //             dst.put(itval, dest_buf);
  //         else
  //             dst.add(itval, dest_buf);
  //     };

  //     auto ec = dst.execution_context();
  //     block_for(*ec, dst(), lambda_hptt);
  // }

private:
  ExecutionContext& ec_;
  // void validate() {
  //     // 1. every tensor used by operarions should be listed in tensors_

  //     // 2. every tensor must be initialized (part of LHS) or be an
  //     // input (ilve_in) tensor before it is used

  //     // 3. every output tensor should be allocated and set (be LHS in
  //     // at least one operation or be a live_in tensor)

  //     // 4. every tensor must be allocated before it is used

  //     // 5. every non-output (not in live_out) tensor must be
  //     // deallocated
  // }
  std::vector<std::shared_ptr<Op>> ops_;
  size_t                           start_idx_ = 0;

}; // class Scheduler

} // namespace tamm
