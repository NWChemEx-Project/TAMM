#pragma once

#include "tamm/iteration.hpp"
#include "tamm/perm.hpp"
#include "tamm/tiled_index_space.hpp"
#include <chrono>
#include <map>
#include <vector>

namespace tamm {

class TimerGuard {
public:
  TimerGuard(double* refptr): refptr_{refptr} {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  ~TimerGuard() {
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time =
      std::chrono::high_resolution_clock::now();
    *refptr_ +=
      std::chrono::duration_cast<std::chrono::duration<double>>((end_time - start_time_)).count();
  }

private:
  double*                                                     refptr_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
}; // TimerGuard

namespace internal {

template<typename>
struct is_tuple: std::false_type {};
template<typename... T>
struct is_tuple<std::tuple<T...>>: std::true_type {};
template<typename T>
inline constexpr bool is_tuple_v = is_tuple<T>::value;

template<typename>
struct is_complex: std::false_type {};
template<typename T>
struct is_complex<std::complex<T>>: std::true_type {};
template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

inline void update_fillin_map(std::map<std::string, Label>&   str_to_labels,
                              const std::vector<bool>&        str_map,
                              const std::vector<std::string>& str_labels, int initial_off) {
  const size_t sz = str_labels.size();
  for(size_t i = 0; i < sz; i++) {
    if(str_map[i]) { str_to_labels[str_labels[i]] = -initial_off - i - 1; }
  }
}

template<typename LabelTensorT>
inline void fillin_tensor_label_from_map(LabelTensorT&                       ltensor,
                                         const std::map<std::string, Label>& str_to_labels) {
  IndexLabelVec new_labels = ltensor.labels();
  const size_t  sz         = ltensor.labels().size();
  for(size_t i = 0; i < sz; i++) {
    if(ltensor.str_map()[i]) {
      EXPECTS(str_to_labels.find(ltensor.str_labels()[i]) != str_to_labels.end());
      new_labels[i] = ltensor.tensor().tiled_index_spaces()[i].label(
        str_to_labels.find(ltensor.str_labels()[i])->second);
    }
  }
  ltensor.set_labels(new_labels);
}

// /**
//  * @ingroup perm
//  * @brief Compute permutation to be performed to permute vector @p from to
//  * vector @p to.
//  * @param from Source vector for the permutation
//  * @param to Target vector for the permutation
//  * @pre @p from and @p to are permutations of each other
//  * @pre from.size() == to.size()
//  * @return Vector to permute @p from to @p to.
//  * @post Return ret such that:
//  * ensures 0<=i<from.size(): to[i] = from[ret[i]]
//  */
// template<typename T>
// PermVector perm_compute(const std::vector<T>& from, const std::vector<T>& to) {
//     PermVector layout;

//     EXPECTS(from.size() == to.size());
//     for(auto p : to) {
//         auto itr = std::find(from.begin(), from.end(), p);
//         EXPECTS(itr != from.end());
//         layout.push_back(itr - from.begin());
//     }
//     return layout;
// }

// template<typename T>
// bool are_permutations(const std::vector<T>& vec1, const std::vector<T>& vec2) {
//     if(vec1.size() != vec2.size()) { return false; }
//     std::vector<bool> taken(vec1.size(), false);
//     for(size_t i = 0; i < vec1.size(); i++) {
//         auto it = std::find(vec2.begin(), vec2.end(), vec1[i]);
//         if(it == vec2.end()) { return false; }
//         if(taken[std::distance(vec2.begin(), it)] == true) { return false; }
//         taken[std::distance(vec2.begin(), it)] = true;
//     }
//     return true;
// }

template<typename TiledIndexLabel>
std::vector<TiledIndexLabel>
unique_entries_by_primary_label(const std::vector<TiledIndexLabel>& input_vec) {
  std::vector<TiledIndexLabel> ret;

  for(const auto& val: input_vec) {
    auto it = std::find_if(ret.begin(), ret.end(), [&](const auto& lbl) {
      return lbl.primary_label() == val.primary_label();
    });
    if(it == ret.end()) { ret.push_back(val); }
  }
  return ret;
}

// template<typename T>
// std::vector<size_t> perm_map_compute(const std::vector<T>& unique_vec,
//                                      const std::vector<T>& vec_required) {
//     std::vector<size_t> ret;
//     for(const auto& val : vec_required) {
//         auto it = std::find(unique_vec.begin(), unique_vec.end(), val);
//         EXPECTS(it >= unique_vec.begin());
//         EXPECTS(it != unique_vec.end());
//         ret.push_back(it - unique_vec.begin());
//     }
//     return ret;
// }

template<typename T>
std::vector<size_t> perm_map_compute_by_primary_label(const std::vector<T>& unique_vec,
                                                      const std::vector<T>& vec_required) {
  std::vector<size_t> ret;
  for(const auto& val: vec_required) {
    auto it = std::find_if(unique_vec.begin(), unique_vec.end(), [&](const auto& lbl) {
      return val.primary_label() == lbl.primary_label();
    });
    EXPECTS(it >= unique_vec.begin());
    EXPECTS(it != unique_vec.end());
    ret.push_back(it - unique_vec.begin());
  }
  return ret;
}

// template<typename T, typename Integer>
// std::vector<T> perm_map_apply(const std::vector<T>& input_vec,
//                               const std::vector<Integer>& perm_map) {
//     std::vector<T> ret;
//     for(const auto& pm : perm_map) {
//         EXPECTS(pm < input_vec.size());
//         ret.push_back(input_vec[pm]);
//     }
//     return ret;
// }

// template<typename T, typename Integer>
// void perm_map_apply(std::vector<T>& out_vec, const std::vector<T>& input_vec,
//                     const std::vector<Integer>& perm_map) {
//     out_vec.resize(perm_map.size());
//     for(size_t i = 0; i < perm_map.size(); i++) {
//         EXPECTS(perm_map[i] < input_vec.size());
//         out_vec[i] = input_vec[perm_map[i]];
//     }
// }

inline IndexLabelVec sort_on_dependence(const IndexLabelVec& labels) {
  std::vector<TileLabelElement> primary_labels;
  std::vector<size_t>           sorted_order;
  for(const auto& lbl: labels) { primary_labels.push_back(lbl.primary_label()); }
  for(size_t i = 0; i < labels.size(); i++) {
    const auto& lbl = labels[i];
    for(const auto& slbl: lbl.secondary_labels()) {
      const auto it = std::find(primary_labels.begin(), primary_labels.end(), slbl);
      EXPECTS(it != primary_labels.end());
      const auto sit =
        std::find(sorted_order.begin(), sorted_order.end(), it - primary_labels.begin());
      if(sit == sorted_order.end()) { sorted_order.push_back(it - primary_labels.begin()); }
    }
    const auto it = std::find(sorted_order.begin(), sorted_order.end(), i);
    if(it == sorted_order.end()) { sorted_order.push_back(i); }
  }
  IndexLabelVec ret;
  for(const auto& pos: sorted_order) { ret.push_back(labels[pos]); }
  return ret;
}

inline std::tuple<IndexLabelVec, IndexVector>
extract_blockid_and_label(const IndexLabelVec& input_labels, const IndexVector& input_blockid,
                          const IndexLabelVec& labels_to_match) {
  IndexLabelVec ret_labels;
  IndexVector   ret_blockid;
  for(size_t i = 0; i < labels_to_match.size(); i++) {
    auto lbl = labels_to_match[i];
    // for(const auto& lbl : labels_to_match) {
    auto it = std::find_if(input_labels.begin(), input_labels.end(), [&](const auto& itlbl) {
      return lbl.primary_label() == itlbl.primary_label();
    });
    EXPECTS(it != input_labels.end());
    size_t pos = it - input_labels.begin();
    EXPECTS(pos < input_labels.size() && 0 <= pos);
    // EXPECTS(pos < input_blockid.size() && 0 <= pos);

    ret_labels.push_back(input_labels[pos]);
    ret_blockid.push_back(input_blockid[i]);
  }
  return {ret_labels, ret_blockid};
}

// template<typename T>
// bool cartesian_iteration(std::vector<T>& itr, const std::vector<T>& end) {
//     EXPECTS(itr.size() == end.size());

//     int i;
//     for(i = -1 + itr.size(); i >= 0 && itr[i] + 1 == end[i]; i--) {
//         itr[i] = T{0};
//     }

//     if(i >= 0) {
//         ++itr[i];
//         return true;
//     }
//     return false;
// }

inline IndexVector indep_values(const IndexVector& blockid, const Index& idx,
                                const std::map<size_t, std::vector<size_t>>& dep_map) {
  IndexVector ret{};
  if(dep_map.find(idx) != dep_map.end()) {
    for(const auto& dep_id: dep_map.at(idx)) { ret.push_back(blockid[dep_id]); }
  }
  return ret;
}

template<typename LabeledTensorT>
IndexVector translate_sparse_blockid(const IndexVector& blockid, const LabeledTensorT& ltensor) {
  EXPECTS(blockid.size() == ltensor.labels().size());
  const auto& tensor  = ltensor.tensor();
  const auto& dep_map = tensor.dep_map();
  EXPECTS(blockid.size() == tensor.num_modes());
  IndexVector translate_blockid;
  for(size_t i = 0; i < blockid.size(); i++) {
    auto indep_vals = indep_values(blockid, i, dep_map);
    if(!indep_vals.empty()) {
      auto l_dep_map = ltensor.labels()[i].tiled_index_space().tiled_dep_map();
      auto t_dep_map = tensor.tiled_index_spaces()[i].tiled_dep_map();
      // check if any one of them doesn't have the TIS for indep_values
      if(l_dep_map.find(indep_vals) == l_dep_map.end() ||
         t_dep_map.find(indep_vals) == t_dep_map.end())
        return IndexVector(blockid.size(), -1);
    }
    const auto& label_tis  = ltensor.labels()[i].tiled_index_space()(indep_vals);
    const auto& tensor_tis = label_tis.root_tis();
    Index       val        = label_tis.translate(blockid[i], tensor_tis);
    translate_blockid.push_back(val);
  }
  return translate_blockid;
}

template<typename LabeledTensorT>
IndexVector translate_blockid(const IndexVector& blockid, const LabeledTensorT& ltensor) {
  EXPECTS(blockid.size() == ltensor.labels().size());
  const auto& tensor  = ltensor.tensor();
  const auto& dep_map = tensor.dep_map();
  EXPECTS(blockid.size() == tensor.num_modes());
  IndexVector translate_blockid;
  for(size_t i = 0; i < blockid.size(); i++) {
    auto indep_vals = indep_values(blockid, i, dep_map);
    if(!indep_vals.empty()) {
      auto l_dep_map = ltensor.labels()[i].tiled_index_space().tiled_dep_map();
      auto t_dep_map = tensor.tiled_index_spaces()[i].tiled_dep_map();
      // check if any one of them doesn't have the TIS for indep_values
      if(l_dep_map.find(indep_vals) == l_dep_map.end() ||
         t_dep_map.find(indep_vals) == t_dep_map.end())
        return IndexVector(blockid.size(), -1);
    }
    const auto& label_tis  = ltensor.labels()[i].tiled_index_space()(indep_vals);
    const auto& tensor_tis = tensor.tiled_index_spaces()[i](indep_vals);
    Index       val        = label_tis.translate(blockid[i], tensor_tis);
    translate_blockid.push_back(val);
  }
  return translate_blockid;
}

inline IndexVector translate_blockid_with_labels(const IndexVector&        from_blockid,
                                                 const IndexLabelVec&      from_labels,
                                                 const TiledIndexSpaceVec& to_tis) {
  EXPECTS(from_blockid.size() == from_labels.size());
  EXPECTS(from_labels.size() == to_tis.size());

  IndexVector translated_blockid;
  for(size_t i = 0; i < from_blockid.size(); i++) {
    const auto& from_tis = from_labels[i].tiled_index_space();
    Index       val      = from_tis.translate(from_blockid[i], to_tis[i]);
    translated_blockid.push_back(val);
  }
  return translated_blockid;
}

inline void print_blockid(const IndexVector& blockid, const std::string& name = "blockid") {
  std::cout << name << ": ";
  for(auto i: blockid) std::cout << i << " ";
  std::cout << std::endl;
};

template<typename Iter>
inline std::string join(Iter begin, Iter end, const std::string& sep) {
  std::ostringstream oss;
  if(begin != end) {
    oss << *begin;
    ++begin;
  }
  while(begin != end) {
    oss << sep << *begin;
    begin++;
  }
  return oss.str();
}

template<typename Container>
inline std::string join(const Container& c, const std::string& sep) {
  return join(c.begin(), c.end(), sep);
}

/**
 * @brief Construct a dependence map from a label vector. The returned dependence map returns the
 * list of indices a given index depends on, by comparing the primary labels.
 *
 * For example, the routine returns {0:[], 1:[0]} for (i,j(i)). All values in this map are empty
 * when there are no dependent labels. When duplicates exist (e.g., (i,i,j(i))), one of them is
 * arbitrarily picked.
 */
inline std::map<size_t, std::vector<size_t>>
construct_dep_map(const std::vector<TiledIndexLabel>& tile_labels) {
  std::map<size_t, std::vector<size_t>> dep_map;
  std::vector<TileLabelElement>         primary_labels;
  for(const auto& lbl: tile_labels) { primary_labels.push_back(lbl.primary_label()); }
  for(size_t i = 0; i < tile_labels.size(); i++) {
    std::vector<size_t> deps;
    for(auto& sec_lbl: tile_labels[i].secondary_labels()) {
      auto it = std::find(primary_labels.begin(), primary_labels.end(), sec_lbl);
      EXPECTS(it != primary_labels.end());
      deps.push_back(it - primary_labels.begin());
    }
    dep_map[i] = deps;
  }
  return dep_map;
}

template<typename T>
std::vector<T> topological_sort(const std::map<T, std::vector<T>>& dep_map) {
  size_t            num_ids = dep_map.size();
  std::vector<T>    order(num_ids);
  std::vector<bool> done(num_ids, false);
  std::vector<bool> is_ordered(num_ids, false);
  size_t            ctr = 0;
  for(size_t i = 0; i < num_ids; i++) {
    if(done[i]) continue;
    std::vector<size_t> stack{i};
    while(!stack.empty()) {
      for(auto id: dep_map.find(stack.back())->second) {
        EXPECTS(id != i);
        if(!done[id]) {
          stack.push_back(id);
          continue;
        }
      }

      if(!is_ordered[stack.back()]) {
        order[stack.back()]      = ctr++;
        is_ordered[stack.back()] = true;
        done[stack.back()]       = true;
      }

      stack.pop_back();
    }
    EXPECTS(done[i]);
  }
  EXPECTS(ctr == num_ids);
  std::vector<T> new_order(num_ids);
  for(size_t i = 0; i < num_ids; i++) { new_order[order[i]] = i; }

  return new_order;
  // return order;
}

inline std::tuple<IndexVector, bool> translate_blockid_if_possible(const IndexVector& from_blockid,
                                                                   const IndexLabelVec& from_label,
                                                                   const IndexLabelVec& to_label) {
  EXPECTS(from_blockid.size() == from_label.size());
  EXPECTS(from_label.size() == to_label.size());
  if(from_label == to_label) {
    auto to_blockid = from_blockid;
    return {to_blockid, true};
  }

  for(size_t i = 0; i < from_label.size(); i++) {
    if(!from_label[i].tiled_index_space().is_compatible_with(to_label[i].tiled_index_space())) {
      return {IndexVector{}, false};
    }
  }
  const std::map<size_t, std::vector<size_t>>& from_dep_map = construct_dep_map(from_label);
  const std::map<size_t, std::vector<size_t>>& to_dep_map   = construct_dep_map(to_label);

  // for(auto& [key, value] : from_dep_map) {
  //     std::cout << "key - " << key << std::endl;
  //     for(auto& id : value) {
  //         std::cout << "value - " << id << std::endl;
  //     }
  // }

  // for(auto& [key, value] : to_dep_map) {
  //     std::cout << "key - " << key << std::endl;
  //     for(auto& id : value) {
  //         std::cout << "value - " << id << std::endl;
  //     }
  // }

  std::vector<size_t> compute_order = topological_sort(to_dep_map);
  // std::cout << "compute_order: ";
  // for(auto& i : compute_order) {
  //     std::cout << i << " ";
  // }
  // std::cout << std::endl;

  EXPECTS(compute_order.size() == from_blockid.size());
  IndexVector to_blockid(from_blockid.size(), -1);
  for(size_t i = 0; i < compute_order.size(); i++) {
    IndexVector  from_indep_vec, to_indep_vec;
    const size_t cur_pos = compute_order[i];
    auto         it      = from_dep_map.find(cur_pos);
    EXPECTS(it != from_dep_map.end());
    for(const auto& ipos: it->second) { from_indep_vec.push_back(from_blockid[ipos]); }
    it = to_dep_map.find(cur_pos);
    EXPECTS(it != to_dep_map.end());
    for(const auto& ipos: it->second) { to_indep_vec.push_back(to_blockid[ipos]); }
    size_t to_id;
    bool   valid;
    EXPECTS(from_label[cur_pos].tiled_index_space().is_compatible_with(
      to_label[cur_pos].tiled_index_space()));

    // if(!(from_blockid[cur_pos] < from_blockid.size())){
    //     std::cout << "from_blockid.size() = " << from_blockid.size() << std::endl;
    //     std::cout << "from_blockid[" << cur_pos << "] = " << from_blockid[cur_pos] << std::endl;

    // }
    // EXPECTS(from_blockid[cur_pos] < from_blockid.size());
    // std::cout << "cur_pos = " << cur_pos << std::endl;
    // std::cout << "from_blockid[cur_pos] = " << from_blockid[cur_pos] << std::endl;
    // std::cout << "from_label[cur_pos] = " << &from_label[cur_pos] << std::endl;
    // std::cout << "to_label[cur_pos] = " << &to_label[cur_pos] << std::endl;

    // std::cout << "from_indep_vec: ";
    // for(auto& i : from_indep_vec) {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "to_indep_vec: ";
    // for(auto& i : to_indep_vec) {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;

    std::tie(to_id, valid) = from_label[cur_pos].tiled_index_space().translate_if_possible(
      from_blockid[cur_pos], from_indep_vec, to_label[cur_pos].tiled_index_space(), to_indep_vec);

    if(!valid) { return {to_blockid, false}; }
    else {
      // std::cout << "to_blockid : " << i << " - " << compute_order[i] << std::endl;
      // std::cout << "to_id: " << to_id << std::endl;
      to_blockid[compute_order[i]] = to_id;
    }
  }
  return {to_blockid, true};
}

inline void update_labels(IndexLabelVec& labels) {
  EXPECTS(!labels.empty());
  auto dep_map            = construct_dep_map(labels);
  bool has_new_lbl        = false;
  bool have_other_dep_lbl = false;

  const int        nlabels = labels.size();
  std::vector<int> lbl_map(nlabels, -1);
  for(int i = 0; i < nlabels; i++) {
    auto& lbl = labels[i];
    if(lbl_map[i] != -1) { continue; }
    for(int j = i + 1; j < nlabels; j++) {
      if(labels[j] == lbl) { lbl_map[j] = i; }
    }
    lbl_map[i] = i;
  }

  EXPECTS(labels.size() == lbl_map.size());
  for(auto& i: lbl_map) { EXPECTS(i != -1); }

  for(int i = 0; i < nlabels; i++) {
    if(lbl_map[i] < i) {
      labels[i] = labels[lbl_map[i]];
      continue;
    }

    auto& lbl = labels[i];
    if(lbl.is_dependent() && lbl.secondary_labels().size() == 0) {
      labels[i]   = lbl.tiled_index_space().parent_tis().label();
      has_new_lbl = true;
    }
    else if(lbl.is_dependent() && lbl.secondary_labels().size() > 0) { have_other_dep_lbl = true; }
  }

  if(has_new_lbl && have_other_dep_lbl) {
    // Update dependent labels if a new label is created
    for(int i = 0; i < nlabels; i++) {
      auto& lbl = labels[i];
      if(lbl.is_dependent()) {
        auto primary_label    = lbl.primary_label();
        auto secondary_labels = lbl.secondary_labels();
        EXPECTS(!secondary_labels.empty());
        EXPECTS(dep_map[i].size() == secondary_labels.size());
        auto sec_indices = dep_map[i];
        for(size_t j = 0; j < sec_indices.size(); j++) {
          secondary_labels[j] = labels[sec_indices[j]].primary_label();
        }
        labels[i] = TiledIndexLabel{primary_label, secondary_labels};
      }
    }
  }
}

inline void print_labels(const IndexLabelVec& labels) {
  for(auto& lbl: labels) {
    std::cout << "primary: " << lbl.primary_label().label() << " - secondary: [ ";
    for(const auto& l: lbl.secondary_labels()) { std::cout << l.label() << " "; }
    std::cout << "]" << std::endl;
  }
  std::cout << "-------" << std::endl;
}

inline bool is_dense_labels(const IndexLabelVec& labels) {
  for(auto& lbl: labels) {
    if(lbl.is_dependent()) return false;
  }
  return true;
}
template<typename LabeledTensorT>
inline bool is_slicing(const LabeledTensorT& lt) {
  const auto& tis_vec = lt.tensor().tiled_index_spaces();
  const auto& labels  = lt.labels();
  EXPECTS(tis_vec.size() == labels.size());
  for(size_t i = 0; i < labels.size(); i++) {
    if(!labels[i].tiled_index_space().is_identical(tis_vec[i])) return true;
  }
  return false;
}

template<typename T>
inline bool has_duplicates(const std::vector<T>& vec) {
  std::set<T> set_T(vec.begin(), vec.end());
  return (vec.size() != set_T.size());
}

inline bool empty_reduction_primary_labels(const IndexLabelVec& lhs_labels,
                                           const IndexLabelVec& rhs_labels) {
  std::set<TileLabelElement> lhs_plabels, rhs_plabels;
  for(const auto& lbl: lhs_labels) { lhs_plabels.insert(lbl.primary_label()); }
  for(const auto& lbl: rhs_labels) { rhs_plabels.insert(lbl.primary_label()); }
  std::set<TileLabelElement> result;
  std::set_difference(lhs_plabels.begin(), lhs_plabels.end(), rhs_plabels.begin(),
                      rhs_plabels.end(), std::inserter(result, result.end()));
  return result.empty();
}

// Convert array into a tuple
template<typename Array, std::size_t... I>
inline auto a2t_impl(const Array& a, std::index_sequence<I...>) {
  return std::make_tuple(a[I]...);
}

template<typename T, std::size_t N, typename Indices = std::make_index_sequence<N>>
inline auto array2tuple(const std::array<T, N>& a) {
  return a2t_impl(a, Indices{});
}

template<typename VecT, size_t N>
inline auto split_vector(const VecT& vec, const std::vector<size_t>& sizes) {
  EXPECTS(N == sizes.size());
  EXPECTS(N > 0);
  size_t total_size = 0;
  for(const auto& size: sizes) { total_size += size; }
  EXPECTS(total_size = vec.size());

  std::array<VecT, N> new_vecs;
  size_t              start = 0;
  for(size_t i = 0; i < N; i++) {
    auto start_it = vec.begin() + start;
    auto end_it   = start_it + sizes[i];
    EXPECTS(vec.end() - end_it >= 0);
    new_vecs[i].insert(new_vecs[i].end(), start_it, end_it);
    start += sizes[i];
  }

  return array2tuple(new_vecs);
}

template<typename VecT>
inline void merge_vector_impl(VecT& result, const VecT& last) {
  result.insert(result.end(), last.begin(), last.end());
}

template<typename VecT, typename... Args>
inline void merge_vector_impl(VecT& result, const VecT& next, Args&&... rest) {
  result.insert(result.end(), next.begin(), next.end());
  merge_vector_impl(result, std::forward<Args>(rest)...);
}

template<typename VecT, typename... Args>
inline VecT merge_vector(Args&&... rest) {
  VecT result;

  merge_vector_impl(result, std::forward<Args>(rest)...);

  return result;
}

template<typename T>
inline bool has_repeated_elements(const std::vector<T>& vec) {
  return vec.size() != internal::unique_entries(vec).size();
}

inline std::vector<std::string> split_string(std::string str, char delim) {
  std::vector<std::string> result;

  std::istringstream str_stream(str);
  std::string        split_str;
  while(getline(str_stream, split_str, delim)) { result.push_back(split_str); }

  return result;
}
/**
 * @brief New optimized loop related functions
 *
 * @todo: move these to a new loop function
 *
 */

// template <typename Func>
// inline void loop_nest_exec(const std::vector<Range>& ranges, Func&& func,
//                     std::vector<Index>& itr) {
//   EXPECTS(itr.size() == ranges.size());
//   int N = itr.size();
//   if (N == 1) {
//     loop_nest_y_1(func, &itr[0], &ranges[0])();
//   } else if (N == 2) {
//     loop_nest_y_2(func, &itr[0], &ranges[0])();
//   } else if (N == 3) {
//     loop_nest_y_3(func, &itr[0], &ranges[0])();
//   } else if (N == 4) {
//     loop_nest_y_4(func, &itr[0], &ranges[0])();
//   } else {
//     NOT_IMPLEMENTED();
//   }
// }

} // namespace internal

} // namespace tamm
