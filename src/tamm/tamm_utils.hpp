#pragma once

#include "eigen_includes.hpp"
#include "tamm_io.hpp"

namespace tamm {

template<typename Arg, typename... Args>
void print_varlist(Arg&& arg, Args&&... args) {
  std::cout << std::forward<Arg>(arg);
  ((std::cout << ',' << std::forward<Args>(args)), ...);
  std::cout << std::endl;
}

template<typename T>
double compute_tensor_size(const Tensor<T>& tensor) {
  auto   lt   = tensor();
  double size = 0;
  for(auto it: tensor.loop_nest()) {
    auto blockid = internal::translate_blockid(it, lt);
    if(!tensor.is_non_zero(blockid)) continue;
    size += tensor.block_size(blockid);
  }
  return size;
}

/**
 * @brief Prints a Tensor object
 *
 * @tparam T template type for Tensor element type
 * @param [in] tensor input Tensor object
 */
template<typename T>
void print_tensor(const Tensor<T>& tensor, std::string filename = "") {
  std::stringstream tstring;
  auto              lt = tensor();

  auto nz_check = [=](const T val) {
    if constexpr(tamm::internal::is_complex_v<T>) {
      if(val.real() > 1e-12 || val.real() < -1e-12) return true;
    }
    else if(val > 1e-12 || val < -1e-12) return true;
    return false;
  };

  // int                  ndims = tensor.num_modes();
  std::vector<int64_t> dims;
  for(auto tis: tensor.tiled_index_spaces()) dims.push_back(tis.index_space().num_indices());

  tstring << "tensor dims = " << dims << std::endl;
  tstring << "actual tensor size = " << tensor.size() << std::endl;
  for(auto it: tensor.loop_nest()) {
    auto blockid = internal::translate_blockid(it, lt);
    if(!tensor.is_non_zero(blockid)) continue;
    TAMM_SIZE      size = tensor.block_size(blockid);
    std::vector<T> buf(size);
    tensor.get(blockid, buf);
    auto bdims    = tensor.block_dims(blockid);
    auto boffsets = tensor.block_offsets(blockid);
    tstring << "blockid: " << blockid << ", ";
    tstring << "block_offsets: " << boffsets << ", ";
    tstring << "bdims: " << bdims << ", size: " << size << std::endl;

    for(TAMM_SIZE i = 0; i < size; i++) {
      if(i % 6 == 0) tstring << "\n";
      if(nz_check(buf[i]))
        tstring << std::fixed << std::setw(10) << std::setprecision(10) << buf[i] << " ";
    }
    tstring << std::endl;
  }

  if(!filename.empty()) {
    std::ofstream tos(filename + ".txt", std::ios::out);
    if(!tos) std::cerr << "Error opening file " << filename << std::endl;
    tos << tstring.str() << std::endl;
    tos.close();
  }
  else std::cout << tstring.str();
}

template<typename T>
void print_tensor_all(const Tensor<T>& tensor, std::string filename = "") {
  std::stringstream tstring;
  auto              lt = tensor();

  // int                  ndims = tensor.num_modes();
  std::vector<int64_t> dims;
  for(auto tis: tensor.tiled_index_spaces()) dims.push_back(tis.index_space().num_indices());

  tstring << "tensor dims = " << dims << std::endl;
  tstring << "actual tensor size = " << tensor.size() << std::endl;
  for(auto it: tensor.loop_nest()) {
    auto blockid = internal::translate_blockid(it, lt);
    if(!tensor.is_non_zero(blockid)) continue;
    TAMM_SIZE      size = tensor.block_size(blockid);
    std::vector<T> buf(size);
    tensor.get(blockid, buf);
    auto bdims    = tensor.block_dims(blockid);
    auto boffsets = tensor.block_offsets(blockid);
    tstring << "blockid: " << blockid << ", ";
    tstring << "block_offsets: " << boffsets << ", ";
    tstring << "bdims: " << bdims << ", size: " << size << std::endl;

    for(TAMM_SIZE i = 0; i < size; i++) {
      if(i % 6 == 0) tstring << "\n";
      tstring << std::fixed << std::setw(10) << std::setprecision(10) << buf[i] << " ";
    }
    tstring << std::endl;
  }

  if(!filename.empty()) {
    std::ofstream tos(filename + ".txt", std::ios::out);
    if(!tos) std::cerr << "Error opening file " << filename << std::endl;
    tos << tstring.str() << std::endl;
    tos.close();
  }
  else std::cout << tstring.str();
}

template<typename T>
void print_tensor_reshaped(LabeledTensor<T> l_tensor, const IndexLabelVec& new_labels) {
  EXPECTS(l_tensor.tensor().is_allocated());
  auto ec = l_tensor.tensor().execution_context();

  Tensor<T> new_tensor{new_labels};

  Scheduler sch{*ec};

  sch.allocate(new_tensor)(new_tensor(new_labels) = l_tensor).execute();

  print_tensor_all(new_tensor);
  sch.deallocate(new_tensor).execute();
}

template<typename T>
void print_labeled_tensor(LabeledTensor<T> l_tensor) {
  EXPECTS(l_tensor.tensor().is_allocated());
  auto ec = l_tensor.tensor().execution_context();

  Tensor<T> new_tensor{l_tensor.labels()};

  Scheduler sch{*ec};

  sch.allocate(new_tensor)(new_tensor(l_tensor.labels()) = l_tensor).execute();

  print_tensor_all(new_tensor);
  sch.deallocate(new_tensor).execute();
}

template<typename T>
std::string tensor_to_string(const Tensor<T>& tensor) {
  std::stringstream tstring;
  auto              lt = tensor();

  // int                  ndims = tensor.num_modes();
  std::vector<int64_t> dims;
  for(auto tis: tensor.tiled_index_spaces()) dims.push_back(tis.index_space().num_indices());

  tstring << "tensor dims = " << dims << std::endl;
  tstring << "actual tensor size = " << tensor.size() << std::endl;
  for(auto it: tensor.loop_nest()) {
    auto blockid = internal::translate_blockid(it, lt);
    if(!tensor.is_non_zero(blockid)) continue;
    TAMM_SIZE      size = tensor.block_size(blockid);
    std::vector<T> buf(size);
    tensor.get(blockid, buf);
    auto bdims    = tensor.block_dims(blockid);
    auto boffsets = tensor.block_offsets(blockid);
    tstring << "blockid: " << blockid << ", ";
    tstring << "block_offsets: " << boffsets << ", ";
    tstring << "bdims: " << bdims << ", size: " << size << std::endl;

    for(TAMM_SIZE i = 0; i < size; i++) {
      if(i % 6 == 0) tstring << "\n";
      tstring << std::fixed << std::setw(15) << std::setprecision(15) << buf[i] << " ";
    }
    tstring << std::endl;
  }

  return tstring.str();
}

template<typename T>
void print_vector(std::vector<T> vec, std::string filename = "") {
  std::stringstream tstring;
  for(size_t i = 0; i < vec.size(); i++)
    tstring << i + 1 << "\t" << std::fixed << std::setprecision(12) << vec[i] << std::endl;

  if(!filename.empty()) {
    std::ofstream tos(filename, std::ios::out);
    if(!tos) std::cerr << "Error opening file " << filename << std::endl;
    tos << tstring.str() << std::endl;
    tos.close();
  }
  else std::cout << tstring.str();
}

template<typename T>
void print_max_above_threshold(const Tensor<T>& tensor, double printtol,
                               std::string filename = "") {
  auto              lt = tensor();
  std::stringstream tstring;

  for(auto it: tensor.loop_nest()) {
    auto blockid = internal::translate_blockid(it, lt);
    if(!tensor.is_non_zero(blockid)) continue;
    auto           block_dims   = tensor.block_dims(blockid);
    auto           block_offset = tensor.block_offsets(blockid);
    TAMM_SIZE      size         = tensor.block_size(blockid);
    std::vector<T> buf(size);
    tensor.get(blockid, buf);
    auto bdims  = tensor.block_dims(blockid);
    auto nmodes = tensor.num_modes();

    size_t c = 0;

    if(nmodes == 1) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++, c++) {
        if(std::abs(buf[c]) > printtol)
          tstring << i << "   " << std::fixed << std::setprecision(12) << std::right
                  << std::setw(18) << buf[c] << std::endl;
      }
    }
    else if(nmodes == 2) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++, c++) {
          if(std::abs(buf[c]) > printtol)
            tstring << i << "   " << j << "   " << std::fixed << std::setprecision(12) << std::right
                    << std::setw(18) << buf[c] << std::endl;
        }
      }
    }
    else if(nmodes == 3) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++, c++) {
            if(std::abs(buf[c]) > printtol)
              tstring << i << "   " << j << "   " << k << "   " << std::fixed
                      << std::setprecision(12) << std::right << std::setw(18) << buf[c]
                      << std::endl;
          }
        }
      }
    }
    else if(nmodes == 4) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
            for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++, c++) {
              if(std::abs(buf[c]) > printtol)
                tstring << i << "   " << j << "   " << k << "   " << l << "   " << std::fixed
                        << std::setprecision(12) << std::right << std::setw(18) << buf[c]
                        << std::endl;
            }
          }
        }
      }
    }
    else NOT_IMPLEMENTED();
  }

  if(!filename.empty()) {
    std::ofstream tos(filename, std::ios::out);
    if(!tos) std::cerr << "Error opening file " << filename << std::endl;
    tos << tstring.str() << std::endl;
    tos.close();
  }
  else std::cout << tstring.str();
}

/**
 * @brief Get the scalar value from the Tensor
 *
 * @tparam T template type for Tensor element type
 * @param [in] tensor input Tensor object
 * @returns a scalar value in type T
 *
 * @warning This function only works with scalar (zero dimensional) Tensor
 * objects
 */
template<typename T>
T get_scalar(Tensor<T>& tensor) {
  T scalar;
  EXPECTS(tensor.num_modes() == 0);
  tensor.get({}, {&scalar, 1});
  return scalar;
}

template<typename TensorType>
ExecutionContext& get_ec(LabeledTensor<TensorType> ltensor) {
  return *(ltensor.tensor().execution_context());
}

/**
 * @brief Update input LabeledTensor object with a lambda function
 *
 * @tparam T template type for Tensor element type
 * @tparam Func template type for the lambda function
 * @param [in] labeled_tensor tensor slice to be updated
 * @param [in] lambda function for updating the tensor
 */

template<typename T, typename Func>
void update_tensor(Tensor<T> tensor, Func lambda) {
  update_tensor(tensor(), lambda);
}

template<typename T, typename Func>
void update_tensor(LabeledTensor<T> labeled_tensor, Func lambda) {
  LabelLoopNest loop_nest{labeled_tensor.labels()};

  for(const auto& itval: loop_nest) {
    const IndexVector blockid = internal::translate_blockid(itval, labeled_tensor);
    size_t            size    = labeled_tensor.tensor().block_size(blockid);
    std::vector<T>    buf(size);

    labeled_tensor.tensor().get(blockid, buf);
    lambda(blockid, buf);
    labeled_tensor.tensor().put(blockid, buf);
  }
}

/**
 * @brief Update a value at a given coordinate in a tensor
 * TODO: update local buf directly to avoid get/put
 */
template<typename T>
void update_tensor_val(LabeledTensor<T> ltensor, std::vector<size_t> coord, T val) {
  Tensor<T>    tensor = ltensor.tensor();
  const size_t ndims  = tensor.num_modes();
  EXPECTS(ndims == coord.size());

  if((tensor.execution_context())->pg().rank() == 0) {
    LabelLoopNest loop_nest{ltensor.labels()};

    for(auto it: tensor.loop_nest()) {
      auto blockid = internal::translate_blockid(it, ltensor);
      if(!tensor.is_non_zero(blockid)) continue;
      auto block_dims   = tensor.block_dims(blockid);
      auto block_offset = tensor.block_offsets(blockid);
      // std::cout << "blockid: " << blockid << ", ";
      // std::cout << "block_offsets: " << block_offset << ", ";
      // std::cout << "bdims: " << block_dims << ", size: " << tensor.block_size(blockid) <<
      // std::endl;

      bool vc = true;
      for(size_t x = 0; x < ndims; x++) {
        if(!((coord[x] >= block_offset[x]) && (coord[x] < (block_offset[x] + block_dims[x]))))
          vc = vc && false;
      }
      if(vc) {
        TAMM_SIZE size = tensor.block_size(blockid);

        std::vector<T> buf(size);
        TAMM_SIZE      val_pos = 0;
        for(size_t x = 0; x < ndims; x++) {
          TAMM_SIZE rd = coord[x] - block_offset[x];
          for(size_t y = x + 1; y < ndims; y++) rd *= block_dims[y];
          val_pos += rd;
        }

        tensor.get(blockid, buf);
        buf[val_pos] = val;
        tensor.put(blockid, buf);
        break;
      }
    }
  }
  (tensor.execution_context())->pg().barrier();
}

template<typename T>
void update_tensor_val(Tensor<T>& tensor, std::vector<size_t> coord, T val) {
  update_tensor_val(tensor(), coord, val);
}

/**
 * @brief Update input LabeledTensor object with a lambda function
 *
 * @tparam T template type for Tensor element type
 * @tparam Func template type for the lambda function
 * @param [in] labeled_tensor tensor slice to be updated
 * @param [in] lambda function for updating the tensor
 */

template<typename T, typename Func>
void update_tensor_general(Tensor<T> tensor, Func lambda) {
  update_tensor_general(tensor(), lambda);
}

template<typename T, typename Func>
void update_tensor_general(LabeledTensor<T> labeled_tensor, Func lambda) {
  LabelLoopNest loop_nest{labeled_tensor.labels()};

  for(const auto& itval: loop_nest) {
    const IndexVector blockid = internal::translate_blockid(itval, labeled_tensor);
    size_t            size    = labeled_tensor.tensor().block_size(blockid);
    std::vector<T>    buf(size);

    labeled_tensor.tensor().get(blockid, buf);
    lambda(labeled_tensor.tensor(), blockid, buf);
    labeled_tensor.tensor().put(blockid, buf);
  }
}

/**
 * @brief Construct an ExecutionContext object
 *
 * @returns an Execution context
 * @todo there is possible memory leak as distribution will not be unallocated
 * when Execution context is destructed
 */
/*inline ExecutionContext make_execution_context() {
    ProcGroup* pg = new ProcGroup {ProcGroup::create_world_coll()};
    auto* pMM             = MemoryManagerGA::create_coll(pg);
    Distribution_NW* dist = new Distribution_NW();
    RuntimeEngine* re = new RuntimeEngine{};
    ExecutionContext *ec = new ExecutionContext(*pg, dist, pMM, re);
    return *ec;
} */

/**
 * @brief method for getting the sum of the values on the diagonal
 *
 * @returns sum of the diagonal values
 * @warning only defined for NxN tensors
 */
template<typename TensorType>
TensorType trace(Tensor<TensorType> tensor) {
  return trace(tensor());
}

template<typename TensorType>
TensorType trace(LabeledTensor<TensorType> ltensor) {
  ExecutionContext& ec    = get_ec(ltensor);
  TensorType        lsumd = 0;
  TensorType        gsumd = 0;

  Tensor<TensorType> tensor = ltensor.tensor();
  // Defined only for NxN tensors
  EXPECTS(tensor.num_modes() == 2);

  auto gettrace = [&](const IndexVector& bid) {
    const IndexVector blockid = internal::translate_blockid(bid, ltensor);
    if(blockid[0] == blockid[1]) {
      const TAMM_SIZE         size = tensor.block_size(blockid);
      std::vector<TensorType> buf(size);
      tensor.get(blockid, buf);
      auto   block_dims   = tensor.block_dims(blockid);
      auto   block_offset = tensor.block_offsets(blockid);
      auto   dim          = block_dims[0];
      auto   offset       = block_offset[0];
      size_t i            = 0;
      for(auto p = offset; p < offset + dim; p++, i++) { lsumd += buf[i * dim + i]; }
    }
  };
  block_for(ec, ltensor, gettrace);
  gsumd = ec.pg().allreduce(&lsumd, ReduceOp::sum);
  return gsumd;
}

/**
 * @brief method for getting the sum of the values on the diagonal
 *
 * @returns sum of the diagonal values
 * @warning only defined for NxN tensors
 */
template<typename TensorType>
TensorType trace_sqr(Tensor<TensorType> tensor) {
  return trace_sqr(tensor());
}

template<typename TensorType>
TensorType trace_sqr(LabeledTensor<TensorType> ltensor) {
  ExecutionContext& ec    = get_ec(ltensor);
  TensorType        lsumd = 0;
  TensorType        gsumd = 0;

  Tensor<TensorType> tensor = ltensor.tensor();
  // Defined only for NxN tensors
  EXPECTS(tensor.num_modes() == 2);

  auto gettrace = [&](const IndexVector& bid) {
    const IndexVector blockid = internal::translate_blockid(bid, ltensor);
    if(blockid[0] == blockid[1]) {
      const TAMM_SIZE         size = tensor.block_size(blockid);
      std::vector<TensorType> buf(size);
      tensor.get(blockid, buf);
      auto   block_dims   = tensor.block_dims(blockid);
      auto   block_offset = tensor.block_offsets(blockid);
      auto   dim          = block_dims[0];
      auto   offset       = block_offset[0];
      size_t i            = 0;
      for(auto p = offset; p < offset + dim; p++, i++) {
        // sqr of diagonal
        lsumd += buf[i * dim + i] * buf[i * dim + i];
      }
    }
  };
  block_for(ec, ltensor, gettrace);
  gsumd = ec.pg().allreduce(&lsumd, ReduceOp::sum);
  return gsumd;
}

/**
 * @brief method for converting std::vector to tamm 1D tensor
 */
template<typename TensorType>
void vector_to_tamm_tensor(Tensor<TensorType> tensor, std::vector<TensorType> svec) {
  EXPECTS(tensor.num_modes() == 1);

  for(const auto& blockid: tensor.loop_nest()) {
    const tamm::TAMM_SIZE   size = tensor.block_size(blockid);
    std::vector<TensorType> buf(size);
    tensor.get(blockid, buf);
    auto   block_dims   = tensor.block_dims(blockid);
    auto   block_offset = tensor.block_offsets(blockid);
    size_t c            = 0;
    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++, c++) {
      buf[c] = svec[i];
    }
    tensor.put(blockid, buf);
  }
}

/**
 * @brief method for converting tamm 1D tensor to std::vector
 */
template<typename TensorType>
void tamm_tensor_to_vector(Tensor<TensorType> tensor, std::vector<TensorType>& svec) {
  EXPECTS(tensor.num_modes() == 1);

  for(const auto& blockid: tensor.loop_nest()) {
    const tamm::TAMM_SIZE   size = tensor.block_size(blockid);
    std::vector<TensorType> buf(size);
    tensor.get(blockid, buf);
    auto   block_dims   = tensor.block_dims(blockid);
    auto   block_offset = tensor.block_offsets(blockid);
    size_t c            = 0;
    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++, c++) {
      svec[i] = buf[c];
    }
  }
}

/**
 * @brief method for converting tamm 1D tensor to std::vector
 */
template<typename TensorType>
std::vector<TensorType> tamm_tensor_to_vector(Tensor<TensorType> tensor) {
  EXPECTS(tensor.num_modes() == 1);
  auto                    dim1 = tensor.tiled_index_spaces()[0].max_num_indices();
  std::vector<TensorType> svec(dim1);
  tamm_tensor_to_vector(tensor, svec);
  return svec;
}

/**
 * @brief method for getting the diagonal values in a Tensor
 *
 * @returns the diagonal values
 * @warning only defined for NxN tensors
 */
template<typename TensorType>
std::vector<TensorType> diagonal(Tensor<TensorType> tensor) {
  return diagonal(tensor());
}

template<typename TensorType>
std::vector<TensorType> diagonal(LabeledTensor<TensorType> ltensor) {
  ExecutionContext& ec = get_ec(ltensor);

  std::vector<TensorType> dvec;

  if(ec.pg().rank() == 0) {
    Tensor<TensorType> tensor = ltensor.tensor();
    // Defined only for NxN tensors
    EXPECTS(tensor.num_modes() == 2);

    LabelLoopNest loop_nest{ltensor.labels()};

    for(const IndexVector& bid: loop_nest) {
      const IndexVector blockid = internal::translate_blockid(bid, ltensor);

      if(blockid[0] == blockid[1]) {
        const TAMM_SIZE         size = tensor.block_size(blockid);
        std::vector<TensorType> buf(size);
        tensor.get(blockid, buf);
        auto   block_dims   = tensor.block_dims(blockid);
        auto   block_offset = tensor.block_offsets(blockid);
        auto   dim          = block_dims[0];
        auto   offset       = block_offset[0];
        size_t i            = 0;
        for(auto p = offset; p < offset + dim; p++, i++) { dvec.push_back(buf[i * dim + i]); }
      }
    }
  }

  int dsize = (int) dvec.size();
  ec.pg().broadcast(&dsize, 0);
  if(ec.pg().rank() != 0) dvec.resize(dsize);
  ec.pg().broadcast(dvec.data(), dsize, 0);

  return dvec;
}

template<typename TensorType>
Tensor<TensorType> identity_matrix(ExecutionContext& ec, const TiledIndexSpace& tis) {
  Tensor<TensorType> tensor{tis, tis};
  Tensor<TensorType>::allocate(&ec, tensor);
  Scheduler{ec}(tensor() = 0.0).execute();

  if(ec.pg().rank() == 0) {
    LabelLoopNest loop_nest{tensor().labels()};

    for(const IndexVector& blockid: loop_nest) {
      if(blockid[0] == blockid[1]) {
        const TAMM_SIZE         size = tensor.block_size(blockid);
        std::vector<TensorType> buf(size);
        tensor.get(blockid, buf);
        auto   block_dims   = tensor.block_dims(blockid);
        auto   block_offset = tensor.block_offsets(blockid);
        auto   dim          = block_dims[0];
        auto   offset       = block_offset[0];
        size_t i            = 0;
        for(auto p = offset; p < offset + dim; p++, i++) { buf[i * dim + i] = 1; }
        tensor.put(blockid, buf);
      }
    }
  }
  ec.pg().barrier();
  return tensor;
}

/**
 * @brief method for updating the diagonal values in a Tensor
 *
 * @warning only defined for NxN tensors
 */
template<typename TensorType>
void update_diagonal(Tensor<TensorType> tensor, const std::vector<TensorType>& dvec) {
  update_diagonal(tensor(), dvec);
}

template<typename TensorType>
void update_diagonal(LabeledTensor<TensorType> ltensor, const std::vector<TensorType>& dvec) {
  ExecutionContext& ec = get_ec(ltensor);

  if(ec.pg().rank() == 0) {
    Tensor<TensorType> tensor = ltensor.tensor();
    // Defined only for NxN tensors
    EXPECTS(tensor.num_modes() == 2);

    LabelLoopNest loop_nest{ltensor.labels()};

    for(const IndexVector& bid: loop_nest) {
      const IndexVector blockid = internal::translate_blockid(bid, ltensor);

      if(blockid[0] == blockid[1]) {
        const TAMM_SIZE         size = tensor.block_size(blockid);
        std::vector<TensorType> buf(size);
        tensor.get(blockid, buf);
        auto   block_dims   = tensor.block_dims(blockid);
        auto   block_offset = tensor.block_offsets(blockid);
        auto   dim          = block_dims[0];
        auto   offset       = block_offset[0];
        size_t i            = 0;
        for(auto p = offset; p < offset + dim; p++, i++) { buf[i * dim + i] += dvec[p]; }
        tensor.put(blockid, buf);
      }
    }
  }
}

/**
 * @brief uses a function to fill in elements of a tensor
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param ltensor tensor to operate on
 * @param func function to fill in the tensor with
 */
template<typename TensorType>
void fill_tensor(Tensor<TensorType>                                        tensor,
                 std::function<void(const IndexVector&, span<TensorType>)> func) {
  fill_tensor(tensor(), func);
}

template<typename TensorType>
void fill_tensor(LabeledTensor<TensorType>                                 ltensor,
                 std::function<void(const IndexVector&, span<TensorType>)> func) {
  ExecutionContext&  ec     = get_ec(ltensor);
  Tensor<TensorType> tensor = ltensor.tensor();

  auto lambda = [&](const IndexVector& bid) {
    const IndexVector       blockid = internal::translate_blockid(bid, ltensor);
    const tamm::TAMM_SIZE   dsize   = tensor.block_size(blockid);
    std::vector<TensorType> dbuf(dsize);
    // tensor.get(blockid, dbuf);
    func(blockid, dbuf);
    tensor.put(blockid, dbuf);
  };
  block_for(ec, ltensor, lambda);
}

template<typename TensorType>
void fill_sparse_tensor(Tensor<TensorType>                                        tensor,
                        std::function<void(const IndexVector&, span<TensorType>)> func) {
  fill_sparse_tensor(tensor(), func);
}

template<typename TensorType>
void fill_sparse_tensor(LabeledTensor<TensorType>                                 ltensor,
                        std::function<void(const IndexVector&, span<TensorType>)> func) {
  ExecutionContext&  ec     = get_ec(ltensor);
  Tensor<TensorType> tensor = ltensor.tensor();

  auto lambda = [&](const IndexVector& bid) {
    const tamm::TAMM_SIZE   dsize   = tensor.block_size(bid);
    const IndexVector       blockid = internal::translate_sparse_blockid(bid, ltensor);
    std::vector<TensorType> dbuf(dsize);
    // tensor.get(blockid, dbuf);
    func(blockid, dbuf);

    tensor.put(bid, dbuf);
  };
  block_for(ec, ltensor, lambda);
}

template<typename TensorType>
Tensor<TensorType> redistribute_tensor(Tensor<TensorType> stensor, TiledIndexSpaceVec tis,
                                       std::vector<size_t> spins = {}) {
  ExecutionContext& ec = get_ec(stensor());

#if defined(USE_UPCXX)
  ga_over_upcxx<TensorType>* wmn_ga = tamm_to_ga(ec, stensor);
#else
  int wmn_ga = tamm_to_ga(ec, stensor);
#endif

  Tensor<TensorType> dtensor{tis};
  if(spins.size() > 0) dtensor = Tensor<TensorType>{tis, spins};
  if(stensor.kind() == TensorBase::TensorKind::dense) dtensor.set_dense();
  Tensor<TensorType>::allocate(&ec, dtensor);

  ga_to_tamm(ec, dtensor, wmn_ga);

#if defined(USE_UPCXX)
  wmn_ga->destroy();
#else
  NGA_Destroy(wmn_ga);
#endif

  return dtensor;
}

/**
 * @brief retile a tamm tensor
 *
 * @param stensor source tensor
 * @param dtensor tensor after retiling.
 *  Assumed to be created & allocated using the new tiled index space.
 */
template<typename TensorType>
void retile_tamm_tensor(Tensor<TensorType> stensor, Tensor<TensorType>& dtensor,
                        std::string tname = "") {
  auto io_t1 = std::chrono::high_resolution_clock::now();

  ExecutionContext& ec   = get_ec(stensor());
  int               rank = ec.pg().rank().value();

#if defined(USE_UPCXX)
  ga_over_upcxx<TensorType>* ga_tens = tamm_to_ga(ec, stensor);
  ga_to_tamm(ec, dtensor, ga_tens);
  ga_tens->destroy();
#else
  int ga_tens = tamm_to_ga(ec, stensor);
  ga_to_tamm(ec, dtensor, ga_tens);
  NGA_Destroy(ga_tens);
#endif

  auto io_t2 = std::chrono::high_resolution_clock::now();

  double rt_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((io_t2 - io_t1)).count();
  if(rank == 0 && !tname.empty())
    std::cout << "Time to re-tile " << tname << " tensor: " << rt_time << " secs" << std::endl;
}

template<typename TensorType>
TensorType linf_norm(Tensor<TensorType> tensor) {
  return linf_norm(tensor());
}

template<typename TensorType>
TensorType linf_norm(LabeledTensor<TensorType> ltensor) {
  ExecutionContext& gec = get_ec(ltensor);

  TensorType         linfnorm = 0;
  Tensor<TensorType> tensor   = ltensor.tensor();

#ifdef TU_SG
  int rank                   = gec.pg().rank().value();
  auto [nagg, ppn, subranks] = get_subgroup_info(gec, tensor);
#if defined(USE_UPCXX)
  upcxx::team* sub_comm =
    new upcxx::team(gec.pg().comm()->split(rank < subranks ? 0 : upcxx::team::color_none, 0));
#else
  MPI_Comm sub_comm;
  subcomm_from_subranks(gec, subranks, sub_comm);
#endif

  if(rank < subranks) {
#if defined(USE_UPCXX)
    ProcGroup pg = ProcGroup::create_coll(*sub_comm);
#else
    ProcGroup pg = ProcGroup::create_coll(sub_comm);
#endif
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
#else
  ExecutionContext& ec = gec;
#endif
    auto getnorm = [&](const IndexVector& bid) {
      const IndexVector       blockid = internal::translate_blockid(bid, ltensor);
      const tamm::TAMM_SIZE   dsize   = tensor.block_size(blockid);
      std::vector<TensorType> dbuf(dsize);
      tensor.get(blockid, dbuf);
      for(TensorType val: dbuf) {
        if(std::abs(linfnorm) < std::abs(val)) linfnorm = val;
      }
    };
    block_for(ec, ltensor, getnorm);

#ifdef TU_SG
    ec.flush_and_sync();
    // MemoryManagerGA::destroy_coll(mgr);
#if !defined(USE_UPCXX)
    MPI_Comm_free(&sub_comm);
#endif
    pg.destroy_coll();
  }
#if defined(USE_UPCXX)
  sub_comm->destroy();
#endif
#endif

  gec.pg().barrier();

  auto                            linf_abs = std::abs(linfnorm);
  std::vector<decltype(linf_abs)> linf_pair{linf_abs, gec.pg().rank().value()};
  decltype(linf_pair)             linf_result(2, 0);

  gec.pg().allreduce(linf_pair.data(), linf_result.data(), 1, ReduceOp::maxloc);
  gec.pg().broadcast(&linfnorm, linf_result[1]);

  return linfnorm;
}

/**
 * @brief applies a function elementwise to a tensor
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param ltensor tensor to operate on
 * @param func function to be applied to each element
 */
template<typename TensorType>
void apply_ewise_ip(LabeledTensor<TensorType> ltensor, std::function<TensorType(TensorType)> func) {
  ExecutionContext&  gec    = get_ec(ltensor);
  Tensor<TensorType> tensor = ltensor.tensor();

#ifdef TU_SG
  auto [nagg, ppn, subranks] = get_subgroup_info(gec, tensor);
#if defined(USE_UPCXX)
  upcxx::team* sub_comm = new upcxx::team(
    gec.pg().comm()->split(gec.pg().rank() < subranks ? 0 : upcxx::team::color_none, 0));
#else
  MPI_Comm sub_comm;
  subcomm_from_subranks(gec, subranks, sub_comm);
#endif
  if(gec.pg().rank() < subranks) {
#if defined(USE_UPCXX)
    ProcGroup pg = ProcGroup::create_coll(*sub_comm);
#else
    ProcGroup pg = ProcGroup::create_coll(sub_comm);
#endif
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
#else
  ExecutionContext& ec = gec;
#endif

    auto lambda = [&](const IndexVector& bid) {
      const IndexVector       blockid = internal::translate_blockid(bid, ltensor);
      const tamm::TAMM_SIZE   dsize   = tensor.block_size(blockid);
      std::vector<TensorType> dbuf(dsize);
      tensor.get(blockid, dbuf);
      for(size_t c = 0; c < dsize; c++) dbuf[c] = func(dbuf[c]);
      tensor.put(blockid, dbuf);
    };
    block_for(ec, ltensor, lambda);

#ifdef TU_SG
    ec.flush_and_sync();
    // MemoryManagerGA::destroy_coll(mgr);
#if !defined(USE_UPCXX)
    MPI_Comm_free(&sub_comm);
#endif
    pg.destroy_coll();
  }
#if defined(USE_UPCXX)
  sub_comm->destroy();
#endif
#endif
  gec.pg().barrier();
}

// Several convenience functions using apply_ewise_ip.
// These routines update the tensor in-place
template<typename TensorType>
void conj_ip(LabeledTensor<TensorType> ltensor) {
  std::function<TensorType(TensorType)> func = [&](TensorType a) { return std::conj(a); };
  apply_ewise_ip(ltensor, func);
}

template<typename TensorType>
void conj_ip(Tensor<TensorType> tensor) {
  conj_ip(tensor());
}

template<typename TensorType>
void scale_ip(LabeledTensor<TensorType> ltensor, TensorType alpha) {
  std::function<TensorType(TensorType)> func = [&](TensorType a) { return alpha * a; };
  apply_ewise_ip(ltensor, func);
}

template<typename TensorType>
void scale_ip(Tensor<TensorType> tensor, TensorType alpha) {
  scale_ip(tensor(), alpha);
}

/**
 * @brief applies a function elementwise to a tensor, returns a new tensor
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param oltensor original tensor
 * @param func function to be applied to each element
 * @return resulting tensor after applying func to original tensor
 */
template<typename TensorType>
Tensor<TensorType> apply_ewise(LabeledTensor<TensorType>             oltensor,
                               std::function<TensorType(TensorType)> func, bool is_lt = true) {
  ExecutionContext&  ec      = get_ec(oltensor);
  Tensor<TensorType> otensor = oltensor.tensor();

  Tensor<TensorType> tensor{oltensor.labels()};
  if(otensor.kind() == TensorBase::TensorKind::dense) tensor.set_dense();

  LabeledTensor<TensorType> ltensor = tensor();
  Tensor<TensorType>::allocate(&ec, tensor);
  // if(is_lt) Scheduler{ec}(ltensor = oltensor).execute();

  auto lambda = [&](const IndexVector& bid) {
    const IndexVector       blockid = internal::translate_blockid(bid, oltensor);
    const tamm::TAMM_SIZE   dsize   = tensor.block_size(bid);
    std::vector<TensorType> dbuf(dsize);
    otensor.get(blockid, dbuf);
    for(size_t c = 0; c < dsize; c++) dbuf[c] = func(dbuf[c]);
    tensor.put(bid, dbuf);
  };
  block_for(ec, ltensor, lambda);
  return tensor;
}

// Several convenience functions using apply_ewise
// These routines return a new tensor
template<typename TensorType, typename = std::enable_if_t<internal::is_complex_v<TensorType>>>
Tensor<TensorType> conj(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
  std::function<TensorType(TensorType)> func = [&](TensorType a) { return std::conj(a); };
  return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType, typename = std::enable_if_t<internal::is_complex_v<TensorType>>>
Tensor<TensorType> conj(Tensor<TensorType> tensor) {
  return conj(tensor(), false);
}

template<typename TensorType>
Tensor<TensorType> square(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
  std::function<TensorType(TensorType)> func = [&](TensorType a) { return a * a; };
  return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> square(Tensor<TensorType> tensor) {
  return square(tensor(), false);
}

template<typename TensorType>
Tensor<TensorType> log10(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
  std::function<TensorType(TensorType)> func = [&](TensorType a) { return std::log10(a); };
  return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> log10(Tensor<TensorType> tensor) {
  return log10(tensor(), false);
}

template<typename TensorType>
Tensor<TensorType> log(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
  std::function<TensorType(TensorType)> func = [&](TensorType a) { return std::log(a); };
  return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> log(Tensor<TensorType> tensor) {
  return log(tensor(), false);
}

template<typename TensorType>
Tensor<TensorType> einverse(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
  std::function<TensorType(TensorType)> func = [&](TensorType a) { return 1 / a; };
  return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> einverse(Tensor<TensorType> tensor) {
  return einverse(tensor(), false);
}

template<typename TensorType>
Tensor<TensorType> pow(LabeledTensor<TensorType> ltensor, TensorType alpha, bool is_lt = true) {
  std::function<TensorType(TensorType)> func = [&](TensorType a) { return std::pow(a, alpha); };
  return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> pow(Tensor<TensorType> tensor, TensorType alpha) {
  return pow(tensor(), alpha, false);
}

template<typename TensorType>
Tensor<TensorType> scale(LabeledTensor<TensorType> ltensor, TensorType alpha, bool is_lt = true) {
  std::function<TensorType(TensorType)> func = [&](TensorType a) { return alpha * a; };
  return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> scale(Tensor<TensorType> tensor, TensorType alpha) {
  return scale(tensor(), alpha, false);
}

template<typename TensorType>
Tensor<TensorType> sqrt(LabeledTensor<TensorType> ltensor, bool is_lt = true) {
  std::function<TensorType(TensorType)> func = [&](TensorType a) { return std::sqrt(a); };
  return apply_ewise(ltensor, func, is_lt);
}

template<typename TensorType>
Tensor<TensorType> sqrt(Tensor<TensorType> tensor) {
  return sqrt(tensor(), false);
}

// Normally, a setop is used instead of this routine.
template<typename TensorType>
void set_val_ip(LabeledTensor<TensorType> ltensor, TensorType alpha) {
  std::function<TensorType(TensorType)> func = [&](TensorType a) { return alpha; };
  apply_ewise_ip(ltensor, func);
}

template<typename TensorType>
void set_val_ip(Tensor<TensorType> tensor, TensorType alpha) {
  set_val_ip(tensor(), alpha);
}

template<typename TensorType>
void random_ip(LabeledTensor<TensorType> ltensor, unsigned int seed = 0) {
  std::mt19937                           generator(get_ec(ltensor).pg().rank().value());
  std::uniform_real_distribution<double> tensor_rand_dist(0.0, 1.0);

  if(seed > 0) { generator.seed(seed); }

  if constexpr(!tamm::internal::is_complex_v<TensorType>) {
    std::function<TensorType(TensorType)> func = [&](TensorType a) {
      return tensor_rand_dist(generator);
    };
    apply_ewise_ip(ltensor, func);
  }
  else {
    std::function<TensorType(TensorType)> func = [&](TensorType a) {
      return TensorType(tensor_rand_dist(generator), tensor_rand_dist(generator));
    };
    apply_ewise_ip(ltensor, func);
  }
}

template<typename TensorType>
void random_ip(Tensor<TensorType> tensor, unsigned int seed = 0) {
  random_ip(tensor(), seed);
}

template<typename TensorType>
TensorType sum(LabeledTensor<TensorType> ltensor) {
  ExecutionContext&  gec    = get_ec(ltensor);
  TensorType         lsumsq = 0;
  TensorType         gsumsq = 0;
  Tensor<TensorType> tensor = ltensor.tensor();

  int rank = gec.pg().rank().value();

#ifdef TU_SG
  auto [nagg, ppn, subranks] = get_subgroup_info(gec, tensor);
#if defined(USE_UPCXX)
  upcxx::team* sub_comm = new upcxx::team(
    gec.pg().comm()->split(gec.pg().rank() < subranks ? 0 : upcxx::team::color_none, 0));
#else
  MPI_Comm sub_comm;
  subcomm_from_subranks(gec, subranks, sub_comm);
#endif

  if(rank < subranks) {
#if defined(USE_UPCXX)
    ProcGroup pg = ProcGroup::create_coll(*sub_comm);
#else
    ProcGroup pg = ProcGroup::create_coll(sub_comm);
#endif
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
#else
  ExecutionContext& ec = gec;
#endif
    auto getsum = [&](const IndexVector& bid) {
      const IndexVector       blockid = internal::translate_blockid(bid, ltensor);
      const tamm::TAMM_SIZE   dsize   = tensor.block_size(blockid);
      std::vector<TensorType> dbuf(dsize);
      tensor.get(blockid, dbuf);
      for(TensorType val: dbuf) lsumsq += val;
    };
    block_for(ec, ltensor, getsum);

#ifdef TU_SG
    ec.flush_and_sync();
    // MemoryManagerGA::destroy_coll(mgr);
#if !defined(USE_UPCXX)
    MPI_Comm_free(&sub_comm);
#endif
    pg.destroy_coll();
  }
#if defined(USE_UPCXX)
  sub_comm->destroy();
#endif
#endif

  gec.pg().barrier();

  gsumsq = gec.pg().allreduce(&lsumsq, ReduceOp::sum);
  return gsumsq;
}

template<typename TensorType>
TensorType sum(Tensor<TensorType> tensor) {
  return sum(tensor());
}

template<typename TensorType>
TensorType norm_unused(LabeledTensor<TensorType> ltensor) {
  ExecutionContext&  ec = get_ec(ltensor);
  Scheduler          sch{ec};
  Tensor<TensorType> nval{};
  sch.allocate(nval);

  if constexpr(internal::is_complex_v<TensorType>) {
    auto ltconj = tamm::conj(ltensor);
    sch(nval() = ltconj() * ltensor).deallocate(ltconj).execute();
  }
  else sch(nval() = ltensor * ltensor).execute();

  auto rval = get_scalar(nval);
  sch.deallocate(nval).execute();

  return std::sqrt(rval);
}

template<typename TensorType>
TensorType norm(Tensor<TensorType> tensor) {
  ExecutionContext& gec = get_ec(tensor());
  return norm(gec, tensor());
}

template<typename TensorType>
TensorType norm(LabeledTensor<TensorType> ltensor) {
  ExecutionContext& gec = get_ec(ltensor);
  return norm(gec, ltensor);
}

template<typename TensorType>
TensorType norm(ExecutionContext& gec, Tensor<TensorType> tensor) {
  return norm(gec, tensor());
}

template<typename TensorType>
TensorType norm(ExecutionContext& gec, LabeledTensor<TensorType> ltensor) {
  // ExecutionContext& gec = get_ec(ltensor);

  TensorType         lsumsq = 0;
  TensorType         gsumsq = 0;
  Tensor<TensorType> tensor = ltensor.tensor();

#ifdef TU_SG
  int rank                   = gec.pg().rank().value();
  auto [nagg, ppn, subranks] = get_subgroup_info(gec, tensor);
#if defined(USE_UPCXX)
  upcxx::team* sub_comm =
    new upcxx::team(gec.pg().comm()->split(rank < subranks ? 0 : upcxx::team::color_none, 0));
#else
  MPI_Comm sub_comm;
  subcomm_from_subranks(gec, subranks, sub_comm);
#endif

  if(rank < subranks) {
#if defined(USE_UPCXX)
    ProcGroup pg = ProcGroup::create_coll(*sub_comm);
#else
    ProcGroup pg = ProcGroup::create_coll(sub_comm);
#endif

    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
#else
  ExecutionContext& ec = gec;
#endif

    auto getnorm = [&](const IndexVector& bid) {
      const IndexVector       blockid = internal::translate_blockid(bid, ltensor);
      const tamm::TAMM_SIZE   dsize   = tensor.block_size(blockid);
      std::vector<TensorType> dbuf(dsize);
      tensor.get(blockid, dbuf);
      if constexpr(std::is_same_v<TensorType, std::complex<double>> ||
                   std::is_same_v<TensorType, std::complex<float>>)
        for(TensorType val: dbuf) lsumsq += val * std::conj(val);
      else
        for(TensorType val: dbuf) lsumsq += val * val;
    };
    block_for(ec, ltensor, getnorm);

#ifdef TU_SG
    ec.flush_and_sync();
#if !defined(USE_UPCXX)
    MPI_Comm_free(&sub_comm);
#endif
    pg.destroy_coll();
  }
#if defined(USE_UPCXX)
  sub_comm->destroy();
#endif

#endif
  gec.pg().barrier();

  gsumsq = gec.pg().allreduce(&lsumsq, ReduceOp::sum);
  return std::sqrt(gsumsq);
}

// returns max_element, blockids, coordinates of max element in the block
template<typename TensorType, typename = std::enable_if_t<std::is_arithmetic_v<TensorType> ||
                                                          std::is_floating_point_v<TensorType>>>
std::tuple<TensorType, IndexVector, std::vector<size_t>> max_element(Tensor<TensorType> tensor) {
  return max_element(tensor());
}

template<typename TensorType, typename = std::enable_if_t<std::is_arithmetic_v<TensorType> ||
                                                          std::is_floating_point_v<TensorType>>>
std::tuple<TensorType, IndexVector, std::vector<size_t>>
max_element(LabeledTensor<TensorType> ltensor) {
  ExecutionContext& ec   = get_ec(ltensor);
  const auto        rank = ec.pg().rank().value();
  TensorType        max  = 0.0;

  Tensor<TensorType> tensor = ltensor.tensor();
  auto               nmodes = tensor.num_modes();
  // Works for only up to 6D tensors
  EXPECTS(tensor.num_modes() <= 6);

  IndexVector             maxblockid(nmodes);
  std::vector<size_t>     bfuv(nmodes);
  std::vector<TensorType> lmax(2, 0);
  std::vector<TensorType> gmax(2, 0);

  auto getmax = [&](const IndexVector& bid) {
    const IndexVector       blockid = internal::translate_blockid(bid, ltensor);
    const tamm::TAMM_SIZE   dsize   = tensor.block_size(blockid);
    std::vector<TensorType> dbuf(dsize);
    tensor.get(blockid, dbuf);
    auto block_dims   = tensor.block_dims(blockid);
    auto block_offset = tensor.block_offsets(blockid);

    size_t c = 0;

    if(nmodes == 1) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++, c++) {
        if(lmax[0] < dbuf[c]) {
          lmax[0]    = dbuf[c];
          lmax[1]    = rank;
          bfuv[0]    = i - block_offset[0];
          maxblockid = {blockid[0]};
        }
      }
    }
    else if(nmodes == 2) {
      auto dimi = block_offset[0] + block_dims[0];
      auto dimj = block_offset[1] + block_dims[1];
      for(size_t i = block_offset[0]; i < dimi; i++) {
        for(size_t j = block_offset[1]; j < dimj; j++, c++) {
          if(lmax[0] < dbuf[c]) {
            lmax[0]    = dbuf[c];
            lmax[1]    = rank;
            bfuv[0]    = i - block_offset[0];
            bfuv[1]    = j - block_offset[1];
            maxblockid = {blockid[0], blockid[1]};
          }
        }
      }
    }
    else if(nmodes == 3) {
      auto dimi = block_offset[0] + block_dims[0];
      auto dimj = block_offset[1] + block_dims[1];
      auto dimk = block_offset[2] + block_dims[2];

      for(size_t i = block_offset[0]; i < dimi; i++) {
        for(size_t j = block_offset[1]; j < dimj; j++) {
          for(size_t k = block_offset[2]; k < dimk; k++, c++) {
            if(lmax[0] < dbuf[c]) {
              lmax[0]    = dbuf[c];
              lmax[1]    = rank;
              bfuv[0]    = i - block_offset[0];
              bfuv[1]    = j - block_offset[1];
              bfuv[2]    = k - block_offset[2];
              maxblockid = {blockid[0], blockid[1], blockid[2]};
            }
          }
        }
      }
    }
    else if(nmodes == 4) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
            for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++, c++) {
              if(lmax[0] < dbuf[c]) {
                lmax[0]    = dbuf[c];
                lmax[1]    = rank;
                bfuv[0]    = i - block_offset[0];
                bfuv[1]    = j - block_offset[1];
                bfuv[2]    = k - block_offset[2];
                bfuv[3]    = l - block_offset[3];
                maxblockid = {blockid[0], blockid[1], blockid[2], blockid[3]};
              }
            }
          }
        }
      }
    }
    else if(nmodes == 5) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
            for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++) {
              for(size_t m = block_offset[4]; m < block_offset[4] + block_dims[4]; m++, c++) {
                if(lmax[0] < dbuf[c]) {
                  lmax[0]    = dbuf[c];
                  lmax[1]    = rank;
                  bfuv[0]    = i - block_offset[0];
                  bfuv[1]    = j - block_offset[1];
                  bfuv[2]    = k - block_offset[2];
                  bfuv[3]    = l - block_offset[3];
                  bfuv[4]    = m - block_offset[4];
                  maxblockid = {blockid[0], blockid[1], blockid[2], blockid[3], blockid[4]};
                }
              }
            }
          }
        }
      }
    }

    else if(nmodes == 6) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
            for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++) {
              for(size_t m = block_offset[4]; m < block_offset[4] + block_dims[4]; m++) {
                for(size_t n = block_offset[5]; n < block_offset[5] + block_dims[5]; n++, c++) {
                  if(lmax[0] < dbuf[c]) {
                    lmax[0]    = dbuf[c];
                    lmax[1]    = rank;
                    bfuv[0]    = i - block_offset[0];
                    bfuv[1]    = j - block_offset[1];
                    bfuv[2]    = k - block_offset[2];
                    bfuv[3]    = l - block_offset[3];
                    bfuv[4]    = m - block_offset[4];
                    bfuv[5]    = n - block_offset[5];
                    maxblockid = {blockid[0], blockid[1], blockid[2],
                                  blockid[3], blockid[4], blockid[5]};
                  }
                }
              }
            }
          }
        }
      }
    }
  };
  block_for(ec, ltensor, getmax);

  ec.pg().allreduce(lmax.data(), gmax.data(), 1, ReduceOp::maxloc);
  ec.pg().broadcast(maxblockid.data(), nmodes, gmax[1]);
  ec.pg().broadcast(bfuv.data(), nmodes, gmax[1]);

  return std::make_tuple(gmax[0], maxblockid, bfuv);
}

// returns min_element, blockids, coordinates of min element in the block
template<typename TensorType, typename = std::enable_if_t<std::is_arithmetic_v<TensorType> ||
                                                          std::is_floating_point_v<TensorType>>>
std::tuple<TensorType, IndexVector, std::vector<size_t>> min_element(Tensor<TensorType> tensor) {
  return min_element(tensor());
}

template<typename TensorType, typename = std::enable_if_t<std::is_arithmetic_v<TensorType> ||
                                                          std::is_floating_point_v<TensorType>>>
std::tuple<TensorType, IndexVector, std::vector<size_t>>
min_element(LabeledTensor<TensorType> ltensor) {
  ExecutionContext& ec   = get_ec(ltensor);
  const auto        rank = ec.pg().rank().value();
  TensorType        min  = 0.0;

  Tensor<TensorType> tensor = ltensor.tensor();
  auto               nmodes = tensor.num_modes();
  // Works for only up to 6D tensors
  EXPECTS(tensor.num_modes() <= 6);

  IndexVector             minblockid(nmodes);
  std::vector<size_t>     bfuv(nmodes);
  std::vector<TensorType> lmin(2, 0);
  std::vector<TensorType> gmin(2, 0);

  auto getmin = [&](const IndexVector& bid) {
    const IndexVector       blockid = internal::translate_blockid(bid, ltensor);
    const tamm::TAMM_SIZE   dsize   = tensor.block_size(blockid);
    std::vector<TensorType> dbuf(dsize);
    tensor.get(blockid, dbuf);
    auto   block_dims   = tensor.block_dims(blockid);
    auto   block_offset = tensor.block_offsets(blockid);
    size_t c            = 0;

    if(nmodes == 1) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++, c++) {
        if(lmin[0] > dbuf[c]) {
          lmin[0]    = dbuf[c];
          lmin[1]    = rank;
          bfuv[0]    = i - block_offset[0];
          minblockid = {blockid[0]};
        }
      }
    }
    else if(nmodes == 2) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++, c++) {
          if(lmin[0] > dbuf[c]) {
            lmin[0]    = dbuf[c];
            lmin[1]    = rank;
            bfuv[0]    = i - block_offset[0];
            bfuv[1]    = j - block_offset[1];
            minblockid = {blockid[0], blockid[1]};
          }
        }
      }
    }
    else if(nmodes == 3) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++, c++) {
            if(lmin[0] > dbuf[c]) {
              lmin[0]    = dbuf[c];
              lmin[1]    = rank;
              bfuv[0]    = i - block_offset[0];
              bfuv[1]    = j - block_offset[1];
              bfuv[2]    = k - block_offset[2];
              minblockid = {blockid[0], blockid[1], blockid[2]};
            }
          }
        }
      }
    }
    else if(nmodes == 4) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
            for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++, c++) {
              if(lmin[0] > dbuf[c]) {
                lmin[0]    = dbuf[c];
                lmin[1]    = rank;
                bfuv[0]    = i - block_offset[0];
                bfuv[1]    = j - block_offset[1];
                bfuv[2]    = k - block_offset[2];
                bfuv[3]    = l - block_offset[3];
                minblockid = {blockid[0], blockid[1], blockid[2], blockid[3]};
              }
            }
          }
        }
      }
    }
    else if(nmodes == 5) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
            for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++) {
              for(size_t m = block_offset[4]; m < block_offset[4] + block_dims[4]; m++, c++) {
                if(lmin[0] > dbuf[c]) {
                  lmin[0]    = dbuf[c];
                  lmin[1]    = rank;
                  bfuv[0]    = i - block_offset[0];
                  bfuv[1]    = j - block_offset[1];
                  bfuv[2]    = k - block_offset[2];
                  bfuv[3]    = l - block_offset[3];
                  bfuv[4]    = m - block_offset[4];
                  minblockid = {blockid[0], blockid[1], blockid[2], blockid[3], blockid[4]};
                }
              }
            }
          }
        }
      }
    }

    else if(nmodes == 6) {
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
            for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++) {
              for(size_t m = block_offset[4]; m < block_offset[4] + block_dims[4]; m++) {
                for(size_t n = block_offset[5]; n < block_offset[5] + block_dims[5]; n++, c++) {
                  if(lmin[0] > dbuf[c]) {
                    lmin[0]    = dbuf[c];
                    lmin[1]    = rank;
                    bfuv[0]    = i - block_offset[0];
                    bfuv[1]    = j - block_offset[1];
                    bfuv[2]    = k - block_offset[2];
                    bfuv[3]    = l - block_offset[3];
                    bfuv[4]    = m - block_offset[4];
                    bfuv[5]    = n - block_offset[5];
                    minblockid = {blockid[0], blockid[1], blockid[2],
                                  blockid[3], blockid[4], blockid[5]};
                  }
                }
              }
            }
          }
        }
      }
    }
  };
  block_for(ec, ltensor, getmin);

  ec.pg().allreduce(lmin.data(), gmin.data(), 1, ReduceOp::minloc);
  ec.pg().broadcast(minblockid.data(), nmodes, gmin[1]);
  ec.pg().broadcast(bfuv.data(), nmodes, gmin[1]);

  return std::make_tuple(gmin[0], minblockid, bfuv);
}

// regular 2D tamm tensor to block-cyclic tamm tensor
template<typename TensorType>
void to_block_cyclic_tensor(Tensor<TensorType> tensor, Tensor<TensorType> bc_tensor) {
  int ndims = tensor.num_modes();
  EXPECTS(ndims == 2);
  EXPECTS(bc_tensor.is_block_cyclic());
#if !defined(USE_UPCXX)
  auto ga_tens = bc_tensor.ga_handle();
#endif

  // bc_tensor might be on a smaller process group
  ExecutionContext& ec = get_ec(bc_tensor());

  auto tamm_bc_lambda = [&](const IndexVector& bid) {
    const IndexVector blockid = internal::translate_blockid(bid, tensor());

    auto block_dims   = tensor.block_dims(blockid);
    auto block_offset = tensor.block_offsets(blockid);

    const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
#if defined(USE_UPCXX)
    int64_t lo[2] = {cd_ncast<size_t>(block_offset[0]), cd_ncast<size_t>(block_offset[1])};
    int64_t hi[2] = {cd_ncast<size_t>(block_offset[0] + block_dims[0] - 1),
                     cd_ncast<size_t>(block_offset[1] + block_dims[1] - 1)};
    int64_t ld    = cd_ncast<size_t>(block_dims[1]);
#else
    std::vector<int64_t> lo(ndims), hi(ndims), ld(ndims - 1);

    for(size_t i = 0; i < ndims; i++) lo[i] = cd_ncast<size_t>(block_offset[i]);
    for(size_t i = 0; i < ndims; i++) hi[i] = cd_ncast<size_t>(block_offset[i] + block_dims[i] - 1);
    for(size_t i = 1; i < ndims; i++) ld[i - 1] = cd_ncast<size_t>(block_dims[i]);
#endif

    std::vector<TensorType> sbuf(dsize);
    tensor.get(blockid, sbuf);
#if defined(USE_UPCXX)
    bc_tensor.put_raw_contig(lo, hi, sbuf.data());
#else
    NGA_Put64(ga_tens, &lo[0], &hi[0], &sbuf[0], &ld[0]);
#endif
  };

  block_for(ec, tensor(), tamm_bc_lambda);
}

template<typename TensorType>
void from_block_cyclic_tensor(Tensor<TensorType> bc_tensor, Tensor<TensorType> tensor,
                              bool is_bc = true) {
  const auto ndims = bc_tensor.num_modes();
  EXPECTS(ndims == 2);
  if(is_bc) EXPECTS(bc_tensor.is_block_cyclic());
  EXPECTS(bc_tensor.kind() == TensorBase::TensorKind::dense);
  EXPECTS(bc_tensor.distribution().kind() == DistributionKind::dense);

#if !defined(USE_UPCXX)
  auto ga_tens = bc_tensor.ga_handle();
#endif

  // bc_tensor might be on a smaller process group
  ExecutionContext& ec = get_ec(bc_tensor());

  auto tamm_bc_lambda = [&](const IndexVector& bid) {
    const IndexVector blockid = internal::translate_blockid(bid, tensor());

    auto block_dims   = tensor.block_dims(blockid);
    auto block_offset = tensor.block_offsets(blockid);

    const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);

#if defined(USE_UPCXX)
    int64_t lo[2] = {cd_ncast<size_t>(block_offset[0]), cd_ncast<size_t>(block_offset[1])};
    int64_t hi[2] = {cd_ncast<size_t>(block_offset[0] + block_dims[0] - 1),
                     cd_ncast<size_t>(block_offset[1] + block_dims[1] - 1)};
    int64_t ld    = cd_ncast<size_t>(block_dims[1]);
#else
    std::vector<int64_t> lo(ndims), hi(ndims), ld(ndims - 1);

    for(size_t i = 0; i < ndims; i++) lo[i] = cd_ncast<size_t>(block_offset[i]);
    for(size_t i = 0; i < ndims; i++) hi[i] = cd_ncast<size_t>(block_offset[i] + block_dims[i] - 1);
    for(size_t i = 1; i < ndims; i++) ld[i - 1] = cd_ncast<size_t>(block_dims[i]);
#endif

    std::vector<TensorType> sbuf(dsize);
#if defined(USE_UPCXX)
    bc_tensor.get_raw_contig(lo, hi, sbuf.data());
#else
    NGA_Get64(ga_tens, &lo[0], &hi[0], &sbuf[0], &ld[0]);
#endif
    tensor.put(blockid, sbuf);
  };

  block_for(ec, tensor(), tamm_bc_lambda);
}

// convert dense tamm tensor to regular tamm tensor
template<typename TensorType>
void from_dense_tensor(Tensor<TensorType> d_tensor, Tensor<TensorType> tensor) {
  from_block_cyclic_tensor(d_tensor, tensor, false);
}

template<typename TensorType>
Tensor<TensorType> from_block_cyclic_tensor(Tensor<TensorType> bc_tensor) {
  // TiledIndexSpaceVec tis_vec={}, bool is_dense=true)

  const int ndims = bc_tensor.num_modes();
  EXPECTS(ndims == 2);
  EXPECTS(bc_tensor.is_block_cyclic());
  EXPECTS(bc_tensor.kind() == TensorBase::TensorKind::dense);
  EXPECTS(bc_tensor.distribution().kind() == DistributionKind::dense);

  //   if(!tis_vec.empty()) EXPECTS(tis_vec.size() == ndims);
  //   else tis_vec = bc_tensor.tiled_index_spaces();

  ExecutionContext& ec = get_ec(bc_tensor());
  // this routine works only when the tiled index spaces are exactly the same
  Tensor<TensorType> tensor{bc_tensor.tiled_index_spaces()};
  /*if(is_dense)*/ tensor.set_dense();
  Tensor<TensorType>::allocate(&ec, tensor);
  from_block_cyclic_tensor(bc_tensor, tensor);

  return tensor;
}

template<typename TensorType>
std::tuple<TensorType*, int64_t> access_local_block_cyclic_buffer(Tensor<TensorType> tensor) {
  EXPECTS(tensor.num_modes() == 2);
  int               gah = tensor.ga_handle();
  ExecutionContext& ec  = get_ec(tensor());
  TensorType*       lbufptr;
  int64_t           lbufsize;
  NGA_Access_block_segment64(gah, ec.pg().rank().value(), reinterpret_cast<void*>(&lbufptr),
                             &lbufsize);
  return std::make_tuple(lbufptr, lbufsize);
}

// permute a given tensor (TODO: does not work correctly for >=3D dense tensors)
template<typename TensorType>
Tensor<TensorType> permute_tensor(Tensor<TensorType> tensor, std::vector<int> permute) {
  auto tis   = tensor.tiled_index_spaces();
  int  ndims = tis.size();
  EXPECTS(!permute.empty() && ndims == permute.size());
  EXPECTS(std::all_of(permute.begin(), permute.end(), [&](int i) { return i >= 0 && i < ndims; }));

  std::vector<TiledIndexLabel> til(ndims), ptil(ndims);
  for(int i = 0; i < ndims; i++) til[i] = tis[i].label("all");
  for(int i = 0; i < ndims; i++) ptil[i] = til[permute[i]];

  Tensor<TensorType> ptensor{ptil};
  if(tensor.kind() == TensorBase::TensorKind::dense) ptensor.set_dense();
  ExecutionContext& ec = get_ec(tensor());
  Scheduler{ec}.allocate(ptensor)(ptensor(ptil) = tensor(til)).execute();

  return ptensor; // caller responsible for deallocating this tensor
}

// Extract block of a dense tensor given by [lo, hi)
template<typename TensorType>
Tensor<TensorType> tensor_block(Tensor<TensorType>& tensor, std::vector<int64_t> lo,
                                std::vector<int64_t> hi, std::vector<int> permute = {}) {
  const int ndims = tensor.num_modes();

  EXPECTS(tensor.kind() == TensorBase::TensorKind::dense);
  EXPECTS(tensor.distribution().kind() == DistributionKind::dense);

  auto tis = tensor.tiled_index_spaces();

  for(int i = 0; i < ndims; i++) EXPECTS(hi[i] <= tis[i].index_space().num_indices() && lo[i] >= 0);

  LabeledTensor<TensorType> ltensor = tensor();
  ExecutionContext&         ec      = get_ec(ltensor);
  std::vector<bool>         is_irreg_tis(ndims, false);

  for(int i = 0; i < ndims; i++) is_irreg_tis[i] = !tis[i].input_tile_sizes().empty();

  std::vector<std::vector<Tile>> tiles(ndims);
  for(int i = 0; i < ndims; i++)
    tiles[i] = is_irreg_tis[i] ? tis[i].input_tile_sizes()
                               : std::vector<Tile>{tis[i].input_tile_size()};

  std::vector<Tile> max_ts(ndims);
  for(int i = 0; i < ndims; i++)
    max_ts[i] = is_irreg_tis[i] ? *max_element(tiles[i].begin(), tiles[i].end()) : tiles[i][0];

  TiledIndexSpaceVec btis(ndims);
  for(int i = 0; i < ndims; i++) btis[i] = TiledIndexSpace{range(hi[i] - lo[i]), max_ts[i]};

  Tensor<TensorType> btensor{btis};
  btensor.set_dense();
  Tensor<TensorType>::allocate(&ec, btensor);

#if defined(USE_UPCXX)

  EXPECTS(ndims >= 1 && ndims <= 4);

  std::vector<int64_t> ld(ndims);

  for(int i = 0; i < ndims; ++i) {
    ld[i] = hi[i] - lo[i] - 1;
    hi[i]--;
  }

  // Pad to 4D
  for(int i = ndims; i < 4; ++i) {
    lo.insert(lo.begin(), 0);
    hi.insert(hi.begin(), 0);
    ld.insert(ld.begin(), 0);
  }

  std::vector<TensorType> sbuf(btensor.size());
  tensor.get_raw(lo.data(), hi.data(), sbuf.data());
  btensor.put_raw(std::vector<int64_t>(4, 0).data(), ld.data(), sbuf.data());

#else

  int btensor_gah = btensor.ga_handle();
  int tensor_gah  = tensor.ga_handle();

  char ctrans = 'N';

  int64_t lo_src[ndims], hi_src[ndims], lo_dst[ndims], hi_dst[ndims];
  for(int i = 0; i < ndims; i++) {
    lo_src[i] = lo[i];
    hi_src[i] = hi[i] - 1;
  }

  for(int i = 0; i < ndims; i++) {
    lo_dst[i] = 0;
    hi_dst[i] = hi_src[i] - lo_src[i];
  }

  int ga_pg_default = GA_Pgroup_get_default();
  GA_Pgroup_set_default(ec.pg().ga_pg());

  NGA_Copy_patch64(ctrans, tensor_gah, lo_src, hi_src, btensor_gah, lo_dst, hi_dst);

  GA_Pgroup_set_default(ga_pg_default);
#endif

  Tensor<TensorType> pbtensor = btensor;
  if(!permute.empty()) {
    pbtensor = permute_tensor<TensorType>(btensor, permute);
    Tensor<TensorType>::deallocate(btensor);
  }

  return pbtensor; // Caller responsible for deallocating this tensor
}

inline TiledIndexLabel compose_lbl(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
  auto lhs_tis = lhs.tiled_index_space();
  auto rhs_tis = rhs.tiled_index_space();

  auto res_tis = lhs_tis.compose_tis(rhs_tis);

  return res_tis.label("all");
}

inline TiledIndexSpace compose_tis(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
  return lhs.compose_tis(rhs);
}

inline TiledIndexLabel invert_lbl(const TiledIndexLabel& lhs) {
  auto lhs_tis = lhs.tiled_index_space().invert_tis();

  return lhs_tis.label("all");
}

inline TiledIndexSpace invert_tis(const TiledIndexSpace& lhs) { return lhs.invert_tis(); }

inline TiledIndexLabel intersect_lbl(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
  auto lhs_tis = lhs.tiled_index_space();
  auto rhs_tis = rhs.tiled_index_space();

  auto res_tis = lhs_tis.intersect_tis(rhs_tis);

  return res_tis.label("all");
}

inline TiledIndexSpace intersect_tis(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
  return lhs.intersect_tis(rhs);
}

inline TiledIndexLabel union_lbl(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
  auto lhs_tis = lhs.tiled_index_space();
  auto rhs_tis = rhs.tiled_index_space();

  auto res_tis = lhs_tis.union_tis(rhs_tis);

  return res_tis.label("all");
}

inline TiledIndexSpace union_tis(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
  return lhs.union_tis(rhs);
}

inline TiledIndexLabel project_lbl(const TiledIndexLabel& lhs, const TiledIndexLabel& rhs) {
  auto lhs_tis = lhs.tiled_index_space();
  auto rhs_tis = rhs.tiled_index_space();

  auto res_tis = lhs_tis.project_tis(rhs_tis);

  return res_tis.label("all");
}

inline TiledIndexSpace project_tis(const TiledIndexSpace& lhs, const TiledIndexSpace& rhs) {
  return lhs.project_tis(rhs);
}

/**
 * @brief uses a function to fill in elements of a tensor
 *
 * @tparam TensorType the type of the elements in the tensor
 * @param ec Execution context used in the blockfor
 * @param ltensor tensor to operate on
 * @param func function to fill in the tensor with
 */
template<typename TensorType>
inline size_t hash_tensor(Tensor<TensorType> tensor) {
  auto   ltensor = tensor();
  auto   ec      = tensor.execution_context();
  size_t hash    = tensor.num_modes();
  auto   lambda  = [&](const IndexVector& bid) {
    const IndexVector     blockid = internal::translate_blockid(bid, ltensor);
    const tamm::TAMM_SIZE dsize   = tensor.block_size(blockid);

    internal::hash_combine(hash, tensor.block_size(blockid));
    std::vector<TensorType> dbuf(dsize);
    tensor.get(blockid, dbuf);
    for(auto& val: dbuf) {
      if constexpr(internal::is_complex_v<TensorType>) internal::hash_combine(hash, std::abs(val));
      else internal::hash_combine(hash, val);
    }
  };
  block_for(*ec, ltensor, lambda);

  return hash;
}

// Convert regular tamm tensor to dense tamm tensor
template<typename TensorType>
Tensor<TensorType> to_dense_tensor(ExecutionContext& ec_dense, Tensor<TensorType> tensor) {
  EXPECTS(tensor.distribution().kind() == DistributionKind::nw);
  EXPECTS(tensor.kind() == TensorBase::TensorKind::normal ||
          tensor.kind() == TensorBase::TensorKind::spin);

  Tensor<TensorType> btensor{tensor.tiled_index_spaces()};
  btensor.set_dense();
  Tensor<TensorType>::allocate(&ec_dense, btensor);

  auto f = [&](const auto& blockid) {
    std::vector<TensorType> buffer(tensor.block_size(blockid));
    tensor.get(blockid, buffer);
    btensor.put(blockid, buffer);
  };

  block_for(ec_dense, tensor(), f);

  return btensor; // Caller responsible for deallocating this tensor
}

// Extract a single value specified by index_id from a dense tensor
template<typename TensorType>
TensorType get_tensor_element(Tensor<TensorType> tensor, std::vector<int64_t> index_id) {
  const int ndims = tensor.num_modes();
  EXPECTS(tensor.kind() == TensorBase::TensorKind::dense);

  TensorType val{};

#if defined(USE_UPCXX)
  EXPECTS(ndims >= 1 && ndims <= 4);

  std::vector<int64_t> lo(4, 0), hi(4, 0);

  for(int i = 4 - ndims, j = 0; i < 4; ++i, ++j) {
    lo[i] = index_id[j];
    hi[i] = index_id[j];
  }

  tensor.get_raw_contig(lo.data(), hi.data(), &val);
#else
  std::vector<int64_t> lo(ndims), hi(ndims);
  std::vector<int64_t> ld(ndims - 1, 1);

  for(int i = 0; i < ndims; i++) {
    lo[i] = index_id[i];
    hi[i] = index_id[i];
  }

  NGA_Get64(tensor.ga_handle(), lo.data(), hi.data(), &val, ld.data());
#endif

  return val;
};

/**
 * @brief Prints a dense Tensor object
 *
 * @tparam T template type for Tensor element type
 * @param [in] tensor input Tensor object
 */
template<typename T>
void print_dense_tensor(const Tensor<T>& tensor, std::function<bool(std::vector<size_t>)> func,
                        std::string filename = "", bool append = false) {
  auto              lt    = tensor();
  int               ndims = tensor.num_modes();
  ExecutionContext& ec    = get_ec(lt);

  auto nz_check = [=](const T val) {
    if constexpr(tamm::internal::is_complex_v<T>) {
      if(val.real() > 1e-12 || val.real() < -1e-12) return true;
    }
    else if(val > 1e-12 || val < -1e-12) return true;
    return false;
  };

  if(ec.pg().rank() == 0) {
    std::stringstream tstring;
    EXPECTS(ndims >= 1 && ndims <= 4);

    for(auto it: tensor.loop_nest()) {
      auto blockid = internal::translate_blockid(it, lt);

      if(!tensor.is_non_zero(blockid)) continue;

      TAMM_SIZE      size = tensor.block_size(blockid);
      std::vector<T> buf(size);
      tensor.get(blockid, buf);
      auto block_dims   = tensor.block_dims(blockid);
      auto block_offset = tensor.block_offsets(blockid);

      tstring << std::fixed << std::setprecision(10);

      size_t c = 0;
      if(ndims == 1) {
        for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++, c++) {
          if(func({i}) && nz_check(buf[c])) tstring << i + 1 << "   " << buf[c] << std::endl;
        }
      }
      else if(ndims == 2) {
        for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
          for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++, c++) {
            if(func({i, j}) && nz_check(buf[c]))
              tstring << i + 1 << "   " << j + 1 << "   " << buf[c] << std::endl;
          }
        }
      }
      else if(ndims == 3) {
        for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
          for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
            for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++, c++) {
              if(func({i, j, k}) && nz_check(buf[c]))
                tstring << i + 1 << "   " << j + 1 << "   " << k + 1 << "   " << buf[c]
                        << std::endl;
            }
          }
        }
      }
      else if(ndims == 4) {
        for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
          for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
            for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
              for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++, c++) {
                if(func({i, j, k, l}) && nz_check(buf[c]))
                  tstring << i + 1 << "   " << j + 1 << "   " << k + 1 << "   " << l + 1 << "   "
                          << buf[c] << std::endl;
              }
            }
          }
        }
      }
    }

    if(!filename.empty()) {
      std::ofstream tos;
      if(append) tos.open(filename + ".txt", std::ios::app);
      else tos.open(filename + ".txt", std::ios::out);
      if(!tos) std::cerr << "Error opening file " << filename << std::endl;
      tos << tstring.str() << std::endl;
      tos.close();
    }
    else std::cout << tstring.str();
  }
  ec.pg().barrier();
}

template<typename T>
void print_dense_tensor(const Tensor<T>& tensor, std::string filename = "") {
  std::function<bool(std::vector<size_t>)> func = [&](std::vector<size_t> cond) { return true; };
  print_dense_tensor(tensor, func, filename);
}

template<typename T>
void print_memory_usage(const int64_t rank, std::string mstring = "") {
  auto& memprof = tamm::MemProfiler::instance();

  auto mem_to_string = [&](double mem_size) {
    return std::to_string((mem_size * sizeof(T)) / 1073741824.0) + " GiB";
  };

  if(rank == 0) {
    if(mstring.empty()) mstring = "Memory stats";
    std::cout << mstring << std::endl << std::string(mstring.length(), '-') << std::endl;
    std::cout << "allocation count: " << memprof.alloc_counter << std::endl;
    std::cout << "deallocation count: " << memprof.dealloc_counter << std::endl;
    std::cout << "total memory allocated: " << mem_to_string(memprof.mem_allocated) << std::endl;
    std::cout << "total memory deallocated: " << mem_to_string(memprof.mem_deallocated)
              << std::endl;
    std::cout << "maximum memory in single allocation: "
              << mem_to_string(memprof.max_in_single_allocate) << std::endl;
    std::cout << "maximum memory consumption: " << mem_to_string(memprof.max_total_allocated)
              << std::endl;
  }
}

} // namespace tamm
