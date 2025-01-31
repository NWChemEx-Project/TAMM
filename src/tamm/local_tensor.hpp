#pragma once

#include "tamm/tensor.hpp"

namespace tamm {

/// @brief Creates a local copy of the distributed tensor
/// @tparam T Data type for the tensor being made local
template<typename T>
class LocalTensor: public Tensor<T> { // move to another hpp
public:
  LocalTensor()                              = default;
  LocalTensor(LocalTensor&&)                 = default;
  LocalTensor(const LocalTensor&)            = default;
  LocalTensor& operator=(LocalTensor&&)      = default;
  LocalTensor& operator=(const LocalTensor&) = default;
  ~LocalTensor()                             = default;

  // LocalTensor(Tensor<T> dist_tensor): dist_tensor_(dist_tensor) { construct_local_tensor(); }

  LocalTensor(std::initializer_list<TiledIndexSpace> tiss):
    Tensor<T>(construct_local_tis_vec(TiledIndexSpaceVec(tiss))) {}

  LocalTensor(std::vector<TiledIndexSpace> tiss): Tensor<T>(construct_local_tis_vec(tiss)) {}

  LocalTensor(std::initializer_list<TiledIndexLabel> tis_labels):
    Tensor<T>(construct_local_tis_vec(IndexLabelVec(tis_labels))) {}

  LocalTensor(std::initializer_list<size_t> dim_sizes):
    Tensor<T>(construct_tis_vec(std::vector<size_t>(dim_sizes))) {}

  LocalTensor(std::vector<size_t> dim_sizes): Tensor<T>(construct_tis_vec(dim_sizes)) {}

  /// @brief Operator overload for constructing a LabeledTensor object with list of labels
  /// @tparam ...Args Variadic template for set of input labels
  /// @param ...rest list of labels
  /// @return a LabeledTensor object created with corresponding labels
  template<class... Args>
  LabeledTensor<T> operator()(Args&&... rest) const {
    return LabeledTensor<T>{*this, std::forward<Args>(rest)...};
  }

  /// @brief  Initialized the LocalTensor object with given value
  /// @param val  input value used to set all values in LocalTensor
  void init(T val) {
    EXPECTS_STR(this->is_allocated(), "LocalTensor has to be allocated");

    auto ec = this->execution_context();
    Scheduler{*ec}((*this)() = val).execute();
  }

  /// @brief  An element-wise set operation that uses indices and a value
  /// @param indices  indices for the specific element in the LocalTensor
  /// @param val  the value for to be set
  void set(std::vector<size_t> indices, T val) {
    EXPECTS_STR(this->is_allocated(), "LocalTensor has to be allocated");
    EXPECTS_STR(indices.size() == this->num_modes(),
                "Number of indices must match the number of dimensions");
    size_t linearIndex = compute_linear_index(indices);

    this->access_local_buf()[linearIndex] = val;
  }

  /// @brief An element-wise get operation that returns the value at the specific location
  /// @param indices  the location is described as a set of indices in the LocalTensor
  /// @return the value from the specific location in the LocalTensor
  T get(const std::vector<size_t>& indices) const {
    EXPECTS_STR(indices.size() == this->num_modes(),
                "Number of indices must match the number of dimensions");
    size_t linearIndex = compute_linear_index(indices);

    return this->access_local_buf()[linearIndex];
  }

  /// @brief  An element-wise get operation that returns the value at the specific location
  /// @tparam ...Args variadic template for having the input indices for the location on the
  /// LocalTensor
  /// @param ...args list of indices for representing the specific location
  /// @return the value that is in the input location
  template<typename... Args>
  T get(Args... args) {
    std::vector<size_t> indices;
    unpack(indices, args...);
    EXPECTS_STR(indices.size() == this->num_modes(),
                "Number of indices must match the number of dimensions");
    size_t linearIndex = compute_linear_index(indices);

    return this->access_local_buf()[linearIndex];
  }

  /// @brief Method for resizing the reference LocalTensor
  /// @tparam ...Args variadic template for having the sizes for each dimension
  /// @param ...args the list of sizes for the corresponding re-size operation
  template<typename... Args>
  void resize(Args... args) {
    std::vector<size_t> new_sizes;
    unpack(new_sizes, args...);
    EXPECTS_STR(new_sizes.size() == (*this).num_modes(),
                "Number of new sizes must match the number of dimensions");
    resize(std::vector<size_t>{new_sizes});
  }

  /// @brief  Method for resizing the reference LocalTensor
  /// @param new_sizes  a list of new sizes for the resized LocalTensor
  void resize(const std::vector<size_t>& new_sizes) {
    EXPECTS_STR((*this).is_allocated(), "LocalTensor has to be allocated!");
    auto num_dims = (*this).num_modes();
    EXPECTS_STR(num_dims == new_sizes.size(),
                "Number of new sizes must match the number of dimensions.");

    for(size_t i = 0; i < new_sizes.size(); i++) {
      EXPECTS_STR(new_sizes[i] != 0, "New size should be larger than 0.");
    }

    LocalTensor<T> resizedTensor;

    auto dimensions = (*this).dim_sizes();

    if(dimensions == new_sizes) return;

    if(isWithinOldDimensions(new_sizes)) {
      std::vector<size_t> offsets(new_sizes.size(), 0);
      resizedTensor = (*this).block(offsets, new_sizes);
    }
    else {
      resizedTensor = LocalTensor<T>{new_sizes};
      resizedTensor.allocate((*this).execution_context());
      (*this).copy_to_bigger(resizedTensor);
    }

    auto old_tensor = (*this);
    (*this)         = resizedTensor;
    old_tensor.deallocate();
  }

  /**
   * @brief Method for filling the local tensor data with the original distributed tensor. We first
   * construct a loop nest and to a get on all blocks that are then written to the corresponding
   * place in the new local tensor
   *
   * @param dist_tensor Distributed source tensor to copy
   */
  void from_distributed_tensor(const Tensor<T>& dist_tensor) {
    for(const auto& blockid: dist_tensor.loop_nest()) {
      const tamm::TAMM_SIZE size = dist_tensor.block_size(blockid);
      std::vector<T>        buf(size);
      dist_tensor.get(blockid, buf);
      auto block_dims   = dist_tensor.block_dims(blockid);
      auto block_offset = dist_tensor.block_offsets(blockid);
      patch_copy_local(buf, block_dims, block_offset, true);
    }
  }

  /**
   * @brief Method for filling the original distributed tensor data with the local tensor. We first
   * construct a loop nest and to a get on all blocks that are then written to the corresponding
   * place in the distributed tensor
   *
   * @param dist_tensor Distributed destination tensor to copy to
   */
  void to_distributed_tensor(Tensor<T>& dist_tensor) {
    for(const auto& blockid: dist_tensor.loop_nest()) {
      const tamm::TAMM_SIZE size = dist_tensor.block_size(blockid);
      std::vector<T>        buf(size);
      dist_tensor.get(blockid, buf);
      auto block_dims   = dist_tensor.block_dims(blockid);
      auto block_offset = dist_tensor.block_offsets(blockid);
      patch_copy_local(buf, block_dims, block_offset, false);
      dist_tensor.put(blockid, buf);
    }
  }

  /// @brief A helper method that copy a block of that to a corresponding patch in the local copy
  /// @param sbuf Block data that wants to be copied
  /// @param block_dims Block dimensions to find the accurate location in the linearized local
  /// tensor
  /// @param block_offset The offsets of the input data from the original multidimensional tensor
  void patch_copy_local(std::vector<T>& sbuf, const std::vector<size_t>& block_dims,
                        const std::vector<size_t>& block_offset, bool copy_to_local) {
    auto num_dims = this->num_modes();
    // Compute the total number of elements to copy
    size_t total_elements = 1;
    for(size_t dim: block_dims) { total_elements *= dim; }
    // Initialize indices to the starting offset
    std::vector<size_t> indices(block_offset);

    for(size_t c = 0; c < total_elements; ++c) {
      size_t linearIndex = compute_linear_index(indices);

      // Access the tensor element at the current indices
      if(copy_to_local) this->access_local_buf()[linearIndex] = sbuf[c];
      else sbuf[c] = this->access_local_buf()[linearIndex];

      // Increment indices
      for(int dim = num_dims - 1; dim >= 0; --dim) {
        if(++indices[dim] < block_offset[dim] + block_dims[dim]) { break; }
        indices[dim] = block_offset[dim];
      }
    }
  }

  /// @brief  Method for applying the copy operation from a smaller LocalTensor to a bigger
  /// LocalTensor used for re-sizing
  /// @param bigger_tensor the reference tensor
  void copy_to_bigger(LocalTensor& bigger_tensor) const {
    auto smallerDims = (*this).dim_sizes();

    // Helper lambda to iterate over all indices of a tensor
    auto iterateIndices = [](const std::vector<size_t>& dims) {
      std::vector<size_t> indices(dims.size(), 0);
      bool                done = false;
      return [=]() mutable {
        if(done) return std::vector<size_t>();
        auto current = indices;
        for(int i = indices.size() - 1; i >= 0; --i) {
          if(++indices[i] < dims[i]) break;
          if(i == 0) {
            done = true;
            break;
          }
          indices[i] = 0;
        }
        return current;
      };
    };

    auto smallerIt = iterateIndices(smallerDims);
    while(true) {
      auto indices = smallerIt();
      if(indices.empty()) break;
      auto bigIndices = indices;
      bigger_tensor.set(bigIndices, (*this).get(indices));
    }
  }

  /// @brief Method for extracting a block from a multi-dimensional LocalTensor
  /// @param start_offsets the list of offsets corresponding to the start of the block for each
  /// dimension
  /// @param span_sizes the list of span sizes from the start for each dimension
  /// @return returns a new LocalTensor that is allocated and contains the values from the reference
  LocalTensor<T> block(const std::vector<size_t>& start_offsets,
                       const std::vector<size_t>& span_sizes) const {
    EXPECTS_STR((*this).is_allocated(), "LocalTensor has to be allocated!");
    auto num_dims = (*this).num_modes();
    EXPECTS_STR(num_dims == start_offsets.size(),
                "Number of start offsets should match the number of dimensions.");
    EXPECTS_STR(num_dims == span_sizes.size(),
                "Number of span sizes should match the number of dimensions.");

    // Create a local tensor for the block
    LocalTensor<T> blockTensor{span_sizes};
    blockTensor.allocate(this->execution_context());

    // Iterate over all dimensions to copy the block
    std::vector<size_t> indices(num_dims, 0);
    std::vector<size_t> source_indices = start_offsets;

    bool done = false;
    while(!done) {
      // Copy the element
      blockTensor.set(indices, (*this).get(source_indices));

      // Update indices
      done = true;
      for(size_t i = 0; i < num_dims; ++i) {
        if(++indices[i] < span_sizes[i]) {
          ++source_indices[i];
          done = false;
          break;
        }
        else {
          indices[i]        = 0;
          source_indices[i] = start_offsets[i];
        }
      }
    }

    return blockTensor;
  }

  /// @brief  Method for extracting a block from a 2 dimensional LocalTensor
  /// @param x_offset the starting x-axis offset for the block
  /// @param y_offset the starting y-axis offset for the block
  /// @param x_span the span of the block for x-axis
  /// @param y_span the span of the block for y-axis
  /// @return returns a new LocalTensor that is allocated and contains the values from the reference
  LocalTensor<T> block(size_t x_offset, size_t y_offset, size_t x_span, size_t y_span) const {
    auto num_dims = (*this).num_modes();
    EXPECTS_STR(num_dims == 2, "This block method only works for 2-D tensors!");

    return block({x_offset, y_offset}, {x_span, y_span});
  }

  /// @brief  Method for getting the dimension sizes of the LocalTensor
  /// @return a vector of sizes that corresponds to the size of each dimension in LocalTensor
  std::vector<size_t> dim_sizes() const {
    std::vector<size_t> dimensions;

    for(const auto& tis: (*this).tiled_index_spaces()) {
      dimensions.push_back(tis.max_num_indices());
    }

    return dimensions;
  }

private:
  /// @brief  Method for contructing single tiled TiledIndexSpaces with the input TiledIndexSpaces
  /// @param tiss the input TiledIndexSpaces used for construction of LocalTensor
  /// @return a vector of TiledIndexSpaces with single tiles from the input TiledIndexSpaces
  TiledIndexSpaceVec construct_local_tis_vec(std::vector<TiledIndexSpace> tiss) {
    std::vector<size_t> dim_sizes;

    for(const auto& tis: tiss) { dim_sizes.push_back(tis.max_num_indices()); }

    return construct_tis_vec(dim_sizes);
  }

  /// @brief Method for contructing single tiled TiledIndexSpaces with the input labels
  /// @param tis_labels the input labels used for construction of LocalTensor
  /// @return a vector of TiledIndexSpaces with single tiles from the input labels
  TiledIndexSpaceVec construct_local_tis_vec(std::vector<TiledIndexLabel> tis_labels) {
    std::vector<size_t> dim_sizes;

    for(const auto& tis_label: tis_labels) {
      dim_sizes.push_back(tis_label.tiled_index_space().max_num_indices());
    }

    return construct_tis_vec(dim_sizes);
  }

  /// @brief Method for constructing single tiled TiledIndexSpaces with the given input sizes
  /// @param dim_sizes  the input sizes for each dimension
  /// @return a vector of TiledIndexSpaces corresponding to the input dimension sizes
  TiledIndexSpaceVec construct_tis_vec(std::vector<size_t> dim_sizes) {
    TiledIndexSpaceVec local_tis_vec;
    for(const auto& dim_size: dim_sizes) {
      local_tis_vec.push_back(
        TiledIndexSpace{IndexSpace{range(dim_size)}, static_cast<Tile>(dim_size)});
    }

    return local_tis_vec;
  }

  /// @brief Method for constructing the linearized index for a given location on the local tensor
  /// @param indices The index for the corresponding location wanted to be accessed
  /// @return The linear position to the local memory manager
  size_t compute_linear_index(const std::vector<size_t>& indices) const {
    auto                num_modes = this->num_modes();
    std::vector<size_t> dims      = (*this).dim_sizes();
    size_t              index     = 0;
    size_t              stride    = 1;

    for(size_t i = 0; i < num_modes; ++i) {
      index += indices[num_modes - 1 - i] * stride;
      stride *= dims[num_modes - 1 - i];
    }

    return index;
  }

  /// @brief  Function checks if a given set of indices are within the old dimensions of the
  /// LocalTensor that is being resized
  /// @param indices the indices describing the exact location in the LocalTensor
  /// @return true if the indices are in the old dimensions
  bool isWithinOldDimensions(const std::vector<size_t>& indices) const {
    std::vector<size_t> dimensions = (*this).dim_sizes();

    for(size_t i = 0; i < indices.size(); ++i) {
      if(indices[i] > dimensions[i]) { return false; }
    }
    return true;
  }

  /// @brief Helper method that will unpack the variadic template for operator()
  /// @param indices A reference to the vector of indices
  /// @param index The last index that is provided to the operator()
  void unpack(std::vector<size_t>& indices, size_t index) { indices.push_back(index); }

  /// @brief Helper method that will unpack the variadic template for operator()
  /// @tparam ...Args The variadic template from the arguments to the operator()
  /// @param indices A reference to the vector of indices
  /// @param next Unpacked index for the operator()
  /// @param ...rest The rest of the variadic template that will be unpacked in the recursive calls
  template<typename... Args>
  void unpack(std::vector<size_t>& indices, size_t next, Args... rest) {
    indices.push_back(next);
    unpack(indices, rest...);
  }
};

} // namespace tamm
