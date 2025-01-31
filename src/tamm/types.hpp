// Copyright 2016 Pacific Northwest National Laboratory

#pragma once

#include "ga/ga-mpi.h"
#include "ga/ga.h"
#include "tamm/boundvec.hpp"
#include "tamm/errors.hpp"
#include "tamm/strong_num.hpp"
#include <complex>
#include <iosfwd>
#include <map>
#if defined(USE_UPCXX)
#include <upcxx/upcxx.hpp>
#endif

//#include <mpi.h>

namespace tamm {

// Free functions
#if __cplusplus < 201703L
namespace internal {
template<class F, class Tuple, std::size_t... I>
constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
  return f(std::get<I>(std::forward<Tuple>(t))...);
}
} // namespace internal

template<class F, class Tuple>
constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return internal::apply_impl(
    std::forward<F>(f), std::forward<Tuple>(t),
    std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}
#endif
// End Free functions

namespace internal {
template<typename T>
void hash_combine(size_t& seed, T const& v) {
  seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
} // namespace internal

using TAMM_SIZE = uint64_t;
// IndexSpace related type definitions
using Index         = uint32_t;
using IndexVector   = std::vector<Index>;
using IndexIterator = std::vector<Index>::const_iterator;
using Tile          = uint32_t;
// DAG related Hash
using HashData       = uint64_t;
using StringLabelVec = std::vector<std::string>;

class TiledIndexSpace;
using TiledIndexSpaceVec = std::vector<TiledIndexSpace>;
class TiledIndexLabel;
using IndexLabelVec = std::vector<TiledIndexLabel>;

using Perm       = int32_t;
using PermVector = std::vector<Perm>;

//////////////////////////////////
struct IrrepSpace;
using Irrep = StrongNum<IrrepSpace, uint32_t>;
struct SpinSpace;
using Spin = StrongNum<SpinSpace, uint32_t>;
struct SpatialSpace;
using Spatial    = StrongNum<SpatialSpace, uint32_t>;
using TensorRank = size_t;
struct OffsetSpace;
using Offset = StrongNum<OffsetSpace, uint64_t>;
struct BlockIndexSpace;
using BlockIndex = StrongNum<BlockIndexSpace, uint64_t>;
struct ProcSpace;
using Proc = StrongNum<ProcSpace, int64_t>;
struct SignSpace;
using Sign = StrongNum<SignSpace, int32_t>;

// these are typedefs for usability
using Size       = Offset;
using BlockCount = BlockIndex;
using Label      = int; // needs to support negative values

using IntLabel    = int32_t; // a simple integer label for indices
using IntLabelVec = std::vector<IntLabel>;

using SizeVec  = std::vector<Size>;
using ProcGrid = std::vector<Proc>;
using ProcList = std::vector<int>;

enum class AllocationStatus { invalid, created, attached, deallocated, orphaned };

enum class ElementType {
  invalid,
  single_precision,
  double_precision,
  single_complex,
  double_complex
};

enum class DistributionKind { invalid, nw, dense, simple_round_robin, view, unit_tile };

enum class MemoryManagerKind { invalid, ga, local };

template<typename T>
constexpr ElementType tensor_element_type() {
  return ElementType::invalid;
}

template<>
constexpr ElementType tensor_element_type<double>() {
  return ElementType::double_precision;
}

template<>
constexpr ElementType tensor_element_type<float>() {
  return ElementType::single_precision;
}

template<>
constexpr ElementType tensor_element_type<std::complex<float>>() {
  return ElementType::single_complex;
}

template<>
constexpr ElementType tensor_element_type<std::complex<double>>() {
  return ElementType::double_complex;
}

static inline constexpr size_t element_size(ElementType eltype) {
  size_t ret = 0;
  switch(eltype) {
    case ElementType::single_precision: ret = sizeof(float); break;
    case ElementType::double_precision: ret = sizeof(double); break;
    case ElementType::single_complex: ret = sizeof(std::complex<float>); break;
    case ElementType::double_complex: ret = sizeof(std::complex<double>); break;
    default: UNREACHABLE();
  }
  return ret;
}

const TensorRank maxrank{8};

template<typename T>
using TensorVec = BoundVec<T, maxrank>;

using BlockDimVec = TensorVec<BlockIndex>;
// using DimTypeVec = TensorVec<DimType>;
using PermVec = TensorVec<uint32_t>;

///////////

///////////////////

using RangeValue = int64_t;

// enum class IndexSpaceType { mo, mso, ao, aso, aux };
enum class SpinPosition { ignore, upper, lower };
enum class IndexPosition { upper, lower, neither };

enum class SpinType { ao_spin, mo_spin };

enum class ExecutionHW { CPU, GPU, DEFAULT };

enum class ReduceOp { min, max, sum, maxloc, minloc };

using SpinMask = std::vector<SpinPosition>;

#if defined(USE_UPCXX)
using rtDataHandlePtr = upcxx::future<>*;
using rtDataHandle    = upcxx::future<>;
#else
using rtDataHandlePtr = ga_nbhdl_t*;
using rtDataHandle    = ga_nbhdl_t;
#endif

class DataCommunicationHandle {
public:
  DataCommunicationHandle()  = default;
  ~DataCommunicationHandle() = default;

  void waitForCompletion() {
    if(!getCompletionStatus()) {
#if defined(USE_UPCXX)
      data_handle_.wait();
#else
      NGA_NbWait(&data_handle_);
#endif
      setCompletionStatus();
    }
  }
  void setCompletionStatus() { status_ = true; }
  void resetCompletionStatus() { status_ = false; }

  bool getCompletionStatus() {
    /*
    if(status_ == false)
        status_ = NGA_NbTest(&data_handle_);
    */

    return status_;
  }
  rtDataHandlePtr getDataHandlePtr() { return &data_handle_; }

  rtDataHandle data_handle_;

private:
  bool status_{true};
};

using DataCommunicationHandlePtr = DataCommunicationHandle*;

//////////////////

// namespace SpinType {
// const Spin alpha{1};
// const Spin beta{2};
// }; // namespace SpinType

#if !defined(USE_UPCXX)
template<typename T>
static inline MPI_Datatype mpi_type() {
  using std::is_same_v;

  if constexpr(is_same_v<int, T>) return MPI_INT;
  else if constexpr(is_same_v<bool, T>) return MPI_C_BOOL;
  else if constexpr(is_same_v<char, T>) return MPI_CHAR;
  else if constexpr(is_same_v<int64_t, T>) return MPI_INT64_T;
  else if constexpr(is_same_v<uint32_t, T>) return MPI_UNSIGNED;
  else if constexpr(is_same_v<size_t, T>) return MPI_UNSIGNED_LONG;
  else if constexpr(is_same_v<float, T>) return MPI_FLOAT;
  else if constexpr(is_same_v<double, T>) return MPI_DOUBLE;
  else if constexpr(is_same_v<std::complex<float>, T>) return MPI_COMPLEX;
  else if constexpr(is_same_v<std::complex<double>, T>) return MPI_DOUBLE_COMPLEX;
  else NOT_IMPLEMENTED(); // unhandled type
}

template<typename T>
static inline MPI_Datatype mpi_type_loc() {
  using std::is_same_v;

  if constexpr(is_same_v<int, T>) return MPI_2INT;
  else if constexpr(is_same_v<float, T>) return MPI_2REAL;
  else if constexpr(is_same_v<double, T>) return MPI_2DOUBLE_PRECISION;
  else NOT_IMPLEMENTED(); // unhandled type
}

static inline MPI_Op mpi_op(ReduceOp rop) {
  if(rop == ReduceOp::min) return MPI_MIN;
  else if(rop == ReduceOp::max) return MPI_MAX;
  else if(rop == ReduceOp::sum) return MPI_SUM;
  else if(rop == ReduceOp::minloc) return MPI_MINLOC;
  else if(rop == ReduceOp::maxloc) return MPI_MAXLOC;
  else NOT_IMPLEMENTED(); // unhandled op
}
#endif

namespace internal {
template<typename T, typename... Args>
void unfold_vec(std::vector<T>& v, Args&&... args) {
  static_assert((std::is_constructible_v<T, Args&&> && ...));
  (v.push_back(std::forward<Args>(args)), ...);
}
} // namespace internal

inline Label make_label() {
  static Label lbl = 0;
  return lbl++;
}

/**
 * Convert a TAMM element type to a GA element type
 * @param eltype TAMM element type
 * @return Corresponding GA element type
 */
static int to_ga_eltype(ElementType eltype) {
  int ret;
  switch(eltype) {
    case ElementType::single_precision: ret = C_FLOAT; break;
    case ElementType::double_precision: ret = C_DBL; break;
    case ElementType::single_complex: ret = C_SCPL; break;
    case ElementType::double_complex: ret = C_DCPL; break;
    // case ElementType::invalid: ret = 0; break;
    default: ret = 0; UNREACHABLE();
  }
  return ret;
}

/**
 * Convert a GA element type to a TAMM element type
 * @param eltype GA element type
 * @return Corresponding TAMM element type
 */
static ElementType from_ga_eltype(int eltype) {
  ElementType ret;
  switch(eltype) {
    case C_FLOAT: ret = ElementType::single_precision; break;
    case C_DBL: ret = ElementType::double_precision; break;
    case C_SCPL: ret = ElementType::single_complex; break;
    case C_DCPL: ret = ElementType::double_complex; break;
    default: ret = ElementType::invalid; UNREACHABLE();
  }
  return ret;
}

inline constexpr const char* element_type_to_string(ElementType eltype) {
  switch(eltype) {
    case ElementType::invalid: return "inv"; break;
    // case ElType::i32: return "i32"; break;
    // case ElType::i64: return "i64"; break;
    case ElementType::single_precision: return "f32"; break;
    case ElementType::double_precision: return "f64"; break;
    case ElementType::single_complex: return "cf32"; break;
    case ElementType::double_complex: return "cf64"; break;
  }
  return "NaN";
}

enum class ElType {
  inv   = 0b1000,
  i32   = 0b0000,
  i64   = 0b0001,
  fp32  = 0b0010,
  fp64  = 0b0011,
  cfp32 = 0b0110,
  cfp64 = 0b0111
};

template<typename T>
inline constexpr ElType eltype = ElType::inv;

template<>
inline constexpr ElType eltype<int32_t> = ElType::i32;

template<>
inline constexpr ElType eltype<int64_t> = ElType::i64;

template<>
inline constexpr ElType eltype<float> = ElType::fp32;

template<>
inline constexpr ElType eltype<double> = ElType::fp64;

template<>
inline constexpr ElType eltype<std::complex<float>> = ElType::cfp32;

template<>
inline constexpr ElType eltype<std::complex<double>> = ElType::cfp64;

inline constexpr const char* eltype_to_string(ElType eltype) {
  switch(eltype) {
    case ElType::inv: return "inv"; break;
    case ElType::i32: return "i32"; break;
    case ElType::i64: return "i64"; break;
    case ElType::fp32: return "f32"; break;
    case ElType::fp64: return "f64"; break;
    case ElType::cfp32: return "cf32"; break;
    case ElType::cfp64: return "cf64"; break;
  }
  return "NaN";
}

inline constexpr ElType lub(ElType first, ElType second) {
  return ElType(static_cast<int>(first) | static_cast<int>(second));
}

template<class... Ts>
struct overloaded: Ts... {
  using Ts::operator()...;
};
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

using SymbolTable = std::map<void*, std::string>;

using TranslateFunc = std::function<Index(Index id)>;

// Custom hash function for IndexVector/BlockId
struct IndexVectorHash {
  std::size_t operator()(const IndexVector& vec) const {
    std::size_t seed = vec.size();
    for(Index v: vec) { internal::hash_combine(seed, v); }
    return seed;
  }
};

// Custom equality function for IndexVector/BlockId
struct IndexVectorEqual {
  bool operator()(const IndexVector& lhs, const IndexVector& rhs) const { return lhs == rhs; }
};

} // namespace tamm
