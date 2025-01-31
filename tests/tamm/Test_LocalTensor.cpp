#include <chrono>
#include <tamm/tamm.hpp>
#include <tamm/tamm_git.hpp>

using namespace tamm;

bool check_local_tis_sizes(const TiledIndexSpace& l_tis, size_t expected_size) {
  return (l_tis.max_num_indices() == expected_size && l_tis.tile_size(0) == expected_size &&
          l_tis.input_tile_size() == expected_size);
}

template<typename T>
bool check_local_tensor_sizes(const LocalTensor<T>&      l_tensor,
                              const std::vector<size_t>& expected_sizes) {
  EXPECTS_STR(l_tensor.num_modes() == expected_sizes.size(),
              "Expected sizes should be same as the dimensions of the input LocalTensor.");
  auto tis_vec = l_tensor.tiled_index_spaces();
  bool result  = true;
  for(size_t i = 0; i < tis_vec.size(); i++) {
    if(!check_local_tis_sizes(tis_vec.at(i), expected_sizes.at(i))) {
      result = false;
      break;
    }
  }

  return result;
}

template<typename T>
bool check_local_tensor_values(const LocalTensor<T>& l_tensor, T value) {
  EXPECTS_STR(l_tensor.is_allocated(), "LocalTensor should be allocated to check the values.");

  bool result    = true;
  auto tis_sizes = l_tensor.dim_sizes();

  auto num_elements = 1;

  for(auto tis_sz: tis_sizes) { num_elements *= tis_sz; }
  auto* local_buf = l_tensor.access_local_buf();
  for(size_t i = 0; i < num_elements; i++) {
    if(local_buf[i] != value) {
      result = false;
      break;
    }
  }
  return result;
}

template<typename T>
void test_local_tensor_constructors(Scheduler& sch, size_t N, Tile tilesize) {
  // LocalTensor construction
  // - TIS list
  // - TIS vec
  // - Labels
  // - Sizes

  TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};

  auto [i, j, k, l, m] = tis1.labels<5>("all");

  Tensor<T> A{i, j, k};
  Tensor<T> B{k, l};
  Tensor<T> C{i, j, l};

  sch.allocate(A, B, C).execute();
  EXPECTS_STR(A.is_allocated() && B.is_allocated() && C.is_allocated(),
              "All distributed tensors should be able to allocate!");

  ExecutionContext local_ec{sch.ec().pg(), DistributionKind::nw, MemoryManagerKind::local};
  Scheduler        sch_local{local_ec};

  LocalTensor<T> local_A{tis1, tis1, tis1};
  LocalTensor<T> local_B{B.tiled_index_spaces()};
  LocalTensor<T> local_C{i, j, l};
  LocalTensor<T> local_D{N, N, N};
  LocalTensor<T> local_E{10, 10, 10};

  sch_local.allocate(local_A, local_B, local_C, local_D, local_E).execute();

  EXPECTS_STR(local_A.is_allocated() && local_B.is_allocated() && local_C.is_allocated() &&
                local_D.is_allocated() && local_E.is_allocated(),
              "All local tensors should be able to allocate!");

  EXPECTS_STR(check_local_tensor_sizes(local_A, {N, N, N}), "Local_A is not correctly created!");
  EXPECTS_STR(check_local_tensor_sizes(local_B, {N, N}), "Local_B is not correctly created!");
  EXPECTS_STR(check_local_tensor_sizes(local_C, {N, N, N}), "Local_C is not correctly created!");
  EXPECTS_STR(check_local_tensor_sizes(local_D, {N, N, N}), "Local_D is not correctly created!");
  EXPECTS_STR(check_local_tensor_sizes(local_E, {10, 10, 10}), "Local_E is not correctly created!");
}

template<typename T>
void test_local_tensor_block(ExecutionContext& ec, size_t N) {
  // Block
  // - Tensor - various sizes, test with 0 for any dim size
  // - Matrix - various sizes, test with 0 for any dim size

  ExecutionContext local_ec{ec.pg(), DistributionKind::nw, MemoryManagerKind::local};
  Scheduler        sch_local{local_ec};

  LocalTensor<T> local_A{N, N, N};
  LocalTensor<T> local_B{N, N};

  sch_local.allocate(local_A, local_B)(local_A() = 42.0)(local_B() = 21.0).execute();

  auto local_C = local_A.block({0, 0, 0}, {4, 4, 4});
  auto local_D = local_B.block(0, 0, 4, 4);

  EXPECTS_STR(check_local_tensor_sizes(local_C, {4, 4, 4}), "Local_C is not correctly created!");
  EXPECTS_STR(check_local_tensor_sizes(local_D, {4, 4}), "Local_D is not correctly created!");

  EXPECTS_STR(check_local_tensor_values(local_C, 42.0), "Local_C doesn't have correct values!");
  EXPECTS_STR(check_local_tensor_values(local_D, 21.0), "Local_D doesn't have correct values!");
}

template<typename T>
void test_local_tensor_resize(ExecutionContext& ec, size_t N) {
  // Resize
  // - Smaller
  // - Larger
  // - Same size
  // - all 0 size
  // - change dim?

  ExecutionContext local_ec{ec.pg(), DistributionKind::nw, MemoryManagerKind::local};
  Scheduler        sch_local{local_ec};

  LocalTensor<T> local_A{N, N, N};
  LocalTensor<T> local_B{N, N};

  sch_local.allocate(local_A, local_B)(local_A() = 42.0)(local_B() = 21.0).execute();

  local_A.resize(5, 5, 5);
  EXPECTS_STR(check_local_tensor_sizes(local_A, {5, 5, 5}), "Local_A is not correctly created!");
  EXPECTS_STR(check_local_tensor_values(local_A, 42.0), "Local_A doesn't have correct values!");

  auto* tensor_ptr = local_A.base_ptr();
  local_A.resize(5, 5, 5);
  auto* tensor_resize_ptr = local_A.base_ptr();

  EXPECTS_STR(tensor_ptr == tensor_resize_ptr,
              "Resize into same size should return the old tensor!");

  local_A.resize(N, N, N);
  EXPECTS_STR(check_local_tensor_sizes(local_A, {N, N, N}), "Local_A is not correctly created!");
  EXPECTS_STR(check_local_tensor_values(local_A.block({0, 0, 0}, {5, 5, 5}), 42.0),
              "Local_A doesn't have correct values!");

  // local_A.resize(0,0,0);

  // local_A.resize(5,5);
}

template<typename T>
void test_local_tensor_accessor(ExecutionContext& ec, size_t N) {
  // Set/Get
  // - Single access
  // - Looped access

  ExecutionContext local_ec{ec.pg(), DistributionKind::nw, MemoryManagerKind::local};
  Scheduler        sch_local{local_ec};

  LocalTensor<T> local_A{N, N, N};
  LocalTensor<T> local_B{N, N};

  // clang-format off
  sch_local.allocate(local_A, local_B)
  (local_A() = 42.0)
  (local_B() = 21.0)
  .execute();
  // clang-format on

  EXPECTS_STR(local_A.get(0, 0, 0) == 42.0, "The get value doesn't match the expected value.");

  local_A.set({0, 0, 0}, 1.0);
  EXPECTS_STR(local_A.get(0, 0, 0) == 1.0, "The get value doesn't match the expected value.");
  local_A.set({0, 0, 0}, 42.0);

  for(size_t i = 0; i < N; i++) {
    for(size_t j = 0; j < N; j++) {
      for(size_t k = 0; k < N; k++) {
        EXPECTS_STR(local_A.get(i, j, k) == 42.0,
                    "The get value doesn't match the expected value.");
        local_A.set({i, j, k}, local_B.get(i, j));
        EXPECTS_STR(local_A.get(i, j, k) == 21.0,
                    "The get value doesn't match the expected value.");
      }
    }
  }
}

template<typename T>
void test_local_tensor_copy(ExecutionContext& ec, size_t N, Tile tilesize) {
  ExecutionContext local_ec{ec.pg(), DistributionKind::nw, MemoryManagerKind::local};
  Scheduler        sch_local{local_ec};

  Scheduler       sch_dist{ec};
  TiledIndexSpace tN{IndexSpace{range(N)}, tilesize};

  Tensor<T> dist_A{tN, tN, tN};
  // clang-format off
  sch_dist.allocate(dist_A)
  (dist_A() = 42.0)
  .execute();
  // clang-format on

  LocalTensor<T> local_A{dist_A.tiled_index_spaces()};
  // Copy from distrubuted tensor

  // clang-format off
  sch_local.allocate(local_A)
  (local_A() = 1.0)
  .execute();
  // clang-format on

  std::cout << "local_A before from_distributed_tensor" << std::endl;
  print_tensor(local_A);

  local_A.from_distributed_tensor(dist_A);

  std::cout << "local_A after from_distributed_tensor" << std::endl;
  print_tensor(local_A);

  // Copy to distributed tensor

  // clang-format off
  sch_local
  (local_A() = 21.0)
  .execute();
  // clang-format on

  local_A.to_distributed_tensor(dist_A);

  std::cout << "dist_A after to_distributed_tensor" << std::endl;

  if(ec.print()) print_tensor(dist_A);
}

template<typename T>
void test_local_tensor(Scheduler& sch, size_t N, Tile tilesize) {
  TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};

  auto [i, j, k, l, m] = tis1.labels<5>("all");

  Tensor<T> A{i, j, k};
  Tensor<T> B{k, l};
  Tensor<T> C{i, j, l};

  sch.allocate(A, B, C)(A() = 1.0)(B() = 2.0)(C() = 3.0).execute();

  ExecutionContext local_ec{sch.ec().pg(), DistributionKind::nw, MemoryManagerKind::local};

  Scheduler sch_local{local_ec};

  LocalTensor<T> new_local1{i, j, k};
  LocalTensor<T> new_local2{tis1, tis1, tis1};
  LocalTensor<T> new_local3{N, N, N};
  LocalTensor<T> new_local4{A.tiled_index_spaces()};

  sch_local
    .allocate(new_local1, new_local2, new_local3, new_local4)(new_local1() = 42.0)(
      new_local2() = 21.0)(new_local3() = 1.0)(new_local4() = 2.0)

    // .deallocate()
    .execute();

  // std::cout << "A_local" << std::endl;
  new_local3.init(42.0);

  std::cout << "value at 5,5,5 - " << new_local3.get(5, 5, 5) << std::endl;
  new_local3.set({5, 5, 5}, 1.0);       // memset val
  auto val = new_local3.get({5, 5, 5}); // memset val

  std::cout << "new value at 5,5,5 - " << new_local3.get(5, 5, 5) << std::endl;
  std::cout << "new_local4* before resize - " << new_local4.base_ptr() << std::endl;
  new_local4.resize(N, N, N); // vector.resize()? eigen.resize()?

  std::cout << "new_local4* after resize - " << new_local4.base_ptr() << std::endl;
  std::cout << "----------------------------------------------------" << std::endl;
  std::cout << "new_local4* before resize - " << new_local4.base_ptr() << std::endl;
  new_local4.resize(N + 5, N + 5, N + 5); // vector.resize()? eigen.resize()?
  std::cout << "new_local4* after resize - " << new_local4.base_ptr() << std::endl;
  auto new_local5 = new_local3.block({5, 5, 5}, {4, 4, 4});

  print_tensor(new_local1);
  print_tensor(new_local2);
  print_tensor(new_local3);
  print_tensor(new_local4);
  print_tensor(new_local5);
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  if(argc < 3) { tamm_terminate("Please provide an index space size and tile size"); }

  size_t is_size   = atoi(argv[1]);
  Tile   tile_size = atoi(argv[2]);

  if(is_size < tile_size) tile_size = is_size;

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  ExecutionHW ex_hw = ec.exhw();

  Scheduler sch{ec};

  if(ec.print()) {
    std::cout << tamm_git_info() << std::endl;
    auto current_time   = std::chrono::system_clock::now();
    auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
    auto cur_local_time = localtime(&current_time_t);
    std::cout << std::endl << "date: " << std::put_time(cur_local_time, "%c") << std::endl;

    std::cout << "nnodes: " << ec.nnodes() << ", ";
    std::cout << "nproc: " << ec.nnodes() * ec.ppn() << std::endl;
    std::cout << "dim, tile sizes = " << is_size << ", " << tile_size << std::endl;
    ec.print_mem_info();
    std::cout << std::endl << std::endl;
  }

  // test_local_tensor<double>(sch, is_size, tile_size);
  test_local_tensor_constructors<double>(sch, is_size, tile_size);
  test_local_tensor_copy<double>(ec, is_size, tile_size);
  test_local_tensor_block<double>(ec, is_size);
  test_local_tensor_resize<double>(ec, is_size);
  test_local_tensor_accessor<double>(ec, is_size);

  tamm::finalize();

  return 0;
}