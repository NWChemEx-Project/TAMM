#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <tamm/tamm.hpp>

using namespace tamm;

using T = double;

void lambda_function(const IndexVector& blockid, span<T> buff) {
  for(size_t i = 0; i < static_cast<size_t>(buff.size()); i++) { buff[i] = 42; }
}

template<size_t last_idx>
void l_func(const IndexVector& blockid, span<T> buf) {
  if(blockid[0] == last_idx || blockid[1] == last_idx) {
    for(auto i = 0U; i < buf.size(); i++) buf[i] = -1;
  }
  else {
    for(auto i = 0U; i < buf.size(); i++) buf[i] = 0;
  }

  if(blockid[0] == last_idx && blockid[1] == last_idx) {
    for(auto i = 0U; i < buf.size(); i++) buf[i] = 0;
  }
};

template<typename T>
void check_value(LabeledTensor<T> lt, T val) {
  LabelLoopNest loop_nest{lt.labels()};
  T             ref_val = val;
  for(const auto& itval: loop_nest) {
    const IndexVector blockid = internal::translate_blockid(itval, lt);
    size_t            size    = lt.tensor().block_size(blockid);
    std::vector<T>    buf(size);
    lt.tensor().get(blockid, buf);
    if(lt.tensor().is_non_zero(blockid)) { ref_val = val; }
    else { ref_val = (T) 0; }
    for(TAMM_SIZE i = 0; i < size; i++) { REQUIRE(std::abs(buf[i] - ref_val) < 1.0e-10); }
  }
}

template<typename T>
void check_value(Tensor<T>& t, T val) {
  check_value(t(), val);
}

template<typename T>
void tensor_contruction(const TiledIndexSpace& T_AO, const TiledIndexSpace& T_MO,
                        const TiledIndexSpace& T_ATOM, const TiledIndexSpace& T_AO_ATOM) {
  TiledIndexLabel A, r, s, mu, mu_A;

  A              = T_ATOM.label("all", 1);
  std::tie(r, s) = T_MO.labels<2>("all");
  mu             = T_AO.label("all", 1);
  mu_A           = T_AO_ATOM.label("all", 1);

  // Tensor Q{T_ATOM, T_MO, T_MO}, C{T_AO,T_MO}, SC{T_AO,T_MO};
  Tensor<T> Q{A, r, s}, C{mu, r}, SC{mu, s};

  Q(A, r, s) = 0.5 * C(mu_A(A), r) * SC(mu_A(A), s);
  Q(A, r, s) += 0.5 * C(mu_A(A), s) * SC(mu_A(A), r);
}

TEST_CASE("Block Sparse Tensor Construction") {
  // std::cout << "Starting Block Sparse Tensor tests" << std::endl;
  using T = double;
  IndexSpace SpinIS{
    range(0, 20),
    {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}},
    {{Spin{1}, {range(0, 5), range(10, 15)}}, {Spin{-1}, {range(5, 10), range(15, 20)}}}};

  IndexSpace IS{range(0, 20)};

  TiledIndexSpace SpinTIS{SpinIS, 5};
  TiledIndexSpace TIS{IS, 5};

  std::vector<SpinPosition> spin_mask_2D{SpinPosition::lower, SpinPosition::upper};
  ProcGroup                 pg = ProcGroup::create_world_coll();
  ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};

  bool failed = false;
  try {
    TiledIndexSpaceVec t_spaces{SpinTIS, SpinTIS};
    auto is_non_zero_2D = [t_spaces, spin_mask_2D](const IndexVector& blockid) -> bool {
      Spin upper_total = 0, lower_total = 0, other_total = 0;
      for(size_t i = 0; i < 2; i++) {
        const auto& tis = t_spaces[i];
        if(spin_mask_2D[i] == SpinPosition::upper) { upper_total += tis.spin(blockid[i]); }
        else if(spin_mask_2D[i] == SpinPosition::lower) { lower_total += tis.spin(blockid[i]); }
        else { other_total += tis.spin(blockid[i]); }
      }

      return (upper_total == lower_total);
    };

    TensorInfo tensor_info{t_spaces, is_non_zero_2D};

    Tensor<T> tensor{t_spaces, tensor_info};
    tensor.allocate(ec);
    Scheduler{*ec}(tensor() = 42).execute();
    check_value(tensor, (T) 42);
    // print_tensor_all(tensor);

    tensor.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << '\n';
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    TiledIndexSpaceVec t_spaces{SpinTIS, SpinTIS};
    auto               is_non_zero_2D = [](const IndexVector& blockid) -> bool {
      return blockid[0] == blockid[1];
    };

    TensorInfo tensor_info{t_spaces, is_non_zero_2D};

    Tensor<T> tensor{t_spaces, tensor_info};
    tensor.allocate(ec);
    Scheduler{*ec}(tensor() = 42).execute();
    check_value(tensor, (T) 42);
    // print_tensor_all(tensor);

    tensor.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << '\n';
    failed = true;
  }
  REQUIRE(!failed);

  IndexSpace MO_IS{range(0, 20), {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}}};

  TiledIndexSpace MO{MO_IS, 5};

  auto [i, j, k, l] = MO("occ").labels<4>();
  auto [a, b, c, d] = MO("virt").labels<4>();

  Char2TISMap char2MOstr = {{'i', "occ"},  {'j', "occ"},  {'k', "occ"},  {'l', "occ"},
                            {'a', "virt"}, {'b', "virt"}, {'c', "virt"}, {'d', "virt"}};
  TensorInfo  tensor_info{
    {MO, MO, MO, MO},                                 // Tensor dims
    {"ijab", "iajb", "ijka", "ijkl", "iabc", "abcd"}, // Allowed blocks
    char2MOstr,                                       // Char to named sub-space string
    {"abij",
      "aibj"} // Disallowed blocks - note that allowed blocks will precedence over disallowed blocks
  };

  failed = false;
  try {
    Tensor<T> tensor{{MO, MO, MO, MO}, tensor_info};

    tensor.allocate(ec);

    Scheduler{*ec}(tensor() = 42).execute();
    check_value(tensor, (T) 42);

    // clang-format off
    Scheduler{*ec}
    (tensor(i, j, a, b) = 1.0)
    (tensor(i, a, j, b) = 2.0)
    (tensor(i, j, k, a) = 3.0)
    (tensor(i, j, k, l) = 4.0)
    (tensor(i, a, b, c) = 5.0)
    (tensor(a, b, c, d) = 6.0)
    .execute();
    // clang-format on

    check_value(tensor(i, j, a, b), (T) 1.0);
    check_value(tensor(i, a, j, b), (T) 2.0);
    check_value(tensor(i, j, k, a), (T) 3.0);
    check_value(tensor(i, j, k, l), (T) 4.0);
    check_value(tensor(i, a, b, c), (T) 5.0);
    check_value(tensor(a, b, c, d), (T) 6.0);

    tensor.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << '\n';
    failed = true;
  }

  failed = false;
  try {
    Tensor<T> tensor{{MO, MO, MO, MO}, {"ijab", "ijka", "iajb"}, char2MOstr};

    tensor.allocate(ec);

    Scheduler{*ec}(tensor() = 42).execute();
    check_value(tensor, (T) 42);

    // clang-format off
    Scheduler{*ec}
    (tensor(i, j, a, b) = 1.0)
    (tensor(i, a, j, b) = 2.0)
    (tensor(i, j, k, a) = 3.0)
    .execute();
    // clang-format on

    check_value(tensor(i, j, a, b), (T) 1.0);
    check_value(tensor(i, a, j, b), (T) 2.0);
    check_value(tensor(i, j, k, a), (T) 3.0);

    tensor.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << '\n';
    failed = true;
  }

  failed = false;
  try {
    Tensor<T> tensorA{{MO, MO, MO, MO}, {"ijab", "ijkl"}, char2MOstr};
    Tensor<T> tensorB{{MO, MO, MO, MO}, {"ijka", "iajb"}, char2MOstr};
    Tensor<T> tensorC{{MO, MO, MO, MO}, {"iabc", "abcd"}, char2MOstr};

    tensorA.allocate(ec);
    tensorB.allocate(ec);
    tensorC.allocate(ec);

    Scheduler{*ec}(tensorA() = 2.0)(tensorB() = 4.0)(tensorC() = 0.0).execute();
    check_value(tensorA, (T) 2.0);
    check_value(tensorB, (T) 4.0);
    check_value(tensorC, (T) 0.0);

    // clang-format off
    Scheduler{*ec}
    (tensorC(a, b, c, d) += tensorA(i, j, a, b) * tensorB(j, c, i, d))
    (tensorC(i, a, b, c) += 0.5 * tensorA(j, k, a, b) * tensorB(i, j, k, c))
    .execute();
    // clang-format on

    check_value(tensorC(i, a, b, c), (T) 400.0);
    check_value(tensorC(a, b, c, d), (T) 800.0);

    // std::cout << "Printing TensorC:" << std::endl;
    // print_tensor(tensorC);

    tensorA.deallocate();
    tensorB.deallocate();
    tensorC.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << '\n';
    failed = true;
  }

  failed = false;
  try {
    Tensor<T> tensorA{{MO, MO, MO, MO}, {{i, j, a, b}, {i, j, k, l}}};
    Tensor<T> tensorB{{MO, MO, MO, MO}, {{i, j, k, a}, {i, a, j, b}}};
    Tensor<T> tensorC{{MO, MO, MO, MO}, {{i, a, b, c}, {a, b, c, d}}};

    tensorA.allocate(ec);
    tensorB.allocate(ec);
    tensorC.allocate(ec);

    Scheduler{*ec}(tensorA() = 2.0)(tensorB() = 4.0)(tensorC() = 0.0).execute();
    check_value(tensorA, (T) 2.0);
    check_value(tensorB, (T) 4.0);
    check_value(tensorC, (T) 0.0);

    // clang-format off
    Scheduler{*ec}
    (tensorC(a, b, c, d) += tensorA(i, j, a, b) * tensorB(j, c, i, d))
    (tensorC(i, a, b, c) += 0.5 * tensorA(j, k, a, b) * tensorB(i, j, k, c))
    .execute();
    // clang-format on

    check_value(tensorC(i, a, b, c), (T) 400.0);
    check_value(tensorC(a, b, c, d), (T) 800.0);

    // std::cout << "Printing TensorC:" << std::endl;
    // print_tensor(tensorC);

    tensorA.deallocate();
    tensorB.deallocate();
    tensorC.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << '\n';
    failed = true;
  }

  failed = false;
  try {
    Tensor<T> tensorA{{MO, MO, MO, MO}, {{"occ, occ, virt, virt"}, {"occ, occ, occ, occ"}}};
    Tensor<T> tensorB{{MO, MO, MO, MO}, {{"occ, occ, occ, virt"}, {"occ, virt, occ, virt"}}};
    Tensor<T> tensorC{{MO, MO, MO, MO}, {{"occ, virt, virt, virt"}, {"virt, virt, virt, virt"}}};

    tensorA.allocate(ec);
    tensorB.allocate(ec);
    tensorC.allocate(ec);

    Scheduler{*ec}(tensorA() = 2.0)(tensorB() = 4.0)(tensorC() = 0.0).execute();
    check_value(tensorA, (T) 2.0);
    check_value(tensorB, (T) 4.0);
    check_value(tensorC, (T) 0.0);

    // clang-format off
    Scheduler{*ec}
    (tensorC(a, b, c, d) += tensorA(i, j, a, b) * tensorB(j, c, i, d))
    (tensorC(i, a, b, c) += 0.5 * tensorA(j, k, a, b) * tensorB(i, j, k, c))
    .execute();
    // clang-format on

    check_value(tensorC(i, a, b, c), (T) 400.0);
    check_value(tensorC(a, b, c, d), (T) 800.0);

    // std::cout << "Printing TensorC:" << std::endl;
    // print_tensor(tensorC);

    tensorA.deallocate();
    tensorB.deallocate();
    tensorC.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << '\n';
    failed = true;
  }

  failed = false;
  try {
    TiledIndexSpace Occ  = MO("occ");
    TiledIndexSpace Virt = MO("virt");
    Tensor<T>       tensorA{
      {MO, MO, MO, MO},
      {TiledIndexSpaceVec{Occ, Occ, Virt, Virt}, TiledIndexSpaceVec{Occ, Occ, Occ, Occ}}};
    Tensor<T> tensorB{
      {MO, MO, MO, MO},
      {TiledIndexSpaceVec{Occ, Occ, Occ, Virt}, TiledIndexSpaceVec{Occ, Virt, Occ, Virt}}};
    Tensor<T> tensorC{
      {MO, MO, MO, MO},
      {TiledIndexSpaceVec{Occ, Virt, Virt, Virt}, TiledIndexSpaceVec{Virt, Virt, Virt, Virt}}};

    tensorA.allocate(ec);
    tensorB.allocate(ec);
    tensorC.allocate(ec);

    Scheduler{*ec}(tensorA() = 2.0)(tensorB() = 4.0)(tensorC() = 0.0).execute();
    check_value(tensorA, (T) 2.0);
    check_value(tensorB, (T) 4.0);
    check_value(tensorC, (T) 0.0);

    // clang-format off
    Scheduler{*ec}
    (tensorC(a, b, c, d) += tensorA(i, j, a, b) * tensorB(j, c, i, d))
    (tensorC(i, a, b, c) += 0.5 * tensorA(j, k, a, b) * tensorB(i, j, k, c))
    .execute();
    // clang-format on

    check_value(tensorC(i, a, b, c), (T) 400.0);
    check_value(tensorC(a, b, c, d), (T) 800.0);

    // std::cout << "Printing TensorC:" << std::endl;
    // print_tensor(tensorC);

    tensorA.deallocate();
    tensorB.deallocate();
    tensorC.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << '\n';
    failed = true;
  }

  REQUIRE(!failed);
}
#if 1
TEST_CASE("Spin Tensor Construction") {
  using T = double;
  IndexSpace SpinIS{
    range(0, 20),
    {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}},
    {{Spin{1}, {range(0, 5), range(10, 15)}}, {Spin{-1}, {range(5, 10), range(15, 20)}}}};

  IndexSpace IS{range(0, 20)};

  TiledIndexSpace SpinTIS{SpinIS, 5};
  TiledIndexSpace TIS{IS, 5};

  std::vector<SpinPosition> spin_mask_2D{SpinPosition::lower, SpinPosition::upper};

  TiledIndexLabel i, j, k, l;
  std::tie(i, j) = SpinTIS.labels<2>("all");
  std::tie(k, l) = TIS.labels<2>("all");

  bool failed = false;
  try {
    TiledIndexSpaceVec t_spaces{SpinTIS, SpinTIS};
    Tensor<T>          tensor{t_spaces, spin_mask_2D};
  } catch(const std::string& e) {
    std::cerr << e << '\n';
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    IndexLabelVec t_lbls{i, j};
    Tensor<T>     tensor{t_lbls, spin_mask_2D};
  } catch(const std::string& e) {
    std::cerr << e << '\n';
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    TiledIndexSpaceVec t_spaces{TIS, TIS};

    Tensor<T> tensor{t_spaces, spin_mask_2D};
  } catch(const std::string& e) {
    std::cerr << e << '\n';
    failed = true;
  }
  REQUIRE(!failed);

  {
    REQUIRE((SpinTIS.spin(0) == Spin{1}));
    REQUIRE((SpinTIS.spin(1) == Spin{-1}));
    REQUIRE((SpinTIS.spin(2) == Spin{1}));
    REQUIRE((SpinTIS.spin(3) == Spin{-1}));

    REQUIRE((SpinTIS("occ").spin(0) == Spin{1}));
    REQUIRE((SpinTIS("occ").spin(1) == Spin{-1}));

    REQUIRE((SpinTIS("virt").spin(0) == Spin{1}));
    REQUIRE((SpinTIS("virt").spin(1) == Spin{-1}));
  }

  TiledIndexSpace tis_3{SpinIS, 3};

  {
    REQUIRE((tis_3.spin(0) == Spin{1}));
    REQUIRE((tis_3.spin(1) == Spin{1}));
    REQUIRE((tis_3.spin(2) == Spin{-1}));
    REQUIRE((tis_3.spin(3) == Spin{-1}));
    REQUIRE((tis_3.spin(4) == Spin{1}));
    REQUIRE((tis_3.spin(5) == Spin{1}));
    REQUIRE((tis_3.spin(6) == Spin{-1}));
    REQUIRE((tis_3.spin(7) == Spin{-1}));

    REQUIRE((tis_3("occ").spin(0) == Spin{1}));
    REQUIRE((tis_3("occ").spin(1) == Spin{1}));
    REQUIRE((tis_3("occ").spin(2) == Spin{-1}));
    REQUIRE((tis_3("occ").spin(3) == Spin{-1}));

    REQUIRE((tis_3("virt").spin(0) == Spin{1}));
    REQUIRE((tis_3("virt").spin(1) == Spin{1}));
    REQUIRE((tis_3("virt").spin(2) == Spin{-1}));
    REQUIRE((tis_3("virt").spin(3) == Spin{-1}));
  }

  ProcGroup         pg = ProcGroup::create_world_coll();
  ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};

  failed = false;
  try {
    TiledIndexSpaceVec t_spaces{tis_3, tis_3};
    Tensor<T>          tensor{t_spaces, spin_mask_2D};
    tensor.allocate(ec);
    Scheduler{*ec}(tensor() = 42).execute();
    check_value(tensor, (T) 42);
    tensor.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    TiledIndexSpaceVec t_spaces{tis_3("occ"), tis_3("virt")};
    Tensor<T>          tensor{t_spaces, spin_mask_2D};
    tensor.allocate(ec);

    Scheduler{*ec}(tensor() = 42).execute();
    check_value(tensor, (T) 42);
    tensor.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    TiledIndexSpaceVec t_spaces{tis_3, tis_3};
    Tensor<T>          T1{t_spaces, spin_mask_2D};
    Tensor<T>          T2{t_spaces, spin_mask_2D};
    T1.allocate(ec);
    T2.allocate(ec);

    Scheduler{*ec}(T2() = 3)(T1() = T2()).execute();
    check_value(T2, (T) 3);
    check_value(T1, (T) 3);

    T1.deallocate();
    T2.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    TiledIndexSpaceVec t_spaces{tis_3, tis_3};
    Tensor<T>          T1{t_spaces, {1, 1}};
    Tensor<T>          T2{t_spaces, {1, 1}};
    T1.allocate(ec);
    T2.allocate(ec);

    Scheduler{*ec}(T1() = 42)(T2() = 3)(T1() += T2()).execute();
    check_value(T2, (T) 3);
    check_value(T1, (T) 45);

    T1.deallocate();
    T2.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    TiledIndexSpaceVec t_spaces{tis_3, tis_3};
    Tensor<T>          T1{t_spaces, {1, 1}};
    Tensor<T>          T2{t_spaces, {1, 1}};
    T1.allocate(ec);
    T2.allocate(ec);

    Scheduler{*ec}(T1() = 42)(T2() = 3)(T1() += 2 * T2()).execute();
    check_value(T2, (T) 3);
    check_value(T1, (T) 48);

    T1.deallocate();
    T2.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    TiledIndexSpaceVec t_spaces{tis_3, tis_3};
    Tensor<T>          T1{t_spaces, {1, 1}};
    Tensor<T>          T2{t_spaces, {1, 1}};
    Tensor<T>          T3{t_spaces, {1, 1}};

    T1.allocate(ec);
    T2.allocate(ec);
    T3.allocate(ec);

    Tensor<T> T4 = T3;

    Scheduler{*ec}(T1() = 42)(T2() = 3)(T3() = 4)(T1() += T4() * T2()).execute();
    check_value(T3, (T) 4);
    check_value(T2, (T) 3);
    check_value(T1, (T) 54);

    T1.deallocate();
    T2.deallocate();
    T3.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    auto lambda = [&](const IndexVector& blockid, span<T> buff) {
      for(size_t i = 0; i < static_cast<size_t>(buff.size()); i++) { buff[i] = 42; }
    };
    TiledIndexSpaceVec t_spaces{TIS, TIS};
    Tensor<T>          t{t_spaces, lambda};

    auto lt = t();
    for(auto it: t.loop_nest()) {
      auto           blockid = internal::translate_blockid(it, lt);
      TAMM_SIZE      size    = t.block_size(blockid);
      std::vector<T> buf(size);
      t.get(blockid, buf);
      std::cout << "block" << blockid;
      for(TAMM_SIZE i = 0; i < size; i++) std::cout << buf[i] << " ";
      std::cout << std::endl;
    }

  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    auto lambda = [](const IndexVector& blockid, span<T> buff) {
      for(size_t i = 0; i < static_cast<size_t>(buff.size()); i++) { buff[i] = 42; }
    };
    // TiledIndexSpaceVec t_spaces{TIS, TIS};
    Tensor<T> S{{TIS, TIS}, lambda};
    Tensor<T> T1{{TIS, TIS}};

    T1.allocate(ec);

    Scheduler{*ec}(T1() = 0)(T1() += 2 * S()).execute();

    check_value(T1, (T) 84);

  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    Tensor<T> S{{TIS, TIS}, lambda_function};
    Tensor<T> T1{{TIS, TIS}};

    T1.allocate(ec);

    Scheduler{*ec}(T1() = 0)(T1() += 2 * S()).execute();

    check_value(T1, (T) 84);

  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    std::vector<Tensor<T>> x1(5);
    std::vector<Tensor<T>> x2(5);
    for(int i = 0; i < 5; i++) {
      x1[i] = Tensor<T>{TIS, TIS};
      x2[i] = Tensor<T>{TIS, TIS};
      Tensor<T>::allocate(ec, x1[i], x2[i]);
    }

    auto deallocate_vtensors = [&](auto&&... vecx) {
      //(std::for_each(vecx.begin(), vecx.end(),
      // std::mem_fun(&Tensor<T>::deallocate)), ...);
      //(std::for_each(vecx.begin(), vecx.end(), Tensor<T>::deallocate),
      //...);
    };
    deallocate_vtensors(x1, x2);
  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    IndexSpace      MO_IS{range(0, 7)};
    TiledIndexSpace MO{MO_IS, {1, 1, 3, 1, 1}};

    IndexSpace      MO_IS2{range(0, 7)};
    TiledIndexSpace MO2{MO_IS2, {1, 1, 3, 1, 1}};

    Tensor<T> pT{MO, MO};
    Tensor<T> pV{MO2, MO2};

    pT.allocate(ec);
    pV.allocate(ec);

    auto      tis_list = pT.tiled_index_spaces();
    Tensor<T> H{tis_list};
    H.allocate(ec);

    auto h_tis = H.tiled_index_spaces();

    Scheduler{*ec}(H("mu", "nu") = pT("mu", "nu"))(H("mu", "nu") += pV("mu", "nu")).execute();

  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);

  failed = false;
  try {
    IndexSpace      IS{range(10)};
    TiledIndexSpace TIS{IS, 2};

    Tensor<T>        A{TIS, TIS};
    ProcGroup        pg = ProcGroup::create_world_coll();
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

    A.allocate(&ec);
    A.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;
  try {
    IndexSpace      AO_IS{range(10)};
    TiledIndexSpace AO{AO_IS, 2};
    IndexSpace      MO_IS{range(10)};
    TiledIndexSpace MO{MO_IS, 2};

    Tensor<T>        C{AO, MO};
    ProcGroup        pg = ProcGroup::create_world_coll();
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    tamm::Scheduler  sch{ec};

    sch.allocate(C).execute();
    // Scheduler{&ec}.allocate(C)
    //     (C() = 42.0).execute();

    const auto AOs = C.tiled_index_spaces()[0];
    const auto MOs = C.tiled_index_spaces()[1];
    auto [mu, nu]  = AOs.labels<2>("all");

    // TODO: Take the slice of C that is for the occupied orbitals
    auto [p] = MOs.labels<1>("all");
    Tensor<T> rho{AOs, AOs};

    sch.allocate(rho)(rho() = 0)(rho(mu, nu) += C(mu, p) * C(nu, p)).execute();

    rho.deallocate();
    C.deallocate();
  } catch(const std::string& e) {
    std::cerr << e << std::endl;
    failed = true;
  }
  REQUIRE(!failed);

  // ScaLAPACK test
  failed = false;
  //   try {
  //     std::cout << "------BEGIN block cyclic dist test------\n";
  //     IndexSpace      MO_IS{range(0, 7)};
  //     TiledIndexSpace MO{MO_IS, {1, 1, 5}};
  //     TiledIndexSpace NO{MO_IS, 2};
  //     // IndexSpace MO_IS2{range(0, 7)};
  //     // TiledIndexSpace MO2{MO_IS2, {1, 1, 3, 1, 1}};

  //     Tensor<T> pT{MO, MO};
  //     Tensor<T> pV{NO, NO};

  //     Tensor<T> sca{{1, 1}, {NO, NO}};

  //     pT.allocate(ec);
  //     pV.allocate(ec);
  //     sca.allocate(ec);

  //     auto      tis_list = pT.tiled_index_spaces();
  //     Tensor<T> H{tis_list};
  //     H.allocate(ec);

  //     auto h_tis = H.tiled_index_spaces();
  //     GA_Print_distribution(pT.ga_handle());
  //     // GA_Print(pT.ga_handle());

  //     Scheduler{*ec}(pT("mu", "nu") = 2.2)(H("mu", "nu") = pT("mu", "ku") * pT("ku", "nu"))(sca()
  //     =
  //                                                                                             2.2)
  //       .execute();

  //     // auto x = tamm::norm(H);
  //     GA_Print(H.ga_handle());
  //     auto sca1             = to_block_cyclic_tensor(H, {1, 1}, {2, 2});
  //     auto [lptr, lbufsize] = access_local_block_cyclic_buffer(sca1);
  //     for(auto i = 0L; i < lbufsize; i++) std::cout << lptr[i] << "\n";
  //     // GA_Print(sca1.ga_handle());
  //     from_block_cyclic_tensor(sca1, pT);
  //     // GA_Print(pT.ga_handle());

  //     Tensor<T>::deallocate(H, pT, sca, sca1);

  //     std::cout << "------END block cyclic dist test------\n";

  //   } catch(const std::string& e) {
  //     std::cerr << e << std::endl;
  //     failed = true;
  //   }
  //   REQUIRE(!failed);
  // }

  // TEST_CASE("Non trivial ScaLAPACK test") {
  //   using T = double;
  //   IndexSpace SpinIS{
  //     range(0, 20),
  //     {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}},
  //     {{Spin{1}, {range(0, 5), range(10, 15)}}, {Spin{2}, {range(5, 10), range(15, 20)}}}};

  //   IndexSpace IS{range(0, 20)};

  //   TiledIndexSpace SpinTIS{SpinIS, 5};
  //   TiledIndexSpace TIS{IS, 5};

  //   std::vector<SpinPosition> spin_mask_2D{SpinPosition::lower, SpinPosition::upper};

  //   TiledIndexLabel i, j, k, l;
  //   std::tie(i, j) = SpinTIS.labels<2>("all");
  //   std::tie(k, l) = TIS.labels<2>("all");

  //   bool failed = false;

  //   ProcGroup         pg = ProcGroup::create_world_coll();
  //   ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};

  //   // Non trivial ScaLAPACK test
  //   try {
  //     std::cout << "------BEGIN non trivial block cyclic dist test------\n";

  //     size_t     n  = 512;
  //     tamm::Tile ts = 20;
  //     int64_t    nb = 128;

  //     size_t npr = 1, npc = 1;

  //     IndexSpace      is{range(0, n)};
  //     TiledIndexSpace tis{is, ts};

  //     Tensor<T> A{tis, tis};
  //     A.allocate(ec);
  //     Scheduler{*ec}(A("i", "j") = 1.).execute();

  //     auto A_scal = to_block_cyclic_tensor(A, {npr, npc}, {nb, nb});

  //     std::cout << "------END non trivial block cyclic dist test------\n";

  //   } catch(const std::string& e) {
  //     std::cerr << e << std::endl;
  //     failed = true;
  //   }
  REQUIRE(!failed);
}

TEST_CASE("Hash Based Equality and Compatibility Check") {
  IndexSpace is1{range(0, 20), {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}}};
  IndexSpace is2{range(0, 10)};
  IndexSpace is1_occ = is1("occ");

  TiledIndexSpace tis1{is1};
  TiledIndexSpace tis2{is2};
  TiledIndexSpace tis3{is1_occ};
  TiledIndexSpace sub_tis1{tis1, range(0, 10)};

  REQUIRE(tis2 == tis3);
  REQUIRE(tis2 == tis1("occ"));
  REQUIRE(tis3 == tis1("occ"));
  REQUIRE(tis1 != tis2);
  REQUIRE(tis1 != tis3);
  REQUIRE(tis2 != tis1("virt"));
  REQUIRE(tis3 != tis1("virt"));

  // sub-TIS vs TIS from same IS
  REQUIRE(sub_tis1 == tis2);
  REQUIRE(sub_tis1 == tis3);
  REQUIRE(sub_tis1 == tis1("occ"));
  REQUIRE(sub_tis1 != tis1);
  REQUIRE(sub_tis1 != tis1("virt"));

  REQUIRE(sub_tis1.is_compatible_with(tis1));
  REQUIRE(sub_tis1.is_compatible_with(tis1("occ")));
  REQUIRE(!sub_tis1.is_compatible_with(tis2));
  REQUIRE(!sub_tis1.is_compatible_with(tis3));
  REQUIRE(sub_tis1.is_compatible_with(tis1("virt")));
}

TEST_CASE("GitHub Issues") {
  tamm::ProcGroup        pg = ProcGroup::create_world_coll();
  tamm::ExecutionContext ec(pg, DistributionKind::nw, MemoryManagerKind::ga);
  tamm::TiledIndexSpace  X{tamm::IndexSpace{tamm::range(0, 4)}};
  tamm::TiledIndexSpace  Y{tamm::IndexSpace{tamm::range(0, 3)}};
  auto [i, j] = X.labels<2>("all");
  auto [a]    = Y.labels<1>("all");

  Tensor<double> A{X, X, Y};
  Tensor<double> B{X, X};

  std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
  tamm::Scheduler{ec}
    .allocate(A, B)(A() = 3.0)(B() = 0.0)
    // (B(i,j) += A(i,j,a))
    (B(i, j) += A(i, j, a))
    .execute();

  std::cout << "A tensor" << std::endl;
  print_tensor(A);
  std::cout << "B tensor" << std::endl;
  print_tensor(B);
  std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
  // std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
  // print_tensor(B);
}

TEST_CASE("Slack Issues") {
  using tensor_type = Tensor<double>;
  std::cerr << "Slack Issue Start" << std::endl;
  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  Scheduler        sch{ec};

  tensor_type initialMO_state;

  IndexSpace AOs_{range(0, 10)};
  IndexSpace MOs_{range(0, 10), {{"O", {range(0, 5)}}, {"V", {range(5, 10)}}}};

  TiledIndexSpace tAOs{AOs_};
  TiledIndexSpace tMOs{MOs_};
  TiledIndexSpace tXYZ{IndexSpace{range(0, 3)}};

  tensor_type D{tXYZ, tAOs, tAOs};
  tensor_type C{tAOs, tMOs};
  tensor_type W{tMOs("O"), tMOs("O")};

  sch.allocate(C, W, D)(C() = 42.0)(W() = 1.0)(D() = 1.0).execute();

  auto xyz = tXYZ;
  auto AOs = C.tiled_index_spaces()[0];
  auto MOs = C.tiled_index_spaces()[1]("O");

  initialMO_state = tensor_type{xyz, MOs, MOs};
  tensor_type tmp{xyz, AOs, MOs};

  auto [x]      = xyz.labels<1>("all");
  auto [mu, nu] = AOs.labels<2>("all");
  auto [i, j]   = MOs.labels<2>("all");

  auto tmp_lbls = tmp().labels();
  auto D_lbls   = D().labels();
  auto C_lbls   = C().labels();

  sch
    .allocate(initialMO_state, tmp)(tmp(x, mu, i) = D(x, mu, nu) * C(nu, i))(
      initialMO_state(x, i, j) = C(mu, i) * tmp(x, mu, j))
    .execute();

  // print_tensor(initialMO_state);

  auto X      = initialMO_state.tiled_index_spaces()[0];
  auto n_MOs  = W.tiled_index_spaces()[0];
  auto n_LMOs = W.tiled_index_spaces()[1];

  auto [x_]     = X.labels<1>("all");
  auto [r_, s_] = n_MOs.labels<2>("all", 0);
  auto [i_, j_] = n_LMOs.labels<2>("all", 10);

  tensor_type initW{X, n_MOs, n_LMOs};
  tensor_type WinitW{X, n_LMOs, n_LMOs};

  sch
    .allocate(initW, WinitW)(initW(x_, r_, i_) = initialMO_state(x_, r_, s_) * W(s_, i_))(
      WinitW(x_, i_, j_) = W(r_, i_) * initW(x_, r_, j_))
    .deallocate(initialMO_state, tmp, C, W, D, initW, WinitW)
    .execute();
}

TEST_CASE("Slicing examples") {
  IndexSpace AOs{range(0, 10)};
  IndexSpace MOs{range(0, 10), {{"O", {range(0, 5)}}, {"V", {range(5, 10)}}}};

  TiledIndexSpace tAOs{AOs};
  TiledIndexSpace tMOs{MOs};

  Tensor<double> A{tMOs};
  Tensor<double> B{tMOs, tMOs};

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  Scheduler sch{ec};

  sch.allocate(A, B)(A() = 0.0)(B() = 4.0).execute();

  auto [i] = tMOs.labels<1>("all");
  auto [j] = tMOs.labels<1>("O");
  auto [k] = tMOs.labels<1>("V");

  sch(B(j, j) = 42.0)(B(k, k) = 21.0)(A(i) = B(i, i))
    // (A() = B(i, i))
    // (B(i,i) = A(i))
    .execute();

  // print_tensor(A);
  // print_tensor(B);

  Tensor<double>::deallocate(A, B);
}

TEST_CASE("Fill tensors using lambda functions") {
  IndexSpace AOs{range(0, 10)};
  IndexSpace MOs{range(0, 10), {{"O", {range(0, 5)}}, {"V", {range(5, 10)}}}};

  TiledIndexSpace tAOs{AOs};
  TiledIndexSpace tMOs{MOs};

  Tensor<double> A{tAOs, tAOs};
  Tensor<double> B{tMOs, tMOs};

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  A.allocate(&ec);
  B.allocate(&ec);

  update_tensor(A(), lambda_function);
  // std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
  // print_tensor(A);

  Scheduler{ec}(A() = 0).execute();
  // std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
  // print_tensor(A);

  update_tensor(A(), l_func<9>);
  // std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
  // print_tensor(A);

  auto i = tAOs.label("all");

  update_tensor(A(i, i), lambda_function);
  std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
  print_tensor(A);
  Tensor<double>::deallocate(A, B);
}

/*
TEST_CASE("SCF Example Implementation") {

    using tensor_type = Tensor<double>;
    std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
    std::cerr << "SCF Example Implementation" << std::endl;

    IndexSpace AUXs_{range(0, 7)};
    IndexSpace AOs_{range(0, 7)};
    IndexSpace MOs_{range(0, 10),
                   {{"O", {range(0, 5)}},
                    {"V", {range(5, 10)}}
    }};

    TiledIndexSpace Aux{AUXs_};
    TiledIndexSpace AOs{AOs_};
    TiledIndexSpace tMOs{MOs_};

    std::map<IndexVector, TiledIndexSpace> dep_nu_mu_q{
        {
            {{0}, TiledIndexSpace{AOs, IndexVector{0,3,4}}},
            {{2}, TiledIndexSpace{AOs, IndexVector{0,2}}},
            {{3}, TiledIndexSpace{AOs, IndexVector{1,3,5}}},
            {{4}, TiledIndexSpace{AOs, IndexVector{3,5}}},
            {{5}, TiledIndexSpace{AOs, IndexVector{1,2}}},
            {{6}, TiledIndexSpace{AOs, IndexVector{2}}},

        }
    };

    std::map<IndexVector, TiledIndexSpace> dep_nu_mu_d{
        {
            {{0}, TiledIndexSpace{AOs, IndexVector{1,3,5}}},
            {{1}, TiledIndexSpace{AOs, IndexVector{0,1,2}}},
            {{2}, TiledIndexSpace{AOs, IndexVector{0,2,4}}},
            {{3}, TiledIndexSpace{AOs, IndexVector{1,6}}},
            {{4}, TiledIndexSpace{AOs, IndexVector{3,5}}},
            // {{5}, TiledIndexSpace{AOs, IndexVector{0,1,2}}},
            {{6}, TiledIndexSpace{AOs, IndexVector{0,1,2}}}
        }
    };

    std::map<IndexVector, TiledIndexSpace> dep_nu_mu_c{
        {
            {{0}, TiledIndexSpace{AOs, IndexVector{3}}},
            {{2}, TiledIndexSpace{AOs, IndexVector{0,2}}},
            {{3}, TiledIndexSpace{AOs, IndexVector{1}}},
            {{4}, TiledIndexSpace{AOs, IndexVector{3,5}}},
            // {{5}, TiledIndexSpace{AOs, IndexVector{1,2}}},
            {{6}, TiledIndexSpace{AOs, IndexVector{2}}}
        }
    };

    TiledIndexSpace tSubAO_AO_Q{AOs, {AOs}, dep_nu_mu_q};

    TiledIndexSpace tSubAO_AO_D{AOs, {AOs}, dep_nu_mu_d};

    // TiledIndexSpace tSubAO_AO_C{AOs, {AOs}, dep_nu_mu_c};
    auto tSubAO_AO_C = tSubAO_AO_Q.intersect_tis(tSubAO_AO_D);
    // auto tSubAO_AO_C = tSubAO_AO_D.intersect_tis(tSubAO_AO_Q);

    auto X = Aux.label("all",0);
    auto [mu, nu] = AOs.labels<2>("all",1);
    auto nu_for_Q = tSubAO_AO_Q.label("all",0);
    auto nu_for_D = tSubAO_AO_D.label("all",0);
    auto nu_for_C = tSubAO_AO_C.label("all",0);

    tensor_type Q{X, mu, nu_for_Q(mu)};
    tensor_type D{mu, nu_for_D(mu)};
    tensor_type C{X, mu, nu_for_C(mu)};

    ProcGroup pg = ProcGroup::create_world_coll();
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    Scheduler sch{ec};

    Q.allocate(&ec);
    D.allocate(&ec);
    C.allocate(&ec);

    sch
    (D() = 42.0)
    (Q() = 2.0)
    (C(X, mu, nu_for_C(mu)) = Q(X, mu, nu_for_C(mu)) * D(mu, nu_for_C(mu)))
    .execute();

    std::cerr << "Tensor C:" << std::endl;
    print_tensor(C);
    std::cerr << "Tensor D" << std::endl;
    print_tensor(D);
    std::cerr << "Tensor Q" << std::endl;
    print_tensor(Q);

    sch
        (C() = 1.0)
    .execute();

    print_tensor(C);

    sch
        (C(X, mu, nu) = Q(X, mu, nu) * D(mu, nu))
    .execute();

    print_tensor(C);

    sch.deallocate(C, Q, D);
}
*/
using DepMap = std::map<IndexVector, TiledIndexSpace>;
// using TIS = TiledIndexSpace;

DepMap LMO_domain(const TiledIndexSpace& ref_space) {
  DepMap res = {{{0}, TiledIndexSpace{ref_space, IndexVector{0, 2, 3}}},
                {{1}, TiledIndexSpace{ref_space, IndexVector{0, 1}}}};

  return res;
}

DepMap AO_domain(const TiledIndexSpace& ref_space) {
  DepMap res = {{{1}, TiledIndexSpace{ref_space, IndexVector{1, 2, 4}}},
                {{3}, TiledIndexSpace{ref_space, IndexVector{0, 3, 4}}},
                {{4}, TiledIndexSpace{ref_space, IndexVector{2}}}};

  return res;
}

DepMap fitting_domain(const TiledIndexSpace& ref_space) {
  DepMap res = {{{1}, TiledIndexSpace{ref_space, IndexVector{0, 1, 4}}},
                {{2}, TiledIndexSpace{ref_space, IndexVector{2, 3, 4}}}};

  return res;
}

Tensor<T> cholesky(const Tensor<T>& tens) {
  Tensor<T> res;

  return res;
}

// TEST_CASE("Sample code for Local HF") {
//   // TAMM Scheduler construction
//   ProcGroup        pg = ProcGroup::create_world_coll();
//   ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
//   Scheduler        sch{ec};

//   // Dummy TiledIndexSpaces
//   TiledIndexSpace TAO{IndexSpace{range(5)}};
//   TiledIndexSpace TMO{IndexSpace{range(3)}};

//   // Local SCF TAMM Pseudo-code

//   // Input dense C tensor
//   Tensor<T> LMO{TAO, TMO}; // dense

//   LMO.allocate(&ec);

//   sch(LMO() = 42.0).execute();

//   // LMO_domain(): chooses AOs i -> mu
//   auto lmo_dep_map = LMO_domain(TAO);

//   // TiledIndexSpace lmo_domain{mu(i)}; //construct using explicit loop
//   TiledIndexSpace lmo_domain{TAO, {TMO}, lmo_dep_map}; // construct using explicit loop

//   // LMO_renormalize() {
//   auto [i, j]       = TMO.labels<2>("all");
//   auto [mu, nu]     = TAO.labels<2>("all");
//   auto [mu_p, nu_p] = lmo_domain.labels<2>("all");

//   Tensor<T> S_A{i, mu_p(i), nu_p(i)};
//   Tensor<T> S_v{i, mu, mu_p(i)};
//   Tensor<T> C{i, mu}; // column of LMO

//   sch.allocate(S_A, S_v, C)(S_A() = 1.0)(S_v() = 2.0)(C() = 21.0).execute();

//   // solved using Eigen

//   // Sparsified LMO
//   Tensor<T> LMO_renorm{mu_p(i), i}; // sparsified LMO
//   std::cout << "LMO_renorm - loop nest:" << std::endl;

//   for(const auto& blockid: LMO_renorm.loop_nest()) {
//     std::cout << "Block id: [ ";
//     for(auto& id: blockid) { std::cout << id << " "; }
//     std::cout << "]" << std::endl;
//   }
//   std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
//   sch
//     .allocate(LMO_renorm)
//     // (LMO_renorm(mu, i) = LMO(mu, i))
//     (LMO_renorm(mu, i) = 10.0)
//     .execute();
//   std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
//   print_tensor(LMO_renorm);
//   // }

//   // AO_domain(): constructs ao->ao index space
//   auto ao_screen_dep_map = AO_domain(TAO);

//   // TiledIndexSpace ao_int_screening{nu(mu)}; //ao->ao
//   TiledIndexSpace ao_int_screening{TAO, {TAO}, ao_screen_dep_map};

//   // //chain_maps(): compose lmo->ao and ao->ao
//   auto [nu_mu] = ao_int_screening.labels<1>("all");

//   // TiledIndexSpace ao_domain{nu(i)}; //mo->ao
//   // compose using labels
//   auto ao_domain = compose_lbl(mu_p(i), nu_mu(mu_p)); // nu(i) -> return label

//   // compose using TiledIndexSpaces
//   // auto ao_domain = compose_tis(lmo_domain, ao_int_screening); // -> return tis

//   // fitting domain
//   //  IndexSpace fb; //fitting basis. this is already available and used as input
//   auto lmo_to_fit_dep_map = fitting_domain(TAO);

//   // Output:
//   // TiledIndexSpace lmo_to_fit{A(i)}; // mo-> fitting basis
//   TiledIndexSpace lmo_to_fit{TAO, {TMO}, lmo_to_fit_dep_map}; // mo->fitting basis
//   // continuing with build_K. first contraction “transformation step”

//   // TiledIndexSpace ao_to_lmo{i(mu)}; //
//   // invert using labels
//   auto ao_to_lmo = invert_lbl(mu_p(i)); // i(mu)
//   // invert using TiledIndexSpaces
//   // auto ao_to_lmo= invert_tis(lmo_domain);

//   // IndexLabel i(mu);//ao_to_lmo
//   auto [A, B] = lmo_to_fit.labels<2>("all");

//   // Construct matrix of Coulomb metric, J, only compute for AB pairs which share an lmo
//   auto fit_to_lmo = invert_lbl(A(i)); // i(A)

//   auto fit_to_ao = compose_lbl(fit_to_lmo, mu_p(i)); // mu(A)
//   auto B_p       = compose_lbl(fit_to_lmo, A(i));    // B(A)

//   // auto [B_p] = fit_to_fit.labels<1>("all");

//   // Input X (tensor with lambda function that calls libint)
//   Tensor<T> X{i, A(i), mu_p(i), nu_p(i)}; // internally project on i ?

//   // input J
//   Tensor<T> J{A, B_p(A)};

//   // results
//   Tensor<T> Q{i, A(i), mu_p(i)};
//   Tensor<T> QB{i, B(i), mu_p(i)};
//   Tensor<T> K{i, mu_p(i), nu_p(i)};
//   Tensor<T> Test{i};
//   Tensor<T> Q_inv{A(i), mu_p(i), i};

//   // std::cout << "Q_inv - loop nest" << std::endl;
//   // for(const auto& blockid : Q_inv.loop_nest()) {
//   //     std::cout << "blockid: [ ";
//   //     for(auto& id : blockid) {
//   //         std::cout << id << " ";
//   //     }
//   //     std::cout << "]" << std::endl;
//   // }
//   // std::cout << "Q - loop nest" << std::endl;
//   // for(const auto& blockid : Q.loop_nest()) {
//   //     std::cout << "blockid: [ ";
//   //     for(auto& id : blockid) {
//   //         std::cout << id << " ";
//   //     }
//   //     std::cout << "]" << std::endl;
//   // }
//   std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
//   sch.allocate(X, J, Q, QB, K, Test, Q_inv)(QB() = 10.0)(Q() = 1.0)(Test() = 2.0)(Q_inv() = 3.0)
//     .execute();
//   std::cout << "Printing Q" << std::endl;
//   print_tensor(Q);

//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);

// #if 0

//     sch
//         (Q(i, nu, mu) = 2.0)
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     sch
//         (Q(i, A, mu) = 3.0)
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     sch
//         (Q(i, nu, mu_p) = 4.0)
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     sch
//         (Q(i, A, mu_p) = 5.0)
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     std::cout << "Testing add op" << std::endl;
//     sch
//         (Q() += QB())
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     sch
//         (Q(i, nu, mu) += QB(i, nu, mu))
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     sch
//         (Q(i, A, mu) += QB(i, A, mu))
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     sch
//         (Q(i, nu, mu_p) += QB(i, nu, mu_p))
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     sch
//         (Q(i, A, mu_p) += QB(i, A, mu_p))
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     std::cout << "Testing mult op" << std::endl;
//     sch
//         (Q() += Test() * QB())
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     sch
//         (Q(i, nu, mu) += Test(i) * QB(i, nu, mu))
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     sch
//         (Q(i, A, mu) += Test(i) * QB(i, A, mu))
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     sch
//         (Q(i, nu, mu_p) += Test(i) * QB(i, nu, mu_p))
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);

//     sch
//         (Q(i, A, mu_p) += Test(i) * QB(i, A, mu_p))
//     .execute();
//     std::cout << "Printing Q" << std::endl;
//     print_tensor(Q);
// #endif
// #if 1

//   sch(Q_inv(nu, mu, i) = 2.0).execute();
//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);

//   sch(Q_inv(A, mu, i) = 3.0).execute();
//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);

//   sch(Q_inv(nu, mu_p, i) = 4.0).execute();
//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);

//   sch(Q_inv(A, mu_p, i) = 5.0).execute();
//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);

//   std::cout << "Testing add op" << std::endl;
//   // sch
//   //     (Q() += QB())
//   // .execute();
//   // std::cout << "Printing Q_inv" << std::endl;
//   // print_tensor(Q_inv);

//   sch(Q_inv(nu, mu, i) += QB(i, nu, mu)).execute();
//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);

//   sch(Q_inv(A, mu, i) += QB(i, A, mu)).execute();
//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);

//   sch(Q_inv(nu, mu_p, i) += QB(i, nu, mu_p)).execute();
//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);

//   sch(Q_inv(A, mu_p, i) += QB(i, A, mu_p)).execute();
//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);

//   std::cout << "Testing mult op" << std::endl;
//   // sch
//   //     (Q() += Test() * QB())
//   // .execute();
//   // std::cout << "Printing Q_inv" << std::endl;
//   // print_tensor(Q_inv);

//   sch(Q_inv(nu, mu, i) += Test(i) * QB(i, nu, mu)).execute();
//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);

//   sch(Q_inv(A, mu, i) += Test(i) * QB(i, A, mu)).execute();
//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);

//   sch(Q_inv(nu, mu_p, i) += Test(i) * QB(i, nu, mu_p)).execute();
//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);

//   sch(Q_inv(A, mu_p, i) += Test(i) * QB(i, A, mu_p)).execute();
//   std::cout << "Printing Q_inv" << std::endl;
//   print_tensor(Q_inv);
// #endif
// #if 0

//     sch.allocate(Q, QB, K);
//     // foreach Index i in TMO:
//     for(Index i_val : TMO){
//         Tensor<T> J_i{A(i_val), B(i_val)};
//         Tensor<T> G_i_inv{A(i_val), B(i_val)};
//         sch
//         .allocate(J_i, G_i_inv)         // Q: how to allocate within a loop?
//             (J_i(A(i_val), B(i_val)) = J(A(i_val), B(i_val)))
//         .execute();

//         G_i_inv = invert_tensor(cholesky(J_i));

//         sch
//             (QB(B(i_val), mu(i_val), i_val) += G_i_inv(B(i_val), A(i_val)) * Q(A(i_val),
//             mu(i_val), i_val))
//         .deallocate(J_i, G_i_inv)
//         .execute();
//     }

//     sch
//         (K(mu, nu, i) += QB(A, mu, i) * QB(A, nu, i))
//     .execute();
// #endif

//   Tensor<T>::deallocate(X, J, Q, QB, K, Test, Q_inv, LMO, S_A, S_v, C, LMO_renorm);
// }

TEST_CASE("Test case for getting ExecutionContext from a Tensor") {
  TiledIndexSpace AO{IndexSpace{range(10)}, 2};

  Tensor<double> T0{AO, AO};
  Tensor<double> T1{AO, AO};

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  T0.allocate(&ec);

  auto t0_ec = T0.execution_context();

  std::cout << "EC Ptr: " << &ec << std::endl;
  std::cout << "T0 Ptr: " << t0_ec << std::endl;
  REQUIRE(&ec == t0_ec);

  auto t1_ec = T1.execution_context();
  REQUIRE(t1_ec == nullptr);

  T0.deallocate();
}

// TEST_CASE("Testing Dependent TiledIndexSpace contractions") {
//   using DependencyMap = std::map<IndexVector, TiledIndexSpace>;

// #if 1
//   TiledIndexSpace AO{IndexSpace{range(7)}};
//   TiledIndexSpace MO{IndexSpace{range(10)}};

//   DependencyMap depMO_1 = {{{0}, {TiledIndexSpace{MO, IndexVector{1, 4, 5}}}},
//                            {{2}, {TiledIndexSpace{MO, IndexVector{0, 3, 6, 8}}}},
//                            {{5}, {TiledIndexSpace{MO, IndexVector{2, 4, 6, 9}}}}};

//   DependencyMap depMO_2 = {{{1}, {TiledIndexSpace{MO, IndexVector{0, 1, 4, 5, 8}}}},
//                            {{2}, {TiledIndexSpace{MO, IndexVector{0, 6, 8}}}},
//                            {{3}, {TiledIndexSpace{MO, IndexVector{2, 5, 7}}}}};

//   DependencyMap depMO_3 = {{{0}, {TiledIndexSpace{MO, IndexVector{0, 1, 4, 5, 8}}}},
//                            {{2}, {TiledIndexSpace{MO, IndexVector{0, 6, 8}}}},
//                            {{3}, {TiledIndexSpace{MO, IndexVector{1, 3, 7, 9}}}},
//                            {{4}, {TiledIndexSpace{MO, IndexVector{2, 4, 7}}}},
//                            {{7}, {TiledIndexSpace{MO, IndexVector{1, 5, 7}}}}};
// #else
//   TiledIndexSpace AO{IndexSpace{range(4)}};
//   TiledIndexSpace MO{IndexSpace{range(4)}};

//   DependencyMap depMO_1 = {{{1}, {TiledIndexSpace{MO, IndexVector{1}}}}};

//   DependencyMap depMO_2 = {{{2}, {TiledIndexSpace{MO, IndexVector{2}}}}};
//   DependencyMap depMO_3 = {{{3}, {TiledIndexSpace{MO, IndexVector{3}}}}};
// #endif
//   TiledIndexSpace MO_AO_1{MO, {AO}, depMO_1};
//   TiledIndexSpace MO_AO_2{MO, {AO}, depMO_2};
//   TiledIndexSpace MO_MO_1{MO, {MO}, depMO_3};

//   auto [i, j]         = AO.labels<2>("all");
//   auto [mu, nu]       = MO.labels<2>("all");
//   auto [mu_i, nu_i]   = MO_AO_1.labels<2>("all");
//   auto [mu_k, nu_j]   = MO_AO_2.labels<2>("all");
//   auto [mu_nu, nu_mu] = MO_MO_1.labels<2>("all");

//   ProcGroup        pg = ProcGroup::create_world_coll();
//   ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
//   Scheduler        sch{ec};

//   // Same structure
//   Tensor<double> Q{i, mu_i(i)};
//   Tensor<double> P{i, mu_i(i)};
//   Tensor<double> T{i, mu_k(i)};
//   Tensor<double> FT{i, mu};

//   Q.allocate(&ec);
//   P.allocate(&ec);
//   T.allocate(&ec);
//   FT.allocate(&ec);

//   sch(Q() = 1.0)(P() = 2.0)(T() = 3.0)(FT() = 42.0).execute();
//   std::cerr << "Q Tensor" << std::endl;
//   print_tensor(Q);
//   std::cerr << "P Tensor" << std::endl;
//   print_tensor(P);
//   std::cerr << "T Tensor" << std::endl;
//   print_tensor(T);
//   std::cerr << "FT Tensor" << std::endl;
//   print_tensor(FT);

//   sch(T(i, mu) += FT(i, mu))(Q(i, mu) += FT(i, mu)).execute();

//   std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
//   std::cerr << "T Tensor" << std::endl;
//   print_tensor(T);
//   std::cerr << "Q Tensor" << std::endl;
//   print_tensor(Q);

//   sch
//     //(Q(i, mu_i(i)) += P(i,mu_i(i)))
//     (Q(i, mu) += 2 * P(i, mu))(T(i, mu) += 0.5 * Q(i, mu))(P(i, mu) += T(i, mu) * Q(i, mu))
//       .execute();

//   std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
//   std::cerr << "Q Tensor" << std::endl;
//   print_tensor(Q);
//   std::cerr << "T Tensor" << std::endl;
//   print_tensor(T);
//   std::cerr << "P Tensor" << std::endl;
//   print_tensor(P);

//   Tensor<double>::deallocate(Q, P, T, FT);
// }

TEST_CASE("Test for apply_ewise") {
  IndexSpace      MO{range(10), {{"occ", {range(0, 5)}}, {"virt", {range(5, 10)}}}};
  TiledIndexSpace tMO{MO};

  auto [i, j]           = tMO.labels<2>("all");
  auto [i_virt, j_virt] = tMO.labels<2>("virt");

  Tensor<double> T{i, j};

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  Scheduler sch{ec};

  sch.allocate(T)(T() = 42.0)(T(i_virt, j_virt) = 21.0).execute();

  std::cout << "Printing tensor T" << std::endl;
  print_tensor(T);

  Tensor<double> Temp = tamm::scale(T(i_virt, j_virt), 0.1);
  std::cout << "Printing tensor Temp" << std::endl;
  print_tensor(Temp);
  check_value(Temp, 2.1);

  Tensor<double>::deallocate(T, Temp);
}

TEST_CASE("Testing fill_sparse_tensor") {
  using DependencyMap = std::map<IndexVector, TiledIndexSpace>;

  TiledIndexSpace AO{IndexSpace{range(7)}};
  TiledIndexSpace MO{IndexSpace{range(10)}};

  DependencyMap depMO_1 = {{{0}, {TiledIndexSpace{MO, IndexVector{1, 4, 5}}}},
                           {{2}, {TiledIndexSpace{MO, IndexVector{0, 3, 6, 8}}}},
                           {{5}, {TiledIndexSpace{MO, IndexVector{2, 4, 6, 9}}}}};

  TiledIndexSpace MO_AO_1{MO, {AO}, depMO_1};

  auto [i, j]       = AO.labels<2>("all");
  auto [mu, nu]     = MO.labels<2>("all");
  auto [mu_i, nu_i] = MO_AO_1.labels<2>("all");

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  Scheduler        sch{ec};

  // Same structure
  Tensor<double> Q{i, mu_i(i)};
  Tensor<double> P{i, mu_i(i)};
  Tensor<double> T{i, mu};

  sch.allocate(Q, P, T)(Q() = 1.0)(P() = 2.0)(T() = 3.0).execute();

  std::cout << "Q Tensor" << std::endl;
  print_tensor(Q);
  std::cout << "P Tensor" << std::endl;
  print_tensor(P);
  std::cout << "T Tensor" << std::endl;
  print_tensor(T);

  fill_sparse_tensor<double>(Q, lambda_function);

  std::cout << "Q Tensor" << std::endl;
  print_tensor(Q);
}
#endif

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  doctest::Context context(argc, argv);

  int res = context.run();

  tamm::finalize();

  return res;
}
