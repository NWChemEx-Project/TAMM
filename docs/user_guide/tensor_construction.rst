Tensor Construction
====================

This section describes the syntax and semantics for the
tensor notation in TAMM. This section also describes how sparse
tensor construction and usage is done in TAMM through dependent
``TiledIndexSpace`` objects.

Tensor is the main computation and storage data structure in TAMM. The
main constructs for creating a ``Tensor`` object is using
``TiledIndexSpace``\ s or ``TiledIndexLabel``\ s for each dimension. For
dense case, the construction uses independent ``TiledIndexSpace``\ s or
labels related to these spaces. 

Using Labels
------------

``TiledIndexLabel`` pairs a TiledIndexSpace with an integer label. These
labels can be created using TiledIndexSpace methods: ``labels<N>(...)``
and ``label(...)``. (Note that ``labels<N>(...)`` method starts label
value ``0`` by default if no value is provided,( this might end up
problematic label creation if used on the same tiled index space
multiple times.). These objects are the main components for describing a
Tensor storage and computation over these tensors.

Labeling comes handy in the use of dependent ``TiledIndexSpace`` for
constructing sparse tensors or describing sparse computation over the
tensors. A label for a dependent ``TiledIndexLabel`` uses secondary
labels within parenthesis to describe the dependency in the tensor
description. For dense cases, ``TiledIndexSpace``\ s can used for
constructing the tensors but in case of sparse tensor construction the
only option is to use labeling for describing the dependencies between
dimensions.

.. code:: cpp

   using tensor_type = Tensor<double>;

   // Create labels assuming MO and depAO are defined
   auto [i, j] = MO.labels<2>("all");
   auto mu = depAO.label("all");

   // Dense tensor construction
   tensor_type T1{i, j}; // MO x MO
   tensor_type T2{MO, MO}; // MO x MO

   // Sparse tensor construction
   // mu(i) will construct a dependent TiledIndexLabel which is validated internally.
   tensor_type T3{i, mu(i)} // Perhaps don't use dependancies in the first example?


Using TiledIndexSpace
---------------------

The main construction for ``Tensor`` objects are based on a list of
``TiledIndexSpace`` objects for each mode (dimension) of the tensor.
Users can use operator overloads to get a specific portion of
``TiledIndexSpace`` (any named sub-space specified in index space is
tiled as well e.g. \ ``occ``, ``virt`` etc.) while constructing Tensor
objects. Note that the generated ``TiledIndexSpace`` will have the same
tiling size as the parent index space.

.. code:: cpp

   // Create TiledIndexSpace objects assuming an index space 
   // for AO and MO is constructed 
   TiledIndexSpace MO{MO_is, 10}, AO{AO_is, 10};

   Tensor<double> A{MO, AO}; // 2-mode tensor (MO by AO index space) with double elements

   Tensor<double> B{AO, AO, AO}; // 3-mode tensor (AO index space for all three modes) with double elements 

   Tensor<double> C{MO("occ"), AO}; // 2-mode tensor (occupied MO by AO) with double elements

Using TiledIndexLabel
---------------------

Users can also construct a ``Tensor`` object using ``TiledIndexLabel``
object related to a ``TiledIndexSpace``. This is just a convenience
constructor for independent index spaces, internally ``Tensor`` object
will use ``TiledIndexSpace`` objects.

.. code:: cpp

   // Create TiledIndexSpace objects
   TiledIndexSpace MO{MO_is, 10}, AO{AO_is, 10};

   // Create TiledIndexLabel objects
   TiledIndexLabel i, j, k, l, a, b;

   // Relate labels with TiledIndexSpace objects
   // Multiple labels at once
   std::tie(i, j) = MO.labels<2>("all");
   // Single label for different portions
   k = MO.label("occ");
   l = MO.label("virt");

   std::tie(a, b) = AO.labels<2>("all");

   Tensor<double> A{i, a}; // 2-mode tensor (MO by AO index space) with double elements

   Tensor<double> B{a, a, a}; // 3-mode tensor (AO index space for all three modes) with double elements 

   Tensor<double> C{k, a}; // 2-mode tensor (occupied MO by AO) with double elements

For ``Tensor`` objects over **dependent** index spaces can only be done
using ``TiledIndexLabel``\ s as the construction will dependent on the
relation between ``TiledIndexSpace``\ s.

.. code:: cpp

   // Creating index spaces MO, AO, and Atom
   IndexSpace MO_is{range(0, 100),
               {{"occ", range(0, 50)},
               {"virt", range(50, 100)}}};

   IndexSpace Atom_is{range(0, 5)};
   // Tile Atom space with tiling size of 3
   TiledIndexSpace T_Atom{Atom_is, 3};

   // Construct dependency relation for Atom indices
   std::map<IndexVector, IndexSpace> dep_relation{
       {IndexVector{0}, MO_is("occ")},                   
       {IndexVector{1}, MO_is("virt")}
   };


   // IndexSpace(const std::vector<TiledIndexSpace>& dep_spaces,
   //            const std::map<IndexVector, IndexSpace> dep_relation)
   IndexSpace subMO_Atom_is{{T_Atom}, dep_relation};

   TiledIndexSpace T_subMO_Atom{subMO_Atom_is, 3}

   TiledIndexLabel a = T_subMO_Atom.label("all");
   TiledIndexLabel i = T_Atom.label("all");

   // 2-mode tensor (subMO_Atom by Atom index space) with double elements
   Tensor<double> T{i, a(i)}; 

Specialized constructors
------------------------

For now only specialization for ``Tensor`` object construction is having
a lambda expression for on-the-fly calculated ``Tensor``\ s. **Note
that** these tensors are not stored in memory, they are only read-only
objects that can only by on the right hand side of a computation.

.. code:: cpp

   // Create TiledIndexSpace objects 
   TiledIndexSpace MO{MO_is, 10}, AO{AO_is, 10};

   // 2-mode tensor (MO by AO index space) with 
   // double elements and specialized lambda expression
   Tensor<double> A{{MO, AO}, [] (const IndexVector& block_id, span<T> buf){ /* lambda body*/ }};

   // Lambda expression definition
   auto one_body_overlap_integral_lambda = [] (const IndexVector& block_id, span<T> buf) { /* lambda body*/ };

   // 2-mode tensor (AO by MO index space) with
   // double elements and specialized lambda expression
   Tensor<double> B{{AO, MO}, one_body_overlap_integral_lambda};

Tensor Allocation and Deallocation
----------------------------------

For allocating and deallocating a ``Tensor`` object is explicitly done
using an ``ExecutionContext`` constructed by TAMM memory manager and
distribution:

.. code:: cpp

   // Constructing process group, memory manager, distribution to construct 
   // an execution context for allocation
   ProcGroup        pg = ProcGroup::create_world_coll();
   ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

   // We also provide a utility function that constructs 
   // an ExecutionContext object with default process group, 
   // memory manager and distribution
   auto ec_default = tamm::make_execution_context(); 

   TiledIndexSpace MO{/*...*/};

   auto O = MO("occ");
   auto V = MO("virt");
   auto N = MO("all");

   Tensor<double> d_f1{N, N, N, N};
   Tensor<double> d_r1{O, O, O, O};

   // Tensor allocation using static methods
   Tensor<double>::allocate(&ec, d_r1, d_f1);

   /* Do work on tensors */

   // Deallocation for tensors d_r1 and d_f1
   Tensor<double>::deallocate(d_r1, d_f1);


   // Tensor allocation using Tensor object member functions
   d_r1.allocate(&ec);
   d_f1.allocate(&ec);

   /* Do work on tensors */

   // Deallocation for tensors d_r1 and d_f1
   d_r1.deallocate();
   d_f1.deallocate();

   // Tensor allocation using Scheduler member functions

   Scheduler{&ec}
   // Allocate tensors
   .allocate(d_r1, d_f1)
   (/*Do work on tensors*/)
   // Deallocate the tensors (unless will be used afterwards)
   .deallocate(d_r1, d_f1)
   .execute();


**Note:** The tensors are has to be explicitly allocated using the
specified execution context before being used and they should be
deallocated once their use is finished. Furthermore, allocating a tensor
that is either allocated or has been deallocated is an error. A tensor
can be allocated and then deallocated only once.

Tensors that are not explicitly deallocated are registered for
deallocation in the execution context that was used to deallocate them.
The member function ``flush_and_sync`` of an execution context can be
used to deallocate tensors that cannot be referenced anymore. Finally,
if any tensors were allocated but not deallocated, ``flush_and_sync``
should be called to avoid memory and resource leaks. When calling
library functions that can create tensors, ``flush_and_sync`` should be
called unless it is known that the called functions did not postpone
deallocation of any tensors.

Tensor Accessors
-----------------

TAMM provides tensor accessors based on the ``TiledIndexSpace``\ s used
for construction, as a result the block IDs provided to any accessor
will correspond to the tile ID for each mode of ``Tensor`` object.

.. code:: cpp

   TiledIndexSpace MO{/*...*/};

   TiledIndexSpace O = MO("occ");
   TiledIndexSpace V = MO("virt");
   TiledIndexSpace N = MO("all");

   Tensor<double> d_f1{N, N, N, N};
   Tensor<double> d_r1{O, O, O, O};

   // Allocation for the tensors d_r1 and d_f1
   Tensor<double>::allocate(&ec, d_r1, d_f1);

   // Construct a block ID using the tile indices for each mode
   IndexVector blockId{0, 0, 0, 0};

   // Get the size of the corresponding block
   size_t size = d_r1.block_size(blockId);

   // Construct the data to put 
   std::vector<double> buff{size};

   // Read data from a source
   ReadData(buff, size);

   // Put a value to a block of tensor d_r1
   d_r1.put(blockId, buff);   // internally buff will be converted to a span 

   // Similarly, users can read from the tensor 
   std::vector<double> readBuff{size};
   d_r1.get(blockId, readBuff);

   // Or can do an accumulate on the tensor
   d_r1.add(blockId, buff);

   // Deallocation for tensors d_r1 and d_f1
   Tensor<double>::deallocate(d_r1, d_f1);

.. raw:: html

   <!-- ## Operation Syntax

   **SchedulerDAG and execution within method**
   ```c++
   void ccsd_e(ExecutionContext &ec, const TiledIndexSpace &MO, Tensor<T> &de,
               const Tensor<T> &t1, const Tensor<T> &t2, const Tensor<T> &f1,
               const Tensor<T> &v2)
   {
       const TiledIndexSpace &O = MO("occ");
       const TiledIndexSpace &V = MO("virt");
       Tensor<T> i1{O, V};

       TiledIndexLabel p1, p2, p3, p4, p5;
       TiledIndexLabel h3, h4, h5, h6;

       std::tie(p1, p2, p3, p4, p5) = MO.labels<5>("virt");
       std::tie(h3, h4, h5, h6) = MO.labels<4>("occ");

       Scheduler sch{ec};

       SchedulerDAG ccsd_e_dag;
       ccsd_e_dag.input(t1, t2, f1, v2);
       ccsd_e_dag.output(de);

       ccsd_e_dag.set_lambda([...](...) {
           i1(h6, p5) = f1(h6, p5);
           i1(h6, p5) += 0.5 * t1(p3, h4) * v2(h4, h6, p3, p5);
           de() = 0;
           de() += t1(p5, h6) * i1(h6, p5);
           de() += 0.25 * t2(p1, p2, h3, h4) * v2(h3, h4, p1, p2);
       });

       sch.execute(ccsd_e_dag(de, t1, t2, f1, v2));
   }
   ```
   -------
   **Using methods as Lambda Expression for SchedulerDAG**

   ```c++
   auto ccsd_e(const TiledIndexSpace& MO, Tensor<T>& de, const Tensor<T>& t1,
               const Tensor<T>& t2, const Tensor<T>& f1, const Tensor<T>& v2) {
       const TiledIndexSpace& O = MO("occ");
       const TiledIndexSpace& V = MO("virt");
       Tensor<T> i1{O, V};

       TiledIndexLabel p1, p2, p3, p4, p5;
       TiledIndexLabel h3, h4, h5, h6;

       std::tie(p1, p2, p3, p4, p5) = MO.labels<5>("virt");
       std::tie(h3, h4, h5, h6)     = MO.labels<4>("occ");

       i1(h6, p5) = f1(h6, p5);
       i1(h6, p5) += 0.5 * t1(p3, h4) * v2(h4, h6, p3, p5);
       de() = 0;
       de() += t1(p5, h6) * i1(h6, p5);
       de() += 0.25 * t2(p1, p2, h3, h4) * v2(h3, h4, p1, p2);
   }

   void ccsd_driver() {
       IndexSpace MO_IS{range(0, 200),
                        {{"occ", {range(0, 100)}}, {"virt", {range(100, 200)}}}};
       TiledIndexSpace MO{MO_IS, 10};

       const TiledIndexSpace& N = MO("all");
       const TiledIndexSpace& O = MO("occ");
       const TiledIndexSpace& V = MO("virt");

       Tensor<double> de{};
       Tensor<double> f1{N, N};
       Tensor<double> v2{N, N, N, N};
       Tensor<T> d_t1{V, O};
       Tensor<T> d_t2{V, V, O, O};

       ExecutionContext ec;

       Tensor::allocate(de, f1, v2, d_t1, d_t2);

       SchedulerDAG ccsd_e_dag{ccsd_e};
       ccsd_e_dag.input(t1, t2, f1, v2);
       ccsd_e_dag.output(de);

       // option 1
       Scheduler::execute(ec, ccsd_e_dag(MO, de, t1, t2, f1, v2));
       // option 2
       Scheduler sch(ec);
       sch.execute(ccsd_e_dag(MO, de, t1, t2, f1, v2));
       ///////////////////////////////////////////////////

       Tensor<double> new_f1{N, N};
       new_f1.allocate(ec);
       sch.execute(ccsd_e_dag(MO, de, t1, t2, new_f1, v2));
   }
   ``` -->

Local Tensor Construction
------------------------------

TAMM also provides a rank local tensor implementation called ``LocalTensor<T>`` 
that allows to construct a tensor that resides in each rank. While the constructors
for this specialized tensor is very similar to default distributed tensors, users
can have element-wise operaitons over these tensors as they are locally allocated. 
Different than the default tensor constructors, users can choose to use size values
to construct correspond tensors. 

.. code:: cpp

   // Tensor<T> B{tis1, tis1};
   // Local tensor construction using TiledIndexSpaces
   LocalTensor<T> local_A{tis1, tis1, tis1};
   LocalTensor<T> local_B{B.tiled_index_spaces()};
   // Local tensor construction using TiledIndexLabels
   LocalTensor<T> local_C{i, j, l};
   size_t N = 20;
   // Local tensor construction using a size
   LocalTensor<T> local_D{N, N, N};
   LocalTensor<T> local_E{10, 10, 10};

Similar to general tensor objects in TAMM, ``LocalTensor`` objects have to be allocated.
While allocation/deallocation calls are the same with general Tensor constructs, users 
have to use an ``ExecutionContext`` object with ``LocalMemoryManager``. Below is an 
example of how the allocation for these tensors looks like

.. code:: cpp

   // Execution context with LocalMemoryManager
   ExecutionContext local_ec{sch.ec().pg(), DistributionKind::nw, MemoryManagerKind::local};
   // Scheduler constructed with the new local_ec
   Scheduler        sch_local{local_ec};
   // Allocate call using the local scheduler
   sch_local.allocate(local_A, local_B, local_C, local_D, local_E).execute();

Local Tensor Operations
-----------------------
The `LocalTensor` object provides various functionalities, such as retrieving blocks of data, 
resizing tensors, and element-wise access. A `LocalTensor<T>` object allows you to retrieve 
a block of data using the `block` method. This method has two variants: one for general 
multi-dimensional tensors and another specifically for 2-dimensional tensors.

.. code-block:: cpp

   // Extract block from a 3-D Tensor
   auto local_E = local_A.block({0, 0, 0}, {4, 4, 4});
   // Extract block from a 2-D Tensor
   auto local_F = local_B.block(0, 0, 4, 4);

In the example above, the first call to `block` extracts a `4x4x4` block starting at 
the offset `{0, 0, 0}`, while the second call directly specifies the start offset for 
the x and y axes, followed by the block dimensions.

Another special feature of `LocalTensor` objects is the ability to resize the tensor 
to a new size, while maintaining the same number of dimensions. Depending on the new size, 
values from the original tensor are automatically carried over. The examples below demonstrate 
resizing a local tensor to a smaller and then to a larger size. Note that resizing causes a new 
tensor to be allocated, and the corresponding data is copied over.

.. code-block:: cpp

   // Resize tensor to a smaller size
   local_A.resize(5, 5, 5);
   // Resize tensor to a larger size
   local_A.resize(N, N, N);

`LocalTensor` objects also support element-wise accessor methods, `get` and `set`. 
Unlike default TAMM tensors, all data in a `LocalTensor` resides in local memory, 
enabling element access via index location.

.. code-block:: cpp

   // Set values for the entire tensor using the local scheduler
   sch_local.allocate(local_A, local_B)
   (local_A() = 42.0)
   (local_B() = 21.0)
   .execute();

   // Set a specific value in the tensor
   local_A.set({0, 0, 0}, 1.0);

   // Retrieve a value from the tensor
   auto val = local_B.get(0, 0, 0);

   // Looping through tensor elements
   for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
         for (size_t k = 0; k < N; k++) {
            local_A.set({i, j, k}, local_B.get(i, j));
         }
      }
   }
   
The examples above illustrate element-wise operations. Users can perform scheduler-based 
operations with the local scheduler or define element-wise updates using loops.

`LocalTensor` object also allows copying from or to a distributed tensor object. This is
particularly useful in situations where users need a local copy of distributed
tensors to apply element-wise updates. Below is an example usage of this scenario:

.. code-block:: cpp

   // Distributed tensor constructor
   Tensor<T> dist_A{tN, tN, tN}; 
   // ... 
  
   // Local tensor construction
   LocalTensor<T> local_A{dist_A.tiled_index_spaces()};

   sch_local.allocate(local_A)
   .execute();

   // Copy from distributed tensor
   local_A.from_distributed_tensor(dist_A);

   // Apply updates
   sch_local
   (local_A() = 21.0)
   .execute();

   // Copy back to distributed tensor
   local_A.to_distributed_tensor(dist_A);


Block Sparse Tensor Construction
--------------------------------

TAMM supports the construction of general block sparse tensors using underlying 
`TiledIndexSpace` constructs. Users can specify non-zero blocks by providing a 
lambda function that replaces the block-wise `is_non_zero` check, which is internally 
called for each block operation (e.g., allocation, element-wise operations, 
tensor operations). This approach allows for efficient allocation of only non-zero 
blocks and optimized tensor operations on these portions.

The following code demonstrates how to define a custom lambda function to check 
for block sparsity and construct a block sparse tensor:

.. code-block:: cpp

   // List of index spaces for the tensor construction
   TiledIndexSpaceVec t_spaces{SpinTIS, SpinTIS};
   // Spin mask for the dimensions
   std::vector<SpinPosition> spin_mask_2D{SpinPosition::lower, SpinPosition::upper};

   // Custom lambda function for the is_non_zero check
   auto is_non_zero_2D = [t_spaces, spin_mask_2D](const IndexVector& blockid) -> bool {
       Spin upper_total = 0, lower_total = 0, other_total = 0;
       for (size_t i = 0; i < 2; i++) {
           const auto& tis = t_spaces[i];
           if (spin_mask_2D[i] == SpinPosition::upper) {
               upper_total += tis.spin(blockid[i]);
           } else if (spin_mask_2D[i] == SpinPosition::lower) {
               lower_total += tis.spin(blockid[i]);
           } else {
               other_total += tis.spin(blockid[i]);
           }
       }

       return (upper_total == lower_total);
   };

   // TensorInfo construction
   TensorInfo tensor_info{t_spaces, is_non_zero_2D};

   // Tensor constructor
   Tensor<T> tensor{t_spaces, tensor_info};

TAMM offers a more convenient `TensorInfo` struct to describe non-zero blocks 
using stringed sub-space constructs in `TiledIndexSpace`s. This simplifies the 
process of constructing block sparse tensors.

Here's an example of using `TensorInfo`:

.. code-block:: cpp

   // Map labels to corresponding sub-space strings
   Char2TISMap char2MOstr = {{'i', "occ"}, {'j', "occ"}, {'k', "occ"}, {'l', "occ"},
                             {'a', "virt"}, {'b', "virt"}, {'c', "virt"}, {'d', "virt"}};

   // Construct TensorInfo
   TensorInfo tensor_info{
       {MO, MO, MO, MO},                                 // Tensor dimensions
       {"ijab", "iajb", "ijka", "ijkl", "iabc", "abcd"}, // Allowed blocks
       char2MOstr                                        // Character to sub-space string mapping
       // ,{"abij", "aibj"} // Disallowed blocks (optional)
   };

   // Block Sparse Tensor construction
   Tensor<T> tensor{{MO, MO, MO, MO}, tensor_info};

TAMM also provides a simplified constructor that only requires a list of allowed 
blocks and the character-to-sub-space string map:

.. code-block:: cpp

   // Block Sparse Tensor construction using allowed blocks
   Tensor<T> tensor{{MO, MO, MO, MO}, {"ijab", "ijka", "iajb"}, char2MOstr};

Block Sparse `Tensor` inherits from general TAMM tensor constructs, enabling the application
of standard tensor operations to block sparse tensors. Users can employ labels over the entire 
`TiledIndexSpace` for general computations or use sub-space labels to access specific blocks.

The following code illustrates how to allocate, set values, and perform operations on different 
blocks of block sparse tensors:

.. code-block:: cpp

   // Construct Block Sparse Tensors with different allowed blocks
   Tensor<T> tensorA{{MO, MO, MO, MO}, {"ijab", "ijkl"}, char2MOstr};
   Tensor<T> tensorB{{MO, MO, MO, MO}, {"ijka", "iajb"}, char2MOstr};
   Tensor<T> tensorC{{MO, MO, MO, MO}, {"iabc", "abcd"}, char2MOstr};

   // Allocate and set values
   sch.allocate(tensorA, tensorB, tensorC)
       (tensorA() = 2.0)
       (tensorB() = 4.0)
       (tensorC() = 0.0)
   .execute();

   // Use different blocks to update output tensor
   // a, b, c, d: MO virtual space labels
   // i, j, k, l: MO virtual space labels
   sch
       (tensorC(a, b, c, d) += tensorA(i, j, a, b) * tensorB(j, c, i, d))
       (tensorC(i, a, b, c) += 0.5 * tensorA(j, k, a, b) * tensorB(i, j, k, c))
   .execute();

   // De-allocate tensors
   tensorA.deallocate();
   tensorB.deallocate();
   tensorC.deallocate();

TAMM also provides block sparse constructors similar to the general tensor construction 
by allowing use of TiledIndexLabels, TiledIndexSpaces, and strings corresponding to the 
sub-space names in TiledIndexSpaces for representing only the allowed blocks. With these 
constructors users don't have to provide a mapping from char to corresponding sub-space 
names as they are provided explicitly. Below code shows the use of this constructions, 
similar to previous case block sparse tensors constructed using these methods can be 
directly used in any tensor operations for general tensors:

.. code-block:: cpp

   // Construct Block Sparse Tensors with different allowed blocks
   // Using TiledIndexLabels for allowed blocks 
   Tensor<T> tensorA{{MO, MO, MO, MO}, {{i, j, a, b}, {i, j, k, l}}}; 
   // Using TiledIndexSpaces for allowed blocks
   TiledIndexSpace Occ = MO("occ");
   TiledIndexSpace Virt = MO("virt");
   Tensor<T> tensorB{{MO, MO, MO, MO}, 
               {TiledIndexSpaceVec{Occ, Occ, Occ, Occ}, 
                TiledIndexSpaceVec{Occ, Virt, Occ, Virt}}};
   // Using list of comma seperated strings representing sub-space names
   Tensor<T> tensorC{{MO, MO, MO, MO}, {{"occ, virt, virt, virt"}, 
                                        {"virt, virt, virt, virt"}}};

   // ...

Example Tensor Constructions
----------------------------

Basic examples
~~~~~~~~~~~~~~

   1. scalar

.. code:: cpp

   // Construct a scalar value 
   Tensor T_1{};

..

   2. vector of length 10

.. code:: cpp

   // Create an index space of length 10
   IndexSpace is_2{range(10)};
   // Apply default tiling
   TiledIndexSpace tis_2{is_2};
   // Create a vector with index space is_2
   Tensor T_2{tis_2};

..

   3. matrix that is 10 by 20

.. code:: cpp

   // Create an index space of length 10 and 20
   IndexSpace is1_3{range(10)};
   IndexSpace is2_3{range(20)};
   // Apply default tiling
   TiledIndexSpace tis1_3{is1_3}, tis2_3{is2_3};
   // Create a matrix on tiled index spaces tis1_3, tis2_3
   Tensor T_3{tis1_3, tis2_3};

..

   4. order 3 tensor that is 10 by 20 by 30

.. code:: cpp

   // Create an index space of length 10, 20 and 30
   IndexSpace is1_4{range(10)};
   IndexSpace is2_4{range(20)};
   IndexSpace is3_4{range(30)};
   // Apply default tiling
   TiledIndexSpace tis1_4{is1_4}, tis2_4{is2_4}, tis3_4{is3_4};
   // Construct order 3 tensor in tiled index spaces tis1_4, tis2_4 and tis3_4
   Tensor T_4{tis1_4, tis2_4, tis3_4};

..

   5. vector from 2 with subspaces of length 4 and 6

.. code:: cpp

   // Spliting is_2 into two sub-spaces with 4 and 6 elements
   IndexSpace is1_5{is_2, range(0, 4)};
   IndexSpace is2_5{is_2, range(4, is_2.size())};
   // Create index space combining sub-spaces
   IndexSpace is3_5{{is1_5, is2_5}};
   // Apply default tiling 
   TiledIndexSpace tis_5{is3_5};
   // Create a vector over combined index space
   Tensor T_5{tis1_5};

..

   6. matrix from 3 whose rows are split into two subspaces of length 4
      and 6

.. code:: cpp

   // Spliting is1_3 from 3 into two sub-spaces with 4 and 6 elements
   IndexSpace is1_6{is1_3, range(0, 4)};  
   IndexSpace is2_6{is1_3, range(4, is1_3.size()}; 
   // Create index space combining sub-spaces
   IndexSpace is3_6{{is1_6, is2_6}};
   // Apply default tiling
   TiledIndexSpace tis_6{is3_6};
   // Create a matrix with rows on combined tiled index space
   // columns on tis2_3 from 3
   Tensor T_6{tis_6, tis2_3};

..

   7. matrix from 3 whose columns are split into two subspaces of
      lengths 12 and 8

.. code:: cpp

   // Spliting is2_3 from 3 into two sub-spaces with 12 and 8 elements
   IndexSpace is1_7{is2_3, range(0, 12)};  
   IndexSpace is2_7{is2_3, range(12, is2_3.size())}; 
   // Create index space combining sub-spaces
   IndexSpace is3_7{{is1_7, is2_7}};
   // Apply default tiling
   TiledIndexSpace tis_7{is3_7};
   // Create a matrix with rows on tis1_3 from 3
   // columns on combined tiled index space
   Tensor T_7{tis1_3, tis_7};

..

   8. matrix from 3 having subspaces of both 6 and 7

.. code:: cpp

   // Create matrix on tis_6 from 6 and tis_7 from 7
   Tensor T_8{tis_6, tis_7};

..

   9. tensor with mode 0 split into subspaces of 4 and 6

.. code:: cpp

   // Create order 3 tensor using split version from 5
   // and full spaces from 4
   Tensor T_9{tis_5, tis2_4, tis3_4};

..

   10. tensor with mode 1 split into subspaces of 12 and 8

.. code:: cpp

   // Create order 3 tensor using split version from 7
   // and full spaces from 4
   Tensor T_10{tis1_4, tis_7, tis3_4};

..

   11. tensor with mode 2 split into subspaces of 13 and 17

.. code:: cpp

   // Split the index space form 4 into sub-spaces of length 13 and 17
   IndexSpace is1_11{is3_4, range(0, 13)};
   IndexSpace is2_11{is3_4, range(13, is3_4.size())};
   // Combine the sub-spaces into another index space
   IndexSpace is3_11{{is1_11, is2_11}};
   // Apply default tiling
   TiledIndexSpace tis_11{is3_11};
   // Create order 3 tensor using new split version
   // and full spaces from 4
   Tensor T_11{tis1_4, tis2_4, tis_11};

..

   12. Combine 9 and 10

.. code:: cpp

   // Create order 3 tensor using splits from 9 and 10
   // tis_5  --> split length 4 and 6
   // tis_7  --> split length 12 and 8
   // tis3_4 --> length 30 index space
   Tensor T12{tis_5,tis_7,tis3_4};

..

   13. Combine 9 and 11

.. code:: cpp

   // Create order 3 tensor using splits from 9 and 11
   // tis_5  --> split length 4 and 6
   // tis2_4 --> length 20 index space
   // tis_11 --> split length 13 and 17
   Tensor T13{tis_5,tis2_4,tis_11};

..

   14. Combine 10 and 11

.. code:: cpp

   // Create order 3 tensor using splits from 9 and 11
   // tis1_4 --> length 10 index space
   // tis_7  --> split length 12 and 8
   // tis_11 --> split length 13 and 17
   Tensor T14{tis1_4,tis_7,tis_11};

..

   15. Combine 9, 10, and 11

.. code:: cpp

   // Create order 3 tensor using splits from 9 and 11
   // tis_5  --> split length 4 and 6
   // tis_7  --> split length 12 and 8
   // tis_11 --> split length 13 and 17
   Tensor T15{tis_5,tis_7,tis_11};

..

   16. Vector from 2 with the first subspace split again into a
       subspaces of size 1 and 3

.. code:: cpp

   // Split the sub-space from 5 into another with size 1 and 3
   // is1_5  --> split of size 4
   // is2_5  --> split of size 6
   IndexSpace is1_16{is1_5, range(0,1)};
   IndexSpace is2_16{is1_5, range(1,3)};
   // Combine all into a full space
   IndexSpace is3_16{{is1_16, is2_16, is2_5}};
   // Apply default tiling
   TiledIndexSpace tis_16{is3_16};
   // Create a vector over new tiled index space
   Tensor T16{tis_16};

..

   17. matrix from 8 with the 4 by 12 subspace split further into a 1 by
       12 and a 3 by 12

.. code:: cpp

   // Create a matrix from splits from 16 and 7 
   // tis_16 --> split of size 1, 3 and 6
   // tis_7  --> split of size 12 and 8
   Tensor T17{tis_16, tis_7};

..

   18. vector from 1 where odd numbered elements are in one space and
       even numbered elements are in another

.. code:: cpp

   // Odd numbered elements from 1 to 9
   IndexSpace is1_18{range(1,10,2)};
   // Even numbered elements from 0 to 8
   IndexSpace is2_18{range(0,10,2)};
   // Aggregate odd and even numbered index spaces 
   IndexSpace is3_18{{is1_18, is2_18}};
   // Apply default tiling
   TiledIndexSpace tis3_18{is3_18};
   // Create a vector with tiled index space
   Tensor T18{tis3_18};

..

   19. matrix from 2 where odd rows are in one space even in another

.. code:: cpp

   // Odd numbered elements from 1 to 9
   IndexSpace is1_19{range(1,10,2)};
   // Even numbered elements from 0 to 8
   IndexSpace is2_19{range(0,10,2)};
   // Aggregate odd and even numbered index spaces 
   IndexSpace is3_19{{is1_19, is2_19}};
   // Apply default tiling
   TiledIndexSpace tis1_19{is3_19};
   // Create a matrix using tiled index space with odd and even numbered
   // elements as the row and tiled index space from 3 a columns
   Tensor T19{tis1_19, tis2_3};

..

   20. matrix from 6 that also has the odd rows in one space and the
       even in another

.. code:: cpp

   // Odd numbered elements from 1 to 9
   IndexSpace is1_20{range(1,10,2)};
   // Even numbered elements from 0 to 8
   IndexSpace is2_20{range(0,10,2)};
   // Aggregate odd and even numbered index spaces 
   IndexSpace is3_20{is1_20, is2_20};
   // Spliting is3_20 into two sub-spaces with 4 and 6 elements
   IndexSpace is4_20{is3_20, range(0, 4)};  
   IndexSpace is5_20{is3_20, range(4, is3_20.size())};
   // Aggregate split indexes
   IndexSpace is6_20{is4_20, is5_20};
   // Apply default tiling
   TiledIndexSpace tis1_20{is6_20};
   // Create a matrix using tiled index space with odd and even numbered
   // elements then splitted as the row and tiled index space 
   // from 3 a columns
   Tensor T20{tis1_20, tis2_3};


Dependent Index Spaces
~~~~~~~~~~~~~~~~~~~~~~

For ease of use, if the user provides a
dependent label without secondary labels the tensor will be constructed
over the reference ``TiledIndexSpace`` of the given dependent
``TiledIndexSpace``.

.. code:: cpp

   using tensor_type = Tensor<double>;
   auto [i, j] = MO.labels<2>("all");
   auto [A, B] = AO.labels<2>("all");
   auto [mu, nu] = depAO.labels<2>("all");

   // Dense tensor construction
   tensor_type T1{i, j};     // MO x MO Tensor
   tensor_type T2{i, A};     // MO x AO Tensor
   tensor_type T3{mu, nu};   // AO x AO Tensor
   tensor_type T4{mu, i};    // AO x MO Tensor

   // Sparse tensor construction
   // mu(i) will construct a dependent TiledIndexLabel which is validated internally.
   tensor_type T5{i, mu(i)}; // MO x depAO Tensor 

.. raw:: html

   <!-- 
   ### Canonical CCSD E

   ```c++
   // Up-to-date version can be found at ccsd/ccsd_driver.cc
   template<typename T>
   void ccsd_e(const TiledIndexSpace& MO, 
               Tensor<T>& de,
               const Tensor<T>& t1,
               const Tensor<T>& t2,
               const Tensor<T>& f1,
               const Tensor<T>& v2) {
       
       const TiledIndexSpace& O = MO("occ");
       const TiledIndexSpace& V = MO("virt");
       Tensor<T> i1{O, V};

       TiledIndexLabel p1, p2, p3, p4, p5;
       TiledIndexLabel h3, h4, h5, h6;

       std::tie(p1, p2, p3, p4, p5) = MO.labels<5>("virt");
       std::tie(h3, h4, h5, h6) = MO.labels<4>("occ");

       i1(h6,p5) = f1(h6,p5);
       i1(h6,p5) +=  0.5  * t1(p3,h4) * v2(h4,h6,p3,p5);
       de() =  0;
       de() += t1(p5,h6) * i1(h6,p5);
       de() +=  0.25  * t2(p1,p2,h3,h4) * v2(h3,h4,p1,p2);
   }

   template<typename T>
   void driver() {
       // Construction of tiled index space MO from skretch
       IndexSpace MO_IS{range(0,200), {"occ", {range(0,100)}, 
                                     "virt", {range(100,200)}}};
       TiledIndexSpace MO{MO_IS, 10};
       
       const TiledIndexSpace& O = MO("occ");
       const TiledIndexSpace& V = MO("virt");
       const TiledIndexSpace& N = MO("all");
       Tensor<T> de{};
       Tensor<T> t1{V, O};
       Tensor<T> t2{V, V, O, O};
       Tensor<T> f1{N, N};
       Tensor<T> v2{N, N, N, N};
       ccsd_e(MO, de, t1, t2, f1, v2);
   }
   ```

   ----

   ### Canonical  T1

   ```c++
   // Up-to-date version can be found at ccsd/ccsd_driver.cc
   template<typename T>
   void  ccsd_t1(const TiledIndexSpace& MO, 
                 Tensor<T>& i0, 
                 const Tensor<T>& t1, 
                 const Tensor<T>& t2,
             const Tensor<T>& f1, 
             const Tensor<T>& v2) { 

     const TiledIndexSpace& O = MO("occ");
     const TiledIndexSpace& V = MO("virt");
     Tensor<T> t1_2_1{O, O};
     Tensor<T> t1_2_2_1{O, V};
     Tensor<T> t1_3_1{V, V};
     Tensor<T> t1_5_1{O, V};
     Tensor<T> t1_6_1{O, O, V, V};

     TiledIndexLabel p2, p3, p4, p5, p6, p7;
     TiledIndexLabel h1, h4, h5, h6, h7, h8;

     std::tie(p2, p3, p4, p5, p6, p7) = MO.labels<6>("virt");
     std::tie(h1, h4, h5, h6, h7, h8) = MO.labels<6>("occ");  

     i0(p2,h1)             =   f1(p2,h1);
     t1_2_1(h7,h1)         =   f1(h7,h1);
     t1_2_2_1(h7,p3)       =   f1(h7,p3);
     t1_2_2_1(h7,p3)      +=   -1 * t1(p5,h6) * v2(h6,h7,p3,p5);
     t1_2_1(h7,h1)        +=   t1(p3,h1) * t1_2_2_1(h7,p3);
     t1_2_1(h7,h1)        +=   -1 * t1(p4,h5) * v2(h5,h7,h1,p4);
     t1_2_1(h7,h1)        +=   -0.5 * t2(p3,p4,h1,h5) * v2(h5,h7,p3,p4);
     i0(p2,h1)            +=   -1 * t1(p2,h7) * t1_2_1(h7,h1);
     t1_3_1(p2,p3)         =   f1(p2,p3);
     t1_3_1(p2,p3)        +=   -1 * t1(p4,h5) * v2(h5,p2,p3,p4);
     i0(p2,h1)            +=   t1(p3,h1) * t1_3_1(p2,p3);
     i0(p2,h1)            +=   -1 * t1(p3,h4) * v2(h4,p2,h1,p3);
     t1_5_1(h8,p7)         =   f1(h8,p7);
     t1_5_1(h8,p7)        +=   t1(p5,h6) * v2(h6,h8,p5,p7);
     i0(p2,h1)            +=   t2(p2,p7,h1,h8) * t1_5_1(h8,p7);
     t1_6_1(h4,h5,h1,p3)   =   v2(h4,h5,h1,p3);
     t1_6_1(h4,h5,h1,p3)  +=   -1 * t1(p6,h1) * v2(h4,h5,p3,p6);
     i0(p2,h1)            +=   -0.5 * t2(p2,p3,h4,h5) * t1_6_1(h4,h5,h1,p3);
     i0(p2,h1)            +=   -0.5 * t2(p3,p4,h1,h5) * v2(h5,p2,p3,p4);
   }

   template<typename T>
   void driver() {
       // Construction of tiled index space MO from skretch
       IndexSpace MO_IS{range(0,200), {"occ", {range(0,100)}, 
                                     "virt", {range(100,200)}}};
       TiledIndexSpace MO{MO_IS, 10};
       
       const TiledIndexSpace& O = MO("occ");
       const TiledIndexSpace& V = MO("virt");
       const TiledIndexSpace& N = MO("all");

       Tensor<T> i0{};
       Tensor<T> t1{V, O};
       Tensor<T> t2{V, V, O, O};
       Tensor<T> f1{N, N};
       Tensor<T> v2{N, N, N, N};
       ccsd_t1(MO, i0, t1, t2, f1, v2);
   }
   ```

   ### Canonical HF (work in progress)

   ***Note: We do not have an implementation of initial hcore guess (e.g., STO-nG basis assumption in Ed's toy code, etc.). What parts of that can use TAMM***

   ```c++
   void compute_2body_fock(const TiledIndexSpace& AO,
               const std::vector<libint2::Shell> &shells, 
               const Tensor<T> &D, Tensor<T> &F) {
     // auto will correspond to a TiledIndexSpace or
     // a TiledIndexRange depending on the decision
     const auto& N = AO("all");
     TiledIndexLabel s1, s2, s3, s4;
     std::tie(s1,s2, s3, s4) = AO.range_labels<4>("all");
     const auto n = shells.nbf();
     Tensor<T> G{N,N};
     //TODO: construct D from C
     // construct the 2-electron repulsion integrals engine
     Tensor<T> ERI{N, N, N, N, coulomb_integral_lambda};
     Scheduler()
     .fuse(PermGroup(,,,).iterator(),
         G(s1, s2) += D(s3, s4) * ERI(s1, s2, s3, s4),
         G(s3, s4) += D(s1, s2) * ERI(s1, s2, s3, s4),
         G(s1, s3) -= 0.25*D(s2,s4) * ERI(s1,s2,s3,s4),
         G(s2, s4) -= 0.25*D(s1,s3) * ERI(s1,s2,s3,s4),
         G(s1, s4) -= 0.25*D(s2,s3) * ERI(s1,s2,s3,s4),
         G(s2, s3) -= 0.25*D(s1,s4) * ERI(s1,s2,s3,s4)
         ).execute();
         

     // symmetrize the result and return   
     //Tensor<T> Gt{N,N};
     //Gt(a,b) = G(b,a); //G.transpose();
     F(s1,s2) += 0.5 * G(s1,s2);
     F(s1,s2) += 0.5 * G(s2,s1);
   }

   template<typename T>
   void hartree_fock(const TiledIndexSpace& AO, 
                     const Tensor<T>& C,
                     Tensor<T>& F) {
     const TiledIndexSpace& N = AO("all");
     const TiledIndexSpace& O = AO("occ");

     TiledIndexLabel a,b,c;
     TiledIndexLabel ao,bo,co;
     std::tie(a,b,c) = AO.range_labels<3>("all");
     std::tie(ao,bo,co) = AO.range_labels<3>("occ");
       
     // compute overlap integrals
     //Tensor<T> S{N,N};
     //S = compute_1body_ints(shells, Operator::overlap);
     Tensor<T> S{N,N, one_body_overlap_integral_lambda};
     // compute kinetic-energy integrals
     Tensor<T> T{N,N,one_body_kinetic_integral_lambda};
     //T = compute_1body_ints(shells, Operator::kinetic);
     // compute nuclear-attraction integrals
     //Tensor<T> V{N,N};
     //V = compute_1body_ints(shells, Operator::nuclear, atoms);
     Tensor<T> V{N,N, one_body_nuclear_integral_lambda};
     // Core Hamiltonian = T + V
     Tensor<T> H{N, N};
     H(a,b) = T(a,b);
     H(a,b) += V(a,b);
     
     Tensor<T> D{N, N};
     compute_soad(atoms, D); 
       
     const auto maxiter = 100;
     const auto conv = 1e-12;
     auto iter = 0;
     Tensor<T> ehf{},ediff{},rmsd{};
     Tensor<T> eps{N,N};

     do {
       ++iter;
       // Save a copy of the energy and the density
       Tensor<T> ehf_last{};
       Tensor<T> D_last{N,N};
       
       ehf_last() = ehf();
       D_last(a,b) = D(a,b);

       // build a new Fock matrix
       F(a,b) = H(a,b);
       compute_2body_fock(shells, D, F); //accumulate into F
       
       // solve F C = e S C
       //Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F, S);
       //eps = gen_eig_solver(F,S).eigenvalues();
       //C = gen_eig_solver(F,S).eigenvectors();
       std::tie(C, eps) = eigen_solve(F, S);

       // compute density, D = C(occ) . C(occ)T
       //C_occ(ao,bo) = C(ao,bo); //C.leftCols(ndocc);
       //C_occ_transpose(ao,bo) = C_occ(bo,ao);
       D(ao, bo) = C(ao, xo) * C(xo, bo);

       Tensor<T> tmp1{a, b}, tmp2{a, b};
       // compute HF energy
       //ehf += D(i, j) * (H(i, j) + F(i, j));
       ehf() = 0.0;
       tmp1(a,b) = H(a, b);
       tmp1(a,b) += F(a, b);
       ehf() = D(a,b) * tmp1(a,b);

       // compute difference with last iteration
       ediff() = ehf();
       ediff() = -1.0 * ehf_last();
       tmp2(a,b) = D(a,b);
       tmp2(a,b) += -1.0 * D_last(a,b);
       norm(tmp2,rmsd); //rmsd() = tmp2(a,b).norm();
       rmsd() = tmp2(a,b) * tmp2(a,b);

       //e.g.:Tensor<T> rmsd_local{AllocationModel::replicated};
       //e.g.:rmsd_local(a) = rmsd(a);
       //e.g.: rmsd(a) +=  rmsd_local(a);
       //TODO: only put rmsd_local in process 0 to rmsd
     } while (((fabs(get_scalar(ediff)) > conv) || (fabs(get_scalar(rmsd)) > conv)) && (iter < maxiter));
   }

   template<typename T>
   void driver() {
       // Construction of tiled index space MO from skretch
       IndexSpace AO_IS{range(0,200), {"occ", {range(0,100)}, 
                                     "virt", {range(100,200)}}};
       TiledIndexSpace AO{AO_IS, 10};

       const TiledIndexSpace& N = AO("all");
       
       Tensor<T> C{N, N};
       Tensor<T> F{N, N};
       hartree_fock(AO, C, F);
   }
   ```


   ### DLPNO CCSD (work in progress)
   ```c++
   double dlpno_ccsd(const TiledIndexSpace& AO, const TiledIndexSpace& MO, 
                     const TiledIndexSpace& AtomSpace, 
                     const TiledIndexSpace& SubMO, const TiledIndexSpace& SubPAO,
                     const Tensor<T>& S,
                     const Tensor<T>& C, const Tensor<T>& Ct,
                     const Tensor<T>& dmat, const Tensor<T>& F,
                     const Tensor<T>& I,
                     const Tensor<T>& Cvirtt){
         
       
       const TiledIndexSpace& N_ao = AO("all"); 
       const TiledIndexSpace& N_atom = AtomSpace("all");
       const TiledIndexSpace& N_pao = SubPAO("all");
       const TiledIndexSpace& O_mo = MO("occ");
       const TiledIndexSpace& O_submo = SubMO("occ");
       
       TiledIndexLabel mu, nu, mu_p, nu_p;
       TiledIndexLabel i, j, i_p, j_p;
       TiledIndexLabel A, B;

       std::tie(mu, nu) = AO.range_labels<2>("all");
       std::tie(mu_p, nu_p) = SubPAO.range_labels<2>("all");
       std::tie(i,j) = MO.range_labels<2>("occ");
       std::tie(i_p,j_p) = SubMO.range_labels<2>("occ");
       std::tie(A, B) = AtomSpace.range_labels<2>("all");
       
       Tensor<T> F{N_ao, N_ao};
       //Tensor<T> C{A, mu, i_p(A)};
       Tensor<T> C{N_ao, O_mo};
       Tensor<T> S{N_ao, N_ao};
       Tensor<T> TC_2e{N_ao, N_ao, N_aux, lambda};
       
       //Step 31
       Tensor<T> SC{N_ao, O_mo};
       Tensor<T> P{i, A};
       SC(mu, i) = S(mu, nu) * C(nu, i);
       P(i, A) = C(mu(A), i) * SC(mu(A), i);
       
       //middle step
       //make a map from the number of occupied MO to vector of atoms it is on. Now we know which atoms are associated with which occupied MO.
       
       //we now have SubMO dependent on AtomSpace
       TiledIndexLabel i_p; //..define
       Tensor<T> SCCS{N_ao, N_ao};
       //auto SC(A, mu, i_p) = S(mu, nu) * C(A, nu, i);    auto SCCS(mu, nu) = SC(mu, i) * SC(nu, i_p);
       
       Tensor<T> L{N_ao, N_ao};
       L(mu, nu) = S(mu, nu);
       L(mu, nu) += -1 * SCCS(mu,nu)

       //now we interpret L to construct the mu_tilde (mu_p) index space 
       
       //Step 2
       Tensor<T> l{N_atom, N_atom}; 
       //TODO: support for absolute sum is need
       l(A,B) = L(mu_subatom(A), mu_subatom(B));
       //l(i,j) = outer(fn(i,j)) * A(i, j);
       //e.g.: T1(mu_pp(m,n)) = A(n,m);
       //e.g.: Tensor<T> T5{i, a(i)};
       //e.g.: T5(x, y) = 0;
       
       //here we do a map from PAO to set of atoms around the PAO
       
       //Step 4 - ??
       // auto pairdoms =  form\_pair\_domains(occ2atom, atom2PAO);

   //Now we have the pair index space in terms of pairs of MOs: mu_p(i,j), nu_p(i,j), ... where (i,j) pairs are defined here.

       //Step 5: skip for now
       Tensor<T> Fpao{}, Focc{}, Tmp1{}, Tmp2{};

       Tmp1(mu_p, nu) = L(mu_subatom(A), mnu) * F(mu, nu);
       Fpao(mu_p, nu_p) = Tmp1(mu_p, nu) * L(nu, nu_p);
       Tmp(i_p, j_pnu) = C(i_, u) * F(mu, nu);
       Focc(i_p, j_p) = Tmp(i_p, nu) * C(nu, j_p);
       
       //EV: somehere here or above we need a canonicalization step
       
       //Step 6
       // auto I_imuP =  transform_tensor(I, {Ct, Lt}, {0, 1});
       TiledIndexLabel i_t{N_olocalmo};
       TiledIndexLabel P_i{..};
       Tensor<T> Integral{mu, nu, N_aux, lambda};
       //Tensor<T> I_imuP{i, mu_p(i), P_i(i)};
       TMP(it, nu_p(it), P_i(it)) = C(it, mu_p(it)) * Integral(mu_p{it}, nu_p(it), P_i(it));
       I_imup(it, mupao_p(it), P_i(it)) = TMP(it, nu_p(it), P_i(it)) * L(mupao_p(it), nu_p(it));

       //Step 7
       // auto D_ii =  diagonal_mp2_densities(I_imuP, pairdoms, Focc, Fpao);

       Tensor<T> t{mupao_i(it), mupao_i(it), it};
       t(mupao_i(it), nupao_i(it), it) = 
           I_imup(it, mupao_i(it), P_i(it)) * 
           I_imup(it, nupao_i(it), P_i(it));

       //this will be a lambda
       //TODO: F will need to be computed above (Step 5), but in it-specific form. PAO Fock matrix needs to be diagonal.
       t(mupao_i(it), nupao_i(it), it) /= 
           F(mupao_i(it),mupao_i(it)) +
           F(nupao_i(it), nupao_i(it)) -
           2 * F_occ_mo(it, it);

       //EV: Different PAO spaces are disjoint. mupao_i(0) and mupao_i(1) are completely different. Union of these is not well-defined.
       Tensor<T> D{it, mupao_i(it), nupao_i(it)};
       D(it,mupao_i(it), nupao_i(it)) = 
           s(mupao_i(it), nupao_i(it), it) *
           t(mupao_i(it), nupao_i(it), it);
       //EV: here, because t is symmetric, we can diagonalize it directly. so we skip the D computation.
       
       //Step 8
       //auto LOSVs =  make_losvs(D_ii);
       //this step is diagonalization. 

       //Step 9
       //std::vector<tensor_type<2>> Faa;
       //for(auto i=0; i<LOSVs.size(); ++i) {
       //  auto LOSVi = LOSVs\[i\].shuffle({1, 0});
       //  Faa.push_back(transform_tensor(F, {LOSVs\[i\], LOSVi}, {0, 1}));
       }
       //auto ea = canonicalize(Faa);
       TMP(it, it) = d(mupao_i(it), mupao_i(it)) * 
                   F(mupao_i(it),mupao_i(it));
       F(it, it) = TMP(it, it) * ..;
       
       //Step 

       //Step 11

       //Skipped for now b/c they don't do anything for test system
       //Step 12
       //auto EscOSV =  sc\_osv\_mp2(I_iaP, Focc, ea);
       //auto I_ijP =  transform_tensor(I, {Ct, Ct}, {0, 1});

   }
   ```
   ### DLPNO MP2 (from Ed)


   ------

   ### TAMM Code Sketch from DLPNO Google Docs

   ```c++
   TiledIndexLabel i{N_ao}, k{N_ao};
   TiledIndexLabel j_atom; 
   TiledIndexLabel A{N_atom};

   Tensor<T> tC{N_ao, N_ao, N_atom};
   Tensor<T> tA{N_ao,N_ao};
   Tensor<T> tB{N_ao,N_ao};
   tC(i, k, A) = tA(i, j_atom(A)) * tB(j_atom(A), k);

   for i, k in AO {
     for A in Atom {
       Alloc tC(i, k, A) as Cbuf(i0, k0, A0)
       Cbuf(i0,k0,A0) = 0
       for j_atom in DepAO_Atom(A) { 
         Get tA(i, j_atom) to Abuf(i0, j0)
         Get tB(j_atom, k, k) to Bbuf(j0, k0)
         Cbuf(i0,k0,A0) += Abuf(i0, j0) * Bbuf(j0, k0)
       } //for j_atom in DepAO_Atom(A)
       Put Cbuf(i0,k0,A0) to tC(i,k,A)
       Dealloc Abuf, Bbuf, Cbuf
     } // for A
   } //for i, k

   ``` -->

Loop Nest Order and Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default loop nest ordering is from left-hand side (LHS) to
right-hand side (RHS) labels. For example the ordering for a simple
assignment with sum over operation on the ``(T1(i, j) = T6(j, i, k)``
will end up ordering of “:math:`i \to j \to k`” where k is the summed
over index. This ordering becomes more important when the operations are
over dependent index spaces, as there will be conflicting orders with
the dependency order described in the operation and the storage of the
tensors. In case of the dependencies are not satisfied with the default
ordering the ``TiledIndexSpace`` transformations will be used to
eliminate the dependencies.

.. code:: cpp

   // Given an order
   std::vector<Label> order = get_order(EXPR);

   std::vector<TIS> tis_vec{};
   // Intersect for each lbl with reference to storage and usage
   for(auto i : order){
     std::vector<IndexLabel> lbls = EXPR.find(i);

     auto tmp_tis;
     for(auto lbl : lbls){
       if(!is_dependency_resolved(lbl, tis_vec)){
         lbl.project(lbl.seconary_labels());
       }
       tmp_tis = lbl.tis().intersect(lbl.tensor().tis());
     }
     
   }

   LoopNest loop_nest{tis_vec};
