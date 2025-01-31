cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TAMM VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For cmsb_add_library

if(${PROJECT_NAME}_HAS_CUDA OR ${PROJECT_NAME}_HAS_HIP OR ${PROJECT_NAME}_HAS_DPCPP)
  set(tamm_gpu_INCS
    gpu_streams.hpp
    mr/gpu_memory_resource.hpp
    mr/pinned_memory_resource.hpp
    mr/per_device_resource.hpp
  )

  set(tamm_gpu_SRCS kernels/gpu_blas.cpp)
endif()
if(${PROJECT_NAME}_HAS_DPCPP)
  list(APPEND tamm_gpu_INCS sycl_device.hpp)
endif()

#Add the current directory's sources to the list
set(TAMM_SRCS tamm.cpp errors.cpp index_space.cpp execution_context.cpp tensor_base.cpp kernels/cpu_blas.cpp "${tamm_gpu_SRCS}")

#Add the current directory's header files to the list
set(TAMM_INCLUDES
  rmm_memory_pool.hpp
    mr/device_memory_resource.hpp
    mr/stream_ordered_memory_resource.hpp
    mr/aligned.hpp
    mr/new_delete_resource.hpp
    mr/pool_memory_resource.hpp
    mr/host_memory_resource.hpp
    mr/coalescing_free_list.hpp
    mr/free_list.hpp
    atomic_counter.hpp
    work.hpp
    tamm.hpp
    types.hpp
    errors.hpp
    strong_num.hpp
    proc_grid.hpp
    boundvec.hpp
    index_space.hpp
    tiled_index_space.hpp
    index_space_interface.hpp
    attribute.hpp
    range.hpp
    ops.hpp
    scheduler.hpp
    local_tensor.hpp
    tensor.hpp
    tensor_impl.hpp
    tensor_base.hpp
    tensor_info.hpp
    distribution.hpp
    labeled_tensor.hpp
    execution_context.hpp
    dag_impl.hpp
    proc_group.hpp
    memory_manager.hpp
    memory_manager_ga.hpp
    memory_manager_local.hpp
    index_loop_nest.hpp
    utils.hpp
    tamm_io.hpp
    tamm_utils.hpp
    ga_over_upcxx.hpp
    eigen_utils.hpp
    eigen_includes.hpp
    runtime_engine.hpp
    block_buffer.hpp
    lru_cache.hpp
    kernels/assign.hpp
    kernels/multiply.hpp
    kernels/tamm_blas.hpp
    op_dag.hpp
    interfaces.hpp
    op_profiler.hpp
    mem_profiler.hpp
    op_attributes.hpp
    op_visitors.hpp
    scalar.hpp
    block_plan.hpp
    block_set_plan.hpp
    block_lambda_plan.hpp
    block_assign_plan.hpp
    block_mult_plan.hpp
    block_span.hpp
    blockops_cpu.hpp
    perm.hpp
    iteration.hpp
    ip_hptt.hpp
    blockops_blas.hpp
    tensor_variant.hpp
    op_executor.hpp
    symbol.hpp
    op_cost.hpp
    block_operations.hpp
    setop.hpp
    scanop.hpp
    mapop.hpp
    allocop.hpp
    deallocop.hpp
    addop.hpp
    multop.hpp
    op_base.hpp
    label_translator.hpp
    opmin.hpp
    "${tamm_gpu_INCS}"
    # spin_tensor.hpp
   )

set(TAMM_CFLAGS )
set(TAMM_LFLAGS )

cmsb_add_library(tamm TAMM_SRCS TAMM_INCLUDES TAMM_CFLAGS TAMM_LFLAGS)
