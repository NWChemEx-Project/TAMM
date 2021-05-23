
include(TargetMacros)
add_mpi_unit_test(CD_CCSD_CS 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")
add_mpi_unit_test(CD_CCSD_OS 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")
add_mpi_unit_test(CholeskyDecomp 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")

set(CCSD_T_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/cc/ccsd_t)
set(CCSD_T_SRCS
    ${CCSD_T_SRCDIR}/memory.cpp
    ${CCSD_T_SRCDIR}/ccsd_t_common.hpp
    ${CCSD_T_SRCDIR}/hybrid.cpp
    ${CCSD_T_SRCDIR}/ccsd_t_unfused_driver.hpp
    ${CCSD_T_SRCDIR}/ccsd_t_singles_unfused_cpu.hpp
    ${CCSD_T_SRCDIR}/ccsd_t_doubles_unfused_cpu.hpp
    ${CCSD_T_SRCDIR}/sd_t_total_cpu.cpp
    ${CCSD_T_SRCDIR}/ccsd_t_fused_driver.hpp
    ${CCSD_T_SRCDIR}/fused_common.hpp
    )

if(USE_CUDA)
    set(CCSD_T_UNFUSED_SRCS ${CCSD_T_SRCS}
            ${CCSD_T_SRCDIR}/sd_t_total_gpu.cu
            ${CCSD_T_SRCDIR}/sd_t_total_nwc.cu
            ${CCSD_T_SRCDIR}/ccsd_t_singles_unfused.hpp
            ${CCSD_T_SRCDIR}/ccsd_t_doubles_unfused.hpp)

    set(CCSD_T_FUSED_SRCS ${CCSD_T_SRCS}
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused.hpp
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused_gpu.cu)

elseif(USE_DPCPP)
    set(CCSD_T_UNFUSED_SRCS ${CCSD_T_SRCS})
    set(CCSD_T_FUSED_SRCS ${CCSD_T_SRCS}
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused.hpp
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused_sycl.hpp)
else()
    set(CCSD_T_UNFUSED_SRCS ${CCSD_T_SRCS})
    set(CCSD_T_FUSED_SRCS ${CCSD_T_SRCS}
            ${CCSD_T_SRCDIR}/ccsd_t_all_fused_cpu.hpp)
endif()

add_mpi_cuda_unit_test(CCSD_T_Unfused "${CCSD_T_UNFUSED_SRCS}" 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")
add_mpi_cuda_unit_test(CCSD_T_Unfused_Fast "${CCSD_T_UNFUSED_SRCS}" 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")
add_mpi_cuda_unit_test(CCSD_T_Fused "${CCSD_T_FUSED_SRCS}" 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")
add_mpi_cuda_unit_test(CCSD_T_Fused_Fast "${CCSD_T_FUSED_SRCS}" 2 "${CMAKE_SOURCE_DIR}/../inputs/h2o.json")
