find_path(CBLAS_INCLUDE_DIR cblas.h)
find_path(LAPACKE_INCLUDE_DIR lapacke.h)
find_library(CBLAS_LIBRARY NAMES cblas)
find_library(LAPACKE_LIBRARY NAMES lapacke)
find_package_handle_standard_args(
    LAPACKE
    DEFAULT_MSG
    CBLAS_INCLUDE_DIR LAPACKE_INCLUDE_DIR
    CBLAS_LIBRARY LAPACKE_LIBRARY
)
set(LAPACKE_INCLUDE_DIRS ${LAPACKE_INCLUDE_DIR} ${CBLAS_INCLUDE_DIR})
set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARY} ${CBLAS_LIBRARY})