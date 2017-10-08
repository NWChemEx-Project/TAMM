
Requirements
------------
- Git
- autotools
- cmake >= 3.7
- C++14 compiler

**NOTE:** The current build has only been tested with gcc versions >= 6.0 and Intel 18 compilers.

BUILD
-----

```
TAMM_ROOT=/opt/TAMM  
git clone https://github.com/NWChemEx-Project/TAMM.git $TAMM_ROOT  
git checkout devel
```

- Modify any toolchain file (*except old-tamm-config.cmake*) in ${TAMM_ROOT}/cmake/toolchains to  
  adjust compilers and MPI_INCLUDE_PATH, MPI_LIBRARY_PATH, MPI_LIBRARIES.

  Following are optional:
  - GA Configure Options.
  - BLAS include & library paths.
  - TAMM_PROC_COUNT, EIGEN3_INSTALL_PATH, LIBINT_INSTALL_PATH,
  & ANTLR_CPPRUNTIME_PATH.

  **NOTE:** Eigen3, Netlib blas+lapack, Libint, ANTLR, googletest will be
  built if they do not exist. GA will always be built. Pre-exisiting GA setup
  cannot be specified.
  

```
cd ${TAMM_ROOT}/external  
mkdir build && cd build  
cmake .. -DCMAKE_TOOLCHAIN_FILE=${TAMM_ROOT}/cmake/toolchains/gcc-openmpi-netlib.cmake
make  
```

- After missing dependencies are built:

```
cd ${TAMM_ROOT}  
mkdir build && cd build  
cmake ..  -DCMAKE_TOOLCHAIN_FILE=${TAMM_ROOT}/external/build/tamm_build.cmake  
make install
```


BUILD OLD TAMM CODE (OPTIONAL)
------------------------------

```
TAMM_ROOT=/opt/TAMM  
git clone https://github.com/NWChemEx-Project/TAMM.git $TAMM_ROOT  
git checkout devel
```

 - Modify old-tamm-config.cmake in ${TAMM_ROOT}/external/cmake/toolchains to  
  adjust compilers, NWCHEM_TOP (path to nwchem root folder), GA_CONFIG (path to ga_config)
  and NWCHEM_BUILD_TARGET/NWCHEM_BUILD_DIR.

```
cd ${TAMM_ROOT}/external  
mkdir build && cd build  
cmake .. -DBUILD_OLD_TAMM=ON -DCMAKE_TOOLCHAIN_FILE=${TAMM_ROOT}/cmake/toolchains/gcc-openmpi-netlib.cmake
make  
```

- After missing dependencies are built:

```
cd ${TAMM_ROOT}  
mkdir build && cd build  
cmake ..  -DBUILD_OLD_TAMM=ON -DCMAKE_TOOLCHAIN_FILE=${TAMM_ROOT}/external/build/tamm_build.cmake  
make install
```
