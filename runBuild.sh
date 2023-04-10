#!/bin/bash

cmake -DENABLE_MPI=OFF \
 	-DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_CXX_COMPILER=${LLVM_INSTALL}/bin/clang++ \
  -DCMAKE_C_COMPILER=${LLVM_INSTALL}/bin/clang \
	-DENABLE_PERF_CNTRS=ON \
	-DBUILD_SHARED_LIBS=ON \
	-DPAPI_DIR=${PAPI_PATH} \
  -DCMAKE_EXE_LINKER_FLAGS="-L ${LLVM_INSTALL}/lib -Wl,--rpath,${LLVM_INSTALL}/lib" \
	../

# -DPAPI_DIR=/usr/WS2/bolet1/spack/opt/spack/linux-rhel7-broadwell/gcc-10.2.1/papi-6.0.0.1-42m6tpjegu2j7hxrz26ii5t23jcmtmca \
# -DOpenCV_DIR=~/workspace/opencv/build \
# 	-DCMAKE_BUILD_TYPE=Release \
# 	-DCMAKE_BUILD_TYPE=RelWithDebInfo \