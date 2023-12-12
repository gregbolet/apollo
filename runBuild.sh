#!/bin/bash

#module load cmake

#PAPI_PATH=/usr/WS2/bolet1/spack/opt/spack/linux-rhel7-broadwell/gcc-10.2.1/papi-6.0.0.1-42m6tpjegu2j7hxrz26ii5t23jcmtmca

#PAPI_PATH=/usr/WS2/bolet1/spack/opt/spack/linux-rhel7-cascadelake/gcc-10.2.1/papi-6.0.0.1-sl4kpa2prz3scz6nm4r6smfrd5h67z7n
#LLVM_NAME="release-ruby"
#LLVM_INSTALL=/g/g15/bolet1/workspace/clang-apollo/llvm-project/build-${LLVM_NAME}/install
#
#cmake -DENABLE_MPI=OFF \
# 	-DCMAKE_BUILD_TYPE=Release \
#  -DCMAKE_INSTALL_PREFIX=./install \
#  -DCMAKE_CXX_COMPILER=${LLVM_INSTALL}/bin/clang++ \
#  -DCMAKE_C_COMPILER=${LLVM_INSTALL}/bin/clang \
#	-DENABLE_PERF_CNTRS=ON \
#	-DBUILD_SHARED_LIBS=ON \
#	-DPAPI_DIR=${PAPI_PATH} \
#  -DCMAKE_EXE_LINKER_FLAGS="-L ${LLVM_INSTALL}/lib -Wl,--rpath,${LLVM_INSTALL}/lib" \
#	../

# make sure to module load clang and cmake beforehand
#cmake -DENABLE_MPI=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_CXX_COMPILER=clang++ -DBUILD_SHARED_LIBS=ON ../

# BUILDING WITH CUDA: For some reason the tests don't get linked with CUDA properly
# so build with ENABLE_TESTS=OFF so you can install the library. I think we just 
# need to modify the /tests/ CMakeLists to include CUDA.
cmake -DENABLE_MPI=OFF \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_INSTALL_PREFIX=./install \
			-DCMAKE_CXX_COMPILER=clang++ \
			-DBUILD_SHARED_LIBS=ON \
			-DENABLE_TESTS=OFF \
			-DENABLE_BO=ON \
			-DENABLE_CUDA=ON \
			-DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
			-DLIMBO_DIR=/g/g15/bolet1/workspace/BOcpp/limbo \
			-DNLOPT_DIR=/g/g15/bolet1/workspace/BOcpp/nlopt/build/install \
			-DEIGEN_DIR=/g/g15/bolet1/workspace/BOcpp/eigen/eigen-3.4.0/build/install \
			../


			#-DMANUAL_CUDA_DIR=/usr/tce/packages/cuda/cuda-11.8.0/nvidia/targets/ppc64le-linux/include \
			#-DCUDA_INCLUDE_DIRS=/usr/tce/packages/cuda/cuda-11.8.0/nvidia/targets/ppc64le-linux/include \

# -DPAPI_DIR=/usr/WS2/bolet1/spack/opt/spack/linux-rhel7-broadwell/gcc-10.2.1/papi-6.0.0.1-42m6tpjegu2j7hxrz26ii5t23jcmtmca \
# -DOpenCV_DIR=~/workspace/opencv/build \
# 	-DCMAKE_BUILD_TYPE=Release \
# 	-DCMAKE_BUILD_TYPE=RelWithDebInfo \