#!/bin/bash

cd ./build
cmake   -DENABLE_MPI=OFF \
	-DOpenCV_DIR=~/workspace/opencv/build \
	-DENABLE_PERF_CNTRS=ON \
	-DPAPI_DIR=/usr/WS2/bolet1/spack/opt/spack/linux-rhel7-broadwell/gcc-10.2.1/papi-6.0.0.1-42m6tpjegu2j7hxrz26ii5t23jcmtmca \
	../

# 	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
