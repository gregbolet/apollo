# Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC and other
# Apollo project developers. Produced at the Lawrence Livermore National
# Laboratory. See the top-level LICENSE file for details.
# SPDX-License-Identifier: MIT

include_directories(${PROJECT_BINARY_DIR})
include_directories(
  ./connectors/kokkos/include
)
set(APOLLO_HEADERS
    ../include/apollo/Apollo.h
    ../include/apollo/Config.h
    ../include/apollo/Region.h
    ../include/apollo/PolicyModel.h
    ../include/apollo/TimingModel.h
    ../include/apollo/ModelFactory.h
    ../include/apollo/Timer.h
    )

set(APOLLO_SOURCES
    Apollo.cpp
    Region.cpp
    ModelFactory.cpp
    Config.cpp
    helpers/OutputFormatter.cpp
    models/Random.cpp
    models/Static.cpp
    models/RoundRobin.cpp
    models/DecisionTree.cpp
    models/RandomForest.cpp
    models/PolicyNet.cpp
    models/impl/DecisionTreeImpl.cpp
    models/impl/RandomForestImpl.cpp
    models/preprocessing/Preprocessing.cpp
    models/Optimal.cpp
    connectors/kokkos/kokkos-connector.cpp
    timers/TimerSync.cpp
    )

if (ENABLE_OPENCV)
    list(APPEND APOLLO_SOURCES
        models/RegressionTree.cpp
        )
endif()

if (ENABLE_CUDA)
    list(APPEND APOLLO_SOURCES
        timers/TimerCudaAsync.cpp
        )
endif()

if (ENABLE_HIP)
    list(APPEND APOLLO_SOURCES
        timers/TimerHipAsync.cpp
        )
endif()

if (ENABLE_PERF_CNTRS)
    list(APPEND APOLLO_SOURCES
        perfcntrs/PapiCounters.cpp)
endif()

if (BUILD_SHARED_LIBS)
    add_library(apollo SHARED ${APOLLO_SOURCES})
else()
    add_library(apollo STATIC ${APOLLO_SOURCES})
endif()

##### GREG added
#
if(ENABLE_FULL_EXPLORE)
  #add_definitions(-DFULL_EXPLORE)
  target_compile_definitions(apollo PUBLIC -DFULL_EXPLORE)
endif()
#
##### End of GREG added

if (ENABLE_MPI)
    target_link_libraries(apollo PUBLIC MPI::MPI_CXX)
endif()

if(ENABLE_PERF_CNTRS)
    # Supply the PAPI library path
    if(PAPI_DIR)
        message(STATUS "LINKING PAPI!")
        target_link_libraries(apollo PUBLIC ${PAPI_DIR}/lib/libpapi.so)
    endif()
endif()

if (ENABLE_OPENCV)
    target_link_libraries(apollo PRIVATE dl ${OpenCV_LIBS})
endif()

if (ENABLE_JIT_DTREE)
    target_link_libraries(apollo PRIVATE ${CMAKE_DL_LIBS})
endif()

install(FILES ${APOLLO_HEADERS} DESTINATION include/apollo)

install(TARGETS apollo
    EXPORT apollo
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
