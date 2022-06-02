// Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC and other
// Apollo project developers. Produced at the Lawrence Livermore National
// Laboratory. See the top-level LICENSE file for details.
// SPDX-License-Identifier: MIT

#ifndef APOLLO_H
#define APOLLO_H

#include <map>
#include <string>
#include <vector>

#include "apollo/Config.h"
#include "papi.h"

class Apollo
{
public:
  ~Apollo();
  // disallow copy constructor
  Apollo(const Apollo &) = delete;
  Apollo &operator=(const Apollo &) = delete;

  static Apollo *instance(void) noexcept
  {
    static Apollo the_instance;
    return &the_instance;
  }

  class Region;
  struct RegionContext;
  class Timer;
  class PerfCounter;
  class PapiCounters;

  // PAPI_PERF_CNTR
  // These are all indexed by threadID which omp
  // doles out starting at 0, ending at omp_get_max_threads()-1
  int num_eventsets;
  int num_events;
  int is_multiplexed;
  int* EventSets;
  int* EventSet_is_started;
  int* EventSet_just_used;
  int* events_to_track;
  // This should be indexed with a stride of num_events*threadID
  long long* cntr_values;

  //
  int mpiSize;  // 1 if no MPI
  int mpiRank;  // 0 if no MPI

  // NOTE(chad): We default to walk_distance of 2 so we can
  //             step out of this method, then step out of
  //             some portable policy template, and get to the
  //             module name and offset where that template
  //             has been instantiated in the application code.
  std::string getCallpathOffset(int walk_distance = 2);
  void *callpath_ptr;

  // DEPRECATED, use train.
  void flushAllRegionMeasurements(int step);
  void train(int step);

private:
  Apollo();
  //
  void gatherCollectiveTrainingData(int step);
  // Key: region name, value: region raw pointer
  std::map<std::string, Apollo::Region *> regions;
  // Count total number of region invocations
  unsigned long long region_executions;
};  // end: Apollo

extern "C" {
 void *__apollo_region_create(int num_features, char *id, int num_policies) noexcept;
 void __apollo_region_begin(Apollo::Region *r) noexcept;
 void __apollo_region_end(Apollo::Region *r) noexcept;
 void __apollo_region_set_feature(Apollo::Region *r, float feature) noexcept;
 int __apollo_region_get_policy(Apollo::Region *r) noexcept;
 void __apollo_region_thread_begin(Apollo::Region *r) noexcept;
 void __apollo_region_thread_end(Apollo::Region *r) noexcept;
}

#endif
