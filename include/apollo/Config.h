// Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC and other
// Apollo project developers. Produced at the Lawrence Livermore National
// Laboratory. See the top-level LICENSE file for details.
// SPDX-License-Identifier: MIT

#ifndef APOLLO_CONFIG_H
#define APOLLO_CONFIG_H

#include <string>
#include <vector>

class Config
{
public:
  static int APOLLO_COLLECTIVE_TRAINING;
  static int APOLLO_LOCAL_TRAINING;
  static int APOLLO_SINGLE_MODEL;
  static int APOLLO_REGION_MODEL;
  static int APOLLO_TRACE_POLICY;
  static int APOLLO_RETRAIN_ENABLE;
  static float APOLLO_RETRAIN_TIME_THRESHOLD;
  static float APOLLO_RETRAIN_REGION_THRESHOLD;
  static int APOLLO_STORE_MODELS;
  static int APOLLO_TRACE_RETRAIN;
  static int APOLLO_TRACE_ALLGATHER;
  static int APOLLO_TRACE_BEST_POLICIES;
  static int APOLLO_GLOBAL_TRAIN_PERIOD;
  static int APOLLO_PER_REGION_TRAIN_PERIOD;
  static int APOLLO_TRACE_CSV;
  static int APOLLO_PERSISTENT_DATASETS;
  static std::string APOLLO_POLICY_MODEL;
  static std::string APOLLO_OUTPUT_DIR;
  static std::string APOLLO_DATASETS_DIR;
  static std::string APOLLO_TRACES_DIR;
  static std::string APOLLO_MODELS_DIR;
  static std::string APOLLO_OPTIM_READ_DIR;

  static int APOLLO_MIN_TRAINING_DATA;
  //static int APOLLO_DTREE_DEPTH;
  static int APOLLO_ENABLE_PERF_CNTRS;
  static int APOLLO_PERF_CNTRS_MLTPX;
  static int APOLLO_PERF_CNTRS_SAMPLE_EACH_ITER;
  static std::vector<std::string> APOLLO_PERF_CNTRS;
  static std::string APOLLO_SINGLE_MODEL_TO_LOAD;

    private:
        Config();
};

#endif
