// Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC and other
// Apollo project developers. Produced at the Lawrence Livermore National
// Laboratory. See the top-level LICENSE file for details.
// SPDX-License-Identifier: MIT

#include "apollo/Config.h"

#include <string>

int Config::APOLLO_COLLECTIVE_TRAINING;
int Config::APOLLO_LOCAL_TRAINING;
int Config::APOLLO_SINGLE_MODEL;
int Config::APOLLO_REGION_MODEL;
int Config::APOLLO_TRACE_POLICY;
int Config::APOLLO_RETRAIN_ENABLE;
float Config::APOLLO_RETRAIN_TIME_THRESHOLD;
float Config::APOLLO_RETRAIN_REGION_THRESHOLD;
int Config::APOLLO_STORE_MODELS;
int Config::APOLLO_TRACE_RETRAIN;
int Config::APOLLO_TRACE_ALLGATHER;
int Config::APOLLO_TRACE_BEST_POLICIES;
int Config::APOLLO_GLOBAL_TRAIN_PERIOD;
int Config::APOLLO_PER_REGION_TRAIN_PERIOD;
int Config::APOLLO_TRACE_CSV;
std::string Config::APOLLO_POLICY_MODEL;
std::string Config::APOLLO_TRACE_CSV_FOLDER_SUFFIX;

// Added to control decision tree depth
int Config::APOLLO_DTREE_DEPTH;
std::string Config::APOLLO_SINGLE_MODEL_TO_LOAD;

#ifdef PERF_CNTR_MODE
    int Config::APOLLO_ENABLE_PERF_CNTRS;
    int Config::APOLLO_PERF_CNTRS_MLTPX;
    
    std::vector<std::string> Config::APOLLO_PERF_CNTRS;
#endif
