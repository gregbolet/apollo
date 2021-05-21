
// Copyright (c) 2020, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// This file is part of Apollo.
// OCEC-17-092
// All rights reserved.
//
// Apollo is currently developed by Chad Wood, wood67@llnl.gov, with the help
// of many collaborators.
//
// Apollo was originally created by David Beckingsale, david@llnl.gov
//
// For details, see https://github.com/LLNL/apollo.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <memory>
#include <utility>
#include <algorithm>
#include <sstream>

#include <sys/types.h>
#include <sys/stat.h>

#include "assert.h"

#include "apollo/Apollo.h"
#include "apollo/Region.h"
#include "apollo/Logging.h"
#include "apollo/ModelFactory.h"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif //ENABLE_MPI

static inline bool fileExists(std::string path) {
    struct stat stbuf;
    return (stat(path.c_str(), &stbuf) == 0);
}

void Apollo::Region::train(int step)
{

//#ifdef PERF_CNTR_MODE
  //// If we ran the performance counters, don't add the data to training
  //if(this->shouldRunCounters) {
      //std::cout << "skipping training!" << std::endl;
      //std::cout.flush();
      //return;
  //}
  //else{
      //std::cout << "DOING training!" << std::endl;
      //std::cout.flush();
  //}
//#endif

  if (!model->training) return;

  collectPendingContexts();
  reduceBestPolicies(step);
  if (best_policies.size() <= 0) return;

  measures.clear();

  std::vector<std::vector<float> > train_features;
  std::vector<int> train_responses;

  std::vector<std::vector<float> > train_time_features;
  std::vector<float> train_time_responses;

  if (Config::APOLLO_REGION_MODEL) {
    std::cout << "TRAIN MODEL PER REGION " << name << std::endl;
    // Prepare training data
    for (auto &it2 : best_policies) {
      train_features.push_back(it2.first);
      train_responses.push_back(it2.second.first);

      std::vector<float> feature_vector = it2.first;
      feature_vector.push_back(it2.second.first);
      if (Config::APOLLO_RETRAIN_ENABLE) {
        train_time_features.push_back(feature_vector);
        train_time_responses.push_back(it2.second.second);
      }
    }
  } else {
      assert(false && "Expected per-region model.");
  }

  if (Config::APOLLO_TRACE_BEST_POLICIES) {
    std::stringstream trace_out;
    trace_out << "=== Rank " << apollo->mpiRank << " BEST POLICIES Region "
              << name << " ===" << std::endl;
    for (auto &b : best_policies) {
      trace_out << "[ ";
      for (auto &f : b.first)
        trace_out << (int)f << ", ";
      trace_out << "] P:" << b.second.first << " T: " << b.second.second
                << std::endl;
    }
    trace_out << ".-" << std::endl;
    std::cout << trace_out.str();
    std::ofstream fout("step-" + std::to_string(step) + "-rank-" +
                       std::to_string(apollo->mpiRank) + "-" + name +
                       "-best_policies.txt");
    fout << trace_out.str();
    fout.close();
  }

  model = ModelFactory::createDecisionTree(apollo->num_policies,
                                           train_features,
                                           train_responses);

  if (Config::APOLLO_RETRAIN_ENABLE)
    time_model = ModelFactory::createRegressionTree(train_time_features,
                                                    train_time_responses);

  if (Config::APOLLO_STORE_MODELS) {
    model->store("dtree-step-" + std::to_string(step) + "-rank-" +
                 std::to_string(apollo->mpiRank) + "-" + name + ".yaml");
    model->store(
        "dtree-latest"
        "-rank-" +
        std::to_string(apollo->mpiRank) + "-" + name + ".yaml");

    if (Config::APOLLO_RETRAIN_ENABLE) {
      time_model->store("regtree-step-" + std::to_string(step) + "-rank-" +
                        std::to_string(apollo->mpiRank) + "-" + name + ".yaml");
      time_model->store(
          "regtree-latest"
          "-rank-" +
          std::to_string(apollo->mpiRank) + "-" + name + ".yaml");
    }
  }
}

int
Apollo::Region::getPolicyIndex(Apollo::RegionContext *context)
{

#ifdef PERF_CNTR_MODE
    // If we're measuring perf cntrs, use the default policy
    int choice = (this->shouldRunCounters) ? 0 : model->getIndex(context->features);
#else
    int choice = model->getIndex( context->features );
#endif

    if( Config::APOLLO_TRACE_POLICY ) {
        std::stringstream trace_out;
        int rank;
        rank = apollo->mpiRank;
        trace_out << "Rank " << rank \
            << " region " << name \
            << " model " << model->name \
            << " features [ ";
        for(auto &f: context->features)
            trace_out << (int)f << ", ";
        trace_out << "] policy " << choice << std::endl;
        std::cout << trace_out.str();
        std::ofstream fout( "rank-" + std::to_string(rank) + "-policies.txt", std::ofstream::app );
        fout << trace_out.str();
        fout.close();
    }

#if 0
    if (choice != context->policy) {
        std::cout << "Change policy " << context->policy\
            << " -> " << choice << " region " << name \
            << " training " << model->training \
            << " features: "; \
            for(auto &f : apollo->features) \
                std::cout << (int)f << ", "; \
            std::cout << std::endl; //ggout
    } else {
        //std::cout << "No policy change for region " << name << ", policy " << current_policy << std::endl; //gout
    }
#endif
    context->policy = choice;
    //log("getPolicyIndex took ", evaluation_time_total, " seconds.\n");
    return choice;
}

// initializaiton function used by overloaded region constructors
void Apollo::Region::initRegion(
        const int num_features,
        const char  *regionName,
        int          numAvailablePolicies,
        Apollo::CallbackDataPool *callbackPool,
        const std::string &modelYamlFile){

    apollo = Apollo::instance();
    if( Config::APOLLO_NUM_POLICIES ) {
        apollo->num_policies = Config::APOLLO_NUM_POLICIES;
    }
    else {
        apollo->num_policies = numAvailablePolicies;
    }

    strncpy(name, regionName, sizeof(name)-1 );
    name[ sizeof(name)-1 ] = '\0';

    if (!modelYamlFile.empty()) {
        model = ModelFactory::loadDecisionTree(apollo->num_policies, modelYamlFile);
    }
    else {
        // TODO use best_policies to train a model for new region for which there's training data
        size_t pos = Config::APOLLO_INIT_MODEL.find(",");
        std::string model_str = Config::APOLLO_INIT_MODEL.substr(0, pos);
        if ("Static" == model_str)
        {
            int policy_choice = std::stoi(Config::APOLLO_INIT_MODEL.substr(pos + 1));
            if (policy_choice < 0 || policy_choice >= numAvailablePolicies)
            {
                std::cerr << "Invalid policy_choice " << policy_choice << std::endl;
                abort();
            }
            model = ModelFactory::createStatic(apollo->num_policies, policy_choice);
            //std::cout << "Model Static policy " << policy_choice << std::endl;
        }
        else if ("Load" == model_str)
        {
            std::string model_file;
            if (pos == std::string::npos)
            {
                // Load per region model using the region name for the model file.
                model_file = "dtree-latest-rank-" + std::to_string(apollo->mpiRank) + "-" + std::string(name) + ".yaml";
            }
            else
            {
                // Load the same model for all regions.
                model_file = Config::APOLLO_INIT_MODEL.substr(pos + 1);
            }

            if (fileExists(model_file))
                //std::cout << "Model Load " << model_file << std::endl;
                model = ModelFactory::loadDecisionTree(apollo->num_policies, model_file);
            else {
                // Fallback to default model.
                std::cout << "WARNING: could not load file " << model_file
                          << ", falling back to default Static, 0" << std::endl;
                model = ModelFactory::createStatic(apollo->num_policies, 0);
            }
        }
        else if ("Random" == model_str)
        {
            model = ModelFactory::createRandom(apollo->num_policies);
            //std::cout << "Model Random" << std::endl;
        }
        else if ("RoundRobin" == model_str)
        {
            model = ModelFactory::createRoundRobin(apollo->num_policies);
            //std::cout << "Model RoundRobin" << std::endl;
        }
#ifdef FULL_EXPLORE
        else if ("FullExplore" == model_str)
        {
            model = ModelFactory::createFullExplore(apollo->num_policies);
        }
#endif
        else if ("Optimal" == model_str) {
           std::string file = "opt-" + std::string(name) + "-rank-" + std::to_string(apollo->mpiRank) + ".txt";
           if (!fileExists(file)) {
               std::cerr << "Optimal policy file " << file << " does not exist" << std::endl;
               abort();
           }

           model = ModelFactory::createOptimal(file);
        }
        else
        {
            std::cerr << "Invalid model env var: " + Config::APOLLO_INIT_MODEL << std::endl;
            abort();
        }
    }

    if( Config::APOLLO_TRACE_CSV ) {
        // TODO: assumes model comes from env, fix to use model provided in the constructor
        // TODO: convert to filesystem C++17 API when Apollo moves to it
        int ret;
        ret = mkdir(
            std::string("./trace" + Config::APOLLO_TRACE_CSV_FOLDER_SUFFIX)
                .c_str(),
            S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (ret != 0 && errno != EEXIST) {
            perror("TRACE_CSV mkdir");
            abort();
        }
        std::string fname("./trace" + Config::APOLLO_TRACE_CSV_FOLDER_SUFFIX +
                          "/trace-" + Config::APOLLO_INIT_MODEL + "-region-" +
                          name + "-rank-" + std::to_string(apollo->mpiRank) +
                          ".csv");
        std::cout << "TRACE_CSV fname " << fname << std::endl;
        trace_file.open(fname);
        if(trace_file.fail()) {
            std::cerr << "Error opening trace file " + fname << std::endl;
            abort();
        }
        // Write header.
        trace_file << "rankid training region idx";
        //trace_file << "features";
        for(int i=0; i<num_features; i++)
            trace_file << " f" << i;
        trace_file << " policy xtime\n";
    }

#ifdef PERF_CNTR_MODE
    //std::string events[1] = {"PAPI_TOT_INS"};
    //std::string events[1] = {"PAPI_DP_OPS"};
    //std::vector<std::string> events = {"PAPI_DP_OPS", "PAPI_SP_OPS"};
    //this->papiPerfCnt = new PapiCounters(1, events);
#endif

    //std::cout << "Insert region " << name << " ptr " << this << std::endl;
    const auto ret = apollo->regions.insert( { name, this } );

    return;

}

Apollo::Region::Region(
        const int num_features,
        const char  *regionName,
        int          numAvailablePolicies,
        Apollo::CallbackDataPool *callbackPool,
        const std::string &modelYamlFile)
    :
        num_features(num_features), 
        current_context(nullptr), 
        idx(0), callback_pool(callbackPool),
        papiPerfCnt(nullptr),
        shouldRunCounters(0)
{
    this->initRegion(num_features, regionName, numAvailablePolicies, callbackPool, modelYamlFile);
}

#ifdef PERF_CNTR_MODE
Apollo::Region::Region(
        const int num_features,
        const char  *regionName,
        int          numAvailablePolicies,
        std::vector<std::string> papi_cntr_events,
        int isMultiplexed,
        Apollo::CallbackDataPool *callbackPool,
        const std::string &modelYamlFile)
    :
        num_features(num_features), 
        current_context(nullptr), 
        idx(0), callback_pool(callbackPool),
        shouldRunCounters(1)
{
    this->initRegion(num_features, regionName, numAvailablePolicies, callbackPool, modelYamlFile);
    this->papiPerfCnt = new Apollo::PapiCounters(isMultiplexed, papi_cntr_events);
}

void Apollo::Region::apolloThreadBegin(){
    if(this->papiPerfCnt && this->shouldRunCounters){
        this->papiPerfCnt->startThread();
    }
}
void Apollo::Region::apolloThreadEnd(){
    if(this->papiPerfCnt && this->shouldRunCounters){
        this->papiPerfCnt->stopThread();
    }
}

// This will allow the user to query the constructed model
int Apollo::Region::queryPolicyModel(std::vector<float> feats){
    return this->model->getIndex(feats);
}
#else
// If counters don't get enabled, just return immediately
void Apollo::Region::apolloThreadBegin(){ return; }
void Apollo::Region::apolloThreadEnd(){ return; }
#endif

Apollo::Region::~Region()
{

#ifdef PERF_CNTR_MODE
    if(this->papiPerfCnt){
        delete this->papiPerfCnt;
    }
#endif

    // Disable period based flushing.
    Config::APOLLO_GLOBAL_TRAIN_PERIOD = 0;
    while(pending_contexts.size() > 0)
       collectPendingContexts();

    if(callback_pool)
        delete callback_pool;

    if( Config::APOLLO_TRACE_CSV )
        trace_file.close();


    return;
}

Apollo::RegionContext *
Apollo::Region::begin()
{
    Apollo::RegionContext *context = new Apollo::RegionContext();
    current_context = context;
    context->idx = this->idx;
    this->idx++;

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    context->exec_time_begin = ts.tv_sec + ts.tv_nsec/1e9;
    context->isDoneCallback = nullptr;
    context->callback_arg = nullptr;


    return context;
}

Apollo::RegionContext *
Apollo::Region::begin(std::vector<float> features)
{
    Apollo::RegionContext *context = begin();
    context->features = features;

#ifdef PERF_CNTR_MODE
    //std::cout << "USING THREAD COUNT: " << omp_get_max_threads() << std::endl;

    this->lastFeats = context->features;

    // Check to see if we've already seen this feature set before
    // If not, let's set the flag to run the counters 
    // If so, we should just not run the counter code
    // as we know the counter values already since they're policy-independent
    if(this->papiPerfCnt){
        this->shouldRunCounters = (this->feats_to_cntr_vals.find(context->features) == this->feats_to_cntr_vals.end());

        // If we already have run the counters, then translate the passed-in
        // feature vector so we can properly getPolicyIndex() if a tree has
        // already been trained
        if(!this->shouldRunCounters){
            context->features = this->feats_to_cntr_vals[context->features];
            this->lastFeats = context->features;
        }
    }
#endif

    return context;
}

void
Apollo::Region::collectContext(Apollo::RegionContext *context, double metric)
{
#ifdef PERF_CNTR_MODE
  // If the performance counters were setup
  if(this->papiPerfCnt){

    // The counters were run, this was the first time we saw this user-supplied
    // feature vector. Thus, we should save it to the feature-counter mapping
    // along with not adding the measure to the region measures due to the
    // slight execution time overhead PAPI adds. 
    if(this->shouldRunCounters){

        // Check whether or not the counters were actually called!
        // If they were not actually called, this region doesn't have
        // start/stopApolloThread calls, so let's drop the papiPerfCnt
        // object. Otherwise, continue as normal with adding the counters.
        if(this->papiPerfCnt->all_cntr_values.size() == 0){
            //delete this->papiPerfCnt;
            //this->papiPerfCnt = nullptr;
            //goto skipCounterAdding;
        }

        // First calculate the sum of each counter
        // These summary statistics are calculated across threads, so we
        // always have the same feature dimensions irregardless of thread count
        std::vector<float> vals = this->papiPerfCnt->getSummaryStats();

        // Clear the PapiCounters counter values
        this->papiPerfCnt->clearAllCntrValues();

        // Map the user-provided features to the counter values
        this->feats_to_cntr_vals[context->features] = vals;

        // Add the new features to the feature list
        //for(int i = 0; i < vals.size(); ++i){
            //context->features.push_back(vals[i]);
        //}

        // Store these features for use after Region->end() call finishes
        // and the context gets deleted (so we lose our context->features vector)
        this->lastFeats = context->features;

        // We're gonna change this later, just want something working for now
        goto dontAddMeasure;
    }

    // This case happens if we already got counter values for the 
    // user-provided features (and we've run this feature config once before) 
    else{
        // Get the counter values for this feature set
        //std::vector<float> vals = this->feats_to_cntr_vals[context->features];

        // Set our features to the counter values
        //context->features = vals;

        // Store these features for use after Region->end() call finishes
        // and the context gets deleted (so we lose our context->features vector)
        //this->lastFeats = context->features;

        // continue on to add the measure
    }
  }

skipCounterAdding:
        // Store these features for use after Region->end() call finishes
        // and the context gets deleted (so we lose our context->features vector)
        this->lastFeats = context->features;
#endif

// Hack around the goto statement, wrap this in it's own scope
// Of course this is messy, but we will clean up later. Just want
// to get something working for now. 
{
  // std::cout << "COLLECT CONTEXT " << context->idx << " REGION " << name \
            << " metric " << metric << std::endl;

    // Check if we already have seen this feature+policy combination.
    // If we have, add to the total_time for its measure, otherwise
    // create a new measurement for it.
    auto iter = measures.find({context->features, context->policy});
    if (iter == measures.end()) {
      iter = measures
               .insert(std::make_pair(
                   std::make_pair(context->features, context->policy),
                   std::move(
                       std::make_unique<Apollo::Region::Measure>(1, metric))))
               .first;
    } 
    else {
        iter->second->exec_count++;
        iter->second->time_total += metric;
    }

    if( Config::APOLLO_TRACE_CSV ) {
        trace_file << apollo->mpiRank << " ";
        trace_file << model->name << " ";
        trace_file << this->name << " ";
        trace_file << context->idx << " ";
        for(auto &f : context->features)
            trace_file << f << " ";
        trace_file << context->policy << " ";
        trace_file << metric << "\n";
    }
}

    // Try only counting a region execution as having been a measurement
    apollo->region_executions++;

#ifdef PERF_CNTR_MODE
dontAddMeasure:
#endif

    if( Config::APOLLO_GLOBAL_TRAIN_PERIOD && ( apollo->region_executions%Config::APOLLO_GLOBAL_TRAIN_PERIOD) == 0 ) {
        //std::cout << "FLUSH PERIOD! region_executions " << apollo->region_executions<< std::endl; //ggout
        apollo->flushAllRegionMeasurements(apollo->region_executions);
    }
    else if( Config::APOLLO_PER_REGION_TRAIN_PERIOD && (idx%Config::APOLLO_PER_REGION_TRAIN_PERIOD) == 0 ) {
        train(idx);
    }

    // delete the RegionContext struct
    delete context;
    current_context = nullptr;
}

void
Apollo::Region::end(Apollo::RegionContext *context, double metric)
{
    //std::cout << "END REGION " << name << " metric " << metric << std::endl;

    collectContext(context, metric);

    collectPendingContexts();

    return;
}

void Apollo::Region::collectPendingContexts() {
  auto isDone = [this](Apollo::RegionContext *context) {
    bool returnsMetric;
    double metric;
    if (context->isDoneCallback(context->callback_arg, &returnsMetric, &metric)) {
      if (returnsMetric)
        collectContext(context, metric);
      else {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        context->exec_time_end = ts.tv_sec + ts.tv_nsec/1e9;
        double duration = context->exec_time_end - context->exec_time_begin;
        collectContext(context, duration);
      }
      return true;
    }

    return false;
  };

  pending_contexts.erase(
      std::remove_if(pending_contexts.begin(), pending_contexts.end(), isDone),
      pending_contexts.end());
}

void
Apollo::Region::end(Apollo::RegionContext *context)
{
    if(context->isDoneCallback)
        pending_contexts.push_back(context);
    else {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      context->exec_time_end = ts.tv_sec + ts.tv_nsec/1e9;
      double duration = context->exec_time_end - context->exec_time_begin;
      collectContext(context, duration);
    }

    collectPendingContexts();
}


// DEPRECATED
int
Apollo::Region::getPolicyIndex(void)
{
    return getPolicyIndex(current_context);
}

// DEPRECATED
void
Apollo::Region::end(double metric)
{
    end(current_context, metric);
}

// DEPRECATED
void
Apollo::Region::end(void)
{

    end(current_context);
}

int
Apollo::Region::reduceBestPolicies(int step)
{
    std::stringstream trace_out;
    int rank;
    if( Config::APOLLO_TRACE_MEASURES ) {
#ifdef ENABLE_MPI
        rank = apollo->mpiRank;
#else
        rank = 0;
#endif //ENABLE_MPI
        trace_out << "=================================" << std::endl \
            << "Rank " << rank << " Region " << name << " MEASURES "  << std::endl;
    }
    for (auto iter_measure = measures.begin();
            iter_measure != measures.end();   iter_measure++) {

        const std::vector<float>& feature_vector = iter_measure->first.first;
        const int policy_index                   = iter_measure->first.second;
        auto                           &time_set = iter_measure->second;

        if( Config::APOLLO_TRACE_MEASURES ) {
            trace_out << "features: [ ";
            for(auto &f : feature_vector ) { \
                trace_out << (float)f << ", ";
            }
            trace_out << " ]: "
                << "policy: " << policy_index
                << " , count: " << time_set->exec_count
                << " , total: " << time_set->time_total
                << " , time_avg: " <<  ( time_set->time_total / time_set->exec_count ) << std::endl;
        }
        double time_avg = ( time_set->time_total / time_set->exec_count );

        auto iter =  best_policies.find( feature_vector );
        if( iter ==  best_policies.end() ) {
            best_policies.insert( { feature_vector, { policy_index, time_avg } } );
        }
        else {
            // Key exists
            // If the execution time of this feature+policy is faster than the best,
            // then replace the best policy.
            if(  best_policies[ feature_vector ].second > time_avg ) {
                best_policies[ feature_vector ] = { policy_index, time_avg };
            }
        }
    }

    if( Config::APOLLO_TRACE_MEASURES ) {
        trace_out << ".-" << std::endl;
        trace_out << "Rank " << rank << " Region " << name << " Reduce " << std::endl;
        for( auto &b : best_policies ) {
            trace_out << "features: [ ";
            for(auto &f : b.first )
                trace_out << (float) f << ", ";
            trace_out << "]: P:"
                << b.second.first << " T: " << b.second.second << std::endl;
        }
        trace_out << ".-" << std::endl;
        std::cout << trace_out.str();
        std::ofstream fout("step-" + std::to_string(step) +
                "-rank-" + std::to_string(rank) + "-" + name + "-measures.txt"); \
            fout << trace_out.str();
        fout.close();
    }

    return best_policies.size();
}

void
Apollo::Region::setFeature(Apollo::RegionContext *context, float value)
{
    context->features.push_back(value);


    return;
}

// DEPRECATED
void
Apollo::Region::setFeature(float value)
{
    setFeature(current_context, value);
}
