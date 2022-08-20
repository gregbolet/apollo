// Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC and other
// Apollo project developers. Produced at the Lawrence Livermore National
// Laboratory. See the top-level LICENSE file for details.
// SPDX-License-Identifier: MIT

#include "apollo/Region.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cassert>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "apollo/Apollo.h"
#include "apollo/ModelFactory.h"
#include "timers/TimerSync.h"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif  // ENABLE_MPI

static inline bool fileExists(std::string path)
{
  struct stat stbuf;
  return (stat(path.c_str(), &stbuf) == 0);
}

void Apollo::Region::train(int step, bool doCollectPendingContexts, bool force)
{
  if (!force)
    if (!model->isTrainable()) return;

  // Conditionally collect pending contexts. Auto-training must not
  // collect contexts to avoid infinite recursion since auto-training
  // happens within collectContext().
  if (doCollectPendingContexts) collectPendingContexts();

  if (dataset.size() <= 0) return;

  if (!Config::APOLLO_REGION_MODEL)
    throw std::runtime_error("Expected per-region model training");

  if (Config::APOLLO_TRACE_BEST_POLICIES) {
    std::stringstream trace_out;
    trace_out << "=== Rank " << apollo->mpiRank << " BEST POLICIES Region "
              << name << " ===" << std::endl;

    std::vector<std::vector<float>> features;
    std::vector<int> responses;
    std::map<std::vector<float>, std::pair<int, double>> min_metric_policies;
    dataset.findMinMetricPolicyByFeatures(features,
                                          responses,
                                          min_metric_policies);
    for (auto &b : min_metric_policies) {
      trace_out << "[ ";
      for (auto &f : b.first)
        trace_out << (int)f << ", ";
      trace_out << "] P:" << b.second.first << " T: " << b.second.second
                << std::endl;
    }
    std::cout << trace_out.str();
    std::ofstream fout("step-" + std::to_string(step) + "-rank-" +
                       std::to_string(apollo->mpiRank) + "-" + name +
                       "-best_policies.txt");
    fout << trace_out.str();
    fout.close();
  }

  model->train(dataset);

  if (Config::APOLLO_RETRAIN_ENABLE)
#ifdef ENABLE_OPENCV
    time_model = ModelFactory::createRegressionTree(dataset);
#else
    throw std::runtime_error("Retraining requires OpenCV");
#endif

  if (Config::APOLLO_STORE_MODELS) {
    std::string filename =
        Config::APOLLO_OUTPUT_DIR + "/" + Config::APOLLO_MODELS_DIR + "/" +
        model->name + "-step-" + std::to_string(step) + "-rank-" +
        std::to_string(apollo->mpiRank) + "-" + name + ".yaml";
    model->store(filename);
    filename = Config::APOLLO_OUTPUT_DIR + "/" + Config::APOLLO_MODELS_DIR +
               "/" + model->name +
               "-latest"
               "-rank-" +
               std::to_string(apollo->mpiRank) + "-" + name + ".yaml";
    model->store(filename);

    if (Config::APOLLO_RETRAIN_ENABLE) {
#ifdef ENABLE_OPENCV
      time_model->store("regtree-step-" + std::to_string(step) + "-rank-" +
                        std::to_string(apollo->mpiRank) + "-" + name + ".yaml");
      time_model->store(
          "regtree-latest"
          "-rank-" +
          std::to_string(apollo->mpiRank) + "-" + name + ".yaml");
#else
      throw std::runtime_error("Retraining requires OpenCV");
#endif
    }
  }
}

int Apollo::Region::getPolicyIndex(Apollo::RegionContext *context)
{
//#ifdef PERF_CNTR_MODE
//    int choice;
//
//    // If we're using a static policy, we can't use the 0 default policy
//    if(model_name == "Static"){
//      choice = model->getIndex(context->features);      
//    }
//    else{
//      // If we're measuring perf cntrs, use the default policy
//      choice = (this->shouldRunCounters) ? 0 : model->getIndex(context->features);
//    }
//#else
  int choice = model->getIndex( context->features );
//#endif

  if (Config::APOLLO_TRACE_POLICY) {
    std::stringstream trace_out;
    int rank;
    rank = apollo->mpiRank;
    trace_out << "Rank " << rank << " region " << name << " model "
              << model->name << " features [ ";
    for (auto &f : context->features)
      trace_out << (int)f << ", ";
    trace_out << "] policy " << choice << std::endl;
    std::cout << trace_out.str();
    std::ofstream fout("rank-" + std::to_string(rank) + "-policies.txt",
                       std::ofstream::app);
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
  return choice;
}

void Apollo::Region::parsePolicyModel(const std::string &model_info)
{
  size_t pos = model_info.find(",");
  model_name = model_info.substr(0, pos);

  // Parse any parameters, return if there are not any.
  if (std::string::npos == pos) return;

  std::vector<std::string> params_regex;
  if (model_name == "Static")
    params_regex.push_back("(policy)=([0-9]+)");
  else if (model_name == "DecisionTree") {
    params_regex.push_back("(max_depth)=([0-9]+)");
    params_regex.push_back("(explore)=(RoundRobin|Random)");
    params_regex.push_back("(load)");
    params_regex.push_back("(load-dataset)");
    params_regex.push_back("(load-dataset)=([a-zA-Z0-9_\\-\\.]+)");
    params_regex.push_back("(load)=([a-zA-Z0-9_\\-\\.]+)");
  } else if (model_name == "RandomForest") {
    params_regex.push_back("(num_trees|max_depth)=([0-9]+)");
    params_regex.push_back("(explore)=(RoundRobin|Random)");
    params_regex.push_back("(load)");
    params_regex.push_back("(load-dataset)");
    params_regex.push_back("(load-dataset)=([a-zA-Z0-9_\\-\\.]+)");
    params_regex.push_back("(load)=([a-zA-Z0-9_\\-\\.]+)");
  } else if (model_name == "PolicyNet") {
    params_regex.push_back(
        "(lr|beta|beta1|beta2|threshold)=(([+-]?([[:d:]]*\\.?([[:d:]]*)?))([Ee]"
        "[+-]?[[:d:]]+)?)");
    params_regex.push_back("(load)");
    params_regex.push_back("(load)=([a-zA-Z0-9_\\-\\.]+)");
    params_regex.push_back("(load-dataset)");
    params_regex.push_back("(load-dataset)=([a-zA-Z0-9_\\-\\.]+)");
  }

  std::string model_params_str = model_info.substr(pos + 1);
  do {
    pos = model_params_str.find(",");
    std::string keyval_str = model_params_str.substr(0, pos);
    model_params_str = model_params_str.substr(pos + 1);

    bool matched = false;
    std::smatch m;
    for (auto &regex_str : params_regex) {
      std::regex regex(regex_str);
      if (std::regex_match(keyval_str, m, regex)) {
        auto key = m[1];
        auto value = m[2];
        model_params[key] = value;
        matched = true;
        // std::cout << "Found key " << key << " value " << value << "\n";
      }
    }

    if (!matched) {
      std::cerr << "ERROR: Parameter " << keyval_str
                << " is not valid for model " << model_name << std::endl;
      abort();
    }
  } while (std::string::npos != pos);
}

Apollo::Region::Region(const int num_features,
                       const char *regionName,
                       int num_policies,
                       int _min_training_data,
                       const std::string &_model_info,
                       const std::string &modelYamlFile)
    : num_features(num_features),
      num_policies(num_policies),
      model_info(_model_info),
      current_context(nullptr),
      idx(0)
{

  if(Config::APOLLO_MIN_TRAINING_DATA){
    this->min_training_data = Config::APOLLO_MIN_TRAINING_DATA;
  }
  else{
    this->min_training_data = _min_training_data;
  }

  apollo = Apollo::instance();

  strncpy(name, regionName, sizeof(name) - 1);
  name[sizeof(name) - 1] = '\0';

  // If there is no model_info, parse the policy model from the env
  // variable, else parse it from the model_info argument.
  if (model_info.empty()) model_info = Config::APOLLO_POLICY_MODEL;
  parsePolicyModel(model_info);

#ifdef PERF_CNTR_MODE
    if(Config::APOLLO_ENABLE_PERF_CNTRS){
        //this->num_features = (int) Config::APOLLO_PERF_CNTRS.size();
        //this->shouldRunCounters = 1;
        this->papiPerfCnt = new Apollo::PapiCounters(this->apollo);
    }
    else{
        this->shouldRunCounters = 0;
        this->papiPerfCnt = nullptr;
    }
#else
    this->shouldRunCounters = 0;
    this->papiPerfCnt = nullptr;
#endif 


#ifdef PERF_CNTR_MODE
    if(Config::APOLLO_ENABLE_PERF_CNTRS){
  model = ModelFactory::createPolicyModel(model_name,
                                          apollo->num_events,
                                          num_policies,
                                          model_params);

//#ifdef PAPI_ZERO_VECTOR
    // setup our zero vector to pass to model->getIndex(...) when
    // we run with PA
    this->zero_vector = std::vector<float>(apollo->num_events, 0);

    // let's add the 0 vector to the dataset for this region
    // this is so that in case any training is done, we have 
    // the 0 vector available to make predictions for PA
    // We insert it mapping to policy 0 and with an xtime of 0
    this->dataset.insert(this->zero_vector, 0, 0);
//#endif

    }
    else{
  model = ModelFactory::createPolicyModel(model_name,
                                          num_features,
                                          num_policies,
                                          model_params);
    }
#else
  model = ModelFactory::createPolicyModel(model_name,
                                          num_features,
                                          num_policies,
                                          model_params);
#endif

  if (!modelYamlFile.empty()) model->load(modelYamlFile);

  if (model_params.count("load")) {
    std::string model_file;
    if (model_params["load"].empty())
      model_file = Config::APOLLO_OUTPUT_DIR + "/" + Config::APOLLO_MODELS_DIR +
                   "/" + model_name + "-latest-rank-" +
                   std::to_string(apollo->mpiRank) + "-" + std::string(name) +
                   ".yaml";
    else
      model_file = model_params["load"];

    if (fileExists(model_file)) {
      // std::cout << "Model Load " << model_file << std::endl;
      model->load(model_file);
    } else {
      std::cerr << "ERROR: could not load model file " << model_file
                << ", abort" << std::endl;
      abort();
    }
  }

  if (Config::APOLLO_PERSISTENT_DATASETS) {
    std::string dataset_file = Config::APOLLO_OUTPUT_DIR + "/" +
                               Config::APOLLO_DATASETS_DIR + "/Dataset-" +
                               std::string(name) + ".yaml";
    std::ifstream ifs(dataset_file);
    if (ifs) dataset.load(ifs);
    ifs.close();

    // Check if auto-training applies: min_training_data is set and dataset
    // size is large enough.
    autoTrain();
  }

  if (model_params.count("load-dataset")) {
    std::string dataset_file;
    if (model_params["load-dataset"].empty()){
      dataset_file = Config::APOLLO_OUTPUT_DIR + "/" +
                               Config::APOLLO_DATASETS_DIR + "/Dataset-" +
                               std::string(name) + ".yaml";
    }
    else{
      dataset_file = model_params["load-dataset"];
    }
    std::ifstream ifs(dataset_file);
    if (!ifs) {
      std::cerr << "ERROR: could not load dataset file " << dataset_file
                << std::endl;
      abort();
    }

    dataset.load(ifs);

    //std::cout << "creating region from dataset file " << dataset_file << std::endl;

    // Train with whatever data existing in the dataset, ignore
    // min_training_data (if set).
    train(0);
    ifs.close();
  }

  if (model_name == "Optimal") {
    std::string model_file = "opt-" + std::string(name) + "-rank-" +
                             std::to_string(apollo->mpiRank) + ".txt";
    if (!fileExists(model_file)) {
      std::cerr << "Optimal policy file " << model_file << " does not exist"
                << std::endl;
      abort();
    }

    model->load(model_file);
  }

  if (Config::APOLLO_TRACE_CSV) {
    std::string fname(Config::APOLLO_OUTPUT_DIR + "/" +
                      Config::APOLLO_TRACES_DIR + "/trace-" + model_info +
                      "-region-" + name + "-rank-" +
                      std::to_string(apollo->mpiRank) + ".csv");
    std::cout << "TRACE_CSV fname " << fname << std::endl;
    trace_file.open(fname);
    if (trace_file.fail()) {
      std::cerr << "Error opening trace file " + fname << std::endl;
      abort();
    }
    // Write header.
    trace_file << "rankid training region idx globalidx";
    // trace_file << "features";

#ifdef PERF_CNTR_MODE
    if(Config::APOLLO_ENABLE_PERF_CNTRS){
      for (int i = 0; i < papiPerfCnt->apollo->num_events; i++)
         trace_file << " f" << i;
    }
    else{
      for (int i = 0; i < num_features; i++)
         trace_file << " f" << i;

    }
#else
    for (int i = 0; i < num_features; i++)
       trace_file << " f" << i;
#endif
    trace_file << " policy xtime\n";
  }

  // std::cout << "Insert region " << name << " ptr " << this << std::endl;
  const auto ret = apollo->regions.insert({name, this});

  return;
}

#ifdef PERF_CNTR_MODE
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
#else
// If counters don't get enabled, just return immediately
void Apollo::Region::apolloThreadBegin(){ return; }
void Apollo::Region::apolloThreadEnd(){ return; }
#endif

Apollo::Region::~Region()
{
#ifdef PERF_CNTR_MODE
    if(this->papiPerfCnt){
        //fprintf(stderr, "stopping papicntrs for region [%s]\n", this->name);
        delete this->papiPerfCnt;
    }
#endif

  // Disable period based flushing.
  Config::APOLLO_GLOBAL_TRAIN_PERIOD = 0;
  while (pending_contexts.size() > 0)
    collectPendingContexts();

  if (Config::APOLLO_TRACE_CSV) trace_file.close();

  if (Config::APOLLO_PERSISTENT_DATASETS) {
    std::ofstream file_out(Config::APOLLO_OUTPUT_DIR + "/" +
                           Config::APOLLO_DATASETS_DIR + "/Dataset-" +
                           std::string(name) + ".yaml");
    if (!file_out.is_open())
      std::cerr << "ERROR: Cannot write dataset of " + std::string(name) +
                       " to database\n";
    dataset.store(file_out);
    file_out.close();
  }

  return;
}

Apollo::RegionContext *Apollo::Region::begin()
{
  Apollo::RegionContext *context = new Apollo::RegionContext(num_features);
  current_context = context;
  context->idx = this->idx;
  this->idx++;
  // Use the sync timer by default.
  context->timer = Timer::create<Timer::Sync>();
  context->timer->start();

  return context;
}

Apollo::RegionContext *Apollo::Region::begin(std::vector<float> &features)
{
#ifdef PERF_CNTR_MODE
    //std::cout << "USING THREAD COUNT: " << omp_get_max_threads() << std::endl;

    //this->lastFeats = context->features;

    // Check to see if we've already seen this feature set before
    // If not, let's set the flag to run the counters 
    // If so, we should just not run the counter code
    // as we know the counter values already since they're policy-independent
    if(this->papiPerfCnt){
        this->shouldRunCounters = (this->feats_to_cntr_vals.find(features) == this->feats_to_cntr_vals.end());
        //this->shouldRunCounters = !(this->feats_to_cntr_vals.contains(context->features));

        // If we already have run the counters, then translate the passed-in
        // feature vector so we can properly getPolicyIndex() if a tree has
        // already been trained
        if(!this->shouldRunCounters){
            features = this->feats_to_cntr_vals[features];
            //this->lastFeats = context->features;
        }
//#ifdef PAPI_ZERO_VECTOR
        else{
          // set the feature vector to the 0 vector to query the given model
          // case Static model: getPolicy() will return static policy
          // case RoundRobin training: getPolicy() will return next RR policy
          // case DTree model: getPolicy() will return 0 policy
          this->features_backup = features;
          features = this->zero_vector;
        }
//#endif

    }
#endif

    Apollo::RegionContext *context = begin();
    context->features = features;

    return context;
}

void Apollo::Region::autoTrain()
{
  if (!model->isTrainable()) return;

  if (Config::APOLLO_GLOBAL_TRAIN_PERIOD &&
      (apollo->region_executions % Config::APOLLO_GLOBAL_TRAIN_PERIOD) == 0) {
    apollo->train(apollo->region_executions,
                  /* doCollectPendingContexts */ false);
  } else if (Config::APOLLO_PER_REGION_TRAIN_PERIOD &&
             (idx % Config::APOLLO_PER_REGION_TRAIN_PERIOD) == 0) {
    train(idx, /* doCollectPendingContexts */ false);
  } else if (0 < min_training_data && min_training_data <= dataset.size()){
    if(Config::APOLLO_SINGLE_MODEL){
      apollo->train(idx, /* doCollectPendingContexts */ false);
    }
    else{
      train(idx, /* doCollectPendingContexts */ false);
    }
  }
}

void Apollo::Region::collectContext(Apollo::RegionContext *context,
                                    double metric)
{
#ifdef PERF_CNTR_MODE
  // If the performance counters were setup
  if (this->papiPerfCnt && this->shouldRunCounters) {

    // The counters were run, this was the first time we saw this user-supplied
    // feature vector. Thus, we should save it to the feature-counter mapping
    // along with adding the measure to the region measures.

      // First calculate the sum of each counter
      // These summary statistics are calculated across threads, so we
      // always have the same feature dimensions irregardless of thread count
      std::vector<float> vals = this->papiPerfCnt->getSummaryStats();

      // if we ran the counters, we set the context->features to be the 0 vector
      // so we need to grab the backup-up features and map them to the 
      // counter values

      // Map the user-provided features to the counter values
      //this->feats_to_cntr_vals[context->features] = vals;
//#ifdef PAPI_ZERO_VECTOR
      this->feats_to_cntr_vals[this->features_backup] = vals;
//#else
//      this->feats_to_cntr_vals[context->features] = vals;
//#endif

      // Set the current context features to these new values
      // so that we collect the timing measure and write it to
      // the trace file
      context->features = vals;
  }
#endif

  dataset.insert(context->features, context->policy, metric);

  if (Config::APOLLO_TRACE_CSV) {
    trace_file << apollo->mpiRank << " ";
    trace_file << model->name << " ";
    trace_file << this->name << " ";
    trace_file << context->idx << " ";
    trace_file << apollo->region_executions << " ";
    for (auto &f : context->features)
      trace_file << f << " ";
    trace_file << context->policy << " ";
    trace_file << metric << "\n";
  }

  apollo->region_executions++;

  autoTrain();

  delete context;
  current_context = nullptr;
}

void Apollo::Region::end(Apollo::RegionContext *context, double metric)
{
  // std::cout << "END REGION " << name << " metric " << metric << std::endl;

  collectContext(context, metric);

  collectPendingContexts();

  return;
}

void Apollo::Region::collectPendingContexts()
{
  auto isDone = [this](Apollo::RegionContext *context) {
    if (!context->timer)
      throw std::runtime_error("No timer has been set for the context");
    double metric;
    if (context->timer->isDone(metric)) {
      collectContext(context, metric);
      return true;
    }

    return false;
  };

  pending_contexts.erase(std::remove_if(pending_contexts.begin(),
                                        pending_contexts.end(),
                                        isDone),
                         pending_contexts.end());
}

void Apollo::Region::end(Apollo::RegionContext *context)
{
  context->timer->stop();
  pending_contexts.push_back(context);
  collectPendingContexts();
}

// DEPRECATED
int Apollo::Region::getPolicyIndex(void)
{
  return getPolicyIndex(current_context);
}

// DEPRECATED
void Apollo::Region::end(double metric) { end(current_context, metric); }

// DEPRECATED
void Apollo::Region::end(void) { end(current_context); }

void Apollo::Region::setFeature(Apollo::RegionContext *context, float value)
{
  context->features.push_back(value);
#ifdef PERF_CNTR_MODE
  // Check to see if we've already seen this feature set before
  // If not, let's set the flag to run the counters
  // If so, we should just not run the counter code
  // as we know the counter values already since they're policy-independent
  if (this->papiPerfCnt && context->features.size() == this->num_features) {
    this->shouldRunCounters = (this->feats_to_cntr_vals.find(context->features) == this->feats_to_cntr_vals.end());

    // If we already have run the counters, then translate the passed-in
    // feature vector so we can properly getPolicyIndex() if a tree has
    // already been trained
    if (!this->shouldRunCounters) {
      context->features = this->feats_to_cntr_vals[context->features];
    }
    // If we are going to run the counters, we need to make sure
    // we backup the original code-based features and we set the code
    // to execute with the 0 policy by passing the model the 0 vector;
    //
    else{
      this->features_backup = context->features;
      context->features = this->zero_vector;
    }
  }

  // Start the timer now so as to not capture a map search in it
  context->timer->start();
#endif

  return;
}

// DEPRECATED
void Apollo::Region::setFeature(float value)
{
  setFeature(current_context, value);
}
