// Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC and other
// Apollo project developers. Produced at the Lawrence Livermore National
// Laboratory. See the top-level LICENSE file for details.
// SPDX-License-Identifier: MIT

#include "apollo/models/BayesianOptim.h"


#include <cstring>
#include <string>

// The "aggregator" function is used to take the result of n
// Gaussian processes and combine them into a single value 
// for the acquisition function. We use the default FirstElem()
// which is just the identity function.

BO_DECLARE_DYN_PARAM(size_t, eval_func, dim_in);
BO_DECLARE_DYN_PARAM(int, Params::stop_maxiterations, iterations);

BO_DECLARE_DYN_PARAM(bool, Params::bayes_opt_bobase, bounded);
BO_DECLARE_DYN_PARAM(int, Params::bayes_opt_bobase, stats_enabled);

BO_DECLARE_DYN_PARAM(double, Params::kernel_squared_exp_ard, sigma_sq);
BO_DECLARE_DYN_PARAM(double, Params::kernel_maternthreehalves, sigma_sq);
BO_DECLARE_DYN_PARAM(double, Params::kernel_maternfivehalves, sigma_sq);
BO_DECLARE_DYN_PARAM(double, Params::kernel_exp, sigma_sq);

BO_DECLARE_DYN_PARAM(int, Params::kernel_squared_exp_ard, k);
BO_DECLARE_DYN_PARAM(double, Params::kernel_maternthreehalves, l);
BO_DECLARE_DYN_PARAM(double, Params::kernel_maternfivehalves, l);
BO_DECLARE_DYN_PARAM(double, Params::kernel_exp, l);

BO_DECLARE_DYN_PARAM(double, Params::kernel, noise);
BO_DECLARE_DYN_PARAM(bool, Params::kernel, optimize_noise);

BO_DECLARE_DYN_PARAM(double, Params::acqui_ei, jitter);
BO_DECLARE_DYN_PARAM(double, Params::acqui_ucb, alpha);

// This doesn't work, because they use a constexpr with delta
//BO_DECLARE_DYN_PARAM(double, Params::acqui_gpucb, delta);


namespace apollo
{

BayesianOptim::BayesianOptim(int num_policies, int num_features,
                std::string &kernel, std::string &acqui, 
                double acqui_hyper, double sigma_sq, double l, 
                int k, double whiteKernel, int seed)
  						: PolicyModel(num_policies, "BayesianOptim"),
        			      num_features(num_features), first_execution(1) {

	//std::cout << "Set up BO!" << std::endl;

    // This is how many optimize() calls to make -- we set it to 1
	Params::stop_maxiterations::set_iterations(1);

    // This is how many dimensions of the search space we have, set it to 1 for now
    // since we're mapping 'policy' to 'xtime'
	eval_func::set_dim_in(1);
    Params::bayes_opt_bobase::set_bounded(true);
    Params::bayes_opt_bobase::set_stats_enabled(0);
    Params::kernel::set_noise(whiteKernel);


    if(kernel == "sqexp"){
        if     (acqui == "ei"){    boptimizer = new BO_SQEXP_EI   (seed, sigma_sq, l, acqui_hyper); }
        else if(acqui == "ucb"){   boptimizer = new BO_SQEXP_UCB  (seed, sigma_sq, l, acqui_hyper); }
        else if(acqui == "gpucb"){ boptimizer = new BO_SQEXP_GPUCB(seed, sigma_sq, l, acqui_hyper); }
        else{ throw std::runtime_error("Invalid acquisition function" + acqui + " for SQEXP kernel!"); }
    }
    else if(kernel == "sqexpard"){
        if     (acqui == "ei"){    boptimizer = new BO_SQEXPARD_EI   (seed, sigma_sq, k, acqui_hyper); }
        else if(acqui == "ucb"){   boptimizer = new BO_SQEXPARD_UCB  (seed, sigma_sq, k, acqui_hyper); }
        else if(acqui == "gpucb"){ boptimizer = new BO_SQEXPARD_GPUCB(seed, sigma_sq, k, acqui_hyper); }
        else{ throw std::runtime_error("Invalid acquisition function" + acqui + " for SQEXPARD kernel!"); }
    }
    else if(kernel == "mat32"){
        if     (acqui == "ei"){    boptimizer = new BO_MATERN32_EI   (seed, sigma_sq, l, acqui_hyper); }
        else if(acqui == "ucb"){   boptimizer = new BO_MATERN32_UCB  (seed, sigma_sq, l, acqui_hyper); }
        else if(acqui == "gpucb"){ boptimizer = new BO_MATERN32_GPUCB(seed, sigma_sq, l, acqui_hyper); }
        else{ throw std::runtime_error("Invalid acquisition function" + acqui + " for MATERN 3/2 kernel!"); }
    }
    else if(kernel == "mat52"){
        if     (acqui == "ei"){    boptimizer = new BO_MATERN52_EI   (seed, sigma_sq, l, acqui_hyper); }
        else if(acqui == "ucb"){   boptimizer = new BO_MATERN52_UCB  (seed, sigma_sq, l, acqui_hyper); }
        else if(acqui == "gpucb"){ boptimizer = new BO_MATERN52_GPUCB(seed, sigma_sq, l, acqui_hyper); }
        else{ throw std::runtime_error("Invalid acquisition function" + acqui + " for MATERN 5/2 kernel!"); }
    }
    else{
        std::cout << "No kernel and acqui function given, using default: SQEXP and EI" << std::endl;
        boptimizer = new BO_SQEXP_EI(seed, sigma_sq, l, acqui_hyper);
    }
    //boptimizer->setSeed(seed);


};

BayesianOptim::~BayesianOptim(){
    boptimizer->writeGPVizFiles();
    delete boptimizer;
    return;
};


int BayesianOptim::getIndex(std::vector<float> &features) {

	// This function will simply query the BO model
    last_point = boptimizer->getNextPoint(eval_func(), FirstElem(), first_execution);

    if(first_execution){ first_execution = 0; }

    // policy will be in range of [0,1]
    // need to convert it to integers by mapping [0,policy_count) <--> [0,1]
    double raw_policy = last_point(0);

    int policy = std::min((int) (raw_policy*policy_count), policy_count-1);

    return policy;
}


void BayesianOptim::train(Apollo::Dataset &dataset){
    // Apollo gives us the dataset of features and their mapping
    // Need to convert it to VectorXd format for the optimizer.

    //std::cout << "training samps:" << boptimizer->getNumSamples() << "\ndataset\n";

    const auto &data = dataset.toVectorOfTuples();

    // upate the model with all the data... This is redundantly adding points
    // we need some way to always just add only the NEW points
    for(auto item : data){
        std::vector<float> &feats = std::get<0>(item); 
        std::vector<double> features(feats.begin(), feats.end());

        //std::cout << "[";
        //for(double j : features){
        //    std::cout << j << ",";
        //}
        //std::cout << "]";

        int &policy = std::get<1>(item);
        double &metric = std::get<2>(item);
        //std::cout << "policy " << policy << " xtime " << metric << std::endl;

        // we need to map the x 'policy' value between the range of [0,1]
        double mapped_policy = ((double) policy)/(policy_count-1);

        //std::cout << "policy " << policy << " (" << mapped_policy << ")" << " xtime " << metric << std::endl;

        Eigen::VectorXd x = Eigen::VectorXd(1);
        x(0) = mapped_policy;

        // it does maximization by default, so we need to invert the y value
        Eigen::VectorXd y = Eigen::VectorXd(1);
        y(0) = -metric;

        //std::cout << "policy " << policy << " (" << x(0) << ")" << " xtime " << y(0) << std::endl;

        boptimizer->updateModel(x,y,FirstElem());
    }

    // this forces only new training data to be around on the next train() call
    // this breaks if we have multiple models on the same region -- some models might
    // not get any data...
    dataset.clear();

    //first_execution = 1;
    return;
}

}  // end namespace apollo.