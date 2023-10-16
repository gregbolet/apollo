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

struct eval_func {
    BO_DYN_PARAM(size_t, dim_in);
    BO_PARAM(size_t, dim_out, 1);

    eval_func() {
        std::cout << "Eval Funciton Init Called" << std::endl;
    }

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        std::cout << "THIS SHOULD NEVER GET CALLED!" << std::endl;
        return x + Eigen::VectorXd::Ones(x.rows());
    }
};

BO_DECLARE_DYN_PARAM(size_t, eval_func, dim_in);
BO_DECLARE_DYN_PARAM(int, Params::stop_maxiterations, iterations);


namespace apollo
{

BayesianOptim::BayesianOptim(int num_policies, int num_features)
  						: PolicyModel(num_policies, "BayesianOptim"),
        			      num_features(num_features), first_execution(1) {

	std::cout << "Set up BO!" << std::endl;

    // This is how many optimize() calls to make -- we set it to 1
	Params::stop_maxiterations::set_iterations(1);

    // This is how many dimensions of the search space we have, set it to 1 for now
    // since we're mapping 'policy' to 'xtime'
	eval_func::set_dim_in(1);
    boptimizer.setSeed(775);
};


int BayesianOptim::getIndex(std::vector<float> &features) {

	// This function will simply query the BO model
    last_point = boptimizer.getNextPoint(eval_func(), FirstElem(), first_execution);

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

    std::cout << "training samps:" << boptimizer.getNumSamples() << "\ndataset\n";

    const auto &data = dataset.toVectorOfTuples();

    // upate the model with all the data... This is redundantly adding points
    // we need some way to always just add only the NEW points
    for(auto item : data){
        std::vector<float> &feats = std::get<0>(item); 
        std::vector<double> features(feats.begin(), feats.end());

        std::cout << "[";
        for(double j : features){
            std::cout << j << ",";
        }
        std::cout << "]";

        int &policy = std::get<1>(item);
        double &metric = std::get<2>(item);
        //std::cout << "policy " << policy << " xtime " << metric << std::endl;

        // we need to map the x 'policy' value between the range of [0,1]
        double mapped_policy = ((double) policy)/policy_count;

        //std::cout << "policy " << policy << " (" << mapped_policy << ")" << " xtime " << metric << std::endl;

        Eigen::VectorXd x = Eigen::VectorXd(1);
        x(0) = mapped_policy;

        // it does maximization by default, so we need to invert the y value
        Eigen::VectorXd y = Eigen::VectorXd(1);
        y(0) = -metric;

        std::cout << "policy " << policy << " (" << x(0) << ")" << " xtime " << y(0) << std::endl;

        boptimizer.updateModel(x,y,FirstElem());
    }

    //first_execution = 1;
    return;
}

}  // end namespace apollo.