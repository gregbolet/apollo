// Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC and other
// Apollo project developers. Produced at the Lawrence Livermore National
// Laboratory. See the top-level LICENSE file for details.
// SPDX-License-Identifier: MIT

#ifndef APOLLO_MODELS_BAYESIANOPTIM_H
#define APOLLO_MODELS_BAYESIANOPTIM_H

#include <string>

#include "apollo/PolicyModel.h"

#include <limbo/limbo.hpp>

using namespace limbo;

struct Params {
#ifdef USE_NLOPT
    struct opt_nloptnograd : public defaults::opt_nloptnograd {};
#else
    struct opt_gridsearch : public defaults::opt_gridsearch {};
#endif
    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {};
    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_PARAM(int, stats_enabled, true);
        BO_PARAM(bool, bounded, true);
    };
    struct stop_maxiterations { BO_DYN_PARAM(int, iterations); };
    //struct stop_mintolerance { BO_PARAM(double, tolerance, -0.1); };
    struct acqui_ei { BO_PARAM(double, jitter, 0.0); };
    //struct init_randomsampling { BO_PARAM(int, samples, 0); };
    struct kernel : public defaults::kernel { BO_PARAM(double, noise, 1e-10); };
    //struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {};
    struct kernel_maternthreehalves : public defaults::kernel_maternthreehalves {};
    struct opt_rprop : public defaults::opt_rprop {};
};


//using kernel_t = kernel::SquaredExpARD<Params>;
using kernel_t = kernel::MaternThreeHalves<Params>;
using mean_t = mean::Data<Params>;
using gp_t = model::GP<Params, kernel_t, mean_t>;
using acqui_t = acqui::EI<Params, gp_t>;
using init_t = init::NoInit<Params>;

using boptimizer_signature = boost::parameter::parameters<boost::parameter::optional<tag::acquiopt>,
    boost::parameter::optional<tag::statsfun>,
    boost::parameter::optional<tag::initfun>,
    boost::parameter::optional<tag::acquifun>,
    boost::parameter::optional<tag::stopcrit>,
    boost::parameter::optional<tag::modelfun>>;

template <class Params,
          class A1 = boost::parameter::void_,
          class A2 = boost::parameter::void_,
          class A3 = boost::parameter::void_,
          class A4 = boost::parameter::void_,
          class A5 = boost::parameter::void_,
          class A6 = boost::parameter::void_>
class CustomBOptimizer : public bayes_opt::BOptimizer<Params, A1, A2, A3, A4, A5, A6>{
        public:

            int getNumSamples(){
                // return this->_current_iteration;
                return static_cast<int>(this->_samples.size());
            }

            // we want to create a random number generator here for use in optimization
            // when we sample a point at random each time
            void setSeed(int seed){
                _seed = seed;
                _bound_rng = tools::rgen_double_t(0.0, 1.0, _seed);
                // this will be used when bo_opts 'bounded' is set to false, 
                // samples from a gaussian centered at 0 with a stddev of 10
                _unbound_rng = tools::rgen_gauss_t(0.0, 10.0, _seed);
                nlopt::srand(_seed);
            }

            struct defaults {
#ifdef USE_NLOPT
                using acquiopt_t = opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND>;
#elif defined(USE_LIBCMAES)
                using acquiopt_t = opt::Cmaes<Params>;
#else
#warning NO NLOpt, and NO Libcmaes: the acquisition function will be optimized by a grid search algorithm (which is usually bad). Please install at least NLOpt or libcmaes to use limbo!.
                using acquiopt_t = opt::GridSearch<Params>;
#endif
            };
            
            /// link to the corresponding BoBase (useful for typedefs)
            using base_t = bayes_opt::BoBase<Params, A1, A2, A3, A4, A5, A6>;
            using model_t = typename base_t::model_t;
            using acquisition_function_t = typename base_t::acquisition_function_t;
            // extract the types
            using args = typename boptimizer_signature::bind<A1, A2, A3, A4, A5, A6>::type;
            using acqui_optimizer_t = typename boost::parameter::binding<args, tag::acquiopt, typename defaults::acquiopt_t>::type;

            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            Eigen::VectorXd getNextPoint(const StateFunction& sfun, const AggregatorFunction& afun = AggregatorFunction(), bool reset=true)
            {
                if (reset) { this->setSeed(_seed); }

                std::cout << "total iters " << this->_total_iterations << std::endl;

                this->_init(sfun, afun, reset);

                if (!this->_observations.empty())
                    _model.compute(this->_samples, this->_observations);
                else
                    _model = model_t(StateFunction::dim_in(), StateFunction::dim_out());

                acqui_optimizer_t acqui_optimizer;
                
                if (!this->_stop(*this, afun)) {
                    acquisition_function_t acqui(_model, this->_current_iteration);

                    auto acqui_optimization =
                        [&](const Eigen::VectorXd& x, bool g) { return acqui(x, afun, g); };

                    Eigen::VectorXd starting_point;
                    if(Params::bayes_opt_bobase::bounded()){ starting_point = tools::random_vec(StateFunction::dim_in(), _bound_rng); }
                                                       else{ starting_point = tools::random_vec(StateFunction::dim_in(), _unbound_rng); }
                    Eigen::VectorXd new_sample = acqui_optimizer(acqui_optimization, starting_point, Params::bayes_opt_bobase::bounded());
                    return new_sample;
                }
                return Eigen::VectorXd(StateFunction::dim_out());
            }

            template <typename AggregatorFunction = FirstElem>
            void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const AggregatorFunction& afun = AggregatorFunction())
            {
              // appends the sample and val to _samples and _observations
              this->add_new_sample(sample, val);
              this->_update_stats(*this, afun);

              // add sample and update the GP model
              _model.add_sample(this->_samples.back(), this->_observations.back());

              if (Params::bayes_opt_boptimizer::hp_period() > 0
                  && (this->_current_iteration + 1) % Params::bayes_opt_boptimizer::hp_period() == 0)
                  _model.optimize_hyperparams();

              this->_current_iteration++;
              this->_total_iterations++;
            }


            void writeGPVizFiles(){
                // assume a 1D input and 1D output
                gp_t gp(1,1);

                // compute the GP from the data
                gp.compute(this->_samples, this->_observations);

                // use the GP to predict 100 points in the target space for plotting
                // these are 100 equally-spaced points
                // our GP input values are bound between 0-1
                // we need to map these back to the target space
                std::ofstream ofs("gp.dat");
                for (int i = 0; i < 100; ++i) {
                    Eigen::VectorXd v = tools::make_vector(i / 100.0).array();
                    Eigen::VectorXd mu;
                    double sigma;
                    std::tie(mu, sigma) = gp.query(v);
                    // an alternative (slower) is to query mu and sigma separately:
                    //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
                    //  double s2 = gp.sigma(v);
                    // we write the x value, mean-value, stddev-value
                    ofs << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
                }

                acquisition_function_t acqui(_model, this->_current_iteration);

                // let's do the same sampling process for the acquisition function
                std::ofstream ofs_acqui("acqui.dat");
                for (int i = 0; i < 100; ++i) {
                    Eigen::VectorXd v = tools::make_vector(i / 100.0).array();

                    opt::eval_t res = acqui(v, FirstElem(), false);

                    double x = std::get<0>(res);

                    ofs_acqui << v.transpose() << " " << x << std::endl;
                }

                // these are the points we actually sampled
                std::ofstream ofs_data("data.dat");
                for (size_t i = 0; i < this->_samples.size(); ++i)
                    ofs_data << this->_samples[i].transpose() << " " << this->_observations[i].transpose() << std::endl;

                return;
            }

        protected:
            model_t _model;
            int _seed = -1;
            tools::rgen_double_t _bound_rng = tools::rgen_double_t(0.0, 1.0);
            tools::rgen_gauss_t _unbound_rng = tools::rgen_gauss_t(0.0, 10.0);
};





namespace apollo
{
class BayesianOptim : public PolicyModel
{
public:
  BayesianOptim(int num_policies, int num_features);
  ~BayesianOptim(){boptimizer.writeGPVizFiles();};

  int getIndex(std::vector<float> &features);
  void load(const std::string &filename){};
  void store(const std::string &filename){};
  bool isTrainable() { return true; }
  void train(Apollo::Dataset &dataset);

private:
  int policy_choice;
  int num_features;
  int first_execution;
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> boptimizer;
  Eigen::VectorXd last_point;

};  

}  // end namespace apollo.

#endif
