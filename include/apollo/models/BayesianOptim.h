// Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC and other
// Apollo project developers. Produced at the Lawrence Livermore National
// Laboratory. See the top-level LICENSE file for details.
// SPDX-License-Identifier: MIT

#ifndef APOLLO_MODELS_BAYESIANOPTIM_H
#define APOLLO_MODELS_BAYESIANOPTIM_H

#include <string>

#include "apollo/PolicyModel.h"
#include <chrono>

#include <limbo/limbo.hpp>

// for boost::any usasge
#include <boost/any.hpp>

using namespace limbo;

struct Params {
#ifdef USE_NLOPT
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        // added these in to follow limbo/src/benchmarks/limbo/bench.cpp
        // to re-create their benchmark timing results
        BO_PARAM(double, fun_tolerance, 1e-6);
        BO_PARAM(double, xrel_tolerance, 1e-6);
    };
#else
    struct opt_gridsearch : public defaults::opt_gridsearch {};
#endif
    // This does no GP hyperparameter optimization with -1
    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
        BO_PARAM(int, hp_period, -1);
    };
    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_DYN_PARAM(int, stats_enabled);
        BO_DYN_PARAM(bool, bounded);
    };
    struct stop_maxiterations { BO_DYN_PARAM(int, iterations); };
    //struct stop_mintolerance { BO_PARAM(double, tolerance, -0.1); };
    struct acqui_ei : public defaults::acqui_ei { BO_DYN_PARAM(double, jitter); };
    struct acqui_ucb : public defaults::acqui_ucb { BO_DYN_PARAM(double, alpha); };
    //struct acqui_gpucb : public defaults::acqui_gpucb { BO_DYN_PARAM(double, delta); };
    struct acqui_gpucb : public defaults::acqui_gpucb { BO_PARAM(double, delta, 0.1); };
    //struct init_randomsampling { BO_PARAM(int, samples, 0); };
    struct kernel : public defaults::kernel { 
        BO_DYN_PARAM(double, noise); 
        BO_DYN_PARAM(bool, optimize_noise); 
    };
    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
        BO_DYN_PARAM(double, sigma_sq);
        BO_DYN_PARAM(int, k);
    };
    struct kernel_maternthreehalves : public defaults::kernel_maternthreehalves {
        BO_DYN_PARAM(double, sigma_sq);
        BO_DYN_PARAM(double, l);
    };
    struct kernel_maternfivehalves : public defaults::kernel_maternfivehalves {
        BO_DYN_PARAM(double, sigma_sq);
        BO_DYN_PARAM(double, l);
    };
    struct kernel_exp : public defaults::kernel_exp {
        BO_DYN_PARAM(double, sigma_sq);
        BO_DYN_PARAM(double, l);
    };
    struct opt_rprop : public defaults::opt_rprop {};
};

// BO_DYN_PARAM makes dim_in a static type, this might not be
// a good idea for later when we have multiple eval_func instances...
struct eval_func {
    BO_DYN_PARAM(size_t, dim_in);
    BO_PARAM(size_t, dim_out, 1);

    eval_func() {
        return;
        //std::cout << "Eval Funciton Init Called" << std::endl;
    }

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        std::cout << "THIS SHOULD NEVER GET CALLED!" << std::endl;
        return x + Eigen::VectorXd::Ones(x.rows());
    }
};


//using kernel_t = kernel::Exp<Params>; // AKA: RBF (Radial Basis Function)
///using kernel_t = kernel::SquaredExpARD<Params>;
//using kernel_t = kernel::MaternThreeHalves<Params>;
//using kernel_t = kernel::MaternFiveHalves<Params>;
///using gp_t = model::GP<Params, kernel_t, mean_t>;
///using acqui_t = acqui::EI<Params, gp_t>;
//using acqui_t = acqui::UCB<Params, gp_t>;
//using acqui_t = acqui::GP_UCB<Params, gp_t>;

using mean_t = mean::Data<Params>;
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

            //template <typename StateFunction, typename AggregatorFunction = FirstElem>
            //Eigen::VectorXd getNextPoint(const StateFunction& sfun, const AggregatorFunction& afun = AggregatorFunction(), bool reset=true)
            Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun = FirstElem(), bool reset=true)
            {
                double timer_start, timer_end;
                struct timespec ts;

                if (reset) { 
                    this->setSeed(_seed); 
                    this->_init(sfun, afun, reset);
                    if (!this->_observations.empty())
                        _model.compute(this->_samples, this->_observations, false);
                    else
                        _model = model_t(eval_func::dim_in(), eval_func::dim_out());
                }

                // this is necessary here :/
                this->_current_iteration = 0;

                //std::cout << "total iters " << this->_total_iterations << std::endl;


                //clock_gettime(CLOCK_MONOTONIC, &ts);
                //timer_start = ts.tv_sec + ts.tv_nsec / 1e9;

                //clock_gettime(CLOCK_MONOTONIC, &ts);
                //timer_end = ts.tv_sec + ts.tv_nsec / 1e9;
                //std::cout << (timer_end - timer_start) << ",";

                acqui_optimizer_t acqui_optimizer;
                
                // we only do one sample at a time, so we change the `while` to an `if`
                if (!this->_stop(*this, afun)) {
                    acquisition_function_t acqui(_model, this->_current_iteration);

                    auto acqui_optimization =
                        [&](const Eigen::VectorXd& x, bool g) { return acqui(x, afun, g); };

                    Eigen::VectorXd starting_point;
                    if(Params::bayes_opt_bobase::bounded()){ starting_point = tools::random_vec(eval_func::dim_in(), _bound_rng); }
                                                       else{ starting_point = tools::random_vec(eval_func::dim_in(), _unbound_rng); }

                    clock_gettime(CLOCK_MONOTONIC, &ts);
                    timer_start = ts.tv_sec + ts.tv_nsec / 1e9;

                    Eigen::VectorXd new_sample = acqui_optimizer(acqui_optimization, starting_point, Params::bayes_opt_bobase::bounded());

                    clock_gettime(CLOCK_MONOTONIC, &ts);
                    timer_end = ts.tv_sec + ts.tv_nsec / 1e9;
                    std::cout << (timer_end - timer_start) << std::endl;

                    return new_sample;
                }
                return Eigen::VectorXd(eval_func::dim_out());
            }

            //template <typename AggregatorFunction = FirstElem>
            //void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const AggregatorFunction& afun = AggregatorFunction())
            void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun = FirstElem())
            {

              // appends the sample and val to _samples and _observations
              this->add_new_sample(sample, val);
              this->_update_stats(*this, afun);

              // add sample and update the GP model
              _model.add_sample(this->_samples.back(), this->_observations.back());

              // We don't do hyperparam optimization, so let's leave this out.
              //if (Params::bayes_opt_boptimizer::hp_period() > 0
              //    && (this->_current_iteration + 1) % Params::bayes_opt_boptimizer::hp_period() == 0)
              //    _model.optimize_hyperparams();

              this->_current_iteration++;
              this->_total_iterations++;

            }


            void writeGPVizFiles(std::string regionName){
                // assume a 1D input and 1D output
                // gp_t gp(1,1);
                model_t gp(1,1);

                // compute the GP from the data
                gp.compute(this->_samples, this->_observations, true);

                // compute the acquisition function from our internal model
                acquisition_function_t acqui(_model, this->_current_iteration);

                // use the GP to predict 100 points in the target space for plotting
                // these are 100 equally-spaced points
                // our GP input values are bound between 0-1
                // we need to map these back to the target space
                std::cout << "Writing replay/viz files for: " << regionName << std::endl;

                std::ofstream ofs(regionName+".bo");
                int viz_samples = this->_samples.size()*10;
                for (int i = 0; i < viz_samples; ++i) {
                    Eigen::VectorXd v = tools::make_vector(i / (float)viz_samples).array();

                    // surrogate function
                    Eigen::VectorXd mu;
                    double sigma;
                    std::tie(mu, sigma) = gp.query(v);

                    // acquisition function 
                    opt::eval_t res = acqui(v, FirstElem(), false);
                    double acq_val = std::get<0>(res);

                    // an alternative (slower) is to query mu and sigma separately:
                    //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
                    //  double s2 = gp.sigma(v);
                    // we write the x value, mean-value, stddev-value, and acqusition function value
                    ofs << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << " " << acq_val << std::endl;
                }

                // these are the points we actually sampled
                std::ofstream ofs_data(regionName+".dat");
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












struct CustomBOptimBase{
    virtual int getNumSamples() = 0;
    virtual void setSeed(int seed) = 0;
    virtual void writeGPVizFiles(std::string regionName) = 0;
    virtual void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun) = 0;
    virtual Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true) = 0;
    CustomBOptimBase(){};
    virtual ~CustomBOptimBase(){};
};









struct BO_SQEXP_EI : CustomBOptimBase{
    public: 
        BO_SQEXP_EI(int seed, double sigma_sq, double l, double jitter) : CustomBOptimBase() {
            std::cout << "setup BO_SQEXP_EI!" << std::endl;
            std::cout << "using seed: " << seed << " sigma_sq: " << sigma_sq << " l: " << l << " jitter: " << jitter << std::endl;
            MY_BO.setSeed(seed);

            Params::kernel_exp::set_sigma_sq(sigma_sq);
            Params::kernel_exp::set_l(l);
            Params::acqui_ei::set_jitter(jitter);
        }
        ~BO_SQEXP_EI(){
            //std::cout << "destroying BO_SQEXP_EI!" << std::endl;
        }
        int getNumSamples() {return MY_BO.getNumSamples();};
        void setSeed(int seed) {MY_BO.setSeed(seed);};
        void writeGPVizFiles(std::string regionName){MY_BO.writeGPVizFiles(regionName);};
        void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun){
            MY_BO.updateModel(sample, val, afun);
        };
        Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true){
            return MY_BO.getNextPoint(sfun, afun, reset);
        };
    private:
        #undef kernel_t
        #undef acqui_t
        #undef gp_t
        #define kernel_t kernel::Exp<Params>
        #define gp_t model::GP<Params, kernel_t, mean_t>
        #define acqui_t acqui::EI<Params, gp_t>
        CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> MY_BO;
};

struct BO_SQEXP_UCB : CustomBOptimBase{
    public: 
        BO_SQEXP_UCB(int seed, double sigma_sq, double l, double alpha) : CustomBOptimBase() {
            std::cout << "setup BO_SQEXP_UCB!" << std::endl;
            std::cout << "using seed: " << seed << " sigma_sq: " << sigma_sq << " l: " << l << " alpha: " << alpha << std::endl;
            MY_BO.setSeed(seed);

            Params::kernel_exp::set_sigma_sq(sigma_sq);
            Params::kernel_exp::set_l(l);
            Params::acqui_ucb::set_alpha(alpha);
        }
        ~BO_SQEXP_UCB(){
            //std::cout << "destroying BO_SQEXP_UCB!" << std::endl;
        }
        int getNumSamples() {return MY_BO.getNumSamples();};
        void setSeed(int seed) {MY_BO.setSeed(seed);};
        void writeGPVizFiles(std::string regionName){MY_BO.writeGPVizFiles(regionName);};
        void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun){
            MY_BO.updateModel(sample, val, afun);
        };
        Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true){
            return MY_BO.getNextPoint(sfun, afun, reset);
        };
    private:
        #undef kernel_t
        #undef acqui_t
        #undef gp_t
        #define kernel_t kernel::Exp<Params>
        #define gp_t model::GP<Params, kernel_t, mean_t>
        #define acqui_t acqui::UCB<Params, gp_t>
        CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> MY_BO;
};

struct BO_SQEXP_GPUCB : CustomBOptimBase{
    public: 
        BO_SQEXP_GPUCB(int seed, double sigma_sq, double l, double delta) : CustomBOptimBase() {
            std::cout << "setup BO_SQEXP_GPUCB!" << std::endl;
            std::cout << "using seed: " << seed << " sigma_sq: " << sigma_sq << " l: " << l << " delta: " << delta << std::endl;
            MY_BO.setSeed(seed);

            Params::kernel_exp::set_sigma_sq(sigma_sq);
            Params::kernel_exp::set_l(l);
            // can't set this due to acqui_gpucb constexpr on delta...
            // might need to redefine acqui_gpucb myself...
            //Params::acqui_gpucb::set_delta(delta);
        }
        ~BO_SQEXP_GPUCB(){
            //std::cout << "destroying BO_SQEXP_GPUCB!" << std::endl;
        }
        int getNumSamples() {return MY_BO.getNumSamples();};
        void setSeed(int seed) {MY_BO.setSeed(seed);};
        void writeGPVizFiles(std::string regionName){MY_BO.writeGPVizFiles(regionName);};
        void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun){
            MY_BO.updateModel(sample, val, afun);
        };
        Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true){
            return MY_BO.getNextPoint(sfun, afun, reset);
        };
    private:
        #undef kernel_t
        #undef acqui_t
        #undef gp_t
        #define kernel_t kernel::Exp<Params>
        #define gp_t model::GP<Params, kernel_t, mean_t>
        #define acqui_t acqui::GP_UCB<Params, gp_t>
        CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> MY_BO;
};










struct BO_SQEXPARD_EI : CustomBOptimBase{
    public: 
        BO_SQEXPARD_EI(int seed, double sigma_sq, int k, double jitter) : CustomBOptimBase() {
            std::cout << "setup BO_SQEXPARD_EI!" << std::endl;
            std::cout << "using seed: " << seed << " sigma_sq: " << sigma_sq << " k: " << k << " jitter: " << jitter << std::endl;
            MY_BO.setSeed(seed);

            Params::kernel_squared_exp_ard::set_sigma_sq(sigma_sq);
            Params::kernel_squared_exp_ard::set_k(k);
            Params::acqui_ei::set_jitter(jitter);
        }
        ~BO_SQEXPARD_EI(){
            //std::cout << "destroying BO_SQEXPARD_EI!" << std::endl;
        }
        int getNumSamples() {return MY_BO.getNumSamples();};
        void setSeed(int seed) {MY_BO.setSeed(seed);};
        void writeGPVizFiles(std::string regionName){MY_BO.writeGPVizFiles(regionName);};
        void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun){
            MY_BO.updateModel(sample, val, afun);
        };
        Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true){
            return MY_BO.getNextPoint(sfun, afun, reset);
        };
    private:
        #undef kernel_t
        #undef acqui_t
        #undef gp_t
        #define kernel_t kernel::SquaredExpARD<Params>
        #define gp_t model::GP<Params, kernel_t, mean_t>
        #define acqui_t acqui::EI<Params, gp_t>
        CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> MY_BO;
};

struct BO_SQEXPARD_UCB : CustomBOptimBase{
    public: 
        BO_SQEXPARD_UCB(int seed, double sigma_sq, int k, double alpha) : CustomBOptimBase() {
            std::cout << "setup BO_SQEXPARD_UCB!" << std::endl;
            std::cout << "using seed: " << seed << " sigma_sq: " << sigma_sq << " k: " << k << " alpha: " << alpha << std::endl;
            MY_BO.setSeed(seed);

            Params::kernel_squared_exp_ard::set_sigma_sq(sigma_sq);
            Params::kernel_squared_exp_ard::set_k(k);
            Params::acqui_ucb::set_alpha(alpha);
        }
        ~BO_SQEXPARD_UCB(){
            //std::cout << "destroying BO_SQEXPARD_UCB!" << std::endl;
        }
        int getNumSamples() {return MY_BO.getNumSamples();};
        void setSeed(int seed) {MY_BO.setSeed(seed);};
        void writeGPVizFiles(std::string regionName){MY_BO.writeGPVizFiles(regionName);};
        void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun){
            MY_BO.updateModel(sample, val, afun);
        };
        Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true){
            return MY_BO.getNextPoint(sfun, afun, reset);
        };
    private:
        #undef kernel_t
        #undef acqui_t
        #undef gp_t
        #define kernel_t kernel::SquaredExpARD<Params>
        #define gp_t model::GP<Params, kernel_t, mean_t>
        #define acqui_t acqui::UCB<Params, gp_t>
        CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> MY_BO;
};


struct BO_SQEXPARD_GPUCB : CustomBOptimBase{
    public: 
        BO_SQEXPARD_GPUCB(int seed, double sigma_sq, int k, double delta) : CustomBOptimBase() {
            std::cout << "setup BO_SQEXPARD_GPUCB!" << std::endl;
            std::cout << "using seed: " << seed << " sigma_sq: " << sigma_sq << " k: " << k << " delta: " << delta << std::endl;
            MY_BO.setSeed(seed);

            Params::kernel_squared_exp_ard::set_sigma_sq(sigma_sq);
            Params::kernel_squared_exp_ard::set_k(k);
            //Params::acqui_gpucb::set_delta(delta);
        }
        ~BO_SQEXPARD_GPUCB(){
            //std::cout << "destroying BO_SQEXPARD_GPUCB!" << std::endl;
        }
        int getNumSamples() {return MY_BO.getNumSamples();};
        void setSeed(int seed) {MY_BO.setSeed(seed);};
        void writeGPVizFiles(std::string regionName){MY_BO.writeGPVizFiles(regionName);};
        void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun){
            MY_BO.updateModel(sample, val, afun);
        };
        Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true){
            return MY_BO.getNextPoint(sfun, afun, reset);
        };
    private:
        #undef kernel_t
        #undef acqui_t
        #undef gp_t
        #define kernel_t kernel::SquaredExpARD<Params>
        #define gp_t model::GP<Params, kernel_t, mean_t>
        #define acqui_t acqui::GP_UCB<Params, gp_t>
        CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> MY_BO;
};









struct BO_MATERN32_EI : CustomBOptimBase{
    public: 
        BO_MATERN32_EI(int seed, double sigma_sq, double l, double jitter) : CustomBOptimBase() {
            std::cout << "setup BO_MATERN32_EI!" << std::endl;
            std::cout << "using seed: " << seed << " sigma_sq: " << sigma_sq << " l: " << l << " jitter: " << jitter << std::endl;
            MY_BO.setSeed(seed);

            Params::kernel_maternthreehalves::set_sigma_sq(sigma_sq);
            Params::kernel_maternthreehalves::set_l(l);
            Params::acqui_ei::set_jitter(jitter);
        }
        ~BO_MATERN32_EI(){
            //std::cout << "destroying BO_MATERN32_EI!" << std::endl;
        }
        int getNumSamples() {return MY_BO.getNumSamples();};
        void setSeed(int seed) {MY_BO.setSeed(seed);};
        void writeGPVizFiles(std::string regionName){MY_BO.writeGPVizFiles(regionName);};
        void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun){
            MY_BO.updateModel(sample, val, afun);
        };
        Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true){
            return MY_BO.getNextPoint(sfun, afun, reset);
        };
    private:
        #undef kernel_t
        #undef acqui_t
        #undef gp_t
        #define kernel_t kernel::MaternThreeHalves<Params>
        #define gp_t model::GP<Params, kernel_t, mean_t>
        #define acqui_t acqui::EI<Params, gp_t>
        CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> MY_BO;
};

struct BO_MATERN32_UCB : CustomBOptimBase{
    public: 
        BO_MATERN32_UCB(int seed, double sigma_sq, double l, double alpha) : CustomBOptimBase() {
            std::cout << "setup BO_MATERN32_UCB!" << std::endl;
            std::cout << "using seed: " << seed << " sigma_sq: " << sigma_sq << " l: " << l << " alpha: " << alpha << std::endl;
            MY_BO.setSeed(seed);

            Params::kernel_maternthreehalves::set_sigma_sq(sigma_sq);
            Params::kernel_maternthreehalves::set_l(l);
            Params::acqui_ucb::set_alpha(alpha);
        }
        ~BO_MATERN32_UCB(){
            //std::cout << "destroying BO_MATERN32_UCB!" << std::endl;
        }
        int getNumSamples() {return MY_BO.getNumSamples();};
        void setSeed(int seed) {MY_BO.setSeed(seed);};
        void writeGPVizFiles(std::string regionName){MY_BO.writeGPVizFiles(regionName);};
        void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun){
            MY_BO.updateModel(sample, val, afun);
        };
        Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true){
            return MY_BO.getNextPoint(sfun, afun, reset);
        };
    private:
        #undef kernel_t
        #undef acqui_t
        #undef gp_t
        #define kernel_t kernel::MaternThreeHalves<Params>
        #define gp_t model::GP<Params, kernel_t, mean_t>
        #define acqui_t acqui::UCB<Params, gp_t>
        CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> MY_BO;
};

struct BO_MATERN32_GPUCB : CustomBOptimBase{
    public: 
        BO_MATERN32_GPUCB(int seed, double sigma_sq, double l, double delta) : CustomBOptimBase() {
            std::cout << "setup BO_MATERN32_GPUCB!" << std::endl;
            std::cout << "using seed: " << seed << " sigma_sq: " << sigma_sq << " l: " << l << " delta: " << delta << std::endl;
            MY_BO.setSeed(seed);

            Params::kernel_maternthreehalves::set_sigma_sq(sigma_sq);
            Params::kernel_maternthreehalves::set_l(l);
            // can't set this due to acqui_gpucb constexpr on delta...
            // might need to redefine acqui_gpucb myself...
            //Params::acqui_gpucb::set_delta(delta);
        }
        ~BO_MATERN32_GPUCB(){
            //std::cout << "destroying BO_MATERN32_GPUCB!" << std::endl;
        }
        int getNumSamples() {return MY_BO.getNumSamples();};
        void setSeed(int seed) {MY_BO.setSeed(seed);};
        void writeGPVizFiles(std::string regionName){MY_BO.writeGPVizFiles(regionName);};
        void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun){
            MY_BO.updateModel(sample, val, afun);
        };
        Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true){
            return MY_BO.getNextPoint(sfun, afun, reset);
        };
    private:
        #undef kernel_t
        #undef acqui_t
        #undef gp_t
        #define kernel_t kernel::MaternThreeHalves<Params>
        #define gp_t model::GP<Params, kernel_t, mean_t>
        #define acqui_t acqui::GP_UCB<Params, gp_t>
        CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> MY_BO;
};










struct BO_MATERN52_EI : CustomBOptimBase{
    public: 
        BO_MATERN52_EI(int seed, double sigma_sq, double l, double jitter) : CustomBOptimBase() {
            std::cout << "setup BO_MATERN52_EI!" << std::endl;
            std::cout << "using seed: " << seed << " sigma_sq: " << sigma_sq << " l: " << l << " jitter: " << jitter << std::endl;
            MY_BO.setSeed(seed);

            Params::kernel_maternfivehalves::set_sigma_sq(sigma_sq);
            Params::kernel_maternfivehalves::set_l(l);
            Params::acqui_ei::set_jitter(jitter);
        }
        ~BO_MATERN52_EI(){
            //std::cout << "destroying BO_MATERN52_EI!" << std::endl;
        }
        int getNumSamples() {return MY_BO.getNumSamples();};
        void setSeed(int seed) {MY_BO.setSeed(seed);};
        void writeGPVizFiles(std::string regionName){MY_BO.writeGPVizFiles(regionName);};
        void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun){
            MY_BO.updateModel(sample, val, afun);
        };
        Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true){
            return MY_BO.getNextPoint(sfun, afun, reset);
        };
    private:
        #undef kernel_t
        #undef acqui_t
        #undef gp_t
        #define kernel_t kernel::MaternFiveHalves<Params>
        #define gp_t model::GP<Params, kernel_t, mean_t>
        #define acqui_t acqui::EI<Params, gp_t>
        CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> MY_BO;
};

struct BO_MATERN52_UCB : CustomBOptimBase{
    public: 
        BO_MATERN52_UCB(int seed, double sigma_sq, double l, double alpha) : CustomBOptimBase() {
            std::cout << "setup BO_MATERN52_UCB!" << std::endl;
            std::cout << "using seed: " << seed << " sigma_sq: " << sigma_sq << " l: " << l << " alpha: " << alpha << std::endl;
            MY_BO.setSeed(seed);

            Params::kernel_maternfivehalves::set_sigma_sq(sigma_sq);
            Params::kernel_maternfivehalves::set_l(l);
            Params::acqui_ucb::set_alpha(alpha);
        }
        ~BO_MATERN52_UCB(){
            //std::cout << "destroying BO_MATERN52_UCB!" << std::endl;
        }
        int getNumSamples() {return MY_BO.getNumSamples();};
        void setSeed(int seed) {MY_BO.setSeed(seed);};
        void writeGPVizFiles(std::string regionName){MY_BO.writeGPVizFiles(regionName);};
        void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun){
            MY_BO.updateModel(sample, val, afun);
        };
        Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true){
            return MY_BO.getNextPoint(sfun, afun, reset);
        };
    private:
        #undef kernel_t
        #undef acqui_t
        #undef gp_t
        #define kernel_t kernel::MaternFiveHalves<Params>
        #define gp_t model::GP<Params, kernel_t, mean_t>
        #define acqui_t acqui::UCB<Params, gp_t>
        CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> MY_BO;
};

struct BO_MATERN52_GPUCB : CustomBOptimBase{
    public: 
        BO_MATERN52_GPUCB(int seed, double sigma_sq, double l, double delta) : CustomBOptimBase() {
            std::cout << "setup BO_MATERN52_GPUCB!" << std::endl;
            std::cout << "using seed: " << seed << " sigma_sq: " << sigma_sq << " l: " << l << " delta: " << delta << std::endl;
            MY_BO.setSeed(seed);

            Params::kernel_maternfivehalves::set_sigma_sq(sigma_sq);
            Params::kernel_maternfivehalves::set_l(l);
            // can't set this due to acqui_gpucb constexpr on delta...
            // might need to redefine acqui_gpucb myself...
            //Params::acqui_gpucb::set_delta(delta);
        }
        ~BO_MATERN52_GPUCB(){
            //std::cout << "destroying BO_MATERN52_GPUCB!" << std::endl;
        }
        int getNumSamples() {return MY_BO.getNumSamples();};
        void setSeed(int seed) {MY_BO.setSeed(seed);};
        void writeGPVizFiles(std::string regionName){MY_BO.writeGPVizFiles(regionName);};
        void updateModel(Eigen::VectorXd& sample, Eigen::VectorXd& val, const FirstElem& afun){
            MY_BO.updateModel(sample, val, afun);
        };
        Eigen::VectorXd getNextPoint(const eval_func& sfun, const FirstElem& afun, bool reset=true){
            return MY_BO.getNextPoint(sfun, afun, reset);
        };
    private:
        #undef kernel_t
        #undef acqui_t
        #undef gp_t
        #define kernel_t kernel::MaternFiveHalves<Params>
        #define gp_t model::GP<Params, kernel_t, mean_t>
        #define acqui_t acqui::GP_UCB<Params, gp_t>
        CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> MY_BO;
};













namespace apollo
{
class BayesianOptim : public PolicyModel
{
public:
  BayesianOptim(int num_policies, int num_features, 
                std::string &kernel, 
                std::string &acqui,
                double acqui_hyper,
                double sigma_sq,
                double l,
                int k,
                double whiteKernel,
                int seed,
                int max_samples,
                std::string regionName);
  ~BayesianOptim();

  int getIndex(std::vector<float> &features);
  void load(const std::string &filename){};
  void store(const std::string &filename){};
  bool isTrainable() { return ((max_samples == -1) || (num_samples < max_samples)); };
  void train(Apollo::Dataset &dataset);

private:
  int policy_choice;
  int num_features;
  int num_samples;
  int best_policy;
  double best_policy_xtime;
  int max_samples;
  std::string regionName;

  //int first_execution;

  //template<class A, class B, class C, class D>
  //CustomBOptimizer<A,B,C,D> boptimizer;
  //std::shared_ptr<CustomBOptimizer> boptimizer;
  //boost::any boptimizer;
  //CustomBOptimizer boptimizer;
  CustomBOptimBase * boptimizer;

  //void setBOptimizer(CustomBOptimizer<A,B,C,D>* optim);

  // This is really ugly, but I just want something to work for now :(
  // We prefer having a working prototype, then can clean up later

  #undef kernel_t
  #undef acqui_t
  #undef gp_t
  #define kernel_t kernel::SquaredExpARD<Params>
  #define gp_t model::GP<Params, kernel_t, mean_t>
  #define acqui_t acqui::EI<Params, gp_t>
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> bo_SqExpARD_EI;
  //CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> boptimizer;

  #undef acqui_t
  #define acqui_t acqui::UCB<Params, gp_t>
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> bo_SqExpARD_UCB;

  #undef acqui_t
  #define acqui_t acqui::GP_UCB<Params, gp_t>
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> bo_SqExpARD_GPUCB;


  #undef kernel_t
  #undef acqui_t
  #undef gp_t
  #define kernel_t kernel::MaternThreeHalves<Params>
  #define gp_t model::GP<Params, kernel_t, mean_t>
  #define acqui_t acqui::EI<Params, gp_t>
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> bo_MatThreeHalf_EI;

  #undef acqui_t
  #define acqui_t acqui::UCB<Params, gp_t>
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> bo_MatThreeHalf_UCB;

  #undef acqui_t
  #define acqui_t acqui::GP_UCB<Params, gp_t>
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> bo_MatThreeHalf_GPUCB;


  #undef kernel_t
  #undef acqui_t
  #undef gp_t
  #define kernel_t kernel::MaternFiveHalves<Params>
  #define gp_t model::GP<Params, kernel_t, mean_t>
  #define acqui_t acqui::EI<Params, gp_t>
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> bo_MatFiveHalf_EI;

  #undef acqui_t
  #define acqui_t acqui::UCB<Params, gp_t>
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> bo_MatFiveHalf_UCB;

  #undef acqui_t
  #define acqui_t acqui::GP_UCB<Params, gp_t>
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> bo_MatFiveHalf_GPUCB;


  #undef kernel_t
  #undef acqui_t
  #undef gp_t
  #define kernel_t kernel::Exp<Params>
  #define gp_t model::GP<Params, kernel_t, mean_t>
  #define acqui_t acqui::EI<Params, gp_t>
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> bo_SqExp_EI;

  #undef acqui_t
  #define acqui_t acqui::UCB<Params, gp_t>
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> bo_SqExp_UCB;

  #undef acqui_t
  #define acqui_t acqui::GP_UCB<Params, gp_t>
  CustomBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, initfun<init_t>> bo_SqExp_GPUCB;


  //CustomBOptimizer<Params, modelfun< model::GP<Params, kernel_t, mean_t>>, acquifun<acqui::EI<Params, model::GP<Params, kernel_t, mean_t>>> >
  Eigen::VectorXd last_point;

};  

}  // end namespace apollo.

#endif
