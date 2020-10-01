#ifndef APOLLO_MODELS_RANDOM_H
#define APOLLO_MODELS_RANDOM_H

#include <random>
#include <memory>

#include "apollo/PolicyModel.h"

namespace Apollo {

class Random : public PolicyModel {
    public:
        Random(int num_policies);
        ~Random();

        //
        int  getIndex(std::vector<float> &features);
        void store(const std::string &filename) {};

    private:
        std::random_device random_dev;
        std::mt19937 random_gen;
        std::uniform_int_distribution<> random_dist;
}; //end: Apollo::Model::Random (class)

}; //end: Apollo (namespace)

#endif
