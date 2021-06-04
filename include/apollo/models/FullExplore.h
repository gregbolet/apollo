#ifndef APOLLO_MODELS_FULLEXPLORE_H
#define APOLLO_MODELS_FULLEXPLORE_H

#include <string>
#include <vector>
#include <map>

#include "apollo/PolicyModel.h"

class FullExplore : public PolicyModel {
    public:
        FullExplore(int num_policies);
        ~FullExplore();

        int  getIndex(std::vector<float> &features);
        void store(const std::string &filename) {};

    private:
        std::map< std::vector<float>, int > policies;

}; //end: FullExplore (class)


#endif
