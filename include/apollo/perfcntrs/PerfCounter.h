
#ifndef APOLLO_PERF_CNTR_H
#define APOLLO_PERF_CNTR_H

#include <vector>
#include "apollo/Apollo.h"

// This is an abstract class to help structure the
// performance counter interfaces to PAPI and Caliper
// that we plan on having
class Apollo::PerfCounter {

    public:
        PerfCounter() {};
        ~PerfCounter() {};

        virtual void startThread() {};
        virtual void stopThread() {};        
        virtual void clearAllCntrValues() {};        
        virtual std::vector<float> getSummaryStats() {return std::vector<float>();};        

};

#endif