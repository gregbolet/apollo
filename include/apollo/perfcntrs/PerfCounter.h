
#ifndef APOLLO_PERF_CNTR_H
#define APOLLO_PERF_CNTR_H

#include <vector>

// This is an abstract class to help structure the
// performance counter interfaces to PAPI and Caliper
// that we plan on having
class PerfCounter{

    public:
        PerfCounter() {};
        ~PerfCounter() {};

        virtual void startThread() {};
        virtual void stopThread() {};        
        virtual void clearAllCntrValues() {};        
        virtual std::vector<float> getSummaryStats() {};        

};

#endif