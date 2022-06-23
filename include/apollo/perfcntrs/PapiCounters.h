
#ifndef APOLLO_PAPI_CNTRS_H
#define APOLLO_PAPI_CNTRS_H

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <omp.h>
#include <mutex>
#include <vector>
#include <map>
#include "papi.h"
#include "apollo/perfcntrs/PerfCounter.h"
#include "util/spinlock.h"
#include "apollo/Apollo.h"

// This class assumes OMP is being used for threading
// We will change this use case in the future once we get something working
// For now, when setting up threads we don't do error handling
class Apollo::PapiCounters : public Apollo::PerfCounter{
    
    public:
        PapiCounters(Apollo* apollo);
        ~PapiCounters();

        void startThread() override;
        void stopThread() override;        
        //void clearAllCntrValues() override; 
        std::vector<float> getSummaryStats() override;
        
    private:
        friend class Apollo::Region;
        Apollo* apollo;

};

#endif