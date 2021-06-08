
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

// This class assumes OMP is being used for threading
// We will change this use case in the future once we get something working
// For now, when setting up threads we don't do error handling
class PapiCounters : public PerfCounter{

    public:
        PapiCounters(int isMultiplexed, int numEvents, std::string* eventNames);
        //PapiCounters();
        ~PapiCounters();

        void startThread() override;
        void stopThread() override;        
        void clearAllCntrValues() override; 
        std::vector<float> getSummaryStats() override;
        
    private:
        int isMultiplexed;
        int numEvents;

        // Keep our event names in here
        std::string* event_names_to_track;

        
        // Shared spinlock for setting up threads
        mutable util::spinlock thread_lock;

        // Map the threadID to the counter value pointers
        std::vector<long long*> all_cntr_values;

        // Mapping of threadID to eventSet
        std::map<int, int> thread_id_to_eventset;

        // Mapping of threadID to cntr values 
        std::map<int, long long*> thread_id_to_cntr_ptr;

        // At initialization, convert the string 
        // event names to their integer codes
        int* events_to_track;
        
};

#endif