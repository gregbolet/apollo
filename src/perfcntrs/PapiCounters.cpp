#include "apollo/perfcntrs/PapiCounters.h"

Apollo::PapiCounters::PapiCounters(Apollo* apollo)
		: apollo(apollo) { return; }

Apollo::PapiCounters::~PapiCounters() { return; }

void Apollo::PapiCounters::startThread()
{

	int threadId = omp_get_thread_num();
	int retval;

	// Here we need to check if the counters for this thread have
	// been initialized. If not, we set them up. If so, we skip the setup.
	// If the EventSet for this thread hasn't been started, start it
	if(!this->apollo->EventSet_is_started[threadId]){
		// Keep track of the eventset identifier for this thread
		int EventSet = PAPI_NULL;
	
		// Create the Event Set for this thread
		if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK){
			fprintf(stderr, "PAPI eventset creation error! [%d]\n", retval);
		}
	
		// In Component PAPI, EventSets must be assigned a component index
		// before you can fiddle with their internals. 0 is always the cpu component
		if ((retval = PAPI_assign_eventset_component(EventSet, 0)) != PAPI_OK){
			fprintf(stderr, "PAPI assign eventset component error! [%d]\n", retval);
		}
	
		if (this->apollo->is_multiplexed)
		{
			// Convert our EventSet to a multiplexed EventSet
			if ((retval = PAPI_set_multiplex(EventSet)) != PAPI_OK) {
				if (retval == PAPI_ENOSUPP) {
					fprintf(stderr, "PAPI Multiplexing not supported!\n");
				}
				fprintf(stderr, "PAPI error setting up multiplexing!\n");
			}
		}
	
		// Add events to the eventset
		retval = PAPI_add_events(EventSet, this->apollo->events_to_track, this->apollo->num_events);
		if (retval != PAPI_OK) {
			fprintf(stderr, "PAPI add events failed! [%d]\n", retval);
		}
	
		// Start counting events in the Event Set implicitly zeros out counters
		if ((retval = PAPI_start(EventSet)) != PAPI_OK) {
			fprintf(stderr, "Could NOT start eventset counting! [%d]\n", retval);
		}

		this->apollo->EventSets[threadId] = EventSet;
		this->apollo->EventSet_is_started[threadId] = 1;
	}
	else{
		//printf("In startThread()\n");
		int EventSet = this->apollo->EventSets[threadId];

		// Reset the counters in the EventSet to 0
		if ((retval = PAPI_reset(EventSet)) != PAPI_OK) {
			fprintf(stderr, "Could NOT reset eventset counting! [%d]\n", retval);
		}
	}

}

void Apollo::PapiCounters::stopThread()
{
	int threadId = omp_get_thread_num();
	int EventSet = this->apollo->EventSets[threadId];
	long long *cntr_vals = this->apollo->cntr_values + (threadId * this->apollo->num_events);
	int retval;

	// count events in the Event Set
	// Store the resulting values into our counter values array
	if ((retval = PAPI_read(EventSet, cntr_vals)) != PAPI_OK) {
		fprintf(stderr, "Could NOT do eventset counting! [%d]\n", retval);
	}

	this->apollo->EventSet_just_used[threadId] = 1;
}

std::vector<float> Apollo::PapiCounters::getSummaryStats()
{
	// Calculate the min, max, mean of each perf counter
	// return a vector of the values, where every 3 values in the array corresponds to a perf counter
	std::vector<float> toRet;

	// Go through each event
	for (int j = 0; j < this->apollo->num_events; ++j)
	{

		long long sum = 0;
		
		// Go through all the threads that were just run
		for(int threadId = 0; threadId < this->apollo->num_eventsets; ++threadId){

			// If this thread was run, get its data for this counter
			int was_just_run = this->apollo->EventSet_just_used[threadId];

			// If we collected values, add them to the total sum, then mark
			// the eventset as not just used
			if(was_just_run){
				long long* vals = this->apollo->cntr_values + (threadId * this->apollo->num_events);

				sum += vals[j];
			}
		}

		// We want to take the sum of the work for the counters
		toRet.push_back((float) sum);
	}

	// Set all the just_used properties to 0
	for(int threadId = 0; threadId < this->apollo->num_eventsets; ++threadId){
		int was_just_run = this->apollo->EventSet_just_used[threadId];

		if(was_just_run){
			this->apollo->EventSet_just_used[threadId] = 0;
		}
	}

	// We assume we will have the user supply 3 perfcntrs
	// < PAPI_DP_OPS, PAPI_TOT_INS, PAPI_L3_TCM >
	// We simply change the last element to be the 'memory pressure'
	// indicated by (PAPI_L3_TCM*1e6)/(PAPI_TOT_INS)
	// intuitively: total cache misses per million instructions
	// so we have: < PAPI_DP_OPS, PAPI_TOT_INS, (PAPI_L3_TCM*1e6)/(PAPI_TOT_INS) > 

	//float PAPI_L3_TCM_CNTR = toRet.back();
	//toRet.pop_back();
	//PAPI_L3_TCM_CNTR = PAPI_L3_TCM_CNTR / toRet.back();
	//toRet.push_back(PAPI_L3_TCM_CNTR*1000000);

	return toRet;
}