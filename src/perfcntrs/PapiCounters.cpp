#include "apollo/perfcntrs/PapiCounters.h"

// Initialize PAPI and keep the perfcntr names we want to track
Apollo::PapiCounters::PapiCounters(int isMultiplexed,
																	 std::vector<std::string> eventNames)
		: isMultiplexed(isMultiplexed),
			numEvents(eventNames.size()),
			//runWithCounters(1),
			event_names_to_track(eventNames)
{

	//this->numEvents = eventNames.size();

	// Let's set up PAPI from the main thread
	int retval;

	// Check correct versioning
	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT)
	{
		fprintf(stderr, "PAPI_library_init error: code [%d] and version number [%d]\n", retval, PAPI_VER_CURRENT);
	}

	if (isMultiplexed)
	{

		retval = PAPI_multiplex_init();
		if (retval == PAPI_ENOSUPP)
		{
			fprintf(stderr, "Multiplexing not supported!\n");
		}
		else if (retval != PAPI_OK)
		{
			fprintf(stderr, "PAPI multiplexing error: code [%d]\n", retval);
		}
	}

	// Initialize the threading support
	retval = PAPI_thread_init((unsigned long (*)(void))(omp_get_thread_num));
	if (retval == PAPI_ECMP)
	{
		fprintf(stderr, "PAPI thread init error: code [%d] PAPI_ECMP!\n", retval);
	}
	else if (retval != PAPI_OK)
	{
		fprintf(stderr, "PAPI thread init error: code [%d]!\n", retval);
	}

	// Convert all the event string names to their event codes
	// This will make it easier to add the events to each eventset
	// we create

	this->events_to_track = (int *)malloc(sizeof(int) * numEvents);

	// Add each of the events by getting their event code from the string
	int eventCode = PAPI_NULL;
	for (int i = 0; i < numEvents; i++)
	{

		// get address of the first element of the i-th event string name
		//char* event = &((event_names_to_track[i])[0]);
		const char *event = event_names_to_track[i].c_str();

		if ((retval = PAPI_event_name_to_code(event, &eventCode)) != PAPI_OK)
		{
			fprintf(stderr, "Event code for [%s] does not exist! errcode:[%d]\n", event, retval);
		}

		this->events_to_track[i] = eventCode;
	}

	return;
}

Apollo::PapiCounters::~PapiCounters()
{

	int retval;
	std::map<int, int>::iterator it;

	// Destruct each of the thread's EventSets
	for(it = thread_id_to_eventset.begin();  it != thread_id_to_eventset.end(); it++){

		int EventSet = it->second;

		// Remove all events from the eventset
		if ((retval = PAPI_cleanup_eventset(EventSet)) != PAPI_OK)
		{
			fprintf(stderr, "PAPI could not cleanup eventset!\n");
		}

		// Deallocate the empty eventset from memory
		if ((retval = PAPI_destroy_eventset(&EventSet)) != PAPI_OK)
		{
			fprintf(stderr, "PAPI could not destroy eventset!\n");
		}
	}

	// Destroy the events_to_track
	free(this->events_to_track);

	std::map<int, long long *>::iterator cntr_it;
	// need to free the counter value holders
	for(cntr_it = this->thread_id_to_cntr_ptr.begin(); cntr_it != this->thread_id_to_cntr_ptr.end(); ++cntr_it){
		free(cntr_it->second);
	}

	//printf("Finished freeing events to track!\n");
	//printf("Collected: %d measurements\n", this->all_cntr_values.size());

	// Technically not needed anymore, we're not constantly
	// mallocing and freeing pointers to store the CNTR values
	// this->clearAllCntrValues();

	// Let's avoid PAPI_shutdown for now
	//PAPI_shutdown();

	return;
}

void Apollo::PapiCounters::startThread()
{

	int threadId = omp_get_thread_num();

	// Here we need to check if the counters for this thread have
	// been initialized. If not, we set them up. If so, we skip the setup.
	std::map<int, int>::iterator it = thread_id_to_eventset.find(threadId);

	// If it's not in the mapping, add it to the mapping
	if(it == thread_id_to_eventset.end()){
		// Keep track of the eventset identifier for this thread
		int EventSet = PAPI_NULL;
		int retval;
	
		// Register this thread with PAPI
		//if ( ( retval = PAPI_register_thread() ) != PAPI_OK ) {
		//fprintf(stderr, "PAPI thread registration error!\n");
		//}
	
		// Create the Event Set for this thread
		retval = PAPI_create_eventset(&EventSet);
		if (retval != PAPI_OK)
		{
			fprintf(stderr, "PAPI eventset creation error! [%d]\n", retval);
		}
	
		// In Component PAPI, EventSets must be assigned a component index
		// before you can fiddle with their internals. 0 is always the cpu component
		retval = PAPI_assign_eventset_component(EventSet, 0);
		if (retval != PAPI_OK)
		{
			fprintf(stderr, "PAPI assign eventset component error! [%d]\n", retval);
		}
	
		if (this->isMultiplexed)
		{
			// Convert our EventSet to a multiplexed EventSet
			if ((retval = PAPI_set_multiplex(EventSet)) != PAPI_OK)
			{
				if (retval == PAPI_ENOSUPP)
				{
					fprintf(stderr, "PAPI Multiplexing not supported!\n");
				}
				fprintf(stderr, "PAPI error setting up multiplexing!\n");
			}
		}
	
		// Add events to the eventset
		retval = PAPI_add_events(EventSet, this->events_to_track, this->numEvents);
		if (retval != PAPI_OK)
		{
			fprintf(stderr, "PAPI add events failed! [%d]\n", retval);
		}
	
		// Map this threadId to the eventset
		int threadId = omp_get_thread_num();
	
		// Allocate our array of counter values
		long long *cntr_vals = (long long *)malloc(sizeof(long long) * this->numEvents);
	
		// Lock this little scope to update the maps
		{
			std::lock_guard<util::spinlock> g(thread_lock);
	
			this->thread_id_to_eventset[threadId] = EventSet;
			this->thread_id_to_cntr_ptr[threadId] = cntr_vals;
			this->thread_id_just_run[threadId] = 0;
		}

	}

	//printf("In startThread()\n");
	int EventSet = this->thread_id_to_eventset[threadId];

	// Start counting events in the Event Set implicitly zeros out counters
	if (PAPI_start(EventSet) != PAPI_OK)
	{
		fprintf(stderr, "Could NOT start eventset counting!\n");
	}

	//fprintf(stderr, "STARTED THREAD COUNTING!\n");
}

void Apollo::PapiCounters::stopThread()
{

	int threadId = omp_get_thread_num();
	int EventSet = this->thread_id_to_eventset[threadId];
	long long *cntr_vals = this->thread_id_to_cntr_ptr[threadId];
	int retval;

	// stop counting events in the Event Set
	// Store the resulting values into our counter values array
	if ((retval = PAPI_stop(EventSet, cntr_vals)) != PAPI_OK)
	{
		fprintf(stderr, "Could NOT stop eventset counting!\n");
	}

	// Now mark my thread ID as just having run
	{
		std::lock_guard<util::spinlock> g(thread_lock);
		this->thread_id_just_run[threadId] = 1;
	}

	// Store the pointer to that memory into our list
	// Lock this little scope to store the counter values
	//{
		//std::lock_guard<util::spinlock> g(thread_lock);
		//this->all_cntr_values.push_back(cntr_vals);
	//}

}

//void Apollo::PapiCounters::clearAllCntrValues()
//{
//
//	// Free the counter arrays
//	for (std::vector<long long *>::iterator i = std::begin(this->all_cntr_values);
//			 i != std::end(this->all_cntr_values); ++i)
//	{
//		free(*i);
//	}
//
//	this->all_cntr_values.clear();
//}

std::vector<float> Apollo::PapiCounters::getSummaryStats()
{
	// Calculate the min, max, mean of each perf counter
	// return a vector of the values, where every 3 values in the array corresponds to a perf counter
	std::vector<float> toRet;

	long long sum;
	int j; 
	std::map<int, int>::iterator it;

	// Go through each event
	for (j = 0; j < this->numEvents; ++j)
	{

		sum = 0;
		
		// Go through all the threads that were just run
		for(it = this->thread_id_just_run.begin(); it != this->thread_id_just_run.end(); it++){

			// If this thread was run, get its data for this counter
			int threadId = it->first;
			int was_just_run = it->second;

			if(was_just_run){
				long long* thread_cntrs = this->thread_id_to_cntr_ptr[threadId];

				// Add this data to the summary stats
				sum += thread_cntrs[j];
			}
		}

		//mean = sum / num_run;
		// We want to take the sum of the work for the counters
		toRet.push_back((float) sum);

		//printf("min: %lld, max: %lld, mean: %lld, sum: %lld\n", min, max, mean, sum);
	}

	// Now we just need to mark all the just run threads to 0
	for(it = this->thread_id_just_run.begin(); it != this->thread_id_just_run.end(); it++){
		it->second = 0;
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