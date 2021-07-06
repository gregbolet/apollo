
#include "apollo/perfcntrs/PapiCounters.h"


// Initialize PAPI and keep the perfcntr names we want to track
Apollo::PapiCounters::PapiCounters(int isMultiplexed, 
                           std::vector<std::string> eventNames)
      :isMultiplexed(isMultiplexed),
       numEvents(eventNames.size()),
       //runWithCounters(1),
       event_names_to_track(eventNames){

	//this->numEvents = eventNames.size();

    // Let's set up PAPI from the main thread
	int retval;

	// Check correct versioning
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		fprintf(stderr, "PAPI_library_init error: code [%d] and version number [%d]\n", retval, PAPI_VER_CURRENT );
	}

    if(isMultiplexed){

	    retval = PAPI_multiplex_init();
	    if ( retval == PAPI_ENOSUPP) {
	        fprintf(stderr, "Multiplexing not supported!\n");
	    }
	    else if ( retval != PAPI_OK ) {
		    fprintf(stderr, "PAPI multiplexing error: code [%d]\n", retval );
	    }
    }

	// Initialize the threading support
	retval = PAPI_thread_init( ( unsigned long ( * )( void ) ) (omp_get_thread_num) );
	if ( retval == PAPI_ECMP) {
	   fprintf(stderr, "PAPI thread init error: code [%d] PAPI_ECMP!\n", retval);
	}
	else if ( retval != PAPI_OK ) {
	   fprintf(stderr, "PAPI thread init error: code [%d]!\n", retval);
	}

    // Convert all the event string names to their event codes
    // This will make it easier to add the events to each eventset
    // we create

    this->events_to_track = (int *) malloc(sizeof(int) * numEvents);

	// Add each of the events by getting their event code from the string
	int eventCode = PAPI_NULL;
	for(int i = 0; i < numEvents; i++){

        // get address of the first element of the i-th event string name
        //char* event = &((event_names_to_track[i])[0]);
        const char* event = event_names_to_track[i].c_str(); 

		if((retval = PAPI_event_name_to_code(event, &eventCode)) != PAPI_OK){
			fprintf(stderr, "Event code for [%s] does not exist! errcode:[%d]\n", event, retval);
		}

        this->events_to_track[i] = eventCode;
	}

    return;
}

Apollo::PapiCounters::~PapiCounters(){

    // Destroy the events_to_track
    free(this->events_to_track);

    //printf("Finished freeing events to track!\n");
	//printf("Collected: %d measurements\n", this->all_cntr_values.size());

	this->clearAllCntrValues();

    // Let's avoid PAPI_shutdown for now
    //PAPI_shutdown();

    return;
}


void Apollo::PapiCounters::startThread(){

    //if(!this->runWithCounters){
	    //return;
    //}

    //printf("In startThread()\n");

        // Keep track of the eventset identifier for this thread
     int EventSet = PAPI_NULL;
     int retval;

    // Register this thread with PAPI
    //if ( ( retval = PAPI_register_thread() ) != PAPI_OK ) {
	    //fprintf(stderr, "PAPI thread registration error!\n");
    //}

    // Create the Event Set for this thread
    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK){
	    fprintf(stderr, "PAPI eventset creation error! [%d]\n", retval);
    }

    // In Component PAPI, EventSets must be assigned a component index
    // before you can fiddle with their internals. 0 is always the cpu component
    retval = PAPI_assign_eventset_component( EventSet, 0 );
    if ( retval != PAPI_OK ) {
	    fprintf(stderr, "PAPI assign eventset component error! [%d]\n", retval);
    }

    if(this->isMultiplexed){
	    // Convert our EventSet to a multiplexed EventSet
	    if ( ( retval = PAPI_set_multiplex( EventSet ) ) != PAPI_OK ) {
		    if ( retval == PAPI_ENOSUPP) {
			    fprintf(stderr, "PAPI Multiplexing not supported!\n");
	   	    }
		    fprintf(stderr, "PAPI error setting up multiplexing!\n");
	    }
    }

    // Add events to the eventset
    retval = PAPI_add_events(EventSet, this->events_to_track, this->numEvents);
    if(retval != PAPI_OK){
        fprintf(stderr, "PAPI add events failed! [%d]\n", retval);
    }

    // Map this threadId to the eventset
    int threadId = omp_get_thread_num();

    // Allocate our array of counter values
    long long* cntr_vals = (long long *)malloc(sizeof(long long)*this->numEvents);

    // Lock this little scope to update the maps
    {
        std::lock_guard<util::spinlock> g(thread_lock);

        this->thread_id_to_eventset[threadId] = EventSet;
        this->thread_id_to_cntr_ptr[threadId] = cntr_vals;
    }

	// Zero-out all the counters in the eventset
	//if (PAPI_reset(EventSet) != PAPI_OK){
		//fprintf(stderr, "Could NOT reset eventset!\n");
	//}

	// Start counting events in the Event Set implicitly zeros out counters
	if (PAPI_start(EventSet) != PAPI_OK){
		fprintf(stderr, "Could NOT start eventset counting!\n");
	}
}

void Apollo::PapiCounters::stopThread(){

    //if(!this->runWithCounters){
	    //return;
    //}

    //printf("In stopThread()\n");
    //printf("My map size: %d\n", this->thread_id_to_eventset.size());
    //printf("My other map size: %d\n", this->thread_id_to_cntr_ptr.size());

    int threadId = omp_get_thread_num();
    int EventSet = this->thread_id_to_eventset[threadId];
    long long* cntr_vals = this->thread_id_to_cntr_ptr[threadId];
    int retval;

    // stop counting events in the Event Set
    // Store the resulting values into our counter values array
    if ( ( retval = PAPI_stop( EventSet, cntr_vals ) ) != PAPI_OK){
	    fprintf(stderr, "Could NOT stop eventset counting!\n");
    }

    // Store the pointer to that memory into our list
        // Lock this little scope to store the counter values 
        {
            std::lock_guard<util::spinlock> g(thread_lock);
	    this->all_cntr_values.push_back(cntr_vals);
    }

    // Remove all events from the eventset
    if ( ( retval = PAPI_cleanup_eventset( EventSet ) ) != PAPI_OK ) {
	    fprintf(stderr, "PAPI could not cleanup eventset!\n");
    }

    // Deallocate the empty eventset from memory
    if ( ( retval = PAPI_destroy_eventset( &EventSet) ) != PAPI_OK ) {
	    fprintf(stderr, "PAPI could not destroy eventset!\n");
    }

    // Shut down this thread and free the thread ID
    //if ( ( retval = PAPI_unregister_thread(  ) ) != PAPI_OK ) {
	    //fprintf(stderr, "PAPI could not unregister thread!\n");
    //}
}

void Apollo::PapiCounters::clearAllCntrValues(){

    // Free the counter arrays
    for(std::vector<long long*>::iterator i = std::begin(this->all_cntr_values);
        i != std::end(this->all_cntr_values); ++i){

			//print the values before we drop them
			//for(int j = 0; j < this->numEvents; ++j){
				//printf("%lld \t", (*i)[j]);
			//}
			//printf("\n");

            free(*i);
    }

	this->all_cntr_values.clear();
}

std::vector<float> Apollo::PapiCounters::getSummaryStats(){
	// Calculate the min, max, mean of each perf counter
	// return a vector of the values, where every 3 values in the array corresponds to a perf counter
	std::vector<float> toRet;

	long long min, max, mean, sum, val;
	int i, j;

	// Go through each event
	for(j = 0; j < this->numEvents; ++j){

		// Default to the first thread's values
		min = max = mean = this->all_cntr_values[0][j];

		// Loop to calculate the min, max, mean for this counter
		for(i=1; i < this->all_cntr_values.size(); ++i){
			val = this->all_cntr_values[i][j];

			if(val >= max){ max = val; }
			if(val <= min){ min = val; }
			mean += val;
		}

		sum = mean;
		mean = mean/i;

		//toRet.push_back((float) min);
		//toRet.push_back((float) max);
		//toRet.push_back((float) mean);
		toRet.push_back((float) sum);

		//printf("min: %lld, max: %lld, mean: %lld, sum: %lld\n", min, max, mean, sum);
	}

	return toRet;
}

