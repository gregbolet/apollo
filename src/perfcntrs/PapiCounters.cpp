
#include "apollo/perfcntrs/PapiCounters.h"


// Initialize PAPI and keep the perfcntr names we want to track
PapiCounters::PapiCounters(int isMultiplexed, 
                                int numEvents, 
                                std::string* eventNames)
      :isMultiplexed(isMultiplexed), 
       numEvents(numEvents),
       event_names_to_track(eventNames){

    
    // Let's set up PAPI from the main thread
	int retval;

	// Check correct versioning
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		fprintf(stderr, "PAPI_library_init error: code [%d] and version number [%d]\n", retval, PAPI_VER_CURRENT );
	}

    if(isMultiplexed){
	    // Initialize the multiplexing
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
        char* event = &((event_names_to_track[i])[0]);

		if((retval = PAPI_event_name_to_code(event, &eventCode)) != PAPI_OK){
			fprintf(stderr, "Event code for [%s] does not exist! errcode:[%d]\n", event, retval);
		}

        this->events_to_track[i] = eventCode;
	}

    return;
}

// Nothing to free... yet
PapiCounters::~PapiCounters(){

    // Destroy the events_to_track
    free(this->events_to_track);

    // Free the counter arrays
    for(std::vector<long long*>::iterator i = std::begin(this->all_cntr_values);
        i != std::end(this->all_cntr_values); ++i){
            free(*i);
    }

    return;
}

void PapiCounters::startThread(){

    // Keep track of the eventset identifier for this thread
 	int EventSet = PAPI_NULL;
 	int retval;

	// Register this thread with PAPI
	if ( ( retval = PAPI_register_thread() ) != PAPI_OK ) {
		fprintf(stderr, "PAPI thread registration error!\n");
	}

	/* Create the Event Set for this thread */
	if (PAPI_create_eventset(&EventSet) != PAPI_OK){
		fprintf(stderr, "PAPI eventset creation error!\n");
	}

	/* In Component PAPI, EventSets must be assigned a component index
	   before you can fiddle with their internals. 0 is always the cpu component */
	retval = PAPI_assign_eventset_component( EventSet, 0 );
	if ( retval != PAPI_OK ) {
		fprintf(stderr, "PAPI assign eventset component error!\n");
	}

    if(this->isMultiplexed){
	    /* Convert our EventSet to a multiplexed EventSet*/
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
        fprintf(stderr, "PAPI add events failed!\n");
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
	if (PAPI_reset(EventSet) != PAPI_OK){
		fprintf(stderr, "Could NOT reset eventset!\n");
	}

	/* Start counting events in the Event Set */
	if (PAPI_start(EventSet) != PAPI_OK){
		fprintf(stderr, "Could NOT start eventset counting!\n");
	}
}

void PapiCounters::stopThread(){

    int threadId = omp_get_thread_num();
    int EventSet = this->thread_id_to_eventset[threadId];
    long long* cntr_vals = this->thread_id_to_cntr_ptr[threadId];
    int retval;

	/* stop counting events in the Event Set */
	// Store the resulting values into our counter values array
	if ( ( retval = PAPI_stop( EventSet, cntr_vals ) ) != PAPI_OK){
		fprintf(stderr, "Could NOT stop eventset counting!\n");
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
	if ( ( retval = PAPI_unregister_thread(  ) ) != PAPI_OK ) {
		fprintf(stderr, "PAPI could not unregister thread!\n");
	}
}