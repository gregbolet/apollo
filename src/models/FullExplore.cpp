
#include <string>
#include <iostream>

#include "apollo/Config.h"
#include "apollo/models/FullExplore.h"

int
FullExplore::getIndex(std::vector<float> &features)
{
    int choice;

    // If we don't find the given feature, add it in
    if( policies.find( features ) == policies.end() ) {
        policies[ features ] = 0;
    }
    else {
        // made a full circle, stop
        if( policies[ features ] == 0 )
            return 0;
    }

    // Get the current policy choice (to return)
    choice = policies[ features ];

    // Update the policy choice to the next policy
    policies[ features ] = ( policies[ features ] + 1) % policy_count;

    return choice;
}

FullExplore::FullExplore(int num_policies)
    : PolicyModel(num_policies, "FullExplore", true)
{
    int rank = 0;
    char *slurm_procid = getenv("SLURM_PROCID");

    if (slurm_procid != NULL) {
       rank = atoi(slurm_procid);
    } else {
       rank = 0;
    };

    return;
}

//
// ----------
//
// BELOW: Boilerplate code to manage instances of this model:
//

FullExplore::~FullExplore()
{
    return;
}
