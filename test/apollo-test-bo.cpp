// Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC and other
// Apollo project developers. Produced at the Lawrence Livermore National
// Laboratory. See the top-level LICENSE file for details.
// SPDX-License-Identifier: MIT

#include "apollo/Apollo.h"

#include <chrono>
#include <iostream>

#include "apollo/Region.h"

#define NUM_FEATURES 1
#define NUM_POLICIES 50
#define NUM_SAMPLES 30
#define REPS 1
#define DELAY 0.001


static void delay_loop(const double delay)
{
  auto start = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  // Artificial delay when feature mismatches policy.
  while (elapsed.count() < delay) {
    end = std::chrono::steady_clock::now();
    elapsed = end - start;
  }
}

int main()
{
  std::cout << "=== Testing Apollo correctness\n";

  
  //static Apollo *apollo = Apollo::instance(); 
  //static Apollo::Region r(NUM_FEATURES, "test-region", NUM_POLICIES, 1);
  static Apollo::Region *r;
  if(!r){
    r = new Apollo::Region(NUM_FEATURES, "test-region", NUM_POLICIES, 1);
  }
  //Apollo *apollo = Apollo::instance();

  // Create region, DecisionTree with max_depth 3 will always pick the optimal
  // policy.
  //Apollo::Region *r = new Apollo::Region(NUM_FEATURES,
  //                                       "test-region",
  //                                       NUM_POLICIES,
  //                                       /* min_training_data */ 1);

  // Outer loop to simulate iterative execution of inner region, install tuned
  // model after first iteration that fully explores features and variants.

  // We're going to form a 'V' in xtimes, where the middle policy (idx 2)
  // has the best xtime, while the others have larger xtimes.
  for (int n = 0; n < REPS; n++) {
    for (int i = 0; i < NUM_SAMPLES; i++) {
        r->begin();
        //r.begin();

        // ignoring feature
        r->setFeature(float(0));
        //r.setFeature(float(0));

        int policy = r->getPolicyIndex();
        //int policy = r.getPolicyIndex();


        delay_loop(DELAY * ((6-policy)*(6-policy)));

        r->end();
        //r.end();
    }
  }

  std::cout << "=== Testing complete\n";

  return 0;
}
