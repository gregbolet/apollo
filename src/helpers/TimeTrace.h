#ifndef APOLLO_HELPERS_TIMETRACE_H
#define APOLLO_HELPERS_TIMETRACE_H

#include <chrono>
#include <string>
#include <iostream>

// Simple time tracing class, outputs time elapsed in a block scope.
class TimeTrace
{
  std::chrono::time_point<std::chrono::steady_clock> start;
  std::string ref;

public:
  TimeTrace(std::string ref) : ref(ref)
  {
    start = std::chrono::steady_clock::now();
  }
  ~TimeTrace()
  {
    auto end = std::chrono::steady_clock::now();
    std::cout << "=== T ref " << ref << " = "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       start)
                     .count()
              << " us" << std::endl;
  }
};

#endif