
#include <gismo.h>

#include <chrono>
#include <iomanip>
#include <iostream>

namespace pygadjoints {

class Timer {
#if 1
 public:
  const std::string name;
  const std::chrono::time_point<std::chrono::high_resolution_clock>
      starting_time;
  Timer(const std::string &function_name)
      : name(function_name),
        starting_time{std::chrono::high_resolution_clock::now()} {
    std::cout << "[Timer] : " << std::setw(40) << name << "\tstarted..."
              << std::endl;
  }

  ~Timer() {
    std::cout << "[Timer] : " << std::setw(40) << name << "\tElapsed time : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starting_time)
                     .count()
              << "ms" << std::endl;
  }
#endif
};
}  // namespace pygadjoints