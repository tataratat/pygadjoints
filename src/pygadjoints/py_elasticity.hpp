#include <gismo.h>

namespace test::printer {
class Printer {
 public:
  Printer() = default;

  void print(const int i) {
    for (int j{}; j < i; j++) {
      gsWarn << "TEST IF GISMO IS THERE SUCCEEDED, I AM YOUR\n";
    }
  }
};

}  // namespace test::printer
