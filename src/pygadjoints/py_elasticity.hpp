#include

// namespace and stuff




template<typename Type>
void add_adjoint_class(py::module& m, const char* class_name) {
  //add general class attribs

  py::class_</* cpp class comes here. z.b. Type */> klasse(m, class_name);

  klasse.def(py::init<>())
    .def("");
    
}

