#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "liegroups.h"
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

template <class T, int n, int dof>
void bind_matrix_lie_group(py::module &m, const std::string &name) {

    py::class_<T>(m, name.c_str())
        .def(py::init<>())
        .def_static("Exp", &T::Exp)
        .def_static("Log", &T::Log)
        .def_static("exp", &T::exp)
        .def_static("log", &T::log)
        .def_static("wedge", &T::wedge)
        .def_static("vee", &T::vee)
        .def_static("inverse", &T::inverse)
        .def_static("adjoint", &T::adjoint)
        .def_static("odot", &T::odot)
        .def_static("left_jacobian", &T::leftJacobian)
        .def_static("left_jacobian_inv", &T::leftJacobianInverse)
        .def_static("random", &T::random)
        .def_static("identity", &T::identity)
        .def_readonly_static("dof", &T::dof)
        .def_readonly_static("matrix_size", &T::matrix_size)
        .def_readonly_static("small_angle_tol", &T::small_angle_tol);

};

PYBIND11_MODULE(_impl, m) {

    bind_matrix_lie_group<SO3, 3, 3>(m, "SO3");
    bind_matrix_lie_group<SE3, 3, 3>(m, "SE3");
    bind_matrix_lie_group<SE23, 3, 3>(m, "SE23");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}