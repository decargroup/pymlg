#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "liegroups.h"
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

template <class T, int n, int dof>
void bind_matrix_lie_group(py::module &m, const std::string &name) {

    // using BaseType = MatrixLieGroup<n, dof>;
    // py::class_<BaseType> base(m, "_MatrixLieGroup");
    // base.def(py::init<>())
    // .def_static("Exp", &BaseType::Exp)
    // .def_static("Log", &BaseType::Log)
    // .def_static("wedge", &BaseType::wedge)
    // .def_static("vee", &BaseType::vee)
    // .def_static("inverse", &BaseType::inverse)
    // .def_static("adjoint", &BaseType::adjoint)
    // .def_static("odot", &BaseType::odot)
    // .def_static("left_jacobian", &BaseType::leftJacobian)
    // .def_static("left_jacobian_inv", &BaseType::leftJacobianInverse)
    // .def_readonly_static("dof", &BaseType::dof)
    // .def_readonly_static("matrix_size", &BaseType::matrix_size)
    // .def_readonly_static("small_angle_tol", &BaseType::small_angle_tol);


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

    // py::class_<MatrixLieGroup<3, 3>>(m, "MatrixLieGroup33")
    //     .def("right_jacobian", &MatrixLieGroup<3, 3>::rightJacobian);

    // py::class_<SO3>(m, "SO3")
    // .def_static("Exp", &SO3::Exp)
    // .def_static("Log", &SO3::Log)
    // .def_static("wedge", &SO3::wedge)
    // .def_static("vee", &SO3::vee)
    // .def_static("inverse", &SO3::inverse)
    // .def_static("adjoint", &SO3::adjoint)
    // .def_static("odot", &SO3::odot)
    // .def_static("left_jacobian", &SO3::leftJacobian)
    // .def_static("left_jacobian_inv", &SO3::leftJacobianInverse);
    // .def_readonly_static("small_angle_tol", &SO3::small_angle_tol)
    // .def_readonly_static("dof", &SO3::dof);

    // py::class_<SE3, MatrixLieGroup<4, 6>>(m, "SE3")
    // .def("Exp", &SE3::Exp)
    // .def("Log", &SE3::Log)
    // .def("wedge", &SE3::wedge)
    // .def("vee", &SE3::vee)
    // .def("inverse", &SE3::inverse)
    // .def("adjoint", &SE3::adjoint)
    // .def("left_jacobian", &SE3::leftJacobian)
    // .def("left_jacobian_inv", &SE3::leftJacobianInverse)
    // .def_readonly_static("small_angle_tol", &SE3::small_angle_tol)
    // .def_readonly_static("dof", &SE3::dof);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}