#pragma once 
#include <stdio.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> iou_bev(py::array_t<double> boxes_a_numpy, py::array_t<double> boxes_b_numpy);


PYBIND11_MODULE(numiou, mod) {
    mod.def("bev", &iou_bev, "bev iou algorithm.");
}