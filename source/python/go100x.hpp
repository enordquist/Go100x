//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

#pragma once

//======================================================================================//

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include "timemory/timemory.hpp"

#include "Go100x/kernels.hpp"
#include "Go100x/macros.hpp"
#include "Go100x/profiler.hpp"

namespace py = pybind11;
using namespace std::placeholders;
using namespace py::literals;

using farray_t = py::array_t<float, py::array::c_style | py::array::forcecast>;

//======================================================================================//

#if _PYTHON_MAJOR_VERSION > 2
#    define PYOBJECT_SELF
#    define PYOBJECT_SELF_PARAM
#else
#    define PYOBJECT_SELF py::object,
#    define PYOBJECT_SELF_PARAM py::object
#endif

//======================================================================================//

namespace gox
{
using string_t = std::string;

}  // namespace gox

//======================================================================================//
