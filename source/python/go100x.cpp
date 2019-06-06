// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
#if !defined(DEBUG)
#define DEBUG
#endif

#include "go100x.hpp"
#include "Go100x/kernels.hpp"
#include <pybind11/pybind11.h>

#include "Go100x/macros.hpp"

using namespace tim::component;

using auto_tuple_t =
    tim::auto_tuple<real_clock, system_clock, user_clock, cpu_clock, cpu_util>;

// unlike most components, "cuda_event" does not support nesting
using cuda_tuple_t =
    tim::auto_tuple<real_clock, system_clock, user_clock, cpu_clock, cuda_event>;

//======================================================================================//
//  Python wrappers
//======================================================================================//


PYBIND11_MODULE(go100x, gox)
{
    //----------------------------------------------------------------------------------//
    using gox::string_t;
    py::add_ostream_redirect(gox, "ostream_redirect");

    auto launch_cpu_calculate = [](farray_t matrix_a, farray_t matrix_b) {
        if(matrix_a.size() != matrix_b.size())
        {
            std::cerr << "Error! matrix A size does not match matrix B size: "
                      << matrix_a.size() << " vs. " << matrix_b.size() << std::endl;
            throw std::runtime_error("Matrix input error");
        }

        auto         result    = farray_t(matrix_a.size());
        const float* fmatrix_a = matrix_a.data();
        const float* fmatrix_b = matrix_b.data();
        // time the execution on the CPU
        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[CPU<<<", 0, ",", 0, ">>>]");
            cpu_calculate(fmatrix_a, fmatrix_b, result.mutable_data(), matrix_a.size());
        }
        return result;
    };
    
    auto launch_fun_calculate = [](farray_t R, farray_t r, int J) {

        const int     N = R.size();
        auto         fD = farray_t(N);
        const float* fR = R.data();
        const float* fr = r.data();
        // time the execution on the CPU
        {
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[CPU]");
            cpu_fun(fR, fr, fD.mutable_data(), J, N);
        }
        return fD;
    };

    auto launch_gpu_calculate = [](int block, int grid, farray_t matrix_a,
                                   farray_t matrix_b) {
        if(matrix_a.size() != matrix_b.size())
        {
            std::cerr << "Error! matrix A size does not match matrix B size: "
                      << matrix_a.size() << " vs. " << matrix_b.size() << std::endl;
            throw std::runtime_error("Matrix input error");
        }

        auto         result    = farray_t(matrix_a.size());
        const float* fmatrix_a = matrix_a.data();
        const float* fmatrix_b = matrix_b.data();
        // time the execution on the GPU
        float *fmatrix_a_d, *fmatrix_b_d, *output_d;
        int size = matrix_a.size();
        cudaSetDevice(0);
        CUDA_CHECK_CALL( cudaMalloc(&fmatrix_a_d, size*sizeof(float)) );
        CUDA_CHECK_CALL( cudaMalloc(&fmatrix_b_d, size*sizeof(float)) );
        CUDA_CHECK_CALL( cudaMalloc(&output_d, size*sizeof(float)) );

        CUDA_CHECK_CALL( cudaMemcpy(fmatrix_a_d, fmatrix_a, size*sizeof(float), cudaMemcpyHostToDevice) );
        CUDA_CHECK_CALL( cudaMemcpy(fmatrix_b_d, fmatrix_b, size*sizeof(float), cudaMemcpyHostToDevice) );

  
        {
            CUDA_CHECK_LAST_ERROR();
            TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_t, "[GPU<<<", grid, ",", block, ">>>]");
            gpu_calculate(block, grid, fmatrix_a_d, fmatrix_b_d, output_d,
                          matrix_a.size());
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK_CALL( cudaDeviceSynchronize() );
        }


        CUDA_CHECK_CALL( cudaFree(fmatrix_a_d) );
        CUDA_CHECK_CALL( cudaFree(fmatrix_b_d) );
        CUDA_CHECK_CALL( cudaMemcpy(result.mutable_data(), output_d, size*sizeof(float), cudaMemcpyDeviceToHost) );
        CUDA_CHECK_CALL( cudaFree(output_d) );

        return result;
    };

    gox.def("calculate_cpu", launch_cpu_calculate, "launch the calculation on cpu");
    gox.def("calculate_gpu", launch_gpu_calculate, "launch the calculation on gpu");
    gox.def("calculate_fun", launch_fun_calculate, "launch the calculation on cpu, too");
}
