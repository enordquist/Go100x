//  Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.
//  Copyright 2015. UChicago Argonne, LLC. This software was produced
//  under U.S. Government contract DE-AC02-06CH11357 for Argonne National
//  Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
//  UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
//  ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
//  modified to produce derivative works, such modified software should
//  be clearly marked, so as not to confuse it with the version available
//  from ANL.
//  Additionally, redistribution and use in source and binary forms, with
//  or without modification, are permitted provided that the following
//  conditions are met:
//      * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//      * Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in
//        the documentation andwith the
//        distribution.
//      * Neither the name of UChicago Argonne, LLC, Argonne National
//        Laboratory, ANL, the U.S. Government, nor the names of its
//        contributors may be used to endorse or promote products derived
//        from this software without specific prior written permission.
//  THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
//  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
//  Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
//  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//  ---------------------------------------------------------------
//   TOMOPY header

/** \file macros.hh
 * \headerfile macros.hh "include/macros.hh"
 * Include files + some standard macros available to C++
 */

#pragma once

//======================================================================================//
//  headers
//
#include "profiler.hpp"
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <sstream>
#include <string>

//======================================================================================//
// this function is used by a macro -- returns a unique identifier to the thread
//
inline uintmax_t GetThisThreadID()
{
    static std::atomic<uintmax_t> tcounter;
    static thread_local auto      tid = tcounter++;
    return tid;
}

//======================================================================================//
// get the number of hardware threads
//
#if !defined(HW_CONCURRENCY)
#    define HW_CONCURRENCY std::thread::hardware_concurrency()
#endif

//======================================================================================//
// debugging
//
#if !defined(PRINT_HERE)
#    define PRINT_HERE(extra)                                                            \
        printf("[%lu]> %s@'%s':%i %s\n", GetThisThreadID(), __FUNCTION__, __FILE__,      \
               __LINE__, extra)
#endif

//======================================================================================//
// debugging
//
#if !defined(PRINT_ERROR_HERE)
#    define PRINT_ERROR_HERE(extra)                                                      \
        fprintf(stderr, "[%lu]> %s@'%s':%i %s\n", GetThisThreadID(), __FUNCTION__,       \
                __FILE__, __LINE__, extra)
#endif

//======================================================================================//

#if defined(__NVCC__)

//--------------------------------------------------------------------------------------//
// this is always defined, even in release mode
//
#    if !defined(CUDA_CHECK_CALL)
#        define CUDA_CHECK_CALL(err)                                                     \
            {                                                                            \
                if(cudaSuccess != err)                                                   \
                {                                                                        \
                    std::stringstream ss;                                                \
                    ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'"          \
                       << __FILE__ << "':" << __LINE__ << " : "                          \
                       << cudaGetErrorString(err);                                       \
                    fprintf(stderr, "%s\n", ss.str().c_str());                           \
                    throw std::runtime_error(ss.str().c_str());                          \
                }                                                                        \
            }
#    endif

// this is only defined in debug mode
//
#    if !defined(CUDA_CHECK_LAST_ERROR)
#        if defined(DEBUG)
#            define CUDA_CHECK_LAST_ERROR()                                              \
                {                                                                        \
                    cudaStreamSynchronize(0);                                            \
                    cudaError err = cudaGetLastError();                                  \
                    if(cudaSuccess != err)                                               \
                    {                                                                    \
                        std::stringstream ss;                                            \
                        ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'"      \
                           << __FILE__ << "':" << __LINE__ << " : "                      \
                           << cudaGetErrorString(err);                                   \
                        fprintf(stderr, "cudaCheckError() failed at %s@'%s':%i : %s\n",  \
                                __FUNCTION__, __FILE__, __LINE__,                        \
                                cudaGetErrorString(err));                                \
                        throw std::runtime_error(ss.str());                              \
                    }                                                                    \
                }
#        else
#            define CUDA_CHECK_LAST_ERROR()                                              \
                {                                                                        \
                    ;                                                                    \
                }
#        endif
#    endif

// this is only defined in debug mode
//
#    if !defined(CUDA_CHECK_LAST_STREAM_ERROR)
#        if defined(DEBUG)
#            define CUDA_CHECK_LAST_STREAM_ERROR(stream)                                 \
                {                                                                        \
                    cudaStreamSynchronize(stream);                                       \
                    cudaError err = cudaGetLastError();                                  \
                    if(cudaSuccess != err)                                               \
                    {                                                                    \
                        std::stringstream ss;                                            \
                        ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'"      \
                           << __FILE__ << "':" << __LINE__ << " : "                      \
                           << cudaGetErrorString(err);                                   \
                        fprintf(stderr, "cudaCheckError() failed at %s@'%s':%i : %s\n",  \
                                __FUNCTION__, __FILE__, __LINE__,                        \
                                cudaGetErrorString(err));                                \
                        throw std::runtime_error(ss.str());                              \
                    }                                                                    \
                }
#        else
#            define CUDA_CHECK_LAST_STREAM_ERROR(stream)                                 \
                {                                                                        \
                    ;                                                                    \
                }
#        endif
#    endif

#endif  // NVCC

//======================================================================================//
