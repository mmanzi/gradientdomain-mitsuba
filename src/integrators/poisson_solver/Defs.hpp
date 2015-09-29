/*
 *  Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     *  Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *     *  Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *     *  Neither the name of the NVIDIA CORPORATION nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include <math.h>
#include <float.h>
#include <assert.h>
#include <string>

namespace poisson
{
//------------------------------------------------------------------------
// Common types and definitions.
//------------------------------------------------------------------------

void        fail    (const char* fmt, ...);
std::string sprintf (const char* fmt, ...);

//------------------------------------------------------------------------

#ifdef __CUDACC__
#   define CUDA_FUNC    __host__ __device__ __forceinline__
#else
#   define CUDA_FUNC    inline
#endif

//------------------------------------------------------------------------

template <class T> CUDA_FUNC T      min     (T a, T b)      { return (a < b) ? a : b; }
template <class T> CUDA_FUNC T      max     (T a, T b)      { return (a > b) ? a : b; }
template <class T> CUDA_FUNC void   swap    (T& a, T& b)    { T t = a; a = b; b = t; }

//------------------------------------------------------------------------

class Vec2i
{
public:
    CUDA_FUNC   Vec2i   (void)          : x(0), y(0) {}
    CUDA_FUNC   Vec2i   (int x, int y)  : x(x), y(y) {}

public:
    int x;
    int y;
};

//------------------------------------------------------------------------

class Vec3f
{
public:
    CUDA_FUNC           Vec3f       (void)                      : x(0.0f), y(0.0f), z(0.0f) {}
    CUDA_FUNC           Vec3f       (float xyz)                 : x(xyz), y(xyz), z(xyz) {}
    CUDA_FUNC           Vec3f       (float x, float y, float z) : x(x), y(y), z(z) {}

    CUDA_FUNC Vec3f     operator+   (Vec3f b) const             { return Vec3f(x + b.x, y + b.y, z + b.z); }
    CUDA_FUNC Vec3f     operator-   (Vec3f b) const             { return Vec3f(x - b.x, y - b.y, z - b.z); }
    CUDA_FUNC Vec3f     operator*   (Vec3f b) const             { return Vec3f(x * b.x, y * b.y, z * b.z); }
    CUDA_FUNC Vec3f     operator/   (Vec3f b) const             { return Vec3f(x / b.x, y / b.y, z / b.z); }

    CUDA_FUNC Vec3f     operator+=  (Vec3f b)                   { *this = *this + b; return *this; }
    CUDA_FUNC Vec3f     operator-=  (Vec3f b)                   { *this = *this - b; return *this; }

    CUDA_FUNC float&    operator[]  (int i)                     { assert(i >= 0 && i < 3); return (i == 0) ? x : (i == 1) ? y : z; }
    CUDA_FUNC float     operator[]  (int i) const               { assert(i >= 0 && i < 3); return (i == 0) ? x : (i == 1) ? y : z; }

public:
    float   x;
    float   y;
    float   z;
};

static CUDA_FUNC Vec3f  operator+   (Vec3f a, float b)          { return a + Vec3f(b); }
static CUDA_FUNC Vec3f  operator-   (Vec3f a, float b)          { return a - Vec3f(b); }
static CUDA_FUNC Vec3f  operator*   (Vec3f a, float b)          { return a * Vec3f(b); }
static CUDA_FUNC Vec3f  operator/   (Vec3f a, float b)          { return a / Vec3f(b); }

static CUDA_FUNC Vec3f  operator+   (float a, Vec3f b)          { return Vec3f(a) + b; }
static CUDA_FUNC Vec3f  operator-   (float a, Vec3f b)          { return Vec3f(a) - b; }
static CUDA_FUNC Vec3f  operator*   (float a, Vec3f b)          { return Vec3f(a) * b; }
static CUDA_FUNC Vec3f  operator/   (float a, Vec3f b)          { return Vec3f(a) / b; }

static CUDA_FUNC float  lenSqr      (Vec3f a)                   { return a.x * a.x + a.y * a.y + a.z * a.z; }
static CUDA_FUNC float  length      (Vec3f a)                   { return sqrtf(lenSqr(a)); }

static CUDA_FUNC Vec3f  min         (Vec3f a, Vec3f b)          { return Vec3f(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
static CUDA_FUNC Vec3f  max         (Vec3f a, Vec3f b)          { return Vec3f(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }

//------------------------------------------------------------------------
}
