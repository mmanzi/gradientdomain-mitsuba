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

#include "BackendCUDA.hpp"
#include <stdio.h>

namespace poisson
{
//------------------------------------------------------------------------

#define globalThreadIdx (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y)))

#if __CUDA_ARCH__ < 350
template <class T> __device__ __forceinline__ T __ldg   (const T* in)   { return *in; }
#endif

template <class T> __device__ __forceinline__ T ld4     (const T& in)   { T out; for (int ofs = 0; ofs < sizeof(T); ofs += 4) *(float*)((char*)&out + ofs) = __ldg((float*)((char*)&in + ofs)); return out; }
template <class T> __device__ __forceinline__ T ld8     (const T& in)   { T out; for (int ofs = 0; ofs < sizeof(T); ofs += 8) *(float2*)((char*)&out + ofs) = __ldg((float2*)((char*)&in + ofs)); return out; }
template <class T> __device__ __forceinline__ T ld16    (const T& in)   { T out; for (int ofs = 0; ofs < sizeof(T); ofs += 16) *(float4*)((char*)&out + ofs) = __ldg((float4*)((char*)&in + ofs)); return out; }

//------------------------------------------------------------------------

BackendCUDA::BackendCUDA(int device)
{
    assert(device >= -1);

    // No device specified => choose one.

    if (device == -1)
    {
        device = chooseDevice();
        if (device == -1)
            fail("No suitable CUDA device found!");
    }

    // Initialize.

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    cudaSetDevice(device);
    checkError();

    printf("Using CUDA device %d: %s\n", device, prop.name);
    if (prop.major < 3)
        fail("Compute capability 3.0 or higher is required!");

    m_maxGridWidth = prop.maxGridSize[0];
    m_blockDim = (prop.major >= 5) ? dim3(32, 2, 1) : dim3(32, 4, 1);
}

//------------------------------------------------------------------------

BackendCUDA::~BackendCUDA(void)
{
}

//------------------------------------------------------------------------

int BackendCUDA::chooseDevice(void)
{
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);

    int bestDevice = -1;
    cudaDeviceProp bestProp;
    memset(&bestProp, 0, sizeof(bestProp));

    for (int d = 0; d < numDevices; d++)
    {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);

        if (p.major < 3) // need at least sm_30
            continue;

        if (bestDevice == -1 ||
            (p.major != bestProp.major) ? (p.major > bestProp.major) :
            (p.minor != bestProp.minor) ? (p.minor > bestProp.minor) :
            (p.multiProcessorCount > bestProp.multiProcessorCount))
        {
            bestDevice = d;
            bestProp = p;
        }
    }

    checkError();
    return bestDevice;
}

//------------------------------------------------------------------------

void BackendCUDA::checkError(void)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        fail("CUDA runtime error: %s!", cudaGetErrorString(error));
}

//------------------------------------------------------------------------

BackendCUDA::Vector* BackendCUDA::allocVector(int numElems, size_t bytesPerElem)
{
    assert(numElems >= 0);
    assert(bytesPerElem > 0);

    Vector* x       = new Vector;
    x->numElems     = numElems;
    x->bytesPerElem = bytesPerElem;
    x->bytesTotal   = numElems * bytesPerElem;
    x->ptr          = NULL;

    cudaMalloc(&x->ptr, x->bytesTotal);
    checkError();
    return x;
}

//------------------------------------------------------------------------

void BackendCUDA::freeVector(Vector* x)
{
    if (x && x->ptr)
    {
        cudaFree(x->ptr);
        checkError();
    }
    delete x;
}

//------------------------------------------------------------------------

void* BackendCUDA::map(Vector* x)
{
    assert(x);
    void* ptr = NULL;
    cudaMallocHost(&ptr, x->bytesTotal);
    cudaMemcpy(ptr, x->ptr, x->bytesTotal, cudaMemcpyDeviceToHost);
    checkError();
    return ptr;
}

//------------------------------------------------------------------------

void BackendCUDA::unmap(Vector* x, void* ptr, bool modified)
{
    if (ptr)
    {
        if (modified && x)
            cudaMemcpy(x->ptr, ptr, x->bytesTotal, cudaMemcpyHostToDevice);
        cudaFreeHost(ptr);
        checkError();
    }
}

//------------------------------------------------------------------------

__global__ void kernel_set(float* x, float y, int numElems)
{
    int i = globalThreadIdx;
    if (i >= numElems)
        return;

    x[i] = y;
}

//------------------------------------------------------------------------

void BackendCUDA::set(Vector* x, float y)
{
    assert(x && x->bytesPerElem % sizeof(float) == 0);
    int numElems = (int)(x->bytesTotal / sizeof(float));

    cudaFuncSetCacheConfig(&kernel_set, cudaFuncCachePreferL1);
    kernel_set<<<gridDim(numElems), m_blockDim>>>
        ((float*)x->ptr, y, numElems);

    checkError();
}

//------------------------------------------------------------------------

void BackendCUDA::copy(Vector* x, Vector* y)
{
    assert(x && y && x->bytesTotal == y->bytesTotal);
    cudaMemcpy(x->ptr, y->ptr, x->bytesTotal, cudaMemcpyDeviceToDevice);
    checkError();
}

//------------------------------------------------------------------------

void BackendCUDA::read(void* ptr, Vector* x)
{
    assert(ptr && x);
    cudaMemcpy(ptr, x->ptr, x->bytesTotal, cudaMemcpyDeviceToHost);
    checkError();
}

//------------------------------------------------------------------------

void BackendCUDA::write(Vector* x, const void* ptr)
{
    assert(x && ptr);
    cudaMemcpy(x->ptr, ptr, x->bytesTotal, cudaMemcpyHostToDevice);
    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_Px(Vec3f* Px, const Vec3f* x, Vec2i size, float alpha)
{
    int xx = threadIdx.x + blockIdx.x * blockDim.x;
    int yy = threadIdx.y + blockIdx.y * blockDim.y;
    if (xx >= size.x || yy >= size.y)
        return;

    int i = xx + yy * size.x;
    int n = size.x * size.y;

    Vec3f xi = ld4(x[i]);
    Px[n * 0 + i] = xi * alpha;
    Px[n * 1 + i] = (xx != size.x - 1) ? ld4(x[i + 1]) - xi : 0.0f;
    Px[n * 2 + i] = (yy != size.y - 1) ? ld4(x[i + size.x]) - xi : 0.0f;
}

//------------------------------------------------------------------------

void BackendCUDA::calc_Px(Vector* Px, PoissonMatrix P, Vector* x)
{
    assert(Px && Px->numElems == P.size.x * P.size.y * 3 && Px->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);
    assert(x && x->numElems == P.size.x * P.size.y && x->bytesPerElem == sizeof(Vec3f));

    cudaFuncSetCacheConfig(&kernel_Px, cudaFuncCachePreferL1);
    kernel_Px<<<gridDim(P.size.x, P.size.y), m_blockDim>>>
        ((Vec3f*)Px->ptr, (const Vec3f*)x->ptr, P.size, P.alpha);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_PTW2x(Vec3f* PTW2x, const float* w2, const Vec3f* x, Vec2i size, float alpha)
{
    int xx = threadIdx.x + blockIdx.x * blockDim.x;
    int yy = threadIdx.y + blockIdx.y * blockDim.y;
    if (xx >= size.x || yy >= size.y)
        return;

    int i = xx + yy * size.x;
    int n = size.x * size.y;

    Vec3f PTW2xi = ld4(w2[n * 0 + i]) * ld4(x[n * 0 + i]) * alpha;
    if (xx != 0)            PTW2xi += ld4(w2[n * 1 + i - 1])        * ld4(x[n * 1 + i - 1]);
    if (xx != size.x - 1)   PTW2xi -= ld4(w2[n * 1 + i])            * ld4(x[n * 1 + i]);
    if (yy != 0)            PTW2xi += ld4(w2[n * 2 + i - size.x])   * ld4(x[n * 2 + i - size.x]);
    if (yy != size.y - 1)   PTW2xi -= ld4(w2[n * 2 + i])            * ld4(x[n * 2 + i]);
    PTW2x[i] = PTW2xi;
}

//------------------------------------------------------------------------

void BackendCUDA::calc_PTW2x(Vector* PTW2x, PoissonMatrix P, Vector* w2, Vector* x)
{
    assert(PTW2x && PTW2x->numElems == P.size.x * P.size.y && PTW2x->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);
    assert(w2 && w2->numElems == P.size.x * P.size.y * 3 && w2->bytesPerElem == sizeof(float));
    assert(x && x->numElems == P.size.x * P.size.y * 3 && x->bytesPerElem == sizeof(Vec3f));

    cudaFuncSetCacheConfig(&kernel_PTW2x, cudaFuncCachePreferL1);
    kernel_PTW2x<<<gridDim(P.size.x, P.size.y), m_blockDim>>>
        ((Vec3f*)PTW2x->ptr, (const float*)w2->ptr, (const Vec3f*)x->ptr, P.size, P.alpha);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_Ax_xAx(Vec3f* Ax, Vec3f* xAx, const float* w2, const Vec3f* x, Vec2i size, int elemsPerThread, float alphaSqr)
{
    int xxbegin = threadIdx.x + blockIdx.x * (elemsPerThread * 32);
    int xxend   = ::min(xxbegin + elemsPerThread * 32, size.x);
    int yy      = threadIdx.y + blockIdx.y * blockDim.y;
    if (yy >= size.y || xxbegin >= xxend)
        return;

    Vec3f sum = 0.0f;
    int n = size.x * size.y;

    for (int xx = xxbegin; xx < xxend; xx += 32)
    {
        int i = xx + yy * size.x;

        Vec3f xi = ld4(x[i]);
        Vec3f Axi = ld4(w2[n * 0 + i]) * xi * alphaSqr;
        if (xx != 0)            Axi += ld4(w2[n * 1 + i - 1])       * (xi - ld4(x[i - 1]));
        if (xx != size.x - 1)   Axi += ld4(w2[n * 1 + i])           * (xi - ld4(x[i + 1]));
        if (yy != 0)            Axi += ld4(w2[n * 2 + i - size.x])  * (xi - ld4(x[i - size.x]));
        if (yy != size.y - 1)   Axi += ld4(w2[n * 2 + i])           * (xi - ld4(x[i + size.x]));
        Ax[i] = Axi;
        sum += xi * Axi;
    }

    for (int c = 0; c < 3; c++)
    {
        float t = sum[c];
        for (int i = 1; i < 32; i *= 2) t += __shfl_xor(t, i);
        if (threadIdx.x == 0) atomicAdd(&xAx->x + c, t);
    }
}

//------------------------------------------------------------------------

void BackendCUDA::calc_Ax_xAx(Vector* Ax, Vector* xAx, PoissonMatrix P, Vector* w2, Vector* x)
{
    assert(Ax && Ax->numElems == P.size.x * P.size.y && Ax->bytesPerElem == sizeof(Vec3f));
    assert(xAx && xAx->numElems == 1 && xAx->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);
    assert(w2 && w2->numElems == P.size.x * P.size.y * 3 && w2->bytesPerElem == sizeof(float));
    assert(x && x->numElems == P.size.x * P.size.y && x->bytesPerElem == sizeof(Vec3f));

    int elemsPerThread = 2;
    int totalThreadsX = (P.size.x - 1) / elemsPerThread + 1;
    set(xAx, 0.0f);

    cudaFuncSetCacheConfig(&kernel_Ax_xAx, cudaFuncCachePreferL1);
    kernel_Ax_xAx<<<gridDim(totalThreadsX, P.size.y), m_blockDim>>>
        ((Vec3f*)Ax->ptr, (Vec3f*)xAx->ptr, (const float*)w2->ptr, (const Vec3f*)x->ptr, P.size, elemsPerThread, P.alpha * P.alpha);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_axpy(Vec3f* axpy, Vec3f a, const Vec3f* x, const Vec3f* y, int numElems)
{
    int i = globalThreadIdx;
    if (i >= numElems)
        return;

    axpy[i] = a * ld4(x[i]) + ld4(y[i]);
}

//------------------------------------------------------------------------

void BackendCUDA::calc_axpy(Vector* axpy, Vec3f a, Vector* x, Vector* y)
{
    assert(axpy && axpy->numElems == x->numElems && axpy->bytesPerElem == sizeof(Vec3f));
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(y && y->numElems == x->numElems && y->bytesPerElem == sizeof(Vec3f));

    cudaFuncSetCacheConfig(&kernel_axpy, cudaFuncCachePreferL1);
    kernel_axpy<<<gridDim(x->numElems), m_blockDim>>>
        ((Vec3f*)axpy->ptr, a, (const Vec3f*)x->ptr, (const Vec3f*)y->ptr, x->numElems);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_xdoty(Vec3f* xdoty, const Vec3f* x, const Vec3f* y, int numElems, int elemsPerThread)
{
    int begin = globalThreadIdx;
    begin = (begin & 31) + (begin & ~31) * elemsPerThread;
    int end = ::min(begin + elemsPerThread * 32, numElems);
    if (begin >= end)
        return;

    Vec3f sum = 0.0f;
    for (int i = begin; i < end; i += 32)
        sum += ld4(x[i]) * ld4(y[i]);

    for (int c = 0; c < 3; c++)
    {
        float t = sum[c];
        for (int i = 1; i < 32; i *= 2) t += __shfl_xor(t, i);
        if (threadIdx.x == 0) atomicAdd(&xdoty->x + c, t);
    }
}

//------------------------------------------------------------------------

void BackendCUDA::calc_xdoty(Vector* xdoty, Vector* x, Vector* y)
{
    assert(xdoty && xdoty->numElems == 1 && xdoty->bytesPerElem == sizeof(Vec3f));
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(y && y->numElems == x->numElems && y->bytesPerElem == sizeof(Vec3f));

    int elemsPerThread = 2;
    int totalThreads = (x->numElems - 1) / elemsPerThread + 1;
    set(xdoty, 0.0f);

    cudaFuncSetCacheConfig(&kernel_xdoty, cudaFuncCachePreferL1);
    kernel_xdoty<<<gridDim(totalThreads), m_blockDim>>>
        ((Vec3f*)xdoty->ptr, (const Vec3f*)x->ptr, (const Vec3f*)y->ptr, x->numElems, elemsPerThread);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_r_rz(Vec3f* r, Vec3f* rz, const Vec3f* Ap, const Vec3f* rz2, const Vec3f* pAp, int numPixels, int elemsPerThread)
{
    int begin = globalThreadIdx;
    begin = (begin & 31) + (begin & ~31) * elemsPerThread;
    int end = ::min(begin + elemsPerThread * 32, numPixels);
    if (begin >= end)
        return;

    Vec3f a = ld4(*rz2) / max(ld4(*pAp), FLT_MIN);
    Vec3f sum = 0.0f;

    for (int i = begin; i < end; i += 32)
    {
        Vec3f ri = ld4(r[i]) - ld4(Ap[i]) * a;
        r[i] = ri;
        sum += ri * ri;
    }

    for (int c = 0; c < 3; c++)
    {
        float t = sum[c];
        for (int i = 1; i < 32; i *= 2) t += __shfl_xor(t, i);
        if (threadIdx.x == 0) atomicAdd(&rz->x + c, t);
    }
}

//------------------------------------------------------------------------

void BackendCUDA::calc_r_rz(Vector* r, Vector* rz, Vector* Ap, Vector* rz2, Vector* pAp)
{
    assert(r && r->bytesPerElem == sizeof(Vec3f));
    assert(rz && rz->numElems == 1 && rz->bytesPerElem == sizeof(Vec3f));
    assert(Ap && Ap->numElems == r->numElems && Ap->bytesPerElem == sizeof(Vec3f));
    assert(rz2 && rz2->numElems == 1 && rz2->bytesPerElem == sizeof(Vec3f));
    assert(pAp && pAp->numElems == 1 && pAp->bytesPerElem == sizeof(Vec3f));

    int elemsPerThread = 2;
    int totalThreads = (r->numElems - 1) / elemsPerThread + 1;
    set(rz, 0.0f);

    cudaFuncSetCacheConfig(&kernel_r_rz, cudaFuncCachePreferL1);
    kernel_r_rz<<<gridDim(totalThreads), m_blockDim>>>
        ((Vec3f*)r->ptr, (Vec3f*)rz->ptr, (const Vec3f*)Ap->ptr, (const Vec3f*)rz2->ptr, (const Vec3f*)pAp->ptr, r->numElems, elemsPerThread);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_x_p(Vec3f* x, Vec3f* p, const Vec3f* r, const Vec3f* rz, const Vec3f* rz2, const Vec3f* pAp, int numElems)
{
    int i = globalThreadIdx;
    if (i >= numElems)
        return;

    Vec3f rzv = ld4(*rz);
    Vec3f rz2v = ld4(*rz2);
    Vec3f pApv = ld4(*pAp);
    Vec3f a = rz2v / max(pApv, FLT_MIN);
    Vec3f b = rzv / max(rz2v, FLT_MIN);

    Vec3f pi = ld4(p[i]);
    x[i] += pi * a;
    p[i] = ld4(r[i]) + pi * b;
}

//------------------------------------------------------------------------

void BackendCUDA::calc_x_p(Vector* x, Vector* p, Vector* r, Vector* rz, Vector* rz2, Vector* pAp)
{
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(p && p->numElems == x->numElems && p->bytesPerElem == sizeof(Vec3f));
    assert(r && r->numElems == x->numElems && r->bytesPerElem == sizeof(Vec3f));
    assert(rz && rz->numElems == 1 && rz->bytesPerElem == sizeof(Vec3f));
    assert(rz2 && rz2->numElems == 1 && rz2->bytesPerElem == sizeof(Vec3f));
    assert(pAp && pAp->numElems == 1 && pAp->bytesPerElem == sizeof(Vec3f));

    cudaFuncSetCacheConfig(&kernel_x_p, cudaFuncCachePreferL1);
    kernel_x_p<<<gridDim(x->numElems), m_blockDim>>>
        ((Vec3f*)x->ptr, (Vec3f*)p->ptr, (const Vec3f*)r->ptr, (const Vec3f*)rz->ptr, (const Vec3f*)rz2->ptr, (const Vec3f*)pAp->ptr, x->numElems);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_w2sum(float* w2sum, const Vec3f* e, float reg, int numElems, int elemsPerThread)
{
    int begin = globalThreadIdx;
    begin = (begin & 31) + (begin & ~31) * elemsPerThread;
    int end = ::min(begin + elemsPerThread * 32, numElems);
    if (begin >= end)
        return;

    float sum = 0.0f;
    for (int i = begin; i < end; i += 32)
        sum += 1.0f / (length(ld4(e[i])) + reg);

    for (int i = 1; i < 32; i *= 2) sum += __shfl_xor(sum, i);
    if (threadIdx.x == 0) atomicAdd(w2sum, sum);
}

//------------------------------------------------------------------------

__global__ void kernel_w2(float* w2, const Vec3f* e, float reg, int numElems, float coef)
{
    int i = globalThreadIdx;
    if (i >= numElems)
        return;

    w2[i] = coef / (length(ld4(e[i])) + reg);
}

//------------------------------------------------------------------------

void BackendCUDA::calc_w2(Vector* w2, Vector* e, float reg)
{
    assert(w2 && w2->bytesPerElem == sizeof(float));
    assert(e && e->numElems == w2->numElems && e->bytesPerElem == sizeof(Vec3f));

    int elemsPerThread = 2;
    int totalThreads = (w2->numElems - 1) / elemsPerThread + 1;
    cudaMemset(w2->ptr, 0, sizeof(float));

    cudaFuncSetCacheConfig(&kernel_w2sum, cudaFuncCachePreferL1);
    kernel_w2sum<<<gridDim(totalThreads), m_blockDim>>>
        ((float*)w2->ptr, (const Vec3f*)e->ptr, reg, w2->numElems, elemsPerThread);

    float w2sum = 0.0f;
    cudaMemcpy(&w2sum, w2->ptr, sizeof(float), cudaMemcpyDeviceToHost);
    float coef = (float)w2->numElems / w2sum; // normalize so that average(w2) = 1

    cudaFuncSetCacheConfig(&kernel_w2, cudaFuncCachePreferL1);
    kernel_w2<<<gridDim(w2->numElems), m_blockDim>>>
        ((float*)w2->ptr, (const Vec3f*)e->ptr, reg, w2->numElems, coef);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_tonemapSRGB(unsigned int* out, const Vec3f* in, int numPixels, float scale, float bias)
{
    int i = globalThreadIdx;
    if (i >= numPixels)
        return;

    Vec3f color = ld4(in[i]);
    for (int c = 0; c < 3; c++)
    {
        float& t = color[c];
        t = t * scale + bias;
        t = (t <= 0.0031308f) ? 12.92f * t : 1.055f * powf(t, 1.0f / 2.4f) - 0.055f; // linear to sRGB
    }

    out[i] = 0xFF000000 |
        ((int)min(max(color.x * 255.0f + 0.5f, 0.0f), 255.0f) << 0) |
        ((int)min(max(color.y * 255.0f + 0.5f, 0.0f), 255.0f) << 8) |
        ((int)min(max(color.z * 255.0f + 0.5f, 0.0f), 255.0f) << 16);
}

//------------------------------------------------------------------------

void BackendCUDA::tonemapSRGB(Vector* out, Vector* in, int idx, float scale, float bias)
{
    assert(out && out->bytesPerElem == sizeof(unsigned int));
    assert(in && in->numElems % out->numElems == 0 && in->bytesPerElem == sizeof(Vec3f));

    cudaFuncSetCacheConfig(&kernel_tonemapSRGB, cudaFuncCachePreferL1);
    kernel_tonemapSRGB<<<gridDim(out->numElems), m_blockDim>>>
        ((unsigned int*)out->ptr, (const Vec3f*)in->ptr + idx * out->bytesTotal, out->numElems, scale, bias);

    checkError();
}

//------------------------------------------------------------------------

__global__ void kernel_tonemapLinearA(unsigned int* minmaxU32, const float* in, int numPixels, int numComponents)
{
    int i = globalThreadIdx;
    if (i >= numPixels)
        return;

    float inMin = +FLT_MAX;
    float inMax = -FLT_MAX;
    for (int c = 0; c < numComponents; c++)
    {
        float t = ld4(in[i * numComponents + c]);
        inMin = fminf(inMin, t);
        inMax = fmaxf(inMax, t);
    }

    unsigned int minU32 = __float_as_int(inMin);
    unsigned int maxU32 = __float_as_int(inMax);
    atomicMin(&minmaxU32[0], minU32 ^ (((int)minU32 >> 31) | 0x80000000u));
    atomicMax(&minmaxU32[1], maxU32 ^ (((int)maxU32 >> 31) | 0x80000000u));
}

//------------------------------------------------------------------------

__global__ void kernel_tonemapLinearB(unsigned int* out, const float* in, int numPixels, int numComponents, float scale, float bias)
{
    int i = globalThreadIdx;
    if (i >= numPixels)
        return;

    Vec3f color;
    for (int c = 0; c < 3; c++)
        color[c] = (c < numComponents) ? fabsf(ld4(in[i * numComponents + c]) * scale + bias) : color[c - 1];

    out[i] = 0xFF000000 |
        ((int)min(max(color.x * 255.0f + 0.5f, 0.0f), 255.0f) << 0) |
        ((int)min(max(color.y * 255.0f + 0.5f, 0.0f), 255.0f) << 8) |
        ((int)min(max(color.z * 255.0f + 0.5f, 0.0f), 255.0f) << 16);
}

//------------------------------------------------------------------------

void BackendCUDA::tonemapLinear(Vector* out, Vector* in, int idx, float scaleMin, float scaleMax, bool hasNegative)
{
    assert(out && out->numElems >= 2 && out->bytesPerElem == sizeof(unsigned int));
    assert(in && in->numElems % out->numElems == 0 && in->bytesPerElem % sizeof(float) == 0);
    int numComponents = (int)(in->bytesPerElem / sizeof(float));

    Vec2i minmaxU32(~0u, 0u);
    cudaMemcpy(out->ptr, &minmaxU32, sizeof(Vec2i), cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(&kernel_tonemapLinearA, cudaFuncCachePreferL1);
    kernel_tonemapLinearA<<<gridDim(out->numElems), m_blockDim>>>
        ((unsigned int*)out->ptr, (const float*)in->ptr + idx * out->bytesTotal, out->numElems, numComponents);

    cudaMemcpy(&minmaxU32, out->ptr, sizeof(Vec2i), cudaMemcpyDeviceToHost);
    minmaxU32.x ^= ~((int)minmaxU32.x >> 31) | 0x80000000u;
    minmaxU32.y ^= ~((int)minmaxU32.y >> 31) | 0x80000000u;
    float inMin = *(float*)&minmaxU32.x;
    float inMax = *(float*)&minmaxU32.y;

    float scale = min(max((hasNegative) ? 0.5f / max(max(-inMin, inMax), FLT_MIN) : 1.0f / max(inMax, FLT_MIN), scaleMin), scaleMax);
    float bias = (hasNegative) ? 0.5f : 0.0f;

    cudaFuncSetCacheConfig(&kernel_tonemapLinearB, cudaFuncCachePreferL1);
    kernel_tonemapLinearB<<<gridDim(out->numElems), m_blockDim>>>
        ((unsigned int*)out->ptr, (const float*)in->ptr + idx * out->bytesTotal, out->numElems, numComponents, scale, bias);

    checkError();
}

//------------------------------------------------------------------------

Backend::Timer* BackendCUDA::allocTimer(void)
{
    TimerCUDA* timerCUDA = new TimerCUDA;
    timerCUDA->beginTicks = 0;
    timerCUDA->beginEvent = NULL;
    timerCUDA->endEvent = NULL;

    cudaEventCreate(&timerCUDA->beginEvent);
    cudaEventCreate(&timerCUDA->endEvent);
    checkError();
    return timerCUDA;
}

//------------------------------------------------------------------------

void BackendCUDA::freeTimer(Timer* timer)
{
    TimerCUDA* timerCUDA = (TimerCUDA*)timer;
    if (timerCUDA)
    {
        cudaEventDestroy(timerCUDA->beginEvent);
        cudaEventDestroy(timerCUDA->endEvent);
        checkError();
        delete timerCUDA;
    }
}

//------------------------------------------------------------------------

void BackendCUDA::beginTimer(Timer* timer)
{
    assert(timer);
    TimerCUDA* timerCUDA = (TimerCUDA*)timer;
    cudaDeviceSynchronize();
    cudaEventRecord(timerCUDA->beginEvent);
    checkError();
}

//------------------------------------------------------------------------

float BackendCUDA::endTimer(Timer* timer)
{
    assert(timer);
    TimerCUDA* timerCUDA = (TimerCUDA*)timer;
    cudaEventRecord(timerCUDA->endEvent);
    cudaDeviceSynchronize();
    float millis = 0.0f;
    cudaEventElapsedTime(&millis, timerCUDA->beginEvent, timerCUDA->endEvent);
    checkError();
    return millis * 1.0e-3f;
}

//------------------------------------------------------------------------

dim3 BackendCUDA::gridDim(int totalThreadsX, int totalThreadsY)
{
    dim3 gd;
    gd.x = (totalThreadsX - 1) / m_blockDim.x + 1;
    gd.y = (totalThreadsY - 1) / m_blockDim.y + 1;
    gd.z = 1;

    while ((int)gd.x > m_maxGridWidth)
    {
        gd.x /= 2;
        gd.y *= 2;
    }
    return  gd;
}

//------------------------------------------------------------------------
} // namespace poisson
