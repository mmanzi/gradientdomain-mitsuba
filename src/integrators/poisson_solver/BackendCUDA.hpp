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
#include "Backend.hpp"
#include <cuda_runtime_api.h>

namespace poisson
{
//------------------------------------------------------------------------
// CUDA-accelerated Solver backend.
//------------------------------------------------------------------------

class BackendCUDA : public Backend
{
public:
    struct TimerCUDA : public Timer
    {
        cudaEvent_t     beginEvent;
        cudaEvent_t     endEvent;
    };

public:
                        BackendCUDA         (int device = -1);
    virtual             ~BackendCUDA        (void);

    static int          chooseDevice        (void); // -1 if none.
    static void         checkError          (void);

    // Buffer manipulation.

    virtual Vector*     allocVector         (int numElems, size_t bytesPerElem);
    virtual void        freeVector          (Vector* x);
    virtual void*       map                 (Vector* x);
    virtual void        unmap               (Vector* x, void* ptr, bool modified);
    virtual void        set                 (Vector* x, float y);
    virtual void        copy                (Vector* x, Vector* y);
    virtual void        read                (void* ptr, Vector* x);
    virtual void        write               (Vector* x, const void* ptr);

    // Matrix/vector math.

    virtual void        calc_Px             (Vector* Px, PoissonMatrix P, Vector* x);
    virtual void        calc_PTW2x          (Vector* PTW2x, PoissonMatrix P, Vector* w2, Vector* x);
    virtual void        calc_Ax_xAx         (Vector* Ax, Vector* xAx, PoissonMatrix P, Vector* w2, Vector* x);
    virtual void        calc_axpy           (Vector* axpy, Vec3f a, Vector* x, Vector* y);
    virtual void        calc_xdoty          (Vector* xdoty, Vector* x, Vector* y);
    virtual void        calc_r_rz           (Vector* r, Vector* rz, Vector* Ap, Vector* rz2, Vector* pAp);
    virtual void        calc_x_p            (Vector* x, Vector* p, Vector* r, Vector* rz, Vector* rz2, Vector* pAp);
    virtual void        calc_w2             (Vector* w2, Vector* e, float reg);

    // Tone mapping.

    virtual void        tonemapSRGB         (Vector* out, Vector* in, int idx, float scale, float bias);
    virtual void        tonemapLinear       (Vector* out, Vector* in, int idx, float scaleMin, float scaleMax, bool hasNegative);

    // Timing.

    virtual Timer*      allocTimer          (void);
    virtual void        freeTimer           (Timer* timer);
    virtual void        beginTimer          (Timer* timer);
    virtual float       endTimer            (Timer* timer);

private:
    dim3                gridDim             (int totalThreadsX, int totalThreadsY = 1);

private:
                        BackendCUDA         (const BackendCUDA&); // forbidden
    BackendCUDA&        operator=           (const BackendCUDA&); // forbidden

private:
    int                 m_maxGridWidth;
    dim3                m_blockDim;
};

//------------------------------------------------------------------------
}
