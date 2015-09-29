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
#include "Defs.hpp"

namespace poisson
{
//------------------------------------------------------------------------
// Base class for Solver backends.
// Declares a number of virtual methods to be overriden by subclasses,
// and provides a naive CPU-based default implementation for each method.
//------------------------------------------------------------------------

class Backend
{
public:
    struct Vector
    {
        int             numElems;           // Number of elements (e.g. floats or Vec3fs) in the vector.
        size_t          bytesPerElem;       // Number of bytes consumed by one element.
        size_t          bytesTotal;         // Total number of bytes consumed by the vector.
        void*           ptr;                // Implementation-specific data pointer. Do not access directly!
    };

    struct PoissonMatrix                    // Parameters defining an implicit Poisson matrix (P) of size (N*3)xN, where N is the total number of pixels.
    {
        Vec2i           size;               // Dimensions of the pixel lattice.
        float           alpha;              // Optimization weight of the throughput image.
    };

    struct Timer
    {
        __int64         beginTicks;
    };

public:
                        Backend             (void);
    virtual             ~Backend            (void);

    // Vector manipulation.

    virtual Vector*     allocVector         (int numElems, size_t bytesPerElem);
    virtual void        freeVector          (Vector* x);
    virtual void*       map                 (Vector* x);
    virtual void        unmap               (Vector* x, void* ptr, bool modified);
    virtual void        set                 (Vector* x, float y);                                               // x = replicate(y)
    virtual void        copy                (Vector* x, Vector* y);                                             // x = y
    virtual void        read                (void* ptr, Vector* x);
    virtual void        write               (Vector* x, const void* ptr);

    // Matrix/vector math.

    virtual void        calc_Px             (Vector* Px, PoissonMatrix P, Vector* x);                           // Px = P*x
    virtual void        calc_PTW2x          (Vector* PTW2x, PoissonMatrix P, Vector* w2, Vector* x);            // PTW2x = P'*diag(w2)*x
    virtual void        calc_Ax_xAx         (Vector* Ax, Vector* xAx, PoissonMatrix P, Vector* w2, Vector* x);  // Ax = A*x, xAx = x'*A*x
    virtual void        calc_axpy           (Vector* axpy, Vec3f a, Vector* x, Vector* y);                      // axpy = a*x + y
    virtual void        calc_xdoty          (Vector* xdoty, Vector* x, Vector* y);                              // xdoty = x'*y (rgb)
    virtual void        calc_r_rz           (Vector* r, Vector* rz, Vector* Ap, Vector* rz2, Vector* pAp);      // r -= Ap*(rz2/pAp), rz = r'*r
    virtual void        calc_x_p            (Vector* x, Vector* p, Vector* r, Vector* rz, Vector* rz2, Vector* pAp); // x += p*(rz2/pAp), p = r + p*(rz/rz2)
    virtual void        calc_w2             (Vector* w2, Vector* e, float reg);                                 // w2 = coef / (length(e) + reg)

    // Preconditioning.

    virtual void        calc_MIx            (Vector* MIx, PoissonMatrix P, Vector* w2, Vector* x);              // Mx = inv(M)*x

    // Tone mapping.

    virtual void        tonemapSRGB         (Vector* out /*ABGR_8888*/, Vector* in, int idx, float scale, float bias);
    virtual void        tonemapLinear       (Vector* out /*ABGR_8888*/, Vector* in, int idx, float scaleMin, float scaleMax, bool hasNegative);

    // Timing.

    virtual Timer*      allocTimer          (void);
    virtual void        freeTimer           (Timer* timer);
    virtual void        beginTimer          (Timer* timer);
    virtual float       endTimer            (Timer* timer); // Returns the number of seconds elapsed between beginTimer() and endTimer().

private:
                        Backend             (const Backend&); // forbidden
    Backend&            operator=           (const Backend&); // forbidden

private:
    double              m_timerTicksToSecs;
};

//------------------------------------------------------------------------
}
