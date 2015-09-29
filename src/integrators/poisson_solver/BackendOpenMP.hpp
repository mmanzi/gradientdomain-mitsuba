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

namespace poisson
{
//------------------------------------------------------------------------
// OpenMP-accelerated Solver backend.
//------------------------------------------------------------------------

class BackendOpenMP : public Backend
{
public:
                        BackendOpenMP       (void);
    virtual             ~BackendOpenMP      (void);

    // Vector manipulation.

    virtual void        set                 (Vector* x, float y);
    virtual void        copy                (Vector* x, Vector* y);

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

private:
                        BackendOpenMP       (const BackendOpenMP&); // forbidden
    BackendOpenMP&      operator=           (const BackendOpenMP&); // forbidden
};

//------------------------------------------------------------------------
}
