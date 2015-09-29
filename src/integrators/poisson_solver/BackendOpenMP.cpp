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
/*
 * This file contains modifications by Markus Kettunen <markus.kettunen@aalto.fi>.
 *
 * Changes: Use the physical CPU core count due to suspecting problems with OpenMP and hyperthreading.
 */
/*
 * The implementation of GetLogicalProcessorInformation is based on the sample code available in MSDN under the following license:
 * 
 * Microsoft Limited Public License (Ms-LPL)
 * 
 * This license governs use of the accompanying software. If you use the software, you accept this license. If you do not accept the license, do not use the software.
 * 
 * 1. Definitions
 * 
 * The terms "reproduce," "reproduction," "derivative works," and "distribution" have the same meaning here as under U.S. copyright law. A "contribution" is the original software, or any additions or changes to the software. A "contributor" is any person that distributes its contribution under this license. "Licensed patents" are a contributor's patent claims that read directly on its contribution.
 * 
 * 2. Grant of Rights
 * 
 * (A) Copyright Grant- Subject to the terms of this license, including the license conditions and limitations in section 3, each contributor grants you a non-exclusive, worldwide, royalty-free copyright license to reproduce its contribution, prepare derivative works of its contribution, and distribute its contribution or any derivative works that you create.
 * (B) Patent Grant- Subject to the terms of this license, including the license conditions and limitations in section 3, each contributor grants you a non-exclusive, worldwide, royalty-free license under its licensed patents to make, have made, use, sell, offer for sale, import, and/or otherwise dispose of its contribution in the software or derivative works of the contribution in the software.
 * 
 * 3. Conditions and Limitations
 * 
 * (A) No Trademark License- This license does not grant you rights to use any contributors' name, logo, or trademarks.
 * (B) If you bring a patent claim against any contributor over patents that you claim are infringed by the software, your patent license from such contributor to the software ends automatically.
 * (C) If you distribute any portion of the software, you must retain all copyright, patent, trademark, and attribution notices that are present in the software.
 * (D) If you distribute any portion of the software in source code form, you may do so only under this license by including a complete copy of this license with your distribution. If you distribute any portion of the software in compiled or object code form, you may only do so under a license that complies with this license.
 * (E) The software is licensed "as-is." You bear the risk of using it. The contributors give no express warranties, guarantees, or conditions. You may have additional consumer rights under your local laws which this license cannot change. To the extent permitted under your local laws, the contributors exclude the implied warranties of merchantability, fitness for a particular purpose and non-infringement.
 * 
 * 4. (F) Platform Limitation- The licenses granted in sections 2(A) & 2(B) extend only to the software or derivative works that you create that run on a Microsoft Windows operating system product.
 */
#include "BackendOpenMP.hpp"
#include <omp.h>


// Markus: Required for getPhysicalCoreCount. For some reason the OpenMP reconstruction gets terribly slow if all logical cores are used. Does this fix the issue also on non-hyperthreading CPUs?
#ifdef _WIN32

#define NOMINMAX
#include <windows.h>

typedef BOOL (WINAPI *LPFN_GLPI)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);

#endif


using namespace poisson;


// Markus: Workaround for OpenMP slowness if all cores are used.
static int getPhysicalCoreCount() {
    // TODO: Use Boost 1.56 and boost::thread::physical_concurrency() for all platforms.
#ifndef _WIN32
    return 1;
#else
    static int coreCount = -1;

    if(coreCount < 0)
    {
        // Modified from MSDN documentation of GetLogicalProcessorInformation. The original sample code is licensed under Ms-LPL. See above.
        LPFN_GLPI glpi;
        BOOL done = FALSE;
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = NULL;
        DWORD returnLength = 0;
        DWORD processorCoreCount = 0;
        DWORD byteOffset = 0;

        while (!done)
        {
            DWORD rc = GetLogicalProcessorInformation(buffer, &returnLength);

            if (!rc) 
            {
                if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) 
                {
                    if (buffer)
                        free(buffer);

                    buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(returnLength);

                    if (buffer == NULL) 
                    {
                        coreCount = 1;
                        return coreCount;
                    }
                } 
                else 
                {
                    coreCount = 1;
                    return coreCount;
                }
            } 
            else
            {
                done = TRUE;
            }
        }

        ptr = buffer;
        while (byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= returnLength) 
        {
            if (ptr->Relationship == RelationProcessorCore)
                processorCoreCount++;

            byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            ptr++;
        }
    
        free(buffer);

        coreCount = processorCoreCount;
    }

    return coreCount;
#endif
}


//------------------------------------------------------------------------

BackendOpenMP::BackendOpenMP(void)
{
}

//------------------------------------------------------------------------

BackendOpenMP::~BackendOpenMP(void)
{
}

//------------------------------------------------------------------------

void BackendOpenMP::set(Vector* x, float y)
{
    assert(x && x->bytesPerElem % sizeof(float) == 0);

    float*  p_x = (float*)x->ptr;
    int     n   = (int)(x->bytesTotal / sizeof(float));

    omp_set_num_threads(getPhysicalCoreCount());
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        p_x[i] = y;
}

//------------------------------------------------------------------------

void BackendOpenMP::copy(Vector* x, Vector* y)
{
    assert(x && y && x->bytesTotal == y->bytesTotal);

    float*          p_x = (float*)x->ptr;
    const float*    p_y = (const float*)y->ptr;
    int             n = (int)(x->bytesTotal / sizeof(float));

    omp_set_num_threads(getPhysicalCoreCount());
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        p_x[i] = p_y[i];
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_Px(Vector* Px, PoissonMatrix P, Vector* x)
{
    assert(Px && Px->numElems == P.size.x * P.size.y * 3 && Px->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);
    assert(x && x->numElems == P.size.x * P.size.y && x->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_Px    = (Vec3f*)Px->ptr;
    const Vec3f*    p_x     = (const Vec3f*)x->ptr;
    int             n       = P.size.x * P.size.y;

    omp_set_num_threads(getPhysicalCoreCount());
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        int xx = i % P.size.x;
        int yy = i / P.size.x;

        Vec3f xi = p_x[i];
        p_Px[n * 0 + i] = xi * P.alpha;
        p_Px[n * 1 + i] = (xx != P.size.x - 1) ? p_x[i + 1] - xi : 0.0f;
        p_Px[n * 2 + i] = (yy != P.size.y - 1) ? p_x[i + P.size.x] - xi : 0.0f;
    }
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_PTW2x(Vector* PTW2x, PoissonMatrix P, Vector* w2, Vector* x)
{
    assert(PTW2x && PTW2x->numElems == P.size.x * P.size.y && PTW2x->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);
    assert(w2 && w2->numElems == P.size.x * P.size.y * 3 && w2->bytesPerElem == sizeof(float));
    assert(x && x->numElems == P.size.x * P.size.y * 3 && x->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_PTW2x = (Vec3f*)PTW2x->ptr;
    const float*    p_w2    = (const float*)w2->ptr;
    const Vec3f*    p_x     = (const Vec3f*)x->ptr;
    int             n       = P.size.x * P.size.y;

    omp_set_num_threads(getPhysicalCoreCount());
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        int xx = i % P.size.x;
        int yy = i / P.size.x;

        Vec3f PTW2xi = p_w2[n * 0 + i] * p_x[n * 0 + i] * P.alpha;
        if (xx != 0)            PTW2xi += p_w2[n * 1 + i - 1]           * p_x[n * 1 + i - 1];
        if (xx != P.size.x - 1) PTW2xi -= p_w2[n * 1 + i]               * p_x[n * 1 + i];
        if (yy != 0)            PTW2xi += p_w2[n * 2 + i - P.size.x]    * p_x[n * 2 + i - P.size.x];
        if (yy != P.size.y - 1) PTW2xi -= p_w2[n * 2 + i]               * p_x[n * 2 + i];
        p_PTW2x[i] = PTW2xi;
    }
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_Ax_xAx(Vector* Ax, Vector* xAx, PoissonMatrix P, Vector* w2, Vector* x)
{
    assert(Ax && Ax->numElems == P.size.x * P.size.y && Ax->bytesPerElem == sizeof(Vec3f));
    assert(xAx && xAx->numElems == 1 && xAx->bytesPerElem == sizeof(Vec3f));
    assert(P.size.x >= 0 && P.size.y >= 0);
    assert(w2 && w2->numElems == P.size.x * P.size.y * 3 && w2->bytesPerElem == sizeof(float));
    assert(x && x->numElems == P.size.x * P.size.y && x->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_Ax        = (Vec3f*)Ax->ptr;
    Vec3f*          p_xAx       = (Vec3f*)xAx->ptr;
    const float*    p_w2        = (const float*)w2->ptr;
    const Vec3f*    p_x         = (const Vec3f*)x->ptr;
    int             n           = P.size.x * P.size.y;
    float           alphaSqr    = P.alpha * P.alpha;

    float xAx_0 = 0.0f;
    float xAx_1 = 0.0f;
    float xAx_2 = 0.0f;

    omp_set_num_threads(getPhysicalCoreCount());
#pragma omp parallel for reduction(+ : xAx_0, xAx_1, xAx_2)
    for (int i = 0; i < n; i++)
    {
        int xx = i % P.size.x;
        int yy = i / P.size.x;

        Vec3f xi = p_x[i];
        Vec3f Axi = p_w2[n * 0 + i] * xi * alphaSqr;
        if (xx != 0)            Axi += p_w2[n * 1 + i - 1]          * (xi - p_x[i - 1]);
        if (xx != P.size.x - 1) Axi += p_w2[n * 1 + i]              * (xi - p_x[i + 1]);
        if (yy != 0)            Axi += p_w2[n * 2 + i - P.size.x]   * (xi - p_x[i - P.size.x]);
        if (yy != P.size.y - 1) Axi += p_w2[n * 2 + i]              * (xi - p_x[i + P.size.x]);
        p_Ax[i] = Axi;

        Vec3f xAxi = xi * Axi;
        xAx_0 += xAxi[0];
        xAx_1 += xAxi[1];
        xAx_2 += xAxi[2];
    }

    p_xAx[0] = Vec3f(xAx_0, xAx_1, xAx_2);
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_axpy(Vector* axpy, Vec3f a, Vector* x, Vector* y)
{
    assert(axpy && axpy->numElems == x->numElems && axpy->bytesPerElem == sizeof(Vec3f));
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(y && y->numElems == x->numElems && y->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_axpy  = (Vec3f*)axpy->ptr;
    const Vec3f*    p_x     = (const Vec3f*)x->ptr;
    const Vec3f*    p_y     = (const Vec3f*)y->ptr;
    int             n       = x->numElems;

    omp_set_num_threads(getPhysicalCoreCount());
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        p_axpy[i] = a * p_x[i] + p_y[i];
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_xdoty(Vector* xdoty, Vector* x, Vector* y)
{
    assert(xdoty && xdoty->numElems == 1 && xdoty->bytesPerElem == sizeof(Vec3f));
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(y && y->numElems == x->numElems && y->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_xdoty = (Vec3f*)xdoty->ptr;
    const Vec3f*    p_x     = (const Vec3f*)x->ptr;
    const Vec3f*    p_y     = (const Vec3f*)y->ptr;
    int             n       = x->numElems;

    float xdoty_0 = 0.0f;
    float xdoty_1 = 0.0f;
    float xdoty_2 = 0.0f;

    omp_set_num_threads(getPhysicalCoreCount());
#pragma omp parallel for reduction(+ : xdoty_0, xdoty_1, xdoty_2)
    for (int i = 0; i < n; i++)
    {
        Vec3f xdotyi = p_x[i] * p_y[i];
        xdoty_0 += xdotyi[0];
        xdoty_1 += xdotyi[1];
        xdoty_2 += xdotyi[2];
    }

    p_xdoty[0] = Vec3f(xdoty_0, xdoty_1, xdoty_2);
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_r_rz(Vector* r, Vector* rz, Vector* Ap, Vector* rz2, Vector* pAp)
{
    assert(r && r->bytesPerElem == sizeof(Vec3f));
    assert(rz && rz->numElems == 1 && rz->bytesPerElem == sizeof(Vec3f));
    assert(Ap && Ap->numElems == r->numElems && Ap->bytesPerElem == sizeof(Vec3f));
    assert(rz2 && rz2->numElems == 1 && rz2->bytesPerElem == sizeof(Vec3f));
    assert(pAp && pAp->numElems == 1 && pAp->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_r     = (Vec3f*)r->ptr;
    Vec3f*          p_rz    = (Vec3f*)rz->ptr;
    const Vec3f*    p_Ap    = (const Vec3f*)Ap->ptr;
    const Vec3f*    p_rz2   = (const Vec3f*)rz2->ptr;
    const Vec3f*    p_pAp   = (const Vec3f*)pAp->ptr;
    int             n       = r->numElems;

    Vec3f a = p_rz2[0] / max(p_pAp[0], Vec3f(FLT_MIN));
    float rz_0 = 0.0f;
    float rz_1 = 0.0f;
    float rz_2 = 0.0f;

    omp_set_num_threads(getPhysicalCoreCount());
#pragma omp parallel for reduction(+ : rz_0, rz_1, rz_2)
    for (int i = 0; i < n; i++)
    {
        Vec3f ri = p_r[i] - p_Ap[i] * a;
        p_r[i] = ri;

        Vec3f rzi = ri * ri;
        rz_0 += rzi[0];
        rz_1 += rzi[1];
        rz_2 += rzi[2];
    }

    p_rz[0] = Vec3f(rz_0, rz_1, rz_2);
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_x_p(Vector* x, Vector* p, Vector* r, Vector* rz, Vector* rz2, Vector* pAp)
{
    assert(x && x->bytesPerElem == sizeof(Vec3f));
    assert(p && p->numElems == x->numElems && p->bytesPerElem == sizeof(Vec3f));
    assert(r && r->numElems == x->numElems && r->bytesPerElem == sizeof(Vec3f));
    assert(rz && rz->numElems == 1 && rz->bytesPerElem == sizeof(Vec3f));
    assert(rz2 && rz2->numElems == 1 && rz2->bytesPerElem == sizeof(Vec3f));
    assert(pAp && pAp->numElems == 1 && pAp->bytesPerElem == sizeof(Vec3f));

    Vec3f*          p_x     = (Vec3f*)x->ptr;
    Vec3f*          p_p     = (Vec3f*)p->ptr;
    const Vec3f*    p_r     = (const Vec3f*)r->ptr;
    const Vec3f*    p_rz    = (const Vec3f*)rz->ptr;
    const Vec3f*    p_rz2   = (const Vec3f*)rz2->ptr;
    const Vec3f*    p_pAp   = (const Vec3f*)pAp->ptr;
    int             n       = x->numElems;

    Vec3f a = p_rz2[0] / max(p_pAp[0], Vec3f(FLT_MIN));
    Vec3f b = p_rz[0] / max(p_rz2[0], Vec3f(FLT_MIN));

    omp_set_num_threads(getPhysicalCoreCount());
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        Vec3f pi = p_p[i];
        p_x[i] += pi * a;
        p_p[i] = p_r[i] + pi * b;
    }
}

//------------------------------------------------------------------------

void BackendOpenMP::calc_w2(Vector* w2, Vector* e, float reg)
{
    assert(w2 && w2->bytesPerElem == sizeof(float));
    assert(e && e->numElems == w2->numElems && e->bytesPerElem == sizeof(Vec3f));

    float*          p_w2    = (float*)w2->ptr;
    const Vec3f*    p_e     = (const Vec3f*)e->ptr;
    int             n       = w2->numElems;

    float w2sum = 0.0f;

    omp_set_num_threads(getPhysicalCoreCount());
#pragma omp parallel for reduction(+ : w2sum)
    for (int i = 0; i < n; i++)
    {
        float w2i = 1.0f / (length(p_e[i]) + reg);
        p_w2[i] = w2i;
        w2sum += w2i;
    }

    float coef = (float)n / w2sum; // normalize so that average(w2) = 1

    omp_set_num_threads(getPhysicalCoreCount());
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        p_w2[i] *= coef;
}

//------------------------------------------------------------------------

void BackendOpenMP::tonemapSRGB(Vector* out, Vector* in, int idx, float scale, float bias)
{
    assert(out && out->bytesPerElem == sizeof(unsigned int));
    assert(in && in->numElems % out->numElems == 0 && in->bytesPerElem == sizeof(Vec3f));

    unsigned int*   p_out   = (unsigned int*)out->ptr;
    const Vec3f*    p_in    = (const Vec3f*)in->ptr;

    omp_set_num_threads(getPhysicalCoreCount());
#pragma omp parallel for
    for (int i = 0; i < out->numElems; i++)
    {
        Vec3f color = p_in[i + idx * out->numElems];
        for (int c = 0; c < 3; c++)
        {
            float& t = color[c];
            t = t * scale + bias;
            t = (t <= 0.0031308f) ? 12.92f * t : 1.055f * powf(t, 1.0f / 2.4f) - 0.055f; // linear to sRGB
        }

        p_out[i] = 0xFF000000 |
            ((int)min(max(color.x * 255.0f + 0.5f, 0.0f), 255.0f) << 0) |
            ((int)min(max(color.y * 255.0f + 0.5f, 0.0f), 255.0f) << 8) |
            ((int)min(max(color.z * 255.0f + 0.5f, 0.0f), 255.0f) << 16);
    }
}

//------------------------------------------------------------------------
