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
 * This file contains modifications by Marco Manzi <manzi@iam.unibe.ch> and Markus Kettunen <markus.kettunen@aalto.fi>.
 *
 * Changes:
 *   - Custom logging callbacks.
 *   - Input and output images directly through memory.
 */
#pragma once
#include "Backend.hpp"
//#include "platform.h"
//#include "DebugWindow.hpp"

#include <functional>

namespace poisson
{
//------------------------------------------------------------------------
// Poisson solver main class.
//------------------------------------------------------------------------

class Solver
{
public:
    struct Params
    {
        // Input PFM files.

        std::string         dxPFM;
        std::string         dyPFM;
        std::string         throughputPFM;
        std::string         directPFM;
        std::string         referencePFM;
        float               alpha;

        // Output PFM files.

        std::string         indirectPFM;
        std::string         finalPFM;

        // Output PNG files.

        std::string         dxPNG;
        std::string         dyPNG;
        std::string         throughputPNG;
        std::string         directPNG;
        std::string         referencePNG;
        std::string         indirectPNG;
        std::string         finalPNG;
        float               brightness;

        // Other parameters.

        std::string         backend;        // "Auto", "CUDA", "OpenMP", "Naive".
        int                 cudaDevice;
        bool                verbose;
        bool                display;

        // Solver configuration.

        int                 irlsIterMax;
        float               irlsRegInit;
        float               irlsRegIter;
        int                 cgIterMax;
        int                 cgIterCheck;
        bool                cgPrecond;
        float               cgTolerance;

        // Debugging and information output.

        typedef std::function<void(const std::string&)> LogFunction;
        LogFunction         logFunc;

        // Methods.

                            Params          (void);
        void                setDefaults     (void);
        bool                setConfigPreset (const char* preset); // "L1D", "L1Q", "L1L", "L2D", "L2Q".
        void                sanitize        (void);

        void                setLogFunction  (LogFunction function);

    };

public:
                            Solver          (const Params& params);
                            ~Solver         (void);

    void					importImagesMTS(float *dx, float *dy, float *tp, float *direct, int width, int height);
    void                    setupBackend    (void);
    void                    solveIndirect   (void);
    void					evaluateMetricsMTS(float *err, float &errL1, float &errL2);
    void                    exportImagesMTS (float *rec);

private:
    void                    display         (const char* title);

private:
    void                    log             (const std::string& message);
    void                    log             (const char* fmt, ...);

private:
                            Solver          (const Solver&); // forbidden
    Solver&                 operator=       (const Solver&); // forbidden

private:
    Params                  m_params;
  //  DebugWindow             m_debugWindow;

    Vec2i                   m_size;
    Vec3f*                  m_dx;
    Vec3f*                  m_dy;
    Vec3f*                  m_throughput;
    Vec3f*                  m_direct;
    Vec3f*                  m_reference;

    Backend*                m_backend;
    Backend::PoissonMatrix  m_P;
    Backend::Vector*        m_b;
    Backend::Vector*        m_e;
    Backend::Vector*        m_w2;
    Backend::Vector*        m_x;
    Backend::Vector*        m_r;
    Backend::Vector*        m_z;
    Backend::Vector*        m_p;
    Backend::Vector*        m_Ap;
    Backend::Vector*        m_rr;
    Backend::Vector*        m_rz;
    Backend::Vector*        m_rz2;
    Backend::Vector*        m_pAp;
    Backend::Vector*        m_tonemapped;
    Backend::Timer*         m_timerTotal;
    Backend::Timer*         m_timerIter;
};

//------------------------------------------------------------------------
}
