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
 *   - Ability to disable CUDA.
 *   - Custom logging callbacks.
 *   - Input and output images directly through memory.
 */
#define HAS_CUDA 0 // Marco: Set to one if we have CUDA support. Need to adapt SCons files.

#include "Solver.hpp"
#include "BackendOpenMP.hpp"
#if HAS_CUDA
#include "BackendCUDA.hpp"
#endif
//#include "ImagePfmIO.hpp"
//#include "lodepng.h"
#include <stdio.h>
#include <stdarg.h>

using namespace poisson;

//------------------------------------------------------------------------

Solver::Params::Params(void)
{
    setDefaults();
}

//------------------------------------------------------------------------

void Solver::Params::setDefaults(void)
{
    dxPFM           = "";
    dyPFM           = "";
    throughputPFM   = "";
    directPFM       = "";
    referencePFM    = "";
    alpha           = 0.2f; // importance ratio for reconstruction of primal image versus gradients

    indirectPFM     = "";
    finalPFM        = "";

    dxPNG           = "";
    dyPNG           = "";
    throughputPNG   = "";
    directPNG       = "";
    referencePNG    = "";
    indirectPNG     = "";
    finalPNG        = "";
    brightness      = 1.0f;

    verbose         = false;
    display         = false;
    backend         = "Auto";
    cudaDevice      = -1;

	logFunc = LogFunction([](const std::string& message) { printf(message.c_str()); });

    setConfigPreset("L1D");
}

//------------------------------------------------------------------------

bool Solver::Params::setConfigPreset(const char* preset)
{
    // Base config.

    irlsIterMax = 1;
    irlsRegInit = 0.0f;
    irlsRegIter = 0.0f;
    cgIterMax   = 1;
    cgIterCheck = 100;
    cgPrecond   = false;
    cgTolerance = 0.0f;

    // "L1D" = L1 default config
    // ~1s for 1280x720 on GTX980, L1 error lower than MATLAB reference.

    if (strcmp(preset, "L1D") == 0)
    {
        irlsIterMax = 20;
        irlsRegInit = 0.05f;
        irlsRegIter = 0.5f;
        cgIterMax   = 50;
        return true;
    }

    // "L1Q" = L1 high-quality config
    // ~50s for 1280x720 on GTX980, L1 error as low as possible.

    if (strcmp(preset, "L1Q") == 0)
    {
        irlsIterMax = 64;
        irlsRegInit = 1.0f;
        irlsRegIter = 0.7f;
        cgIterMax   = 1000;
        return true;
    }

    // "L1L" = L1 legacy config
    // ~89s for 1280x720 on GTX980, L1 error equal to MATLAB reference.

    if (strcmp(preset, "L1L") == 0)
    {
        irlsIterMax = 7;
        irlsRegInit = 1.0e-4f;
        irlsRegIter = 1.0e-1f;
        cgIterMax   = 20000;
        cgTolerance = 1.0e-20f;
        return true;
    }

    // "L2D" = L2 default config
    // ~0.1s for 1280x720 on GTX980, L2 error equal to MATLAB reference.

    if (strcmp(preset, "L2D") == 0)
    {
        irlsIterMax = 1;
        irlsRegInit = 0.0f;
        irlsRegIter = 0.0f;
        cgIterMax   = 50;
        return true;
    }

    // "L2Q" = L2 high-quality config
    // ~0.5s for 1280x720 on GTX980, L2 error as low as possible.

    if (strcmp(preset, "L2Q") == 0)
    {
        irlsIterMax = 1;
        irlsRegInit = 0.0f;
        irlsRegIter = 0.0f;
        cgIterMax   = 500;
        return true;
    }

    return false;
}

//------------------------------------------------------------------------

void Solver::Params::sanitize(void)
{
    alpha       = max(alpha, 0.0f);
    brightness  = max(brightness, 0.0f);
    irlsIterMax = max(irlsIterMax, 1);
    irlsRegInit = max(irlsRegInit, 0.0f);
    irlsRegIter = max(irlsRegIter, 0.0f);
    cgIterMax   = max(cgIterMax, 1);
    cgIterCheck = max(cgIterCheck, 1);
    cgTolerance = max(cgTolerance, 0.0f);
}

//------------------------------------------------------------------------

void Solver::Params::setLogFunction(LogFunction function)
{
	logFunc = function;
}

//------------------------------------------------------------------------

Solver::Solver(const Params& params)
:   m_params        (params),

    m_size          (-1, -1),
    m_dx            (NULL),
    m_dy            (NULL),
    m_throughput    (NULL),
    m_direct        (NULL),
    m_reference     (NULL),

    m_backend       (NULL),
    m_b             (NULL),
    m_e             (NULL),
    m_w2            (NULL),
    m_x             (NULL),
    m_r             (NULL),
    m_z             (NULL),
    m_p             (NULL),
    m_Ap            (NULL),
    m_rr            (NULL),
    m_rz            (NULL),
    m_rz2           (NULL),
    m_pAp           (NULL),
    m_tonemapped    (NULL),
    m_timerTotal    (NULL),
    m_timerIter     (NULL)
{
    m_params.sanitize();
}

/* Marco: Make this Mitsuba friendlier. */
void Solver::importImagesMTS(float *dx, float *dy, float *tp, float *direct, int width, int height){

	m_size = Vec2i(width, height);

	m_dx = reinterpret_cast<Vec3f*>(dx);
	m_dy = reinterpret_cast<Vec3f*>(dy);
	m_throughput = reinterpret_cast<Vec3f*>(tp);
	m_direct = reinterpret_cast<Vec3f*>(direct);
}

//------------------------------------------------------------------------

Solver::~Solver(void)
{
    if (m_backend)
    {
        m_backend->freeVector(m_b);
        m_backend->freeVector(m_e);
        m_backend->freeVector(m_w2);
        m_backend->freeVector(m_x);
        m_backend->freeVector(m_r);
        m_backend->freeVector(m_z);
        m_backend->freeVector(m_p);
        m_backend->freeVector(m_Ap);
        m_backend->freeVector(m_rr);
        m_backend->freeVector(m_rz);
        m_backend->freeVector(m_rz2);
        m_backend->freeVector(m_pAp);
        m_backend->freeVector(m_tonemapped);
        m_backend->freeTimer(m_timerTotal);
        m_backend->freeTimer(m_timerIter);
        delete m_backend;
    }
}

//------------------------------------------------------------------------

void Solver::setupBackend(void)
{
    assert(m_dx && m_dy);
    assert(m_size.x > 0 && m_size.y > 0);

    // CUDA backend?

#if HAS_CUDA
    if (!m_backend && (m_params.backend == "CUDA" || m_params.backend == "Auto"))
    {
        int device = m_params.cudaDevice;
        if (device < 0)
            device = BackendCUDA::chooseDevice();

        if (m_params.backend == "CUDA" || device != -1)
            m_backend = new BackendCUDA(device);
    }
#endif
    // OpenMP backend?

    if (!m_backend && (m_params.backend == "OpenMP" || m_params.backend == "Auto"))
    {
        log("Using OpenMP backend\n");
        m_backend = new BackendOpenMP;
    }

    // Naive backend?

    if (!m_backend && (m_params.backend == "Naive" || m_params.backend == "Auto"))
    {
        log("Using naive CPU backend\n");
        m_backend = new Backend;
    }

    // Otherwise => error.

    if (!m_backend)
        fail("Invalid backend specified '%s'!", m_params.backend);

    // Allocate backend objects.

    int n           = m_size.x * m_size.y;
    m_b             = m_backend->allocVector(n * 3, sizeof(Vec3f));
    m_e             = m_backend->allocVector(n * 3, sizeof(Vec3f));
    m_w2            = m_backend->allocVector(n * 3, sizeof(float));
    m_x             = m_backend->allocVector(n,     sizeof(Vec3f));
    m_r             = m_backend->allocVector(n,     sizeof(Vec3f));
    m_z             = m_backend->allocVector(n,     sizeof(Vec3f));
    m_p             = m_backend->allocVector(n,     sizeof(Vec3f));
    m_Ap            = m_backend->allocVector(n,     sizeof(Vec3f));
    m_rr            = m_backend->allocVector(1,     sizeof(Vec3f));
    m_rz            = m_backend->allocVector(1,     sizeof(Vec3f));
    m_rz2           = m_backend->allocVector(1,     sizeof(Vec3f));
    m_pAp           = m_backend->allocVector(1,     sizeof(Vec3f));
    m_tonemapped    = m_backend->allocVector(n,     sizeof(unsigned int));
    m_timerTotal    = m_backend->allocTimer();
    m_timerIter     = m_backend->allocTimer();

    // Initialize P.

    m_P.size.x  = m_size.x;
    m_P.size.y  = m_size.y;
    m_P.alpha   = (m_throughput) ? m_params.alpha : 0.0f;

    // Initialize b = vertcat(throughput_rgb * alpha, dx_rgb, dy_rgb)

    Vec3f* p_b = (Vec3f*)m_backend->map(m_b);
    for (int i = 0; i < n; i++)
    {
        p_b[n * 0 + i] = (m_throughput) ? m_throughput[i] * m_P.alpha : 0.0f;
        p_b[n * 1 + i] = m_dx[i];
        p_b[n * 2 + i] = m_dy[i];
    }
    m_backend->unmap(m_b, (void*)p_b, true);

    // Initialize x = throughput_rgb.

    if (m_throughput)
        m_backend->write(m_x, m_throughput);
    else
        m_backend->set(m_x, 0.0f);
}

//------------------------------------------------------------------------
// Solve the indirect light image by minimizing the L1 (or L2) error.
//
// x = min_x L1(b - P*x),
// where
//      x = Nx1 vector representing the solution (elements are RGB triplets)
//      N = total number of pixels
//      b = (N*3)x1 vector representing the concatenation of throughput*alpha, dx, and dy.
//      P = (N*3)xN Poisson matrix that computes the equivalent of b from x.
//      L1() = L1 norm.
//
// We use Iteratively Reweighted Least Squares method (IRLS) to convert
// the L1 optimization problem to a series of L2 optimization problems.
// http://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
//
// x = min_x L2(W*b - W*P*x),
// where
//      W = (N*3)x(N*3) diagonal weight matrix, adjusted after each iteration
//
// We rewrite the L2 optimization problem using the normal equations.
// http://en.wikipedia.org/wiki/Linear_least_squares_%28mathematics%29
//
// (W*P)'*W*P*x = (W*P)'*W*b
// (P'*W^2*P)*x = (P'*W^2*b)
// A*x = bb,
// where
//      A  = P'*W^2*P = P'*diag(w2)*P = NxN matrix
//      bb = P'*W^2*b = P'*diag(w2)*b = Nx1 vector
//      w2 = (N*3)x1 vector of squared weights
//
// We then solve the normal equations using the Conjugate Gradient method (CG)
// or the Preconditioned Conjugate Gradient method (PCG).
// http://en.wikipedia.org/wiki/Conjugate_gradient_method

void Solver::solveIndirect(void)
{
    assert(m_dx && m_dy && m_backend && m_x);
    assert(m_size.x > 0 && m_size.y > 0);
    m_backend->beginTimer(m_timerTotal);

    // Loop over IRLS iterations.

    for (int irlsIter = 0; irlsIter < m_params.irlsIterMax; irlsIter++)
    {
        // Calculate approximation error.

        m_backend->calc_Px(m_e, m_P, m_x);          // e = P*x
        m_backend->calc_axpy(m_e, -1.0f, m_e, m_b); // e = b - P*x

        // Adjust weights.

        if (irlsIter == 0)
            m_backend->set(m_w2, 1.0f);
        else
        {
            float reg = m_params.irlsRegInit * powf(m_params.irlsRegIter, (float)(irlsIter - 1));
            m_backend->calc_w2(m_w2, m_e, reg);     // w2 = coef / (length(e) + reg)
        }

        // Initialize conjugate gradient.

        Backend::Vector* rz = m_rz;
        Backend::Vector* rz2 = m_rz2;
        m_backend->calc_PTW2x(m_r, m_P, m_w2, m_e); // r = P'*diag(w2)*(b - P*x)
        m_backend->calc_xdoty(rz, m_r, m_r);        // rz = r'*r
        m_backend->copy(m_p, m_r);                  // p  = r

        for (int cgIter = 0;; cgIter++)
        {
            // Display status every N iterations.

            if (cgIter % m_params.cgIterCheck == 0 || cgIter == m_params.cgIterMax)
            {
                // Calculate weighted L2 error.

                float errL2W;
                if (!m_params.cgPrecond || cgIter == 0)
                {
                    const Vec3f* p_rz = (const Vec3f*)m_backend->map(rz);
                    errL2W = p_rz->x + p_rz->y + p_rz->z;
                    m_backend->unmap(rz, (void*)p_rz, false);
                }
                else
                {
                    m_backend->calc_xdoty(m_rr, m_r, m_r);
                    const Vec3f* p_rr = (const Vec3f*)m_backend->map(m_rr);
                    errL2W = p_rr->x + p_rr->y + p_rr->z;
                    m_backend->unmap(m_rr, (void*)p_rr, false);
                }

                // Print status.

                std::string status = sprintf("IRLS = %-3d/ %d, CG = %-4d/ %d, errL2W = %9.2e",
                    irlsIter, m_params.irlsIterMax, cgIter, m_params.cgIterMax, errL2W);

                if (m_params.verbose)
                    log("%s", status.c_str());

                // Done?

                if (cgIter == m_params.cgIterMax || errL2W <= m_params.cgTolerance)
                {
                    if (m_params.verbose)
                        log("\n");
                    break;
                }

                // Display current solution.

                if (m_params.display)
                {
                    m_backend->tonemapSRGB(m_tonemapped, m_x, 0, m_params.brightness, 0.0f);
                    //m_backend->tonemapLinear(m_tonemapped, m_w2, 0, 0.0f, FLT_MAX, false);
                    display(sprintf("Poisson solver: %s", status.c_str()).c_str());
                }

                // Begin iteration timer.

                if (m_params.verbose)
                    m_backend->beginTimer(m_timerIter);
            }

            // Regular conjugate gradient iteration.

            if (!m_params.cgPrecond)
            {
                swap(rz, rz2);                                          // rz2 = rz
                m_backend->calc_Ax_xAx(m_Ap, m_pAp, m_P, m_w2, m_p);    // Ap = A*p, pAp = p'*A*p
                m_backend->calc_r_rz(m_r, rz, m_Ap, rz2, m_pAp);        // r -= Ap*(rz2/pAp), rz  = r'*r
                m_backend->calc_x_p(m_x, m_p, m_r, rz, rz2, m_pAp);     // x += p*(rz2/pAp), p = r + p*(rz/rz2)
            }

            // Preconditioned conjugate gradient iteration.

            else
            {
                if (cgIter == 0)
                {
                    m_backend->calc_MIx(m_z, m_P, m_w2, m_r);           // z  = inv(M)*r
                    m_backend->calc_xdoty(rz, m_r, m_z);                // rz = r'*z
                    m_backend->copy(m_p, m_z);                          // p  = z
                }

                swap(rz, rz2);                                          // rz2 = rz
                m_backend->calc_Ax_xAx(m_Ap, m_pAp, m_P, m_w2, m_p);    // Ap = A*p, pAp = p'*A*p
                m_backend->calc_r_rz(m_r, rz, m_Ap, rz2, m_pAp);        // r -= Ap*(rz2/pAp)
                m_backend->calc_MIx(m_z, m_P, m_w2, m_r);               // z = inv(M)*r
                m_backend->calc_xdoty(rz, m_r, m_z);                    // rz = r'*z
                m_backend->calc_x_p(m_x, m_p, m_r, rz, rz2, m_pAp);     // x += p*(rz2/pAp), p = r + p*(rz/rz2)
            }

            // Print iteration time every N iterations.

            if (m_params.verbose && cgIter % m_params.cgIterCheck == 0)
                log(", %-5.2fms/iter\n", m_backend->endTimer(m_timerIter) * 1.0e3f);
        }
    }

    // Print total execution time.

    log("Execution time = %.2f s\n", m_backend->endTimer(m_timerTotal));

    // Display final result.

    if (m_params.display)
    {
        m_backend->tonemapSRGB(m_tonemapped, m_x, 0, m_params.brightness, 0.0f);
        display("Poisson solver: Done");
    }
}

void Solver::evaluateMetricsMTS(float *err, float &errL1, float &errL2)
{
	assert(m_dx && m_dy && m_backend && m_x);
	assert(m_size.x > 0 && m_size.y > 0);
	int n = m_size.x * m_size.y;

	// Calculate L1 and L2 error.
	{
		m_backend->calc_Px(m_e, m_P, m_x);          // e = P*x
		m_backend->calc_axpy(m_e, -1.0f, m_e, m_b); // e = b - P*x
		const Vec3f* p_e = (const Vec3f*)m_backend->map(m_e);

		errL1 = 0.0f;
		errL2 = 0.0f;
		for (int i = 0; i < n * 3; i++)
		{
			errL1 += length(p_e[i]);
			errL2 += lenSqr(p_e[i]);
		}
		errL1 /= (float)(n * 3);
		errL2 /= (float)(n * 3);

		for (int i = 0; i < m_size.x*m_size.y; i++){
			err[3 * i] = p_e[i].x;
			err[3 * i + 1] = p_e[i].y;
			err[3 * i + 2] = p_e[i].z;
		}

		m_backend->unmap(m_e, (void*)p_e, false);
	}
}
void Solver::exportImagesMTS(float *rec)
{

	// Indirect.
	/*{
		const Vec3f* p_x = (const Vec3f*)m_backend->map(m_x);
		//exportImage(p_x, m_params.indirectPFM.c_str(), m_params.indirectPNG.c_str(), false, m_z);

		for (int i = 0; i < m_size.x*m_size.y; i++){
			rec[3 * i]		= p_x[i].x;
			rec[3 * i + 1]  = p_x[i].y;
			rec[3 * i + 2]  = p_x[i].z;
		}

		m_backend->unmap(m_x, (void*)p_x, false);
	}*/

	// Final.
	{
		if (!m_direct)
			m_backend->copy(m_r, m_x);
		else
		{
			m_backend->write(m_r, m_direct);
			m_backend->calc_axpy(m_r, 1.0f, m_r, m_x);
		}

		const Vec3f* p_r = (const Vec3f*)m_backend->map(m_r);
		//exportImage(p_r, m_params.finalPFM.c_str(), m_params.finalPNG.c_str(), false, m_z);

		for (int i = 0; i < m_size.x*m_size.y; i++){
			rec[3 * i] = p_r[i].x;
			rec[3 * i + 1] = p_r[i].y;
			rec[3 * i + 2] = p_r[i].z;
		}

		m_backend->unmap(m_r, (void*)p_r, false);
	}

	
}
//------------------------------------------------------------------------

void Solver::display(const char* title)
{
   // m_debugWindow.setTitle(title);
   // m_debugWindow.setSize(m_size.x, m_size.y);
   // m_backend->read(m_debugWindow.getPixelPtr(), m_tonemapped);
   // m_debugWindow.display();
}

//------------------------------------------------------------------------

void Solver::log(const std::string& message)
{
	m_params.logFunc(message);
}

//------------------------------------------------------------------------

void Solver::log(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    int len = _vscprintf(fmt, args);
    std::string str;
    str.resize(len);
    vsprintf_s((char*)str.c_str(), len + 1, fmt, args);
    va_end(args);

	log(str);
}

//------------------------------------------------------------------------
