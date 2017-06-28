Gradient-Domain Path Tracing / Gradient-Domain Bidirectional Path Tracing
-------------------------------------------------------------------------

  This code extends Mitsuba 0.5.0 and implements the algorithms presented
  in papers "Gradient-Domain Path Tracing" by Kettunen et al. and
  "Gradient-Domain Bidirectional Path Tracing" by Manzi et al.

  The algorithms internally first render gradient images (differences of
  colors between neighboring pixels), in addition to the standard noisy
  color images. This is done by variations of standard unidirectional and
  bidirectional path tracing as described in the aforementioned papers. They
  then solve a screened Poisson problem to find the image that best matches
  the sampled gradients and colors. This typically results in much less
  noise.

  The algorithms use the same path sampling machinery as their
  corresponding standard methods, but in addition estimate the differences
  to neighboring pixels in a way that typically produces little noise. Using
  the same path sampling machinery means that if the corresponding
  non-gradient method is absolutely terrible for a given scene, making
  it gradient-domain probably won't be enough to save the day. But, as we
  demonstrate in the papers, assuming that you have a scene for which the
  basic method works, making it gradient-domain often saves very much time.

  As described in the papers, the L2 reconstructions are unbiased, but
  sometimes show annoying dipole artifacts. We recommend the slightly
  biased L1 reconstruction in most cases, as the L1 images are generally
  visually much nicer and the bias tends to go away rather quickly.


  By default, reconstructing the final images from the sampled data is
  done on the CPU for compatibility. We recommend using the provided CUDA
  reconstruction when possible. See the instructions below.


  The code was implemented and tested using Visual C++ 2013 (with update 4)
  and CUDA Toolkit 6.5 and 7.0. Linux and Mac OS X support might require
  more work.

  The integrator implementations are released under the same license
  as the rest of Mitsuba. The screened Poisson reconstruction code from
  NVIDIA is under the new BSD license. See the source code for details.


  Project home pages:
  Gradient-Domain Path Tracing:
  https://mediatech.aalto.fi/publications/graphics/GPT/

  Gradient-Domain Bidirectional Path Tracing:
  http://cgg.unibe.ch/publications/gradient-domain-bidirectional-path-tracing


  In case of problems/questions/comments don't hesitate to contact us
  directly: markus.kettunen@aalto.fi or manzi@iam.unibe.ch.


Features, Gradient-Domain Path Tracing (G-PT):
----------------------------------------------
  This implementation supports diffuse, specular and glossy materials. It
  also supports area and point lights, depth-of-field, pixel filters and
  low discrepancy samplers. There is experimental support for sub-surface
  scattering and motion blur.

  Note that this is still an experimental implementation of
  Gradient-Domain Path Tracing that has not been tested with all of
  Mitsuba's features. Notably there is no support yet for participating
  media or directional lights. Environment maps are supported, though.

  When running in the GUI, the implementation will first display the
  sampled color data. When rendering gets to 100%, it then reconstructs the
  final image with the given reconstruction method. The gradient and color
  buffers are written to disk, so for example NVIDIA's screened Poisson
  reconstruction tool may be used for playing around with reconstruction
  parameters at will. For timing the method, note that if multiple render
  jobs are queued in Mitsuba, Mitsuba will start the next render job
  without waiting for the reconstruction to finish, slowing it down.

  This implementation does not yet support the 'hide emitters' option in
  Mitsuba, even though it is displayed in the GUI!


Features, Gradient-Domain Bidirectional Path Tracing (G-BDPT):
--------------------------------------------------------------
  This implementation supports diffuse, specular and glossy materials. It
  also supports area and point lights, depth-of-field, motion blur and low
  discrepancy samplers. Note that currently only the box pixel filter is
  supported. When rendering has finished, the implementation will solve and
  show the L1 reconstruction in the GUI. However, the L2 reconstruction,
  the primal image and the gradient images are also written to disk.

  Note that it is still an experimental implementation that hasn't been
  tested with all of Mitsuba's features. Notably there is no support yet
  for any kind of participating media. Also smart sampling of the direct
  illumination is not implemented (i.e. no sampleDirect option as in BDPT).


Installing:
-----------
  - Download the dependencies package from the Mitsuba Repository
  https://www.mitsuba-renderer.org/repos/ and extract it into the Mitsuba
  directory as 'dependencies'.
  - If you want to use faster GPU reconstruction with CUDA, extract
  gradientdomain_dependencies_CUDA.zip and follow the instructions in
  chapter 'Faster reconstruction on the GPU' below.
  - Scons requires 32 bit Python 2.7. Newer versions do not work with
  scons. Add python27 directory and python27/script directory to VC++
  directories / executable directories.
  - As mentioned, scons is required.
  - Pywin32 is also required.
  - Compile Mitsuba with Visual C++ 2013 or above.


Troubleshooting:
----------------
  - Make sure that the config.py files use DOUBLE_PRECISON flag instead
  of SINGLE_PRECISION since gradient-rendering is very sensitive to the
  used precision. This will hopefully be fixed at a later time.
  - To create a documentation with doxygen run the gendoc.py script in
    mitsuba/doc. If this fails it might be because some packages of latex
    are missing (especially mathtools).  (With miktex install them with
    the package manager admin tool (under windows the tool can be found
    in MiKTeX\miktex\bin\mpm_mfc_admin).)
  - In case of problems, remove database files like .sconsign.dblite,
    .sconf_temp and everything in build/Release and build/Debug.
  - Adapt the used config.py file in NMake.
  - In case of 'Could not compile a simple C++ fragment, verify that
    cl is installed! This could also mean that the Boost libraries are
    missing. The file "config.log" should contain more information',
    try bypassing the test in build/SConscript.configure by changing
137: conf = Configure(env, custom_tests = { 'CheckCXX' : CheckCXX })
to
137: conf = Configure(env)
  - We noticed that if we let OpenMP use all CPU cores for the
    Poisson reconstruction, OpenMP may end being slower than a
    single core reconstruction. We suspect this to be caused by hyper
    threading. Our workaround is to use only as many threads as there
    are physical cores. If you experience random slowdowns in the
    reconstruction (on CPU), please decrease the number of used cores in
    src/integrators/poisson_solver/BackendOpenMP.cpp by one or two.


Usage:
------
  - The rendering algorithms are implemented as integrator plugins
    "Gradient-Domain Path Tracer" (G-PT) and "Gradient-Domain Bidirectional
    Path Tracer" (G-BDPT).  They can be chosen directly in the Mitsuba
    GUI, or by setting the integrator in the scene description files to
    "gpt" or "gbdpt".
  - If running from the command line, be sure to set the scenes to use
    "multifilm" instead of "hdrfilm", since the methods render to multiple
    buffers simultaneously.
  - Note that while rendering is in progress, what is displayed is only
    the sampled color data. Reconstruction of the final image is done
    only after the color and gradient buffers have been sampled.


Faster reconstruction on the GPU:
---------------------------------
  We recommend using the provided CUDA reconstruction code for faster
    reconstruction using the GPU. To compile with CUDA reconstruction,
    unzip the file 'gradientdomain_dependencies_CUDA.zip' and overwrite
    any files when prompted.

  You additionally need to have the CUDA toolkit installed (at least
    version 6.5) with a suitable NVIDIA GPU, and, you need to *manually*
    copy the static CUDA runtime library cudart_static.lib (which is
    provided in the CUDA toolkit installation folder CUDA/vx.x/lib/x64)
    into folder dependencies/lib/x64_vc12.

  Finally you need to rebuild the solution.

  To switch back to CPU reconstruction, unzip the file
    'gradientdomain_dependencies_DEFAULT.zip', overwrite any files when
    prompted and rebuild the solution.

  The CUDA reconstruction library is compiled for win x64 using
    VC++2013. To rebuild the CUDA reconstruction library, get the source
    release from the project home page of Gradient-Domain Path Tracing.


Change log:
-----------
  2017/06/28: Improve G-PT's config serialization. Fixes network
    rendering.
  2015/12/18: Fix vertex classification for perfectly specular material
    components.
  2015/10/06: Improve G-PT's vertex classification for multi-component
    materials.
  2015/10/05: Fix handling of specular materials in G-PT. Fixes glitches
    in scenes Bottle and MatPreview.


License Information:
--------------------

  All source code files contain their licensing information.

  Most notably, unlike Mitsuba code, the screened Poisson reconstruction
    code is NOT covered by GNU GPL 3, but the following license:

Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
*  Neither the name of the NVIDIA CORPORATION nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
