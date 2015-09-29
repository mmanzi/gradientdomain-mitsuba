/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#if !defined(__GPT_PROC_H)
#define __GPT_PROC_H

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/sfcurve.h>
#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/rectwu.h>

#include "gpt.h"
#include "gpt_wr.h"


MTS_NAMESPACE_BEGIN



/**
 * \brief Renders work units (rectangular image regions) using
 * gradient-domain path tracing
 */
class GPTRenderProcess : public BlockedRenderProcess {
public:
	GPTRenderProcess(const RenderJob *parent, RenderQueue *queue,
		int blockSize, const GradientPathTracerConfig &config);

	inline const GPTWorkResult *getResult() const { return m_result.get(); }

	void processResult(const WorkResult *result, bool cancelled);

	/* ParallelProcess impl. */
	ref<WorkProcessor> createWorkProcessor() const;

	MTS_DECLARE_CLASS()

protected:
	/// Virtual destructor
	virtual ~GPTRenderProcess() { }

private:
	ref<GPTWorkResult> m_result;
	ref<Timer> m_refreshTimer;
	GradientPathTracerConfig m_config;
};


MTS_NAMESPACE_END


#endif /* __GPT_PROC */
