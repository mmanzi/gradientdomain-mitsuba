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

#include <mitsuba/core/statistics.h>
#include <mitsuba/core/sfcurve.h>
#include <mitsuba/bidir/util.h>
#include "gpt_proc.h"
#include "gpt_wr.h"

MTS_NAMESPACE_BEGIN



/**
 * BlockRenderer from renderproc.cpp modified to use a custom work result.
 */
class GPTBlockRenderer : public WorkProcessor {
public:
	GPTBlockRenderer(Bitmap::EPixelFormat pixelFormat, int channelCount, int blockSize,
		int borderSize, bool warnInvalid) : m_pixelFormat(pixelFormat),
		m_channelCount(channelCount), m_blockSize(blockSize),
		m_borderSize(borderSize), m_warnInvalid(warnInvalid) { }

	GPTBlockRenderer(Stream *stream, InstanceManager *manager) {
		m_pixelFormat = (Bitmap::EPixelFormat) stream->readInt();
		m_channelCount = stream->readInt();
		m_blockSize = stream->readInt();
		m_borderSize = stream->readInt();
		m_warnInvalid = stream->readBool();
	}

	ref<WorkUnit> createWorkUnit() const {
		return new RectangularWorkUnit();
	}

	ref<WorkResult> createWorkResult() const {
		return new GPTWorkResult(
			m_sensor->getFilm()->getReconstructionFilter(),
			Vector2i(m_blockSize),
			1
		);
	}

	void prepare() {
		Scene *scene = static_cast<Scene *>(getResource("scene"));
		m_scene = new Scene(scene);
		m_sampler = static_cast<Sampler *>(getResource("sampler"));
		m_sensor = static_cast<Sensor *>(getResource("sensor"));
		m_integrator = static_cast<GradientPathIntegrator *>(getResource("integrator"));
		m_scene->removeSensor(scene->getSensor());
		m_scene->addSensor(m_sensor);
		m_scene->setSensor(m_sensor);
		m_scene->setSampler(m_sampler);
		m_scene->setIntegrator(m_integrator);
		m_integrator->wakeup(m_scene, m_resources);
		m_scene->wakeup(m_scene, m_resources);
		m_scene->initializeBidirectional();
	}

	void process(const WorkUnit *workUnit, WorkResult *workResult,
		const bool &stop) {
		const RectangularWorkUnit *rect = static_cast<const RectangularWorkUnit *>(workUnit);
		GPTWorkResult *block = static_cast<GPTWorkResult *>(workResult);

#ifdef MTS_DEBUG_FP
		enableFPExceptions();
#endif

		block->setOffset(rect->getOffset());
		block->setSize(rect->getSize());
		m_hilbertCurve.initialize(TVector2<uint8_t>(rect->getSize()));
		m_integrator->renderBlock(m_scene, m_sensor, m_sampler,
			block, stop, m_hilbertCurve.getPoints());

#ifdef MTS_DEBUG_FP
		disableFPExceptions();
#endif
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		stream->writeInt(m_pixelFormat);
		stream->writeInt(m_channelCount);
		stream->writeInt(m_blockSize);
		stream->writeInt(m_borderSize);
		stream->writeBool(m_warnInvalid);
	}

	ref<WorkProcessor> clone() const {
		return new GPTBlockRenderer(m_pixelFormat, m_channelCount,
			m_blockSize, m_borderSize, m_warnInvalid);
	}

	MTS_DECLARE_CLASS()

protected:
	virtual ~GPTBlockRenderer() { }

private:
	ref<Scene> m_scene;
	ref<Sensor> m_sensor;
	ref<Sampler> m_sampler;
	ref<GradientPathIntegrator> m_integrator;
	Bitmap::EPixelFormat m_pixelFormat;
	int m_channelCount;
	int m_blockSize;
	int m_borderSize;
	bool m_warnInvalid;
	HilbertCurve2D<uint8_t> m_hilbertCurve;
};


GPTRenderProcess::GPTRenderProcess(const RenderJob *parent, RenderQueue *queue, int blockSize, const GradientPathTracerConfig &config)
	: BlockedRenderProcess(parent, queue, blockSize),
	  m_config(config)
{
}

ref<WorkProcessor> GPTRenderProcess::createWorkProcessor() const {
	return new GPTBlockRenderer(m_pixelFormat, m_channelCount, m_blockSize, m_borderSize, m_warnInvalid);
}

void GPTRenderProcess::processResult(const WorkResult *result, bool cancelled) {
	const GPTWorkResult* block = static_cast<const GPTWorkResult *>(result);

	UniqueLock lock(m_resultMutex);

	for(int i = 0; i < 5; ++i) {
		m_film->putMulti(block->getImageBlock(i), i);
	}

	m_progress->update(++m_resultCount);
	lock.unlock();
	m_queue->signalWorkEnd(m_parent, block->getImageBlock(0), cancelled);
}


MTS_IMPLEMENT_CLASS(GPTRenderProcess, false, BlockedRenderProcess)
MTS_IMPLEMENT_CLASS_S(GPTBlockRenderer, false, WorkProcessor)
MTS_NAMESPACE_END
