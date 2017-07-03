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

#if !defined(__GPT_H)
#define __GPT_H

#include <mitsuba/mitsuba.h>
#include "gpt_wr.h"


MTS_NAMESPACE_BEGIN


/* ==================================================================== */
/*                         Configuration storage                        */
/* ==================================================================== */


/// Configuration for the gradient path tracer.
struct GradientPathTracerConfig {
    int m_maxDepth;
	int m_minDepth;
    int m_rrDepth;
	bool m_strictNormals;
	Float m_shiftThreshold;
	bool m_reconstructL1;
	bool m_reconstructL2;
	Float m_reconstructAlpha;
	
	inline GradientPathTracerConfig() { }
	
	inline GradientPathTracerConfig(Stream *stream) {
		m_maxDepth = stream->readInt();
		m_minDepth = stream->readInt();
		m_rrDepth = stream->readInt();
		m_strictNormals = stream->readBool();
		m_shiftThreshold = stream->readFloat();
		m_reconstructL1 = stream->readBool();
		m_reconstructL2 = stream->readBool();
		m_reconstructAlpha = stream->readFloat();
	}
	
	inline void serialize(Stream *stream) const {
		stream->writeInt(m_maxDepth);
		stream->writeInt(m_minDepth);
		stream->writeInt(m_rrDepth);
		stream->writeBool(m_strictNormals);
		stream->writeFloat(m_shiftThreshold);
		stream->writeBool(m_reconstructL1);
		stream->writeBool(m_reconstructL2);
		stream->writeFloat(m_reconstructAlpha);
	}
};



/* ==================================================================== */
/*                         Integrator                         */
/* ==================================================================== */


 
class GradientPathIntegrator : public MonteCarloIntegrator {
public:
	GradientPathIntegrator(const Properties &props);

	/// Unserialize from a binary data stream
	GradientPathIntegrator(Stream *stream, InstanceManager *manager);


	/// Starts the rendering process.
	bool render(Scene *scene,
		RenderQueue *queue, const RenderJob *job,
		int sceneResID, int sensorResID, int samplerResID);


	/// Renders a block in the image.
	void renderBlock(const Scene *scene, const Sensor *sensor, Sampler *sampler, GPTWorkResult *block,
		const bool &stop, const std::vector< TPoint2<uint8_t> > &points) const;

	void serialize(Stream *stream, InstanceManager *manager) const;
	std::string toString() const;


	/// Used by Mitsuba for initializing sub-surface scattering.
	Spectrum Li(const RayDifferential &ray, RadianceQueryRecord &rRec) const;

	
	MTS_DECLARE_CLASS()

protected:

	
private:
	GradientPathTracerConfig m_config; 

};


MTS_NAMESPACE_END

#endif /* __GBDPT_H */
