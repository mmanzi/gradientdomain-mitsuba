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

#if !defined(__GBDPT_H)
#define __GBDPT_H

#include <mitsuba/mitsuba.h>


MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                         Configuration storage                        */
/* ==================================================================== */

/**
 * \brief Stores all configuration parameters of the
 * bidirectional path tracer
 */
struct GBDPTConfiguration {
	int maxDepth, blockSize;
	bool lightImage;
	bool sampleDirect;
	size_t sampleCount;
	Vector2i cropSize;
	int rrDepth;

	int extraBorder;
	int nNeighbours;

	float m_shiftThreshold;
	bool m_reconstructL1;
	bool m_reconstructL2;
	float m_reconstructAlpha;

	inline GBDPTConfiguration() { }

	inline GBDPTConfiguration(Stream *stream) {
		maxDepth = stream->readInt();
		blockSize = stream->readInt();
		lightImage = stream->readBool();
		sampleDirect = stream->readBool();
		sampleCount = stream->readSize();
		cropSize = Vector2i(stream);
		rrDepth = stream->readInt();


		extraBorder = stream->readInt();
		nNeighbours = stream->readInt();

		m_shiftThreshold = stream->readFloat();
		m_reconstructL1 = stream->readBool();
		m_reconstructL2 = stream->readBool();
		m_reconstructAlpha = stream->readFloat();
	}

	inline void serialize(Stream *stream) const {
		stream->writeInt(maxDepth);
		stream->writeInt(blockSize);
		stream->writeBool(lightImage);
		stream->writeBool(sampleDirect);
		stream->writeSize(sampleCount);
		cropSize.serialize(stream);
		stream->writeInt(rrDepth);	//possible problem with network rendering?

		stream->writeInt(extraBorder);
		stream->writeInt(nNeighbours);

		stream->writeFloat(m_shiftThreshold);
		stream->writeBool(m_reconstructL1);
		stream->writeBool(m_reconstructL2);
		stream->writeFloat(m_reconstructAlpha);
		
	}

	void dump() const {
		SLog(EInfo, "Gradient-Domain Bidirectional Path Tracer configuration:");
		SLog(EDebug, "   Maximum path depth          : %i", maxDepth);
		SLog(EDebug, "   Image size                  : %ix%i",
			cropSize.x, cropSize.y);
		SLog(EDebug, "   Generate light image        : %s",
			lightImage ? "yes" : "no");
		SLog(EDebug, "   Russian roulette depth      : %i", rrDepth);
		SLog(EDebug, "   Block size                  : %i", blockSize);
		SLog(EDebug, "   Number of samples           : " SIZE_T_FMT, sampleCount);
	}

	bool accumulateData(ref<Bitmap> buff, ref<Film> film, int bufferIdx, int target, int iter, const std::vector<Float> &weights);
	void prepareDataForSolver(float w, float* out, Float * data, int len, Float *data2 = NULL, int offset = 0);
	void setBitmapFromArray(ref<Bitmap> &bitmap, float *img);
};

MTS_NAMESPACE_END

#endif /* __GBDPT_H */
