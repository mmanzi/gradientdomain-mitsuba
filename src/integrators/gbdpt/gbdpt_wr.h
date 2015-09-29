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

#if !defined(__GBDPT_WR_H)
#define __GBDPT_WR_H

#include <mitsuba/render/imageblock.h>
#include <mitsuba/core/fresolver.h>
#include "gbdpt.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                             Work result                              */
/* ==================================================================== */

/**
   Bidirectional path tracing needs its own WorkResult implementation,
   since each rendering thread simultaneously renders to a small 'camera
   image' block and potentially a full-resolution 'light image'.
*/
class GBDPTWorkResult : public WorkResult {
public:
	GBDPTWorkResult(const GBDPTConfiguration &conf, const ReconstructionFilter *filter,
			Vector2i blockSize = Vector2i(-1, -1), int nbuff = 1, int extraBorder = 0);

	// Clear the contents of the work result
	void clear();

	/// Fill the work result with content acquired from a binary data stream
	virtual void load(Stream *stream);

	/// Serialize a work result to a binary data stream
	virtual void save(Stream *stream) const;

	/// Aaccumulate another work result into this one
	void put(const GBDPTWorkResult *workResult);


	inline void putSample(const Point2 &sample, const Spectrum &spec, int buff=0) {
		m_block[buff]->put(sample, spec, 1.0f);
	}

	inline void putLightSample(const Point2 &sample, const Spectrum &spec, int buff=0) {
		m_lightImage[buff]->put(sample, spec, 1.0f);
	}

	inline const ImageBlock *getImageBlock(int buff = 0) const {
		return m_block[buff].get();
	}

	inline const ImageBlock *getLightImage(int buff = 0) const {
		return m_lightImage[buff].get();
	}

	inline void setSize(const Vector2i &size) {
		for (size_t i = 0; i < m_block.size(); ++i)
			m_block[i]->setSize(size);
	}

	inline void setOffset(const Point2i &offset) {
		for (size_t i = 0; i < m_block.size(); ++i)
			m_block[i]->setOffset(offset);
	}

	/// Return a string representation
	std::string toString() const;

	MTS_DECLARE_CLASS()
protected:
	/// Virtual destructor
	virtual ~GBDPTWorkResult();

	inline int strategyIndex(int s, int t) const {
		int above = s+t-2;
		return s + above*(5+above)/2;
	}
protected:
	ref_vector<ImageBlock> m_block, m_lightImage;
	bool m_hasLightImage;
};

MTS_NAMESPACE_END

#endif /* __GBDPT_WR_H */
