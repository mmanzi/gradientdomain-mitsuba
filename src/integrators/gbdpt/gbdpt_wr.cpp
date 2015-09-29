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

#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include "gbdpt_wr.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                             Work result                              */
/* ==================================================================== */

GBDPTWorkResult::GBDPTWorkResult(const GBDPTConfiguration &conf,
		const ReconstructionFilter *rfilter, Vector2i blockSize, int nbuff, int extraBorder) {
	/* Stores the 'camera image' -- this can be blocked when
	   spreading out work to multiple workers */
	if (blockSize == Vector2i(-1, -1))
		blockSize = Vector2i(conf.blockSize, conf.blockSize);

	m_hasLightImage = conf.lightImage;

	m_block.resize((1 + nbuff));
	for (size_t i = 0; i < m_block.size(); ++i){
		m_block[i] = new ImageBlock(Bitmap::ESpectrumAlphaWeight, blockSize, rfilter, -1, true, extraBorder);
		m_block[i]->setOffset(Point2i(0, 0));
		m_block[i]->setSize(blockSize);
	}

	if (m_hasLightImage) {
		/* Stores the 'light image' -- every worker requires a
		   full-resolution version, since contributions of s==0
		   and s==1 paths can affect any pixel of this bitmap */
		m_lightImage.resize((1 + nbuff));
		for (size_t i = 0; i < m_block.size(); ++i){
			m_lightImage[i] = new ImageBlock(Bitmap::ESpectrum, conf.cropSize, rfilter, -1, true);
			m_lightImage[i]->setSize(conf.cropSize);
			m_lightImage[i]->setOffset(Point2i(0, 0));
		}
	}
}

GBDPTWorkResult::~GBDPTWorkResult() { }

void GBDPTWorkResult::put(const GBDPTWorkResult *workResult) {
	for (size_t i = 0; i < m_block.size(); ++i){
		m_block[i]->put(workResult->m_block[i].get());
		if (m_hasLightImage)
			m_lightImage[i]->put(workResult->m_lightImage[i].get());
	}
}

void GBDPTWorkResult::clear() {
	for (size_t i = 0; i < m_block.size(); ++i){
		if (m_hasLightImage)
			m_lightImage[i]->clear();
		m_block[i]->clear();
	}
}


void GBDPTWorkResult::load(Stream *stream) {
	for (size_t i = 0; i < m_block.size(); ++i){
		if (m_hasLightImage)
			m_lightImage[i]->load(stream);
		m_block[i]->load(stream);
	}
}

void GBDPTWorkResult::save(Stream *stream) const {
	for (size_t i = 0; i < m_block.size(); ++i){
		if (m_hasLightImage)
			m_lightImage[i]->save(stream);
		m_block[i]->save(stream);
	}
}

std::string GBDPTWorkResult::toString() const {
	return m_block[0]->toString();	//todo
}

MTS_IMPLEMENT_CLASS(GBDPTWorkResult, false, WorkResult)
MTS_NAMESPACE_END
