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
#include "gpt_wr.h"


MTS_NAMESPACE_BEGIN


/* ==================================================================== */
/*                             Work result                              */
/* ==================================================================== */

GPTWorkResult::GPTWorkResult(const ReconstructionFilter *rfilter, Vector2i blockSize, int extraBorder) {
	if (blockSize == Vector2i(-1, -1))
		blockSize = Vector2i(32, 32);

	m_block.resize(BUFFER_COUNT); // 0: preview, 1: throughput, 2: dx, 3: dy, 4: very direct

	for (size_t i = 0; i < m_block.size(); ++i){
		m_block[i] = new ImageBlock(Bitmap::ESpectrumAlphaWeight, blockSize, rfilter, -1, true, extraBorder);
		m_block[i]->setOffset(Point2i(0, 0));
		m_block[i]->setSize(blockSize);
	}

	m_block[2]->setAllowNegativeValues(true); // dx
	m_block[3]->setAllowNegativeValues(true); // dy
}

GPTWorkResult::~GPTWorkResult() {
}

void GPTWorkResult::put(const GPTWorkResult *workResult) {
	for (size_t i = 0; i < m_block.size(); ++i){
		m_block[i]->put(workResult->m_block[i].get());
	}
}

void GPTWorkResult::clear() {
	for (size_t i = 0; i < m_block.size(); ++i) {
		m_block[i]->clear();
	}
}


void GPTWorkResult::load(Stream *stream) {
	for (size_t i = 0; i < m_block.size(); ++i) {
		m_block[i]->load(stream);
	}
}

void GPTWorkResult::save(Stream *stream) const {
	for (size_t i = 0; i < m_block.size(); ++i) {
		m_block[i]->save(stream);
	}
}

std::string GPTWorkResult::toString() const {
	return m_block[0]->toString();	//todo
}


MTS_IMPLEMENT_CLASS(GPTWorkResult, false, WorkResult)
MTS_NAMESPACE_END
