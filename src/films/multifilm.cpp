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

//#include "multifilm.h"

#include <mitsuba/render/film.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/statistics.h>
#include <boost/algorithm/string.hpp>
#include "banner.h"
#include "annotations.h"

MTS_NAMESPACE_BEGIN

/*!\plugin{multifilm}{High dynamic range film with multiple outputs}
 * \order{5}
 * \parameters{
 *     \parameter{width, height}{\Integer}{
 *       Width and height of the camera sensor in pixels
 *       \default{768, 576}
 *     }
 *     \parameter{fileFormat}{\String}{
 *       Denotes the desired output file format. The options
 *       are \code{openexr} (for ILM's OpenEXR format),
 *       \code{rgbe} (for Greg Ward's RGBE format),
 *       or \code{pfm} (for the Portable Float Map format)
 *       \default{\code{openexr}}
 *     }
 *     \parameter{pixelFormat}{\String}{Specifies the desired pixel format
 *         of output images. The options are \code{luminance},
 *         \code{luminanceAlpha}, \code{rgb}, \code{rgba}, \code{xyz},
 *         \code{xyza}, \code{spectrum}, and \code{spectrumAlpha}.
 *         For the \code{spectrum*} options, the number of written channels depends on
 *         the value assigned to \code{SPECTRUM\_SAMPLES} during compilation
 *         (see \secref{compiling} for details)
 *         \default{\code{rgb}}
 *     }
 *     \parameter{componentFormat}{\String}{Specifies the desired floating
 *         point component format of output images. The options are
 *         \code{float16}, \code{float32}, or \code{uint32}.
 *         \default{\code{float16}}
 *     }
 *     \parameter{cropOffsetX, cropOffsetY, cropWidth, cropHeight}{\Integer}{
 *       These parameters can optionally be provided to select a sub-rectangle
 *       of the output. In this case, Mitsuba will only render the requested
 *       regions. \default{Unused}
 *     }
 *     \parameter{attachLog}{\Boolean}{Mitsuba can optionally attach
 *         the entire rendering log file as a metadata field so that this
 *         information is permanently saved.
 *         \default{\code{true}, i.e. attach it}
 *     }
 *     \parameter{banner}{\Boolean}{Include a small Mitsuba banner in the
 *         output image? \default{\code{true}}
 *     }
 *     \parameter{highQualityEdges}{\Boolean}{
 *        If set to \code{true}, regions slightly outside of the film
 *        plane will also be sampled. This may improve the image
 *        quality at the edges, especially when using very large
 *        reconstruction filters. In general, this is not needed though.
 *        \default{\code{false}, i.e. disabled}
 *     }
 *     \parameter{\Unnamed}{\RFilter}{Reconstruction filter that should
 *     be used by the film. \default{\code{gaussian}, a windowed Gaussian filter}}
 * }
 *
 * This film allows to write several HDR output images at once.  Currently it is only used for gradient-domain (bidirectional) path tracing (\pluginref{gpt} and \pluginref{gbdpt}). 
 * This Film is automatically selected when switching to G-PT or G-BDPT in the GUI. MultiFilm has the exact same parameters as HDRFilm.
 *
 */

class MultiFilm : public Film {
public:
	/* required by the film interface*/
	bool develop(const Point2i &offset, const Vector2i &size, const Point2i &targetOffset, Bitmap *target) const{
		return developMulti(offset, size, targetOffset, target, 0);
	}
	void setBitmap(const Bitmap *bitmap, Float multiplier = 1.0f){
		setBitmapMulti(bitmap, multiplier, 0);
	}
	void addBitmap(const Bitmap *bitmap, Float multiplier = 1.0f){
		addBitmapMulti(bitmap, multiplier, 0);
	}
	void put(const ImageBlock *bitmap){
		putMulti(bitmap, 0);
	}

	MultiFilm(const Properties &props) : Film(props) {
		/* Should an Mitsuba banner be added to the output image? */
		m_banner = props.getBoolean("banner",false);
		/* Attach the log file as the EXR comment attribute? */
		m_attachLog = props.getBoolean("attachLog", true);

		std::string fileFormat = boost::to_lower_copy(
			props.getString("fileFormat", "openexr"));
		std::vector<std::string> pixelFormats = tokenize(boost::to_lower_copy(
			props.getString("pixelFormat", "rgb")), " ,");
		std::vector<std::string> channelNames = tokenize(
			props.getString("channelNames", ""), ", ");
		std::string componentFormat = boost::to_lower_copy(
			props.getString("componentFormat", "float16"));

		if (fileFormat == "openexr") {
			m_fileFormat = Bitmap::EOpenEXR;
		} else if (fileFormat == "rgbe") {
			m_fileFormat = Bitmap::ERGBE;
		} else if (fileFormat == "pfm") {
			m_fileFormat = Bitmap::EPFM;
		} else {
			Log(EError, "The \"fileFormat\" parameter must either be "
				"equal to \"openexr\", \"pfm\", or \"rgbe\"!");
		}

		if (pixelFormats.empty())
			Log(EError, "At least one pixel format must be specified!");

		if ((pixelFormats.size() != 1 && channelNames.size() != pixelFormats.size()) ||
			(pixelFormats.size() == 1 && channelNames.size() > 1))
			Log(EError, "Number of channel names must match the number of specified pixel formats!");

		if (pixelFormats.size() != 1 && m_fileFormat != Bitmap::EOpenEXR)
			Log(EError, "General multi-channel output is only supported when writing OpenEXR files!");

		for (size_t i=0; i<pixelFormats.size(); ++i) {
			std::string pixelFormat = pixelFormats[i];
			std::string name = i < channelNames.size() ? (channelNames[i] + std::string(".")) : "";

			if (pixelFormat == "luminance") {
				m_pixelFormats.push_back(Bitmap::ELuminance);
				m_channelNames.push_back(name + "Y");
			} else if (pixelFormat == "luminancealpha") {
				m_pixelFormats.push_back(Bitmap::ELuminanceAlpha);
				m_channelNames.push_back(name + "Y");
				m_channelNames.push_back(name + "A");
			} else if (pixelFormat == "rgb") {
				m_pixelFormats.push_back(Bitmap::ERGB);
				m_channelNames.push_back(name + "R");
				m_channelNames.push_back(name + "G");
				m_channelNames.push_back(name + "B");
			} else if (pixelFormat == "rgba") {
				m_pixelFormats.push_back(Bitmap::ERGBA);
				m_channelNames.push_back(name + "R");
				m_channelNames.push_back(name + "G");
				m_channelNames.push_back(name + "B");
				m_channelNames.push_back(name + "A");
			} else if (pixelFormat == "xyz") {
				m_pixelFormats.push_back(Bitmap::EXYZ);
				m_channelNames.push_back(name + "X");
				m_channelNames.push_back(name + "Y");
				m_channelNames.push_back(name + "Z");
			} else if (pixelFormat == "xyza") {
				m_pixelFormats.push_back(Bitmap::EXYZA);
				m_channelNames.push_back(name + "X");
				m_channelNames.push_back(name + "Y");
				m_channelNames.push_back(name + "Z");
				m_channelNames.push_back(name + "A");
			} else if (pixelFormat == "spectrum") {
				m_pixelFormats.push_back(Bitmap::ESpectrum);
				for (int i=0; i<SPECTRUM_SAMPLES; ++i) {
					std::pair<Float, Float> coverage = Spectrum::getBinCoverage(i);
					m_channelNames.push_back(name + formatString("%.2f-%.2fnm", coverage.first, coverage.second));
				}
			} else if (pixelFormat == "spectrumalpha") {
				m_pixelFormats.push_back(Bitmap::ESpectrumAlpha);
				for (int i=0; i<SPECTRUM_SAMPLES; ++i) {
					std::pair<Float, Float> coverage = Spectrum::getBinCoverage(i);
					m_channelNames.push_back(name + formatString("%.2f-%.2fnm", coverage.first, coverage.second));
				}
				m_channelNames.push_back(name + "A");
			} else {
				Log(EError, "The \"pixelFormat\" parameter must either be equal to "
					"\"luminance\", \"luminanceAlpha\", \"rgb\", \"rgba\", \"xyz\", \"xyza\", "
					"\"spectrum\", or \"spectrumAlpha\"!");
			}
		}

		for (size_t i=0; i<m_pixelFormats.size(); ++i) {
			if (SPECTRUM_SAMPLES == 3 && (m_pixelFormats[i] == Bitmap::ESpectrum || m_pixelFormats[i] == Bitmap::ESpectrumAlpha))
				Log(EError, "You requested to render a spectral image, but Mitsuba is currently "
					"configured for a RGB flow (i.e. SPECTRUM_SAMPLES = 3). You will need to recompile "
					"it with a different configuration. Please see the documentation for details.");
		}

		if (componentFormat == "float16") {
			m_componentFormat = Bitmap::EFloat16;
		} else if (componentFormat == "float32") {
			m_componentFormat = Bitmap::EFloat32;
		} else if (componentFormat == "uint32") {
			m_componentFormat = Bitmap::EUInt32;
		} else {
			Log(EError, "The \"componentFormat\" parameter must either be "
				"equal to \"float16\", \"float32\", or \"uint32\"!");
		}

		if (m_fileFormat == Bitmap::ERGBE) {
			/* RGBE output; override pixel & component format if necessary */
			if (m_pixelFormats.size() != 1)
				Log(EError, "The RGBE format does not support general multi-channel images!");
			if (m_pixelFormats[0] != Bitmap::ERGB) {
				Log(EWarn, "The RGBE format only supports pixelFormat=\"rgb\". Overriding..");
				m_pixelFormats[0] = Bitmap::ERGB;
			}
			if (m_componentFormat != Bitmap::EFloat32) {
				Log(EWarn, "The RGBE format only supports componentFormat=\"float32\". Overriding..");
				m_componentFormat = Bitmap::EFloat32;
			}
		} else if (m_fileFormat == Bitmap::EPFM) {
			/* PFM output; override pixel & component format if necessary */
			if (m_pixelFormats.size() != 1)
				Log(EError, "The PFM format does not support general multi-channel images!");
			if (m_pixelFormats[0] != Bitmap::ERGB && m_pixelFormats[0] != Bitmap::ELuminance) {
				Log(EWarn, "The PFM format only supports pixelFormat=\"rgb\" or \"luminance\"."
					" Overriding (setting to \"rgb\")..");
				m_pixelFormats[0] = Bitmap::ERGB;
			}
			if (m_componentFormat != Bitmap::EFloat32) {
				Log(EWarn, "The PFM format only supports componentFormat=\"float32\". Overriding..");
				m_componentFormat = Bitmap::EFloat32;
			}
		}

		std::vector<std::string> keys = props.getPropertyNames();
		for (size_t i=0; i<keys.size(); ++i) {
			std::string key = boost::to_lower_copy(keys[i]);
			key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());

			if ((boost::starts_with(key, "metadata['") && boost::ends_with(key, "']")) ||
			    (boost::starts_with(key, "label[") && boost::ends_with(key, "]")))
				props.markQueried(keys[i]);
		}

		//default output settings
	/*	m_numBuffers = 1;
		m_storage.resize(m_numBuffers);
		if (m_pixelFormats.size() == 1) {
			m_storage[0] = new ImageBlock(Bitmap::ESpectrumAlphaWeight, m_cropSize);
		} else {
			m_storage[0] = new ImageBlock(Bitmap::EMultiSpectrumAlphaWeight, m_cropSize,
				NULL, (int) (SPECTRUM_SAMPLES * m_pixelFormats.size() + 2));
		}			
		m_ext_name.resize(m_numBuffers);
		m_ext_name[0] = new std::string("-out");
		*/
		
		std::vector<std::string> defaultNames = {"-image"};
		setBuffers(defaultNames);
	}

	MultiFilm(Stream *stream, InstanceManager *manager)
		: Film(stream, manager) {
		m_banner = stream->readBool();
		m_attachLog = stream->readBool();
		m_fileFormat = (Bitmap::EFileFormat) stream->readUInt();
		m_pixelFormats.resize((size_t) stream->readUInt());
		for (size_t i=0; i<m_pixelFormats.size(); ++i)
			m_pixelFormats[i] = (Bitmap::EPixelFormat) stream->readUInt();
		m_channelNames.resize((size_t) stream->readUInt());
		for (size_t i=0; i<m_channelNames.size(); ++i)
			m_channelNames[i] = stream->readString();
		m_componentFormat = (Bitmap::EComponentFormat) stream->readUInt();
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		Film::serialize(stream, manager);
		stream->writeBool(m_banner);
		stream->writeBool(m_attachLog);
		stream->writeUInt(m_fileFormat);
		stream->writeUInt((uint32_t) m_pixelFormats.size());
		for (size_t i=0; i<m_pixelFormats.size(); ++i)
			stream->writeUInt(m_pixelFormats[i]);
		stream->writeUInt((uint32_t) m_channelNames.size());
		for (size_t i=0; i<m_channelNames.size(); ++i)
			stream->writeString(m_channelNames[i]);
		stream->writeUInt(m_componentFormat);
	}

	bool setBuffers(std::vector<std::string> &names){	
		m_numBuffers = names.size();
		m_storage.resize(m_numBuffers);
		m_ext_name = names;
		for (size_t i = 0; i < m_numBuffers; ++i){	
			m_storage[i] = new ImageBlock(Bitmap::ESpectrumAlphaWeight, m_cropSize, NULL, -1, true);
		//	m_ext_name[i] = new std::string(names.at(i));
		}
		
		clear();

		return true;
	}


	void clear() {
		for (size_t i = 0; i < m_storage.size(); ++i)
			m_storage[i]->clear();
	}

	void putMulti(const ImageBlock *block, int buf) {
		m_storage[buf]->put(block);
	}

	void setBitmapMulti(const Bitmap *bitmap, Float multiplier, int buf) {
		bitmap->convert(m_storage[buf]->getBitmap(), multiplier);
	}

	void addBitmapMulti(const Bitmap *bitmap, Float multiplier, int buf) {
		/* Currently, only accumulating spectrum-valued floating point images
		   is supported. This function basically just exists to support the
		   somewhat peculiar film updates done by BDPT */

		bool hasAlpha = bitmap->getPixelFormat() == Bitmap::ESpectrumAlphaWeight;

		Vector2i size = bitmap->getSize();
		if ((bitmap->getPixelFormat() != Bitmap::ESpectrum && bitmap->getPixelFormat() != Bitmap::ESpectrumAlphaWeight) ||
			bitmap->getComponentFormat() != Bitmap::EFloat ||
			bitmap->getGamma() != 1.0f ||
			size != m_storage[buf]->getSize() ||
			m_pixelFormats.size() != 1) {
			Log(EError, "addBitmap(): Unsupported bitmap format!");
		}

		size_t nPixels = (size_t) size.x * (size_t) size.y;
		const Float *source = bitmap->getFloatData();
		Float *target = m_storage[buf]->getBitmap()->getFloatData();

		if (hasAlpha){
			for (size_t i = 0; i < nPixels; ++i) {
				for (size_t j = 0; j < SPECTRUM_SAMPLES; ++j)
					*target++ += *source++ * multiplier;
				*target++ += *source++;
				*target++ += *source++;
				//target++; 
				//source++;
			}
		}
		else{
			for (size_t i = 0; i < nPixels; ++i) {
				Float weight = target[SPECTRUM_SAMPLES + 1];
				if (weight == 0)
					weight = target[SPECTRUM_SAMPLES + 1] = 1;
				weight *= multiplier;
				for (size_t j = 0; j < SPECTRUM_SAMPLES; ++j)
					*target++ += *source++ * weight;
				target += 2;

			}
		}
	}


	bool developMulti(const Point2i &sourceOffset, const Vector2i &size,
			const Point2i &targetOffset, Bitmap *target, int buf) const {
		if (buf >= m_numBuffers)
			return false;
		const Bitmap *source = m_storage[buf]->getBitmap();
		const FormatConverter *cvt = FormatConverter::getInstance(
			std::make_pair(Bitmap::EFloat, target->getComponentFormat())
		);

		size_t sourceBpp = source->getBytesPerPixel();
		size_t targetBpp = target->getBytesPerPixel();

		const uint8_t *sourceData = source->getUInt8Data()
			+ (sourceOffset.x + sourceOffset.y * source->getWidth()) * sourceBpp;
		uint8_t *targetData = target->getUInt8Data()
			+ (targetOffset.x + targetOffset.y * target->getWidth()) * targetBpp;

		if (EXPECT_NOT_TAKEN(m_pixelFormats.size() != 1)) {
			/* Special case for general multi-channel images -- just develop the first component(s) */
			for (int i=0; i<size.y; ++i) {
				for (int j=0; j<size.x; ++j) {
					Float weight = *((Float *) (sourceData + (j+1)*sourceBpp - sizeof(Float)));
					Float invWeight = weight != 0 ? ((Float) 1 / weight) : (Float) 0;
					cvt->convert(Bitmap::ESpectrum, 1.0f, sourceData + j*sourceBpp,
						target->getPixelFormat(), target->getGamma(), targetData + j * targetBpp,
						1, invWeight);
				}

				sourceData += source->getWidth() * sourceBpp;
				targetData += target->getWidth() * targetBpp;
			}

		} else if (size.x == m_cropSize.x && target->getWidth() == m_storage[0]->getWidth()) {
			/* Develop a connected part of the underlying buffer */
			cvt->convert(source->getPixelFormat(), 1.0f, sourceData,
				target->getPixelFormat(), target->getGamma(), targetData,
				size.x*size.y);
		} else {
			/* Develop a rectangular subregion */
			for (int i=0; i<size.y; ++i) {
				cvt->convert(source->getPixelFormat(), 1.0f, sourceData,
					target->getPixelFormat(), target->getGamma(), targetData,
					size.x);

				sourceData += source->getWidth() * sourceBpp;
				targetData += target->getWidth() * targetBpp;
			}
		}

		return true;
	}

	void setDestinationFile(const fs::path &destFile, uint32_t blockSize) {
		m_destFile = destFile;
	}


	void develop(const Scene *scene, Float renderTime) {
		if (m_destFile.empty())
			return;

		Log(EInfo, "Developing film ..");

		for (int buf = 0; buf<m_numBuffers; buf++){
			ref<Bitmap> bitmap;
			if (m_pixelFormats.size() == 1) {
				bitmap = m_storage[buf]->getBitmap()->convert(m_pixelFormats[0], m_componentFormat);
				bitmap->setChannelNames(m_channelNames);
			} else {
				bitmap = m_storage[buf]->getBitmap()->convertMultiSpectrumAlphaWeight(m_pixelFormats,
						m_componentFormat, m_channelNames);
			}

			if (m_banner && m_cropSize.x > bannerWidth+5 && m_cropSize.y > bannerHeight + 5 && m_pixelFormats.size() == 1) {
				int xoffs = m_cropSize.x - bannerWidth - 5,
					yoffs = m_cropSize.y - bannerHeight - 5;
				for (int y=0; y<bannerHeight; y++) {
					for (int x=0; x<bannerWidth; x++) {
						if (banner[x+y*bannerWidth])
							continue;
						bitmap->setPixel(Point2i(x+xoffs, y+yoffs), Spectrum(1024));
					}
				}
			}
		


			fs::path filename = m_destFile;


			filename.remove_filename(); // Samuli: augment filename by gradient type
			filename /= std::string(m_destFile.filename().string().c_str()) + m_ext_name.at(buf).c_str();


			std::string properExtension;
			if (m_fileFormat == Bitmap::EOpenEXR)
				properExtension = ".exr";
			else if (m_fileFormat == Bitmap::ERGBE)
				properExtension = ".rgbe";
			else
				properExtension = ".pfm";

			std::string extension = boost::to_lower_copy(filename.extension().string());
			if (extension != properExtension)
				filename.replace_extension(properExtension);

			Log(EInfo, "Writing image to \"%s\" ..", filename.string().c_str());
			ref<FileStream> stream = new FileStream(filename, FileStream::ETruncWrite);

			if (m_pixelFormats.size() == 1)
				annotate(scene, m_properties, bitmap, renderTime, 1.0f);

			/* Attach the log file to the image if this is requested */
			Logger *logger = Thread::getThread()->getLogger();
			std::string log;
			if (m_attachLog && logger->readLog(log)) {
				log += "\n\n";
				log += Statistics::getInstance()->getStats();
				bitmap->setMetadataString("log", log);
			}

			bitmap->write(m_fileFormat, stream);
		}

		// Samuli: output the log file as a separate text file
		if (1)
		{
			std::string log;
			Logger *logger = Thread::getThread()->getLogger();
			if (logger->readLog(log))
			{
				fs::path filename = m_destFile;
				filename.remove_filename();
				filename /= std::string(m_destFile.filename().string().c_str()) + "-log";
				filename.replace_extension(".txt");

				Log(EInfo, "Writing log to \"%s\" ..", filename.string().c_str());
				FILE* f = fopen(filename.string().c_str(), "wt");
				fprintf(f, "%s", log.c_str());
				fclose(f);
			}

			fs::path filename = m_destFile;
			filename.remove_filename();
			filename /= std::string(m_destFile.filename().string().c_str()) + "-stats";
			filename.replace_extension(".txt");

			Log(EInfo, "Writing stats to \"%s\" ..", filename.string().c_str());
			FILE* f = fopen(filename.string().c_str(), "wt");
			fprintf(f, "%s", Statistics::getInstance()->getStats().c_str());
			fclose(f);
		}
	}

	bool hasAlpha() const {
		for (size_t i=0; i<m_pixelFormats.size(); ++i) {
			if (m_pixelFormats[i] == Bitmap::ELuminanceAlpha ||
				m_pixelFormats[i] == Bitmap::ERGBA ||
				m_pixelFormats[i] == Bitmap::EXYZA ||
				m_pixelFormats[i] == Bitmap::ESpectrumAlpha)
				return true;
		}
		return false;
	}

	bool destinationExists(const fs::path &baseName) const {
		std::string properExtension;
		if (m_fileFormat == Bitmap::EOpenEXR)
			properExtension = ".exr";
		else if (m_fileFormat == Bitmap::ERGBE)
			properExtension = ".rgbe";
		else
			properExtension = ".pfm";

		fs::path filename = baseName;
		if (boost::to_lower_copy(filename.extension().string()) != properExtension)
			filename.replace_extension(properExtension);
		return fs::exists(filename);
	}

	std::string MultiFilm::toString() const {
		std::ostringstream oss;
		oss << "multiFilm[" << endl
			<< "  size = " << m_size.toString() << "," << endl
			<< "  fileFormat = " << m_fileFormat << "," << endl
			<< "  pixelFormat = ";
		for (size_t i=0; i<m_pixelFormats.size(); ++i)
			oss << m_pixelFormats[i] << ", ";
		oss << endl
			<< "  channelNames = ";
		for (size_t i=0; i<m_channelNames.size(); ++i)
			oss << "\"" << m_channelNames[i] << "\"" << ", ";
		oss << endl
			<< "  componentFormat = " << m_componentFormat << "," << endl
			<< "  cropOffset = " << m_cropOffset.toString() << "," << endl
			<< "  cropSize = " << m_cropSize.toString() << "," << endl
			<< "  banner = " << m_banner << "," << endl
			<< "  filter = " << indent(m_filter->toString()) << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
protected:
	Bitmap::EFileFormat m_fileFormat;
	std::vector<Bitmap::EPixelFormat> m_pixelFormats;
	std::vector<std::string> m_channelNames;
	Bitmap::EComponentFormat m_componentFormat;
	bool m_banner;
	bool m_attachLog;
	fs::path m_destFile;
	ref_vector<ImageBlock> m_storage;

	std::vector<std::string> m_ext_name; //names of the buffers
	int m_numBuffers;
};

MTS_IMPLEMENT_CLASS_S(MultiFilm, false, Film)
MTS_EXPORT_PLUGIN(MultiFilm, "High dynamic range film");
MTS_NAMESPACE_END
