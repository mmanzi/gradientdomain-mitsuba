/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2012 by Wenzel Jakob and others.

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

#include <mitsuba/bidir/util.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/render/renderproc.h>
#include "mitsuba/core/plugin.h"

#include "gpt_proc.h"
#include "gpt_wr.h"
#include "../poisson_solver/Solver.hpp"


MTS_NAMESPACE_BEGIN

/*!\plugin{gpt}{Gradient-domain path tracer}
* \order{5}
* \parameters{
*	   \parameter{reconstructL1}{\Boolean}{If set, the rendering method reconstructs the final image using a reconstruction method 
*           that efficiently kills many image artifacts. The reconstruction is slightly biased, but the bias will go away by increasing sample count. \default{\code{true}}
*     }
*	   \parameter{reconstructL2}{\Boolean}{If set, the rendering method reconstructs the final image using a reconstruction method that is unbiased, 
*			but sometimes introduces severe dipole artifacts. \default{\code{false}}
*     }
*	   \parameter{shiftThreshold}{\Float}{Specifies the roughness threshold for classifying materials as 'diffuse', in contrast to 'specular', 
*			for the purposes of constructing paths pairs for estimating pixel differences. This value should usually be somewhere between 0.0005 and 0.01. 
*			If the result image has noise similar to standard path tracing, increasing or decreasing this value may sometimes help. This implementation assumes that this value is small.\default{\code{0.001}}
*	   }
*	   \parameter{reconstructAlpha}{\Float}{	
*			Higher value makes the reconstruction trust the noisy color image more, giving less weight to the usually lower-noise gradients. 
*			The optimal value tends to be around 0.2, but for scenes with much geometric detail at sub-pixel level a slightly higher value such as 0.3 or 0.4 may be tried.\default{\code{0.2}}
	   }
* }
*
*
* This plugin implements a gradient-domain path tracer (short: G-PT) as described in the paper "Gradient-Domain Path Tracing" by Kettunen et al. 
* It samples difference images in addition to the standard color image, and reconstructs the final image based on these.
* It supports classical materials like diffuse, specular and glossy materials, and area and point lights, depth-of-field, and low discrepancy samplers. 
* There is also experimental support for sub-surface scattering and motion blur. Note that this is still an experimental implementation of Gradient-Domain Path Tracing 
* that has not been tested with all of Mitsuba's features. Notably there is no support yet for any kind of participating media or directional lights. 
* Environment maps are supported, though. Does not support the 'hide emitters' option even though it is displayed.

*
*/

/// A threshold to use in positive denominators to avoid division by zero.
const Float D_EPSILON = (Float)(1e-14);

/// If defined, uses only the central sample for the throughput estimate. Otherwise uses offset paths for estimating throughput too.
//#define CENTRAL_RADIANCE

/// If defined, applies reconstruction after rendering.
#define RECONSTRUCT


static StatsCounter avgPathLength("Gradient Path Tracer", "Average path length", EAverage);


// Output buffer names.
static const size_t BUFFER_FINAL = 0;       ///< Buffer index for the final image. Also used for preview.
static const size_t BUFFER_THROUGHPUT = 1;  ///< Buffer index for the noisy color image.
static const size_t BUFFER_DX = 2;          ///< Buffer index for the X gradients.
static const size_t BUFFER_DY = 3;          ///< Buffer index for the Y gradients.
static const size_t BUFFER_VERY_DIRECT = 4; ///< Buffer index for very direct light.


/// Returns whether point1 sees point2.
bool testVisibility(const Scene* scene, const Point3& point1, const Point3& point2, Float time) {
	Ray shadowRay;
	shadowRay.setTime(time);
	shadowRay.setOrigin(point1);
	shadowRay.setDirection(point2 - point1);
	shadowRay.mint = Epsilon;
	shadowRay.maxt = (Float)1.0 - ShadowEpsilon;

	return !scene->rayIntersect(shadowRay);
}

/// Returns whether the given ray sees the environment.
bool testEnvironmentVisibility(const Scene* scene, const Ray& ray) {
	const Emitter* env = scene->getEnvironmentEmitter();
	if(!env) {
		return false;
	}

	Ray shadowRay(ray);
	shadowRay.setTime(ray.time);
	shadowRay.setOrigin(ray.o);
	shadowRay.setDirection(ray.d);

	DirectSamplingRecord directSamplingRecord;
	env->fillDirectSamplingRecord(directSamplingRecord, shadowRay);

	shadowRay.mint = Epsilon;
	shadowRay.maxt = ((Float)1.0 - ShadowEpsilon) * directSamplingRecord.dist;

	return !scene->rayIntersect(shadowRay);
}






/// Classification of vertices into diffuse and glossy.
enum VertexType {
	VERTEX_TYPE_GLOSSY,     ///< "Specular" vertex that requires the half-vector duplication shift.
	VERTEX_TYPE_DIFFUSE     ///< "Non-specular" vertex that is rough enough for the reconnection shift.
};

enum RayConnection {
	RAY_NOT_CONNECTED,      ///< Not yet connected - shifting in progress.
	RAY_RECENTLY_CONNECTED, ///< Connected, but different incoming direction so needs a BSDF evaluation.
	RAY_CONNECTED           ///< Connected, allows using BSDF values from the base path.
};


/// Describes the state of a ray that is being traced in the scene.
struct RayState {
	RayState()
		: radiance(0.0f),
		  gradient(0.0f),
		  eta(1.0f),
		  pdf(1.0f),
		  throughput(Spectrum(0.0f)),
		  alive(true),
		  connection_status(RAY_NOT_CONNECTED)
	{}

	/// Adds radiance to the ray.
	inline void addRadiance(const Spectrum& contribution, Float weight) {
		Spectrum color = contribution * weight;
		radiance += color;
	}

	/// Adds gradient to the ray.
	inline void addGradient(const Spectrum& contribution, Float weight) {
		Spectrum color = contribution * weight;
		gradient += color;
	}

	RayDifferential ray;             ///< Current ray.

	Spectrum throughput;             ///< Current throughput of the path.
	Float pdf;                       ///< Current PDF of the path.
	
	// Note: Instead of storing throughput and pdf, it is possible to store Veach-style weight (throughput divided by pdf), if relative PDF (offset_pdf divided by base_pdf) is also stored. This might be more stable numerically.
	
	Spectrum radiance;               ///< Radiance accumulated so far.
	Spectrum gradient;               ///< Gradient accumulated so far.

	RadianceQueryRecord rRec;        ///< The radiance query record for this ray.
	Float eta;                       ///< Current refractive index of the ray.
	bool alive;                      ///< Whether the path matching to the ray is still good. Otherwise it's an invalid offset path with zero PDF and throughput.

	RayConnection connection_status; ///< Whether the ray has been connected to the base path, or is in progress.
};

/// Returns the vertex type of a vertex by its roughness value.
VertexType getVertexTypeByRoughness(Float roughness, const GradientPathTracerConfig& config) {
	if(roughness <= config.m_shiftThreshold) {
		return VERTEX_TYPE_GLOSSY;
	} else {
		return VERTEX_TYPE_DIFFUSE;
	}
}

/// Returns the vertex type (diffuse / glossy) of a vertex, for the purposes of determining
/// the shifting strategy.
///
/// A bare classification by roughness alone is not good for multi-component BSDFs since they
/// may contain a diffuse component and a perfect specular component. If the base path
/// is currently working with a sample from a BSDF's smooth component, we don't want to care
/// about the specular component of the BSDF right now - we want to deal with the smooth component.
///
/// For this reason, we vary the classification a little bit based on the situation.
/// This is perfectly valid, and should be done.
VertexType getVertexType(const BSDF* bsdf, Intersection& its, const GradientPathTracerConfig& config, unsigned int bsdfType) {
	// Return the lowest roughness value of the components of the vertex's BSDF.
	// If 'bsdfType' does not have a delta component, do not take perfect speculars (zero roughness) into account in this.

	Float lowest_roughness = std::numeric_limits<Float>::infinity();

	bool found_smooth = false;
	bool found_dirac = false;
	for(int i = 0, component_count = bsdf->getComponentCount(); i < component_count; ++i) {
		Float component_roughness = bsdf->getRoughness(its, i);

		if(component_roughness == Float(0)) {
			found_dirac = true;
			if(!(bsdfType & BSDF::EDelta)) {
				// Skip Dirac components if a smooth component is requested.
				continue;
			}
		} else {
			found_smooth = true;
		}

		if(component_roughness < lowest_roughness) {
			lowest_roughness = component_roughness;
		}
	}

	// Roughness has to be zero also if there is a delta component but no smooth components.
	if(!found_smooth && found_dirac && !(bsdfType & BSDF::EDelta)) {
        lowest_roughness = Float(0);
	}

	return getVertexTypeByRoughness(lowest_roughness, config);
}

VertexType getVertexType(RayState& ray, const GradientPathTracerConfig& config, unsigned int bsdfType) {
	const BSDF* bsdf = ray.rRec.its.getBSDF(ray.ray);
	return getVertexType(bsdf, ray.rRec.its, config, bsdfType);
}


/// Result of a half-vector duplication shift.
struct HalfVectorShiftResult {
	bool success;   ///< Whether the shift succeeded.
	Float jacobian; ///< Local Jacobian determinant of the shift.
	Vector3 wo;     ///< Tangent space outgoing vector for the shift.
};

/// Calculates the outgoing direction of a shift by duplicating the local half-vector.
HalfVectorShiftResult halfVectorShift(Vector3 tangentSpaceMainWi, Vector3 tangentSpaceMainWo, Vector3 tangentSpaceShiftedWi, Float mainEta, Float shiftedEta) {
	HalfVectorShiftResult result;

	if(Frame::cosTheta(tangentSpaceMainWi) * Frame::cosTheta(tangentSpaceMainWo) < (Float)0) {
		// Refraction.

		// Refuse to shift if one of the Etas is exactly 1. This causes degenerate half-vectors.
		if(mainEta == (Float)1 || shiftedEta == (Float)1) {
			// This could be trivially handled as a special case if ever needed.
			result.success = false;
			return result;
		}

		// Get the non-normalized half vector.
		Vector3 tangentSpaceHalfVectorNonNormalizedMain;
		if(Frame::cosTheta(tangentSpaceMainWi) < (Float)0) {
			tangentSpaceHalfVectorNonNormalizedMain = -(tangentSpaceMainWi * mainEta + tangentSpaceMainWo);
		} else {
			tangentSpaceHalfVectorNonNormalizedMain = -(tangentSpaceMainWi + tangentSpaceMainWo * mainEta);
		}

		// Get the normalized half vector.
		Vector3 tangentSpaceHalfVector = normalize(tangentSpaceHalfVectorNonNormalizedMain);

		// Refract to get the outgoing direction.
		Vector3 tangentSpaceShiftedWo = refract(tangentSpaceShiftedWi, tangentSpaceHalfVector, shiftedEta);

		// Refuse to shift between transmission and full internal reflection.
		// This shift would not be invertible: reflections always shift to other reflections.
		if(tangentSpaceShiftedWo.isZero()) {
			result.success = false;
			return result;
		}

		// Calculate the Jacobian.
		Vector3 tangentSpaceHalfVectorNonNormalizedShifted;
		if(Frame::cosTheta(tangentSpaceShiftedWi) < (Float)0) {
			tangentSpaceHalfVectorNonNormalizedShifted = -(tangentSpaceShiftedWi * shiftedEta + tangentSpaceShiftedWo);
		} else {
			tangentSpaceHalfVectorNonNormalizedShifted = -(tangentSpaceShiftedWi + tangentSpaceShiftedWo * shiftedEta);
		}

		Float hLengthSquared = tangentSpaceHalfVectorNonNormalizedShifted.lengthSquared() / (D_EPSILON + tangentSpaceHalfVectorNonNormalizedMain.lengthSquared());
		Float WoDotH = abs(dot(tangentSpaceMainWo, tangentSpaceHalfVector)) / (D_EPSILON + abs(dot(tangentSpaceShiftedWo, tangentSpaceHalfVector)));

		// Output results.
		result.success = true;
		result.wo = tangentSpaceShiftedWo;
		result.jacobian = hLengthSquared * WoDotH;
	} else {
		// Reflection.
		Vector3 tangentSpaceHalfVector = normalize(tangentSpaceMainWi + tangentSpaceMainWo);
		Vector3 tangentSpaceShiftedWo = reflect(tangentSpaceShiftedWi, tangentSpaceHalfVector);

		Float WoDotH = dot(tangentSpaceShiftedWo, tangentSpaceHalfVector) / dot(tangentSpaceMainWo, tangentSpaceHalfVector);
		Float jacobian = abs(WoDotH);

		result.success = true;
		result.wo = tangentSpaceShiftedWo;
		result.jacobian = jacobian;
	}

	return result;
}


/// Result of a reconnection shift.
struct ReconnectionShiftResult {
	bool success;   ///< Whether the shift succeeded.
	Float jacobian; ///< Local Jacobian determinant of the shift.
	Vector3 wo;     ///< World space outgoing vector for the shift.
};

/// Tries to connect the offset path to a specific vertex of the main path.
ReconnectionShiftResult reconnectShift(const Scene* scene, Point3 mainSourceVertex, Point3 targetVertex, Point3 shiftSourceVertex, Vector3 targetNormal, Float time) {
	ReconnectionShiftResult result;

	// Check visibility of the connection.
	if(!testVisibility(scene, shiftSourceVertex, targetVertex, time)) {
		// Since this is not a light sample, we cannot allow shifts through occlusion.
		result.success = false;
		return result;
	}

	// Calculate the Jacobian.
	Vector3 mainEdge = mainSourceVertex - targetVertex;
	Vector3 shiftedEdge = shiftSourceVertex - targetVertex;

	Float mainEdgeLengthSquared = mainEdge.lengthSquared();
	Float shiftedEdgeLengthSquared = shiftedEdge.lengthSquared();

	Vector3 shiftedWo = -shiftedEdge / sqrt(shiftedEdgeLengthSquared);

	Float mainOpposingCosine = dot(mainEdge, targetNormal) / sqrt(mainEdgeLengthSquared);
	Float shiftedOpposingCosine = dot(shiftedWo, targetNormal);

	Float jacobian = std::abs(shiftedOpposingCosine * mainEdgeLengthSquared) / (D_EPSILON + std::abs(mainOpposingCosine * shiftedEdgeLengthSquared));

	// Return the results.
	result.success = true;
	result.jacobian = jacobian;
	result.wo = shiftedWo;
	return result;
}

/// Tries to connect the offset path to a the environment emitter.
ReconnectionShiftResult environmentShift(const Scene* scene, const Ray& mainRay, Point3 shiftSourceVertex) {
	const Emitter* env = scene->getEnvironmentEmitter();

	ReconnectionShiftResult result;

	// Check visibility of the environment.
	if(!testEnvironmentVisibility(scene, mainRay)) {
		// Sampled by BSDF so cannot accept occlusion.
		result.success = false;
		return result;
	}

	// Return the results.
	result.success = true;
	result.jacobian = Float(1);
	result.wo = mainRay.d;

	return result;
}


/// Stores the results of a BSDF sample.
/// Do not confuse with Mitsuba's BSDFSamplingRecord.
struct BSDFSampleResult {
	BSDFSamplingRecord bRec;  ///< The corresponding BSDF sampling record.
	Spectrum weight;          ///< BSDF weight of the sampled direction.
	Float pdf;                ///< PDF of the BSDF sample.
};


/// The actual Gradient Path Tracer implementation.
class GradientPathTracer {
public:
	GradientPathTracer(const Scene* scene, const Sensor* sensor, Sampler* sampler, GPTWorkResult* block, const GradientPathTracerConfig* config)
		: m_scene(scene),
		  m_sensor(sensor),
		  m_sampler(sampler),
		  m_block(block),
		  m_config(config)
	{
	}

	/// Evaluates a sample at the given position.
	///
	/// Outputs direct radiance to be added on top of the final image, the throughput to the central pixel, gradients to all neighbors,
	/// and throughput contribution to the neighboring pixels.
	void evaluatePoint(RadianceQueryRecord& rRec, const Point2& samplePosition, const Point2& apertureSample, Float timeSample, Float differentialScaleFactor,
		Spectrum& out_very_direct, Spectrum& out_throughput, Spectrum *out_gradients, Spectrum *out_neighborThroughputs)
	{
		// Initialize the base path.
		RayState mainRay;
		mainRay.throughput = m_sensor->sampleRayDifferential(mainRay.ray, samplePosition, apertureSample, timeSample);
		mainRay.ray.scaleDifferential(differentialScaleFactor);
		mainRay.rRec = rRec;
		mainRay.rRec.its = rRec.its;

		// Initialize the offset paths.
		RayState shiftedRays[4];
		
		static const Vector2 pixelShifts[4] = {
			Vector2(1.0f, 0.0f),
			Vector2(0.0f, 1.0f),
			Vector2(-1.0f, 0.0f),
			Vector2(0.0f, -1.0f)
		};

		for(int i = 0; i < 4; ++i) {
			shiftedRays[i].throughput = m_sensor->sampleRayDifferential(shiftedRays[i].ray, samplePosition + pixelShifts[i], apertureSample, timeSample);
			shiftedRays[i].ray.scaleDifferential(differentialScaleFactor);
			shiftedRays[i].rRec = rRec;
			shiftedRays[i].rRec.its = rRec.its;
		}

		// Evaluate the gradients. The actual algorithm happens here.
		Spectrum very_direct = Spectrum(0.0f);
		evaluate(mainRay, shiftedRays, 4, very_direct);
		
		// Output results.
		out_very_direct = very_direct;
		out_throughput = mainRay.radiance;

		for(int i = 0; i < 4; i++) {
			out_gradients[i] = shiftedRays[i].gradient;
			out_neighborThroughputs[i] = shiftedRays[i].radiance;
		}
	}

	/// Samples a direction according to the BSDF at the given ray position.
	inline BSDFSampleResult sampleBSDF(RayState& rayState) {
		Intersection& its = rayState.rRec.its;
		RadianceQueryRecord& rRec = rayState.rRec;
		RayDifferential& ray = rayState.ray;

		// Note: If the base path's BSDF evaluation uses random numbers, it would be beneficial to use the same random numbers for the offset path's BSDF.
		//       This is not done currently.

		const BSDF* bsdf = its.getBSDF(ray);

		// Sample BSDF * cos(theta).
		BSDFSampleResult result = {
			BSDFSamplingRecord(its, rRec.sampler, ERadiance),
			Spectrum(),
			(Float)0
		};

		Point2 sample = rRec.nextSample2D();
		result.weight = bsdf->sample(result.bRec, result.pdf, sample);

		// Variable result.pdf will be 0 if the BSDF sampler failed to produce a valid direction.

		SAssert(result.pdf <= (Float)0 || fabs(result.bRec.wo.length() - 1.0) < 0.00001);
		return result;
	}

	/// Constructs a sequence of base paths and shifts them into offset paths, evaluating their throughputs and differences.
	///
	/// This is the core of the rendering algorithm.
	void evaluate(RayState& main, RayState* shiftedRays, int secondaryCount, Spectrum& out_veryDirect) {
		const Scene *scene = main.rRec.scene;
		
		// Perform the first ray intersection for the base path (or ignore if the intersection has already been provided).
		main.rRec.rayIntersect(main.ray);
		main.ray.mint = Epsilon;

		// Perform the same first ray intersection for the offset paths.
		for(int i = 0; i < secondaryCount; ++i) {
			RayState& shifted = shiftedRays[i];
			shifted.rRec.rayIntersect(shifted.ray);
			shifted.ray.mint = Epsilon;
		}

		if (!main.rRec.its.isValid()) {
			// First hit is not in the scene so can't continue. Also there there are no paths to shift.

			// Add potential very direct light from the environment as gradients are not used for that.
			if (main.rRec.type & RadianceQueryRecord::EEmittedRadiance) {
				out_veryDirect += main.throughput * scene->evalEnvironment(main.ray);
			}

			//SLog(EInfo, "Main ray(%d): First hit not in scene.", rayCount);
			return;
		}
			
		// Add very direct light from non-environment.
		{
			// Include emitted radiance if requested.
			if (main.rRec.its.isEmitter() && (main.rRec.type & RadianceQueryRecord::EEmittedRadiance)) {
				out_veryDirect += main.throughput * main.rRec.its.Le(-main.ray.d);
			}

			// Include radiance from a subsurface scattering model if requested. Note: Not tested!
			if (main.rRec.its.hasSubsurface() && (main.rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
				out_veryDirect += main.throughput * main.rRec.its.LoSub(scene, main.rRec.sampler, -main.ray.d, 0);
			}
		}

		// If no intersection of an offset ray could be found, its offset paths can not be generated.
		for(int i = 0; i < secondaryCount; ++i) {
			RayState& shifted = shiftedRays[i];
			if (!shifted.rRec.its.isValid()) {
				shifted.alive = false;
			}
		}

		// Strict normals check to produce the same results as bidirectional methods when normal mapping is used.
		if (m_config->m_strictNormals) {
			// If 'strictNormals'=true, when the geometric and shading normals classify the incident direction to the same side, then the main path is still good.
			if(dot(main.ray.d, main.rRec.its.geoFrame.n) * Frame::cosTheta(main.rRec.its.wi) >= 0) {
				// This is an impossible base path.
				return;
			}

			for(int i = 0; i < secondaryCount; ++i) {
				RayState& shifted = shiftedRays[i];

				if(dot(shifted.ray.d, shifted.rRec.its.geoFrame.n) * Frame::cosTheta(shifted.rRec.its.wi) >= 0) {
					// This is an impossible offset path.
					shifted.alive = false;
				}
			}
		}


		// Main path tracing loop.
		main.rRec.depth = 1;

		while(main.rRec.depth < m_config->m_maxDepth || m_config->m_maxDepth < 0) {

			// Strict normals check to produce the same results as bidirectional methods when normal mapping is used.
			// If 'strictNormals'=true, when the geometric and shading normals classify the incident direction to the same side, then the main path is still good.
			if (m_config->m_strictNormals) {
				if(dot(main.ray.d, main.rRec.its.geoFrame.n) * Frame::cosTheta(main.rRec.its.wi) >= 0) {
					// This is an impossible main path, and there are no more paths to shift.
					return;
				}

				for(int i = 0; i < secondaryCount; ++i) {
					RayState& shifted = shiftedRays[i];

					if(dot(shifted.ray.d, shifted.rRec.its.geoFrame.n) * Frame::cosTheta(shifted.rRec.its.wi) >= 0) {
						// This is an impossible offset path.
						shifted.alive = false;
					}
				}
			}

			// Some optimizations can be made if this is the last traced segment.
			bool lastSegment = (main.rRec.depth + 1 == m_config->m_maxDepth);

			/* ==================================================================== */
			/*                     Direct illumination sampling                     */
			/* ==================================================================== */

			// Sample incoming radiance from lights (next event estimation).
			{
				const BSDF* mainBSDF = main.rRec.its.getBSDF(main.ray);

				if (main.rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance && mainBSDF->getType() & BSDF::ESmooth && main.rRec.depth + 1 >= m_config->m_minDepth) {
					// Sample an emitter and evaluate f = f/p * p for it. */
					DirectSamplingRecord dRec(main.rRec.its);

					mitsuba::Point2 lightSample = main.rRec.nextSample2D();

					std::pair<Spectrum, bool> emitterTuple = m_scene->sampleEmitterDirectVisible(dRec, lightSample);
					Spectrum mainEmitterRadiance = emitterTuple.first * dRec.pdf;
					bool mainEmitterVisible = emitterTuple.second;

					const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

					// If the emitter sampler produces a non-emitter, that's a problem.
					SAssert(emitter != nullptr);

					// Add radiance and gradients to the base path and its offset path.
					// Query the BSDF to the emitter's direction.
					BSDFSamplingRecord mainBRec(main.rRec.its, main.rRec.its.toLocal(dRec.d), ERadiance);

					// Evaluate BSDF * cos(theta).
					Spectrum mainBSDFValue = mainBSDF->eval(mainBRec);

					// Calculate the probability density of having generated the sampled path segment by BSDF sampling. Note that if the emitter is not visible, the probability density is zero.
					// Even if the BSDF sampler has zero probability density, the light sampler can still sample it.
					Float mainBsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle && mainEmitterVisible) ? mainBSDF->pdf(mainBRec) : 0;

					// There values are probably needed soon for the Jacobians.
					Float mainDistanceSquared = (main.rRec.its.p - dRec.p).lengthSquared();
					Float mainOpposingCosine = dot(dRec.n, (main.rRec.its.p - dRec.p)) / sqrt(mainDistanceSquared);

					// Power heuristic weights for the following strategies: light sample from base, BSDF sample from base.
					Float mainWeightNumerator = main.pdf * dRec.pdf;
					Float mainWeightDenominator = (main.pdf * main.pdf) * ((dRec.pdf * dRec.pdf) + (mainBsdfPdf * mainBsdfPdf));

#ifdef CENTRAL_RADIANCE
					main.addRadiance(main.throughput * (mainBSDFValue * mainEmitterRadiance), mainWeightNumerator / (D_EPSILON + mainWeightDenominator));
#endif

					// Strict normals check to produce the same results as bidirectional methods when normal mapping is used.
					if(!m_config->m_strictNormals || dot(main.rRec.its.geoFrame.n, dRec.d) * Frame::cosTheta(mainBRec.wo) > 0) {
						// The base path is good. Add radiance differences to offset paths.
						for(int i = 0; i < secondaryCount; ++i) {
							// Evaluate and apply the gradient.
							RayState& shifted = shiftedRays[i];

							Spectrum mainContribution(Float(0));
							Spectrum shiftedContribution(Float(0));
							Float weight = Float(0);

							bool shiftSuccessful = shifted.alive;

							// Construct the offset path.
							if(shiftSuccessful) {
								// Generate the offset path.
								if(shifted.connection_status == RAY_CONNECTED) {
									// Follow the base path. All relevant vertices are shared. 
									Float shiftedBsdfPdf = mainBsdfPdf;
									Float shiftedDRecPdf = dRec.pdf;
									Spectrum shiftedBsdfValue = mainBSDFValue;
									Spectrum shiftedEmitterRadiance = mainEmitterRadiance;
									Float jacobian = (Float)1;

									// Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
									Float shiftedWeightDenominator = (jacobian * shifted.pdf) * (jacobian * shifted.pdf) * ((shiftedDRecPdf * shiftedDRecPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
									weight = mainWeightNumerator / (D_EPSILON + shiftedWeightDenominator + mainWeightDenominator);

									mainContribution = main.throughput * (mainBSDFValue * mainEmitterRadiance);
									shiftedContribution = jacobian * shifted.throughput * (shiftedBsdfValue * shiftedEmitterRadiance);

									// Note: The Jacobians were baked into shifted.pdf and shifted.throughput at connection phase.
								} else if(shifted.connection_status == RAY_RECENTLY_CONNECTED) {
									// Follow the base path. The current vertex is shared, but the incoming directions differ.
									Vector3 incomingDirection = normalize(shifted.rRec.its.p - main.rRec.its.p);

									BSDFSamplingRecord bRec(main.rRec.its, main.rRec.its.toLocal(incomingDirection), main.rRec.its.toLocal(dRec.d), ERadiance);

									// Sample the BSDF.
									Float shiftedBsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle && mainEmitterVisible) ? mainBSDF->pdf(bRec) : 0; // The BSDF sampler can not sample occluded path segments.
									Float shiftedDRecPdf = dRec.pdf;
									Spectrum shiftedBsdfValue = mainBSDF->eval(bRec);
									Spectrum shiftedEmitterRadiance = mainEmitterRadiance;
									Float jacobian = (Float)1;

									// Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
									Float shiftedWeightDenominator = (jacobian * shifted.pdf) * (jacobian * shifted.pdf) * ((shiftedDRecPdf * shiftedDRecPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
									weight = mainWeightNumerator / (D_EPSILON + shiftedWeightDenominator + mainWeightDenominator);

									mainContribution = main.throughput * (mainBSDFValue * mainEmitterRadiance);
									shiftedContribution = jacobian * shifted.throughput * (shiftedBsdfValue * shiftedEmitterRadiance);

									// Note: The Jacobians were baked into shifted.pdf and shifted.throughput at connection phase.
								} else {
									// Reconnect to the sampled light vertex. No shared vertices.
									SAssert(shifted.connection_status == RAY_NOT_CONNECTED);

									const BSDF* shiftedBSDF = shifted.rRec.its.getBSDF(shifted.ray);

									// This implementation uses light sampling only for the reconnect-shift.
									// When one of the BSDFs is very glossy, light sampling essentially reduces to a failed shift anyway.
									bool mainAtPointLight = (dRec.measure == EDiscrete);

									VertexType mainVertexType = getVertexType(main, *m_config, BSDF::ESmooth);
									VertexType shiftedVertexType = getVertexType(shifted, *m_config, BSDF::ESmooth);

									if(mainAtPointLight || (mainVertexType == VERTEX_TYPE_DIFFUSE && shiftedVertexType == VERTEX_TYPE_DIFFUSE)) {
										// Get emitter radiance.
										DirectSamplingRecord shiftedDRec(shifted.rRec.its);
										std::pair<Spectrum, bool> emitterTuple = m_scene->sampleEmitterDirectVisible(shiftedDRec, lightSample);
										bool shiftedEmitterVisible = emitterTuple.second;

										Spectrum shiftedEmitterRadiance = emitterTuple.first * shiftedDRec.pdf;
										Float shiftedDRecPdf = shiftedDRec.pdf;

										// Sample the BSDF.
										Float shiftedDistanceSquared = (dRec.p - shifted.rRec.its.p).lengthSquared();
										Vector emitterDirection = (dRec.p - shifted.rRec.its.p) / sqrt(shiftedDistanceSquared);
										Float shiftedOpposingCosine = -dot(dRec.n, emitterDirection);

										BSDFSamplingRecord bRec(shifted.rRec.its, shifted.rRec.its.toLocal(emitterDirection), ERadiance);
										
										// Strict normals check, to make the output match with bidirectional methods when normal maps are present.
										if (m_config->m_strictNormals && dot(shifted.rRec.its.geoFrame.n, emitterDirection) * Frame::cosTheta(bRec.wo) < 0) {
											// Invalid, non-samplable offset path.
											shiftSuccessful = false;
										} else {
											Spectrum shiftedBsdfValue = shiftedBSDF->eval(bRec);
											Float shiftedBsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle && shiftedEmitterVisible) ? shiftedBSDF->pdf(bRec) : 0;
											Float jacobian = std::abs(shiftedOpposingCosine * mainDistanceSquared) / (Epsilon + std::abs(mainOpposingCosine * shiftedDistanceSquared));

											// Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
											Float shiftedWeightDenominator = (jacobian * shifted.pdf) * (jacobian * shifted.pdf) * ((shiftedDRecPdf * shiftedDRecPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
											weight = mainWeightNumerator / (D_EPSILON + shiftedWeightDenominator + mainWeightDenominator);

											mainContribution = main.throughput * (mainBSDFValue * mainEmitterRadiance);
											shiftedContribution = jacobian * shifted.throughput * (shiftedBsdfValue * shiftedEmitterRadiance);
										}
									}
								}
							}

							if(!shiftSuccessful) {
								// The offset path cannot be generated; Set offset PDF and offset throughput to zero. This is what remains.

								// Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset. (Offset path has zero PDF)
								Float shiftedWeightDenominator = Float(0);
								weight = mainWeightNumerator / (D_EPSILON + mainWeightDenominator);

								mainContribution = main.throughput * (mainBSDFValue * mainEmitterRadiance);
								shiftedContribution = Spectrum((Float)0);
							}

							// Note: Using also the offset paths for the throughput estimate, like we do here, provides some advantage when a large reconstruction alpha is used,
							// but using only throughputs of the base paths doesn't usually lose by much.

#ifndef CENTRAL_RADIANCE
							main.addRadiance(mainContribution, weight);
							shifted.addRadiance(shiftedContribution, weight); 
#endif
							shifted.addGradient(shiftedContribution - mainContribution, weight);
						} // for(int i = 0; i < secondaryCount; ++i)
					} // Strict normals
				}
			} // Sample incoming radiance from lights.

			/* ==================================================================== */
			/*               BSDF sampling and emitter hits                         */
			/* ==================================================================== */

			// Sample a new direction from BSDF * cos(theta).
			BSDFSampleResult mainBsdfResult = sampleBSDF(main);

			if(mainBsdfResult.pdf <= (Float)0.0) {
				// Impossible base path.
				break;
			}

			const Vector mainWo = main.rRec.its.toWorld(mainBsdfResult.bRec.wo);

			// Prevent light leaks due to the use of shading normals.
			Float mainWoDotGeoN = dot(main.rRec.its.geoFrame.n, mainWo);
			if (m_config->m_strictNormals && mainWoDotGeoN * Frame::cosTheta(mainBsdfResult.bRec.wo) <= 0) {
				break;
			}

			// The old intersection structure is still needed after main.rRec.its gets updated.
			Intersection previousMainIts = main.rRec.its;

			// Trace a ray in the sampled direction.
			bool mainHitEmitter = false;
			Spectrum mainEmitterRadiance = Spectrum((Float)0);

			DirectSamplingRecord mainDRec(main.rRec.its);
			const BSDF* mainBSDF = main.rRec.its.getBSDF(main.ray);


			// Update the vertex types.
			VertexType mainVertexType = getVertexType(main, *m_config, mainBsdfResult.bRec.sampledType);
			VertexType mainNextVertexType;

			main.ray = Ray(main.rRec.its.p, mainWo, main.ray.time);
			
			if (scene->rayIntersect(main.ray, main.rRec.its)) {
				// Intersected something - check if it was a luminaire.
				if (main.rRec.its.isEmitter()) {
					mainEmitterRadiance = main.rRec.its.Le(-main.ray.d);

					mainDRec.setQuery(main.ray, main.rRec.its);
					mainHitEmitter = true;
				}
				
				// Sub-surface scattering.
				if (main.rRec.its.hasSubsurface() && (main.rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
					mainEmitterRadiance += main.rRec.its.LoSub(scene, main.rRec.sampler, -main.ray.d, main.rRec.depth);
				}

				// Update the vertex type.
				mainNextVertexType = getVertexType(main, *m_config, mainBsdfResult.bRec.sampledType);
			} else {
				// Intersected nothing -- perhaps there is an environment map?
				const Emitter *env = scene->getEnvironmentEmitter();
			
				if (env) {
					// Hit the environment map.
					mainEmitterRadiance = env->evalEnvironment(main.ray);
					if (!env->fillDirectSamplingRecord(mainDRec, main.ray))
						break;
					mainHitEmitter = true;

					// Handle environment connection as diffuse (that's ~infinitely far away).

					// Update the vertex type.
					mainNextVertexType = VERTEX_TYPE_DIFFUSE;
				} else {
					// Nothing to do anymore.
					break;
				}
			}

			// Continue the shift.
			Float mainBsdfPdf = mainBsdfResult.pdf;
			Float mainPreviousPdf = main.pdf;

			main.throughput *= mainBsdfResult.weight * mainBsdfResult.pdf;
			main.pdf *= mainBsdfResult.pdf;
			main.eta *= mainBsdfResult.bRec.eta;

			// Compute the probability density of generating base path's direction using the implemented direct illumination sampling technique.
			const Float mainLumPdf = (mainHitEmitter && main.rRec.depth + 1 >= m_config->m_minDepth && !(mainBsdfResult.bRec.sampledType & BSDF::EDelta)) ?
				scene->pdfEmitterDirect(mainDRec) : 0;

			// Power heuristic weights for the following strategies: light sample from base, BSDF sample from base.
			Float mainWeightNumerator = mainPreviousPdf * mainBsdfResult.pdf;
			Float mainWeightDenominator = (mainPreviousPdf * mainPreviousPdf) * ((mainLumPdf * mainLumPdf) + (mainBsdfPdf * mainBsdfPdf));
				
#ifdef CENTRAL_RADIANCE
			if(main.rRec.depth + 1 >= m_config->m_minDepth) {
				main.addRadiance(main.throughput * mainEmitterRadiance, mainWeightNumerator / (D_EPSILON + mainWeightDenominator));
			}
#endif

			// Construct the offset paths and evaluate emitter hits.

			for(int i = 0; i < secondaryCount; ++i) {
				RayState& shifted = shiftedRays[i];

				Spectrum shiftedEmitterRadiance(Float(0));
				Spectrum mainContribution(Float(0));
				Spectrum shiftedContribution(Float(0));
				Float weight(0);

				bool postponedShiftEnd = false; // Kills the shift after evaluating the current radiance.

				if(shifted.alive) {
					// The offset path is still good, so it makes sense to continue its construction.
					Float shiftedPreviousPdf = shifted.pdf;

					if(shifted.connection_status == RAY_CONNECTED) {
						// The offset path keeps following the base path.
						// As all relevant vertices are shared, we can just reuse the sampled values.
						Spectrum shiftedBsdfValue = mainBsdfResult.weight * mainBsdfResult.pdf;
						Float shiftedBsdfPdf = mainBsdfPdf;
						Float shiftedLumPdf = mainLumPdf;
						Spectrum shiftedEmitterRadiance = mainEmitterRadiance;

						// Update throughput and pdf.
						shifted.throughput *= shiftedBsdfValue;
						shifted.pdf *= shiftedBsdfPdf;
						
						// Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
						Float shiftedWeightDenominator = (shiftedPreviousPdf * shiftedPreviousPdf) * ((shiftedLumPdf * shiftedLumPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
						weight = mainWeightNumerator / (D_EPSILON + shiftedWeightDenominator + mainWeightDenominator);

						mainContribution = main.throughput * mainEmitterRadiance;
						shiftedContribution = shifted.throughput * shiftedEmitterRadiance; // Note: Jacobian baked into .throughput.
					} else if(shifted.connection_status == RAY_RECENTLY_CONNECTED) {
						// Recently connected - follow the base path but evaluate BSDF to the new direction.
						Vector3 incomingDirection = normalize(shifted.rRec.its.p - main.ray.o);
						BSDFSamplingRecord bRec(previousMainIts, previousMainIts.toLocal(incomingDirection), previousMainIts.toLocal(main.ray.d), ERadiance);

						// Note: mainBSDF is the BSDF at previousMainIts, which is the current position of the offset path.

						EMeasure measure = (mainBsdfResult.bRec.sampledType & BSDF::EDelta) ? EDiscrete : ESolidAngle; 

						Spectrum shiftedBsdfValue = mainBSDF->eval(bRec, measure);
						Float shiftedBsdfPdf = mainBSDF->pdf(bRec, measure);

						Float shiftedLumPdf = mainLumPdf;
						Spectrum shiftedEmitterRadiance = mainEmitterRadiance;
						
						// Update throughput and pdf.
						shifted.throughput *= shiftedBsdfValue;
						shifted.pdf *= shiftedBsdfPdf;

						shifted.connection_status = RAY_CONNECTED;

						// Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
						Float shiftedWeightDenominator = (shiftedPreviousPdf * shiftedPreviousPdf) * ((shiftedLumPdf * shiftedLumPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
						weight = mainWeightNumerator / (D_EPSILON + shiftedWeightDenominator + mainWeightDenominator);

						mainContribution = main.throughput * mainEmitterRadiance;
						shiftedContribution = shifted.throughput * shiftedEmitterRadiance; // Note: Jacobian baked into .throughput.
					} else {
						// Not connected - apply either reconnection or half-vector duplication shift.

						const BSDF* shiftedBSDF = shifted.rRec.its.getBSDF(shifted.ray);

						// Update the vertex type of the offset path.
						VertexType shiftedVertexType = getVertexType(shifted, *m_config, mainBsdfResult.bRec.sampledType);

						if(mainVertexType == VERTEX_TYPE_DIFFUSE && mainNextVertexType == VERTEX_TYPE_DIFFUSE && shiftedVertexType == VERTEX_TYPE_DIFFUSE) {
							// Use reconnection shift.

							// Optimization: Skip the last raycast and BSDF evaluation for the offset path when it won't contribute and isn't needed anymore.
							if(!lastSegment || mainHitEmitter || main.rRec.its.hasSubsurface()) {
								ReconnectionShiftResult shiftResult;
								bool environmentConnection = false;

								if(main.rRec.its.isValid()) {
									// This is an actual reconnection shift.
									shiftResult = reconnectShift(m_scene, main.ray.o, main.rRec.its.p, shifted.rRec.its.p, main.rRec.its.geoFrame.n, main.ray.time);
								} else {
									// This is a reconnection at infinity in environment direction.
									const Emitter* env = m_scene->getEnvironmentEmitter();
									SAssert(env != NULL);

									environmentConnection = true;
									shiftResult = environmentShift(m_scene, main.ray, shifted.rRec.its.p);
								}

								if(!shiftResult.success) {
									// Failed to construct the offset path.
									shifted.alive = false;
									goto shift_failed;
								}

								Vector3 incomingDirection = -shifted.ray.d;
								Vector3 outgoingDirection = shiftResult.wo;

								BSDFSamplingRecord bRec(shifted.rRec.its, shifted.rRec.its.toLocal(incomingDirection), shifted.rRec.its.toLocal(outgoingDirection), ERadiance);

								// Strict normals check.
								if(m_config->m_strictNormals && dot(outgoingDirection, shifted.rRec.its.geoFrame.n) * Frame::cosTheta(bRec.wo) <= 0) {
									shifted.alive = false;
									goto shift_failed;
								}

								// Evaluate the BRDF to the new direction.
								Spectrum shiftedBsdfValue = shiftedBSDF->eval(bRec);
								Float shiftedBsdfPdf = shiftedBSDF->pdf(bRec);

								// Update throughput and pdf.
								shifted.throughput *= shiftedBsdfValue * shiftResult.jacobian;
								shifted.pdf *= shiftedBsdfPdf * shiftResult.jacobian;
							
								shifted.connection_status = RAY_RECENTLY_CONNECTED;

								if(mainHitEmitter || main.rRec.its.hasSubsurface()) {
									// Also the offset path hit the emitter, as visibility was checked at reconnectShift or environmentShift.

									// Evaluate radiance to this direction.
									Spectrum shiftedEmitterRadiance(Float(0));
									Float shiftedLumPdf = Float(0);

									if(main.rRec.its.isValid()) {
										// Hit an object.
										if(mainHitEmitter) {
											shiftedEmitterRadiance = main.rRec.its.Le(-outgoingDirection);

											// Evaluate the light sampling PDF of the new segment.
											DirectSamplingRecord shiftedDRec;
											shiftedDRec.p = mainDRec.p;
											shiftedDRec.n = mainDRec.n;
											shiftedDRec.dist = (mainDRec.p - shifted.rRec.its.p).length();
											shiftedDRec.d = (mainDRec.p - shifted.rRec.its.p) / shiftedDRec.dist;
											shiftedDRec.ref = mainDRec.ref;
											shiftedDRec.refN = shifted.rRec.its.shFrame.n;
											shiftedDRec.object = mainDRec.object;

											shiftedLumPdf = scene->pdfEmitterDirect(shiftedDRec);
										}

										// Sub-surface scattering. Note: Should use the same random numbers as the base path!
										if (main.rRec.its.hasSubsurface() && (main.rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
											shiftedEmitterRadiance += main.rRec.its.LoSub(scene, shifted.rRec.sampler, -outgoingDirection, main.rRec.depth);
										}
									} else {
										// Hit the environment.
										shiftedEmitterRadiance = mainEmitterRadiance;
										shiftedLumPdf = mainLumPdf;
									}

									// Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
									Float shiftedWeightDenominator = (shiftedPreviousPdf * shiftedPreviousPdf) * ((shiftedLumPdf * shiftedLumPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
									weight = mainWeightNumerator / (D_EPSILON + shiftedWeightDenominator + mainWeightDenominator);

									mainContribution = main.throughput * mainEmitterRadiance;
									shiftedContribution = shifted.throughput * shiftedEmitterRadiance; // Note: Jacobian baked into .throughput.
								}
							}
						} else {
							// Use half-vector duplication shift. These paths could not have been sampled by light sampling (by our decision).
							Vector3 tangentSpaceIncomingDirection = shifted.rRec.its.toLocal(-shifted.ray.d);
							Vector3 tangentSpaceOutgoingDirection;
							Spectrum shiftedEmitterRadiance(Float(0));

							const BSDF* shiftedBSDF = shifted.rRec.its.getBSDF(shifted.ray);

							// Deny shifts between Dirac and non-Dirac BSDFs.
							bool bothDelta = (mainBsdfResult.bRec.sampledType & BSDF::EDelta) && (shiftedBSDF->getType() & BSDF::EDelta);
							bool bothSmooth = (mainBsdfResult.bRec.sampledType & BSDF::ESmooth) && (shiftedBSDF->getType() & BSDF::ESmooth);
							if(!(bothDelta || bothSmooth)) {
								shifted.alive = false;
								goto half_vector_shift_failed;
							}

							SAssert(fabs(shifted.ray.d.lengthSquared() - 1) < 0.000001);

							// Apply the local shift.
							HalfVectorShiftResult shiftResult = halfVectorShift(mainBsdfResult.bRec.wi, mainBsdfResult.bRec.wo, shifted.rRec.its.toLocal(-shifted.ray.d), mainBSDF->getEta(), shiftedBSDF->getEta());

							if(mainBsdfResult.bRec.sampledType & BSDF::EDelta) {
								// Dirac delta integral is a point evaluation - no Jacobian determinant!
								shiftResult.jacobian = Float(1);
							}

							if(shiftResult.success) {
								// Invertible shift, success.
								shifted.throughput *= shiftResult.jacobian;
								shifted.pdf *= shiftResult.jacobian;
								tangentSpaceOutgoingDirection = shiftResult.wo;
							} else {
								// The shift is non-invertible so kill it.
								shifted.alive = false;
								goto half_vector_shift_failed;
							}

							Vector3 outgoingDirection = shifted.rRec.its.toWorld(tangentSpaceOutgoingDirection);

							// Update throughput and pdf.
							BSDFSamplingRecord bRec(shifted.rRec.its, tangentSpaceIncomingDirection, tangentSpaceOutgoingDirection, ERadiance);
							EMeasure measure = (mainBsdfResult.bRec.sampledType & BSDF::EDelta) ? EDiscrete : ESolidAngle;

							shifted.throughput *= shiftedBSDF->eval(bRec, measure);
							shifted.pdf *= shiftedBSDF->pdf(bRec, measure);

							if(shifted.pdf == Float(0)) {
								// Offset path is invalid!
								shifted.alive = false;
								goto half_vector_shift_failed;
							}

							// Strict normals check to produce the same results as bidirectional methods when normal mapping is used.			
							if(m_config->m_strictNormals && dot(outgoingDirection, shifted.rRec.its.geoFrame.n) * Frame::cosTheta(bRec.wo) <= 0) {
								shifted.alive = false;
								goto half_vector_shift_failed;
							}


							// Update the vertex type.
							VertexType shiftedVertexType = getVertexType(shifted, *m_config, mainBsdfResult.bRec.sampledType);

							// Trace the next hit point.
							shifted.ray = Ray(shifted.rRec.its.p, outgoingDirection, main.ray.time);

							if(!scene->rayIntersect(shifted.ray, shifted.rRec.its)) {
								// Hit nothing - Evaluate environment radiance.
								const Emitter *env = scene->getEnvironmentEmitter();
								if(!env) {
									// Since base paths that hit nothing are not shifted, we must be symmetric and kill shifts that hit nothing.
									shifted.alive = false;
									goto half_vector_shift_failed;
								}
								if(main.rRec.its.isValid()) {
									// Deny shifts between env and non-env.
									shifted.alive = false;
									goto half_vector_shift_failed;
								}

								if(mainVertexType == VERTEX_TYPE_DIFFUSE && shiftedVertexType == VERTEX_TYPE_DIFFUSE) {
									// Environment reconnection shift would have been used for the reverse direction!
									shifted.alive = false;
									goto half_vector_shift_failed;
								}

								// The offset path is no longer valid after this path segment.
								shiftedEmitterRadiance = env->evalEnvironment(shifted.ray);
								postponedShiftEnd = true;
							} else {
								// Hit something.
								
								if(!main.rRec.its.isValid()) {
									// Deny shifts between env and non-env.
									shifted.alive = false;
									goto half_vector_shift_failed;
								}

								VertexType shiftedNextVertexType = getVertexType(shifted, *m_config, mainBsdfResult.bRec.sampledType);

								// Make sure that the reverse shift would use this same strategy!
								// ==============================================================

								if(mainVertexType == VERTEX_TYPE_DIFFUSE && shiftedVertexType == VERTEX_TYPE_DIFFUSE && shiftedNextVertexType == VERTEX_TYPE_DIFFUSE) {
									// Non-invertible shift: the reverse-shift would use another strategy!
									shifted.alive = false;
									goto half_vector_shift_failed;
								}

								if(shifted.rRec.its.isEmitter()) {
									// Hit emitter.
									shiftedEmitterRadiance = shifted.rRec.its.Le(-shifted.ray.d);
								}
								// Sub-surface scattering. Note: Should use the same random numbers as the base path!
								if (shifted.rRec.its.hasSubsurface() && (shifted.rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
									shiftedEmitterRadiance += shifted.rRec.its.LoSub(scene, shifted.rRec.sampler, -shifted.ray.d, main.rRec.depth);
								}
							}

							
half_vector_shift_failed:
							if(shifted.alive) {
								// Evaluate radiance difference using power heuristic between BSDF samples from base and offset paths.
								// Note: No MIS with light sampling since we don't use it for this connection type.
								weight = main.pdf / (shifted.pdf * shifted.pdf + main.pdf * main.pdf);
								mainContribution = main.throughput * mainEmitterRadiance;
								shiftedContribution = shifted.throughput * shiftedEmitterRadiance; // Note: Jacobian baked into .throughput.
							} else {
								// Handle the failure without taking MIS with light sampling, as we decided not to use it in the half-vector-duplication case.
								// Could have used it, but so far there has been no need. It doesn't seem to be very useful.
								weight = Float(1) / main.pdf;
								mainContribution = main.throughput * mainEmitterRadiance;
								shiftedContribution = Spectrum(Float(0));

								// Disable the failure detection below since the failure was already handled.
								shifted.alive = true;
								postponedShiftEnd = true;

								// (TODO: Restructure into smaller functions and get rid of the gotos... Although this may mean having lots of small functions with a large number of parameters.)
							}
						}
					}
				}
				
shift_failed:
				if(!shifted.alive) {
					// The offset path cannot be generated; Set offset PDF and offset throughput to zero.
					weight = mainWeightNumerator / (D_EPSILON + mainWeightDenominator);
					mainContribution = main.throughput * mainEmitterRadiance;
					shiftedContribution = Spectrum((Float)0);
				}
				
				// Note: Using also the offset paths for the throughput estimate, like we do here, provides some advantage when a large reconstruction alpha is used,
				// but using only throughputs of the base paths doesn't usually lose by much.
				if(main.rRec.depth + 1 >= m_config->m_minDepth) {
#ifndef CENTRAL_RADIANCE
					main.addRadiance(mainContribution, weight);
					shifted.addRadiance(shiftedContribution, weight);
#endif
					shifted.addGradient(shiftedContribution - mainContribution, weight);
				}

				if(postponedShiftEnd) {
					shifted.alive = false;
				}
			}

			// Stop if the base path hit the environment.
			main.rRec.type = RadianceQueryRecord::ERadianceNoEmission;
			if(!main.rRec.its.isValid() || !(main.rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance)) {
				break;
			}

			if (main.rRec.depth++ >= m_config->m_rrDepth) {
				/* Russian roulette: try to keep path weights equal to one,
				   while accounting for the solid angle compression at refractive
				   index boundaries. Stop with at least some probability to avoid
				   getting stuck (e.g. due to total internal reflection) */

				Float q = std::min((main.throughput / main.pdf).max() * main.eta * main.eta, (Float) 0.95f);
				if (main.rRec.nextSample1D() >= q)
					break;

				main.pdf *= q;
				for(int i = 0; i < secondaryCount; ++i) {
					RayState& shifted = shiftedRays[i];
					shifted.pdf *= q;
				}
			}
		}

		// Store statistics.
		avgPathLength.incrementBase();
		avgPathLength += main.rRec.depth;
	}

private:
	const Scene* m_scene;
	const Sensor* m_sensor;
	Sampler* m_sampler;
	GPTWorkResult* m_block;
	const GradientPathTracerConfig* m_config;
};


GradientPathIntegrator::GradientPathIntegrator(const Properties &props)
	: MonteCarloIntegrator(props)
{
	m_config.m_maxDepth = props.getInteger("maxDepth", -1);
	m_config.m_minDepth = props.getInteger("minDepth", -1);
	m_config.m_rrDepth = props.getInteger("rrDepth", 5);
	m_config.m_strictNormals = props.getBoolean("strictNormals", false);
	m_config.m_shiftThreshold = props.getFloat("shiftThreshold", Float(0.001));
	m_config.m_reconstructL1 = props.getBoolean("reconstructL1", true);
	m_config.m_reconstructL2 = props.getBoolean("reconstructL2", false);
	m_config.m_reconstructAlpha = (Float)props.getFloat("reconstructAlpha", Float(0.2));

	if(m_config.m_reconstructL1 && m_config.m_reconstructL2)
		Log(EError, "Disable 'reconstructL1' or 'reconstructL2': Cannot display two reconstructions at a time!");

	if(m_config.m_reconstructAlpha <= 0.0f)
		Log(EError, "'reconstructAlpha' must be set to a value greater than zero!");

	if (m_config.m_maxDepth <= 0 && m_config.m_maxDepth != -1)
		Log(EError, "'maxDepth' must be set to -1 (infinite) or a value greater than zero!");
}

GradientPathIntegrator::GradientPathIntegrator(Stream *stream, InstanceManager *manager)
	: MonteCarloIntegrator(stream, manager)
{
	m_config = GradientPathTracerConfig(stream);
}


void GradientPathIntegrator::renderBlock(const Scene *scene, const Sensor *sensor, Sampler *sampler, GPTWorkResult* block,
	const bool &stop, const std::vector< TPoint2<uint8_t> > &points) const
{
	GradientPathTracer tracer(scene, sensor, sampler, block, &m_config);

	bool needsApertureSample = sensor->needsApertureSample();
	bool needsTimeSample = sensor->needsTimeSample();


	// Original code from SamplingIntegrator.
	Float diffScaleFactor = 1.0f / std::sqrt((Float) sampler->getSampleCount());

	// Get ready for sampling.
	RadianceQueryRecord rRec(scene, sampler);

	Point2 apertureSample(0.5f);
	Float timeSample = 0.5f;
	RayDifferential sensorRay;

	block->clear();

	// Sample at the given positions.
	Spectrum gradients[4];
	Spectrum shiftedThroughputs[4];

	for (size_t i = 0; i < points.size(); ++i) {
		if (stop) {
			break;
		}

		Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
		sampler->generate(offset);

		for (size_t j = 0; j < sampler->getSampleCount(); ++j) {
			if (stop) {
				break;
			}

			// Get the initial ray to sample.
			rRec.newQuery(RadianceQueryRecord::ESensorRay, sensor->getMedium());

			Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

			if (needsApertureSample) {
				apertureSample = rRec.nextSample2D();
			}
			if (needsTimeSample) {
				timeSample = rRec.nextSample1D();
			}

			// Do the actual sampling.
			Spectrum centralVeryDirect = Spectrum(0.0f);
			Spectrum centralThroughput = Spectrum(0.0f);

			tracer.evaluatePoint(rRec, samplePos, apertureSample, timeSample, diffScaleFactor, centralVeryDirect, centralThroughput, gradients, shiftedThroughputs);

			// Accumulate results.
			const Point2 right_pixel = samplePos + Vector2(1.0f, 0.0f);
			const Point2 bottom_pixel = samplePos + Vector2(0.0f, 1.0f);
			const Point2 left_pixel = samplePos - Vector2(1.0f, 0.0f);
			const Point2 top_pixel = samplePos - Vector2(0.0f, 1.0f);
			const Point2 center_pixel = samplePos;

			static const int RIGHT = 0;
			static const int BOTTOM = 1;
			static const int LEFT = 2;
			static const int TOP = 3;


			// Note: Sampling differences and throughputs to multiple directions is essentially
			//       multiple importance sampling (MIS) between the pixels.
			//
			//       For a sample from a strategy participating in the MIS to be unbiased, we need to
			//       divide its result by the selection probability of that strategy.
			//
			//       As the selection probability is 0.5 for both directions (no adaptive sampling),
			//       we need to multiply the results by two.
				
			// Note: The central pixel is estimated as
			//               1/4 * (throughput estimate sampled from MIS(center, top)
			//                      + throughput estimate sampled from MIS(center, right)
			//                      + throughput estimate sampled from MIS(center, bottom)
			//                      + throughput estimate sampled from MIS(center, left)).
			//
			//       Variable centralThroughput is the sum of four throughput estimates sampled
			//       from each of these distributions, from the central pixel, so it's actually four samples,
			//       and thus its weight is 4.
			//
			//       The other samples from the MIS'd distributions will be sampled from the neighboring pixels,
			//       and their weight is 1.
			//
			//       If this feels too complicated, it should be OK to output a standard throughput sample from
			//       the path tracer.

			// Add the throughput image as a preview. Note: Preview and final buffers are shared.
			{
#ifdef CENTRAL_RADIANCE
				block->put(samplePos, centralVeryDirect + centralThroughput, 1.0f, 1.0f, BUFFER_FINAL); // Standard throughput estimate with direct.
#else
				block->put(samplePos, (8 * centralVeryDirect) + (2 * centralThroughput), 4.0f, 4.0f, 0); // Adds very direct on top of the throughput image.
				
				block->put(left_pixel, (2 * shiftedThroughputs[LEFT]), 1.0f, 1.0f, BUFFER_FINAL);     // Negative x throughput.
				block->put(right_pixel, (2 * shiftedThroughputs[RIGHT]), 1.0f, 1.0f, BUFFER_FINAL);   // Positive x throughput.
				block->put(top_pixel, (2 * shiftedThroughputs[TOP]), 1.0f, 1.0f, BUFFER_FINAL);       // Negative y throughput.
				block->put(bottom_pixel, (2 * shiftedThroughputs[BOTTOM]), 1.0f, 1.0f, BUFFER_FINAL); // Positive y throughput.
#endif
			}

			// Actual throughputs, with MIS between central and neighbor pixels for all neighbors.
			// This can be replaced with a standard throughput sample without much loss of quality in most cases.
			{
#ifdef CENTRAL_RADIANCE
				block->put(samplePos, centralThroughput, 1.0f, 1.0f, BUFFER_THROUGHPUT); // Standard throughput estimate.
#else
				block->put(samplePos, (2 * centralThroughput), 4.0f, 4.0f, BUFFER_THROUGHPUT); // Central throughput.
	
				block->put(left_pixel, (2 * shiftedThroughputs[LEFT]), 1.0f, 1.0f, BUFFER_THROUGHPUT);     // Negative x throughput.
				block->put(right_pixel, (2 * shiftedThroughputs[RIGHT]), 1.0f, 1.0f, BUFFER_THROUGHPUT);   // Positive x throughput.
				block->put(top_pixel, (2 * shiftedThroughputs[TOP]), 1.0f, 1.0f, BUFFER_THROUGHPUT);       // Negative y throughput.
				block->put(bottom_pixel, (2 * shiftedThroughputs[BOTTOM]), 1.0f, 1.0f, BUFFER_THROUGHPUT); // Positive y throughput.
#endif
			}

			// Gradients.
			{
				block->put(left_pixel, -(2 * gradients[LEFT]), 1.0f, 1.0f, BUFFER_DX);    // Negative x gradient.
				block->put(center_pixel, (2 * gradients[RIGHT]), 1.0f, 1.0f, BUFFER_DX);  // Positive x gradient.
				block->put(top_pixel, -(2 * gradients[TOP]), 1.0f, 1.0f, BUFFER_DY);      // Negative y gradient.
				block->put(center_pixel, (2 * gradients[BOTTOM]), 1.0f, 1.0f, BUFFER_DY); // Positive y gradient.
			}

			// Very direct.
			block->put(center_pixel, centralVeryDirect, 1.0f, 1.0f, BUFFER_VERY_DIRECT);
		}
	}
}

/// Custom render function that samples a number of paths for evaluating differences between pixels.
bool GradientPathIntegrator::render(Scene *scene,
	RenderQueue *queue, const RenderJob *job,
	int sceneResID, int sensorResID, int samplerResID)
{
	if(m_hideEmitters) {
		/* Not supported! */
		Log(EError, "Option 'hideEmitters' not implemented for Gradient-Domain Path Tracing!");
	}

	/* Get config from the parent class. */
	m_config.m_maxDepth = m_maxDepth;
	m_config.m_minDepth = 1; // m_minDepth;
	m_config.m_rrDepth = m_rrDepth;
	m_config.m_strictNormals = m_strictNormals;

	/* Code duplicated from SamplingIntegrator::Render. */
	ref<Scheduler> sched = Scheduler::getInstance();
	ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));

	/* Set up MultiFilm. */
	ref<Film> film = sensor->getFilm();

	std::vector<std::string> outNames = {"-final", "-throughput", "-dx", "-dy", "-direct"};
	if (!film->setBuffers(outNames)){
		Log(EError, "Cannot render image! G-PT has been called without MultiFilm.");
		return false;
	}

	size_t nCores = sched->getCoreCount();
	const Sampler *sampler = static_cast<const Sampler *>(sched->getResource(samplerResID, 0));
	size_t sampleCount = sampler->getSampleCount();

	Log(EInfo, "Starting render job (GPT::render) (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT
		" %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
		sampleCount, sampleCount == 1 ? "sample" : "samples", nCores,
		nCores == 1 ? "core" : "cores");


	/* This is a sampling-based integrator - parallelize. */
	ref<BlockedRenderProcess> proc = new GPTRenderProcess(job, queue, scene->getBlockSize(), m_config);

	int integratorResID = sched->registerResource(this);
	proc->bindResource("integrator", integratorResID);
	proc->bindResource("scene", sceneResID);
	proc->bindResource("sensor", sensorResID);
	proc->bindResource("sampler", samplerResID);

	scene->bindUsedResources(proc);
	bindUsedResources(proc);
	sched->schedule(proc);

	m_process = proc;
	sched->wait(proc);

	sched->unregisterResource(integratorResID);
	m_process = NULL;

#ifdef RECONSTRUCT
	/* Reconstruct. */
	if(m_config.m_reconstructL1 || m_config.m_reconstructL2) {
		/* Allocate bitmaps for the solvers. */
		ref<Bitmap> throughputBitmap(new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getCropSize()));
		ref<Bitmap> directBitmap(new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getCropSize()));
		ref<Bitmap> dxBitmap(new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getCropSize()));
		ref<Bitmap> dyBitmap(new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getCropSize()));
		ref<Bitmap> reconstructionBitmap(new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getCropSize()));

		/* Develop primal and gradient data into bitmaps. */
		film->developMulti(Point2i(0, 0), film->getCropSize(), Point2i(0, 0), throughputBitmap, BUFFER_THROUGHPUT);
		film->developMulti(Point2i(0, 0), film->getCropSize(), Point2i(0, 0), dxBitmap, BUFFER_DX);
		film->developMulti(Point2i(0, 0), film->getCropSize(), Point2i(0, 0), dyBitmap, BUFFER_DY);
		film->developMulti(Point2i(0, 0), film->getCropSize(), Point2i(0, 0), directBitmap, BUFFER_VERY_DIRECT);

		/* Transform the data for the solver. */
		size_t subPixelCount = 3 * film->getCropSize().x * film->getCropSize().y;
		std::vector<float> throughputVector(subPixelCount);
		std::vector<float> dxVector(subPixelCount);
		std::vector<float> dyVector(subPixelCount);
		std::vector<float> directVector(subPixelCount);
		std::vector<float> reconstructionVector(subPixelCount);

		std::transform(throughputBitmap->getFloatData(), throughputBitmap->getFloatData() + subPixelCount, throughputVector.begin(), [](Float x) { return (float)x; });
		std::transform(dxBitmap->getFloatData(), dxBitmap->getFloatData() + subPixelCount, dxVector.begin(), [](Float x) { return (float)x; });
		std::transform(dyBitmap->getFloatData(), dyBitmap->getFloatData() + subPixelCount, dyVector.begin(), [](Float x) { return (float)x; });
		std::transform(directBitmap->getFloatData(), directBitmap->getFloatData() + subPixelCount, directVector.begin(), [](Float x) { return (float)x; });

		/* Reconstruct. */
		poisson::Solver::Params params;

		if(m_config.m_reconstructL1) {
			params.setConfigPreset("L1D");
		} else if(m_config.m_reconstructL2) {
			params.setConfigPreset("L2D");
		}

		params.alpha = (float)m_config.m_reconstructAlpha;
		params.setLogFunction(poisson::Solver::Params::LogFunction([](const std::string& message) { SLog(EInfo, "%s", message.c_str()); }));

		poisson::Solver solver(params);
		solver.importImagesMTS(dxVector.data(), dyVector.data(), throughputVector.data(), directVector.data(), film->getCropSize().x, film->getCropSize().y);

		solver.setupBackend();
		solver.solveIndirect();

		solver.exportImagesMTS(reconstructionVector.data());

		/* Give the solution back to Mitsuba. */
		int w = reconstructionBitmap->getSize().x;
		int h = reconstructionBitmap->getSize().y;

		for(int y = 0, p = 0; y < h; ++y) {
			for(int x = 0; x < w; ++x, p += 3) {
				Float color[3] = {(Float)reconstructionVector[p], (Float)reconstructionVector[p+1], (Float)reconstructionVector[p+2]};
				reconstructionBitmap->setPixel(Point2i(x, y), Spectrum(color));
			}
		}

		film->setBitmapMulti(reconstructionBitmap, 1, BUFFER_FINAL); 
	}
#endif

	return proc->getReturnStatus() == ParallelProcess::ESuccess;
}


static Float miWeight(Float pdfA, Float pdfB) {
	pdfA *= pdfA;
	pdfB *= pdfB;
	return pdfA / (pdfA + pdfB);
}

Spectrum GradientPathIntegrator::Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
	// Duplicate of MIPathTracer::Li to support sub-surface scattering initialization.

	/* Some aliases and local variables */
	const Scene *scene = rRec.scene;
	Intersection &its = rRec.its;
	RayDifferential ray(r);
	Spectrum Li(0.0f);
	bool scattered = false;

	/* Perform the first ray intersection (or ignore if the
		intersection has already been provided). */
	rRec.rayIntersect(ray);
	ray.mint = Epsilon;

	Spectrum throughput(1.0f);
	Float eta = 1.0f;

	while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
		if (!its.isValid()) {
			/* If no intersection could be found, potentially return
				radiance from a environment luminaire if it exists */
			if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
				&& (!m_hideEmitters || scattered))
				Li += throughput * scene->evalEnvironment(ray);
			break;
		}

		const BSDF *bsdf = its.getBSDF(ray);

		/* Possibly include emitted radiance if requested */
		if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
			&& (!m_hideEmitters || scattered))
			Li += throughput * its.Le(-ray.d);

		/* Include radiance from a subsurface scattering model if requested */
		if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
			Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

		if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
			|| (m_strictNormals && dot(ray.d, its.geoFrame.n)
				* Frame::cosTheta(its.wi) >= 0)) {

			/* Only continue if:
				1. The current path length is below the specifed maximum
				2. If 'strictNormals'=true, when the geometric and shading
				    normals classify the incident direction to the same side */
			break;
		}

		/* ==================================================================== */
		/*                     Direct illumination sampling                     */
		/* ==================================================================== */

		/* Estimate the direct illumination if this is requested */
		DirectSamplingRecord dRec(its);

		if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
			(bsdf->getType() & BSDF::ESmooth)) {
			Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
			if (!value.isZero()) {
				const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

				/* Allocate a record for querying the BSDF */
				BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

				/* Evaluate BSDF * cos(theta) */
				const Spectrum bsdfVal = bsdf->eval(bRec);

				/* Prevent light leaks due to the use of shading normals */
				if (!bsdfVal.isZero() && (!m_strictNormals
						|| dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

					/* Calculate prob. of having generated that direction
						using BSDF sampling */
					Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
						? bsdf->pdf(bRec) : 0;

					/* Weight using the power heuristic */
					Float weight = miWeight(dRec.pdf, bsdfPdf);
					Li += throughput * value * bsdfVal * weight;
				}
			}
		}

		/* ==================================================================== */
		/*                            BSDF sampling                             */
		/* ==================================================================== */

		/* Sample BSDF * cos(theta) */
		Float bsdfPdf;
		BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
		Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
		if (bsdfWeight.isZero())
			break;

		scattered |= bRec.sampledType != BSDF::ENull;

		/* Prevent light leaks due to the use of shading normals */
		const Vector wo = its.toWorld(bRec.wo);
		Float woDotGeoN = dot(its.geoFrame.n, wo);
		if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
			break;

		bool hitEmitter = false;
		Spectrum value;

		/* Trace a ray in this direction */
		ray = Ray(its.p, wo, ray.time);
		if (scene->rayIntersect(ray, its)) {
			/* Intersected something - check if it was a luminaire */
			if (its.isEmitter()) {
				value = its.Le(-ray.d);
				dRec.setQuery(ray, its);
				hitEmitter = true;
			}
		} else {
			/* Intersected nothing -- perhaps there is an environment map? */
			const Emitter *env = scene->getEnvironmentEmitter();

			if (env) {
				if (m_hideEmitters && !scattered)
					break;

				value = env->evalEnvironment(ray);
				if (!env->fillDirectSamplingRecord(dRec, ray))
					break;
				hitEmitter = true;
			} else {
				break;
			}
		}

		/* Keep track of the throughput and relative
			refractive index along the path */
		throughput *= bsdfWeight;
		eta *= bRec.eta;

		/* If a luminaire was hit, estimate the local illumination and
			weight using the power heuristic */
		if (hitEmitter &&
			(rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
			/* Compute the prob. of generating that direction using the
				implemented direct illumination sampling technique */
			const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
				scene->pdfEmitterDirect(dRec) : 0;
			Li += throughput * value * miWeight(bsdfPdf, lumPdf);
		}

		/* ==================================================================== */
		/*                         Indirect illumination                        */
		/* ==================================================================== */

		/* Set the recursive query type. Stop if no surface was hit by the
			BSDF sample or if indirect illumination was not requested */
		if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
			break;
		rRec.type = RadianceQueryRecord::ERadianceNoEmission;

		if (rRec.depth++ >= m_rrDepth) {
			/* Russian roulette: try to keep path weights equal to one,
				while accounting for the solid angle compression at refractive
				index boundaries. Stop with at least some probability to avoid
				getting stuck (e.g. due to total internal reflection) */

			Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
			if (rRec.nextSample1D() >= q)
				break;
			throughput /= q;
		}
	}

	return Li;
}


void GradientPathIntegrator::serialize(Stream *stream, InstanceManager *manager) const {
	MonteCarloIntegrator::serialize(stream, manager);
	m_config.serialize(stream);
}

std::string GradientPathIntegrator::toString() const {
	std::ostringstream oss;
	oss << "GradientPathTracer[" << endl
		<< "  maxDepth = " << m_maxDepth << "," << endl
		<< "  rrDepth = " << m_rrDepth << "," << endl
		<< "  shiftThreshold = " << m_config.m_shiftThreshold << endl
		<< "  reconstructL1 = " << m_config.m_reconstructL1 << endl
		<< "  reconstuctL2 = " << m_config.m_reconstructL2 << endl
		<< "  reconstructAlpha = " << m_config.m_reconstructAlpha << endl
		<< "]";
	return oss.str();
}


MTS_IMPLEMENT_CLASS_S(GradientPathIntegrator, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(GradientPathIntegrator, "Gradient Path Integrator");
MTS_NAMESPACE_END
