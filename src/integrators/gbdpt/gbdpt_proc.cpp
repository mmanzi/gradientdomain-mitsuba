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
#include "gbdpt_proc.h"
#include "mitsuba/bidir/mut_manifold.h"
MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                         Shift Path Data		                        */
/* ==================================================================== */
class  ShiftPathData
{
public:
	ShiftPathData(int n){
		jacobianDet.resize(n + 3, 1.0);
		genGeomTerm.resize(n + 3, 1.0);
	}
	std::vector<double> jacobianDet;
	std::vector<double> genGeomTerm;
	int vertex_b;
	MutationRecord muRec;
	bool couldConnectAfterB;
	bool success;
};

/* ==================================================================== */
/*                         Worker implementation                        */
/* ==================================================================== */

class GBDPTRenderer : public WorkProcessor {
public:
	GBDPTRenderer(const GBDPTConfiguration &config) : m_config(config) { }

	GBDPTRenderer(Stream *stream, InstanceManager *manager)
		: WorkProcessor(stream, manager), m_config(stream) { }

	virtual ~GBDPTRenderer() { }

	void serialize(Stream *stream, InstanceManager *manager) const {
		m_config.serialize(stream);
	}

	ref<WorkUnit> createWorkUnit() const {
		return new RectangularWorkUnit();
	}

	ref<WorkResult> createWorkResult() const {
		return new GBDPTWorkResult(m_config, m_rfilter.get(), Vector2i(m_config.blockSize), m_config.nNeighbours/*, m_config.nFeatures*/, m_config.extraBorder);
	}

	void prepare() {
		Scene *scene = static_cast<Scene *>(getResource("scene"));
		m_scene = new Scene(scene);
		m_sampler = static_cast<Sampler *>(getResource("sampler"));
		m_sensor = static_cast<Sensor *>(getResource("sensor"));
		m_rfilter = m_sensor->getFilm()->getReconstructionFilter();
		m_scene->removeSensor(scene->getSensor());
		m_scene->addSensor(m_sensor);
		m_scene->setSensor(m_sensor);
		m_scene->setSampler(m_sampler);
		m_scene->wakeup(NULL, m_resources);
		m_scene->initializeBidirectional();

		/* create offset path generator */
		m_offsetGenerator = new ManifoldPerturbation(m_scene, NULL, m_pool, 0.f, true, true, 0, 0, m_config.m_shiftThreshold);
	}

	void process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop) {
		const RectangularWorkUnit *rect = static_cast<const RectangularWorkUnit *>(workUnit);
		GBDPTWorkResult *result = static_cast<GBDPTWorkResult *>(workResult);
		bool needsTimeSample = m_sensor->needsTimeSample();
		Float time = m_sensor->getShutterOpen();

		result->setOffset(rect->getOffset());
		result->setSize(rect->getSize());
		result->clear();
		m_hilbertCurve.initialize(TVector2<uint8_t>(rect->getSize()));

		Path emitterSubpath;
		Path sensorSubpath;

		/*shift direction is hard-coded. future releases should support arbitrary kernels. */
		Vector2 shifts[4] = { Vector2(0, -1), Vector2(-1, 0), Vector2(1, 0), Vector2(0, 1) };

		if (m_config.maxDepth == -1){
			Log(EWarn, "maxDepth is unlimited, set to 12!");
			m_config.maxDepth = 12;
		}

		/* Determine the necessary random walk depths based on properties of
		the endpoints */
		int emitterDepth = m_config.maxDepth,
			sensorDepth = m_config.maxDepth;

		//marco: ensure some required properties (temporary solution)
		m_config.sampleDirect = false;

		/* Go one extra step if the sensor can be intersected */
		if (!m_scene->hasDegenerateSensor() && emitterDepth != -1)
			++emitterDepth;

		/* Sensor subpath legth +1 if there are emitters that can be intersected (to allow very direct sensor paths)*/
		if (!m_scene->hasDegenerateEmitters() && sensorDepth != -1)
			++sensorDepth;


		/*loop over pixels in block*/
		for (size_t i = 0; i<m_hilbertCurve.getPointCount(); ++i) {


			int neighborCount = m_config.nNeighbours;
			std::vector<ShiftPathData> pathData(neighborCount+1, sensorDepth+3);
			pathData[0].success = true;
			pathData[0].couldConnectAfterB = true;

			/*allocate memory depending on number of neighbours*/
			std::vector<Path> emitterSubpath(neighborCount + 1);
			std::vector<Path> sensorSubpath(neighborCount + 1);
			std::vector<double>	jacobianLP(neighborCount);
			std::vector<double>	genGeomTermLP(neighborCount + 1);
			std::vector<Spectrum> value(neighborCount + 1);
			std::vector<Float> miWeight(neighborCount + 1);
			std::vector<Float> valuePdf(neighborCount + 1);
			bool *pathSuccess = (bool *)alloca((neighborCount + 1) * sizeof(bool));

			Point2 samplePos;

			Point2i offset = Point2i(m_hilbertCurve[i]) + Vector2i(rect->getOffset());
			m_sampler->generate(offset);

			int spp = m_sampler->getSampleCount();

			/* For each sample */
			for (size_t j = 0; j<spp; j++) {
				if (stop)
					break;

				if (needsTimeSample)
					time = m_sensor->sampleTime(m_sampler->next1D());


				/* Start new emitter and sensor subpaths */
				emitterSubpath[0].initialize(m_scene, time, EImportance, m_pool);
				sensorSubpath[0].initialize(m_scene, time, ERadiance, m_pool);

				/* Perform a random walk using alternating steps on each path */
				Path::alternatingRandomWalkFromPixel(m_scene, m_sampler,
					emitterSubpath[0], emitterDepth, sensorSubpath[0],
					sensorDepth, offset, m_config.rrDepth, m_pool);


				samplePos = sensorSubpath[0].vertex(1)->getSamplePosition();

				double jx, jy;

				//marco: Hack- Required to store negative gradients...
				for (size_t i = 0; i < (1 + m_config.nNeighbours); ++i){
					const_cast<ImageBlock *>(result->getImageBlock(i))->setAllowNegativeValues(true);
					if (m_config.lightImage)
						const_cast<ImageBlock *>(result->getLightImage(i))->setAllowNegativeValues(true);
				}
				Path connectPath;
				int ptx;

				/* create shift-able path  */
				bool couldConnect = createShiftablePath(connectPath, emitterSubpath[0], sensorSubpath[0], 1, sensorSubpath[0].vertexCount() - 1, ptx);

				/*  geometry term(s) of base  */
				m_offsetGenerator->computeMuRec(connectPath, pathData[0].muRec);
				int idx = 0;
				for (int v = pathData[0].muRec.extra[0] - 1; v >= 0; v--){
					int idx = connectPath.vertexCount() - 1 - v;
					if (Path::isConnectable_GBDPT(connectPath.vertex(v), m_config.m_shiftThreshold) && v >= pathData[0].muRec.extra[2])
						pathData[0].genGeomTerm.at(idx) = connectPath.calcSpecularPDFChange(v, m_offsetGenerator);
					else
						pathData[0].genGeomTerm.at(idx) = pathData[0].genGeomTerm.at(idx - 1);
				}

				/*shift base path if possible*/
				for (int k = 0; k<neighborCount; k++){
					//we cannot shift very direct paths!
					pathData[k + 1].success = pathData[0].muRec.extra[0] <= 2 ? false : m_offsetGenerator->generateOffsetPathGBDPT(connectPath, sensorSubpath[k + 1], pathData[k + 1].muRec, shifts[k], pathData[k + 1].couldConnectAfterB, false);
					
					//if shift successful, compute jacobian and geometry term for each possible connection strategy that affects the shifted sub path
					//for the manifold exploration shift there are only two connectible vertices in the affected chain (v_b and v_c)
					if (pathData[k + 1].success){

						int idx = 0; int a, b, c;
						for (int v = pathData[k + 1].muRec.extra[0] - 1; v >= 0; v--){
							int idx = connectPath.vertexCount() - 1 - v;
							if (Path::isConnectable_GBDPT(connectPath.vertex(v), m_config.m_shiftThreshold) && v >= pathData[k + 1].muRec.extra[2]){
								a = pathData[k + 1].muRec.extra[0];
								b = v >= pathData[k + 1].muRec.extra[1] ? v : pathData[k + 1].muRec.extra[1];
								c = v >= pathData[k + 1].muRec.extra[1] ? v - 1 : pathData[k + 1].muRec.extra[2];
								jx = connectPath.halfJacobian_GBDPT(a, b, c, m_offsetGenerator);
								jy = sensorSubpath[k + 1].halfJacobian_GBDPT(a, b, c, m_offsetGenerator);
								pathData[k+1].jacobianDet.at(idx) = jy / jx;
								pathData[k+1].genGeomTerm.at(idx) = sensorSubpath[k + 1].calcSpecularPDFChange(v, m_offsetGenerator);
							}
							else{
								pathData[k + 1].jacobianDet.at(idx) = pathData[k + 1].jacobianDet.at(idx - 1);
								pathData[k + 1].genGeomTerm.at(idx) = pathData[k + 1].genGeomTerm.at(idx - 1);
							}
						}
					}
					sensorSubpath[k + 1].reverse();
				}

				/*save index of vertex b for evaluation (indexing is reversed)*/
				int v_b = connectPath.vertexCount() - 1 - pathData[0].muRec.extra[1];

				/* evaluate base and offset paths */
				evaluate(result, emitterSubpath[0], sensorSubpath, pathData, v_b,
					value, miWeight, valuePdf, jacobianLP, genGeomTermLP, pathSuccess);

				/* clean up memory */
				connectPath.release(ptx, ptx + 2, m_pool);
				for (int k = 0; k<neighborCount; k++){
					if (pathData[k+1].success){
						sensorSubpath[k + 1].reverse();
						sensorSubpath[k + 1].release(pathData[k + 1].muRec.l, pathData[k + 1].muRec.m + 1, m_pool);
					}
				}
				sensorSubpath[0].release(m_pool);
				emitterSubpath[0].release(m_pool);
				
				for (size_t i = 0; i < (1 + m_config.nNeighbours); ++i){
					const_cast<ImageBlock *>(result->getImageBlock(i))->setAllowNegativeValues(false);
					if (m_config.lightImage)
						const_cast<ImageBlock *>(result->getLightImage(i))->setAllowNegativeValues(false);
				}

				m_sampler->advance();
			}
		}
		/* Make sure that there were no memory leaks */
		Assert(m_pool.unused());
	}

	// Evaluate the contributions of the given eye and light paths
	void evaluate(GBDPTWorkResult *wr,
		Path &emitterSubpath, std::vector<Path> &sensorSubpath, std::vector<ShiftPathData> &pathData, int vert_b,
		std::vector<Spectrum> &value,  std::vector<Float> &miWeight, std::vector<Float> &valuePdf,
		std::vector<double> &jacobianLP, std::vector<double> &genGeomTermLP, bool *pathSuccess) {

		/* we use fixed neighborhood kernel! Future work will be to extend this to structurally-adaptive neighbours!!! */
		Vector2 shifts[4] = { Vector2(0, -1), Vector2(-1, 0), Vector2(1, 0), Vector2(0, 1) };
		int neighbourCount = 4;

		Point2 initialSamplePos = sensorSubpath[0].vertex(1)->getSamplePosition();
		int pixelIndex = (int)initialSamplePos[0] + (int)initialSamplePos[1] * m_sensor->getFilm()->getSize().x;

		const Scene *scene = m_scene;
		PathEdge  connectionEdge;

		/* combined weights along the two subpaths */
		Spectrum *importanceWeights = (Spectrum *)alloca(emitterSubpath.vertexCount() * sizeof(Spectrum));
		Spectrum **radianceWeights = (Spectrum **)alloca((neighbourCount + 1) * sizeof(Spectrum*));

		/* combined Pdfs along the two subpaths */
		Float *importancePdf = (Float*)alloca(emitterSubpath.vertexCount() * sizeof(Float));
		Float **radiancePdf = (Float **)alloca((neighbourCount + 1) * sizeof(Float*));

		for (int k = 0; k <= neighbourCount; k++){
			radianceWeights[k] = (Spectrum *)alloca(sensorSubpath[0].vertexCount()  * sizeof(Spectrum));
			radiancePdf[k] = (Float *)alloca(sensorSubpath[0].vertexCount() * sizeof(Float));
		}

		/* Compute the importance and radiance data */
		combineImportanceData(emitterSubpath, importanceWeights, importancePdf);
		combineRadianceData(sensorSubpath, pathData, neighbourCount, radianceWeights, radiancePdf);

		/* Allocate space for gradients */
		Spectrum primal = Spectrum(0.f);

		Spectrum *gradient = (Spectrum *)alloca(neighbourCount * sizeof(Spectrum));
		for (int k = 0; k<neighbourCount; k++){
			gradient[k] = Spectrum(0.f);
		}

		Path offsetEmitterSubpath, connectedBasePath;
		Spectrum geomTermBase, connectionPartsBase, offsetImportanceWeight;
		PathEdge connectionEdgeBase;
		bool successConnectBase, successOffsetGen, lightpathSuccess, samplePosValid;
		Float offsetImportancePdf, offsetImportanceGeom;
		Point2 samplePos;

		
		Spectrum visibility = Spectrum(0.f);
		Spectrum light = Spectrum(0.f);


		for (int s = (int)emitterSubpath.vertexCount() - 1; s >= 0; --s) {

			/* Determine the range of sensor vertices to be traversed, while respecting the specified maximum path length */
			int minT = std::max(2 - s, m_config.lightImage ? 1 : 2), // disable t=0 paths
				maxT = (int)sensorSubpath[0].vertexCount() - 1;
			if (m_config.maxDepth != -1)
				maxT = std::min(maxT, m_config.maxDepth + 1 - s);

			for (int t = maxT; t >= minT; --t) {

				samplePosValid = true;

				/* neighbour count and sample position for non-light paths */
				samplePos = initialSamplePos;

				/* if light path can be computed: recalculate pixel position and neighbours*/
				if (t == 1){
					if (sensorSubpath[0].vertex(t)->isSensorSample()&& !sensorSubpath[0].vertex(t)->getSamplePosition(emitterSubpath.vertex(s), samplePos)
						|| !Path::isConnectable_GBDPT(emitterSubpath.vertex(s), m_config.m_shiftThreshold))
						continue;
				}

				int memPointer;

				for (int k = 0; k <= neighbourCount; k++){

					miWeight[k] = 1.f / (s + t + 1);
					pathSuccess[k] = pathData[k].success;
					value[k] = Spectrum(0.f);
					valuePdf[k] = 0.f;

					/* for radiance make sure that offset light paths have the correct value */
					const Spectrum *importanceWeightTmp = &importanceWeights[s],
						           *radianceWeightTmp = &radianceWeights[t == 1 ? 0 : k][t];
					 
					const Float	 *importancePdfTmp = &importancePdf[s],
								 *radiancePdfTmp = &radiancePdf[t == 1 ? 0 : k][t];

					const Path	 *sensorSubpathTmp = &sensorSubpath[k],
								 *emitterSubpathTmp = &emitterSubpath;

					MutationRecord muRec;
					

					/* create the base path on which the shift is applied. changes for every light tracing path */
					if (t == 1 && k == 0){
						pathSuccess[0] = createShiftablePath(connectedBasePath, emitterSubpath, sensorSubpath[0], s, 1,  memPointer);
						m_offsetGenerator->computeMuRec(connectedBasePath, muRec);
						genGeomTermLP[0] =  connectedBasePath.calcSpecularPDFChange(muRec.extra[2], m_offsetGenerator, true);
					}	

					/* if we have a light tracing path we need to modify the emitter-sub path */
					if (t == 1 && k>0 && !value[0].isZero()){
						if (!pathSuccess[0])
							pathSuccess[k] = false;
						else{
							createShiftedLightPath(connectedBasePath, offsetEmitterSubpath, jacobianLP[k - 1], pathSuccess[k], offsetImportanceWeight, offsetImportancePdf, muRec, samplePos, shifts[k-1]/*nIdxL[k - 1]*/, s);
							if (pathSuccess[k]){
								genGeomTermLP[k] =  offsetEmitterSubpath.calcSpecularPDFChange(muRec.extra[2], m_offsetGenerator, true);
								importanceWeightTmp = &offsetImportanceWeight;
								importancePdfTmp = &offsetImportancePdf;
								emitterSubpathTmp = &offsetEmitterSubpath;
							}
							sensorSubpathTmp = &sensorSubpath[0];
						}
					}

					Spectrum geomTerm = Spectrum(0.0);


					do{ //this should really go away at some point...
						if (pathSuccess[k] && pathSuccess[0] && (k == 0 || (valuePdf[0]>0 && !value[0].isZero()))){

							if (!pathData[k].couldConnectAfterB && t > vert_b)
								break;

							PathVertex
								*vsPred = emitterSubpathTmp->vertexOrNull(s - 1),
								*vtPred = sensorSubpathTmp->vertexOrNull(t - 1),
								*vs = emitterSubpathTmp->vertex(s),						//connecting vertex from emitter side
								*vt = sensorSubpathTmp->vertex(t);						//connecting vertex from sensor side
							PathEdge
								*vsEdge = emitterSubpathTmp->edgeOrNull(s - 1),
								*vtEdge = sensorSubpathTmp->edgeOrNull(t - 1);

							/* Allowed remaining number of ENull vertices that can be bridged via pathConnect (negative=arbitrarily many) */
							int remaining = m_config.maxDepth - s - t + 1;

							/* Account for the terms of the measurement contribution function that are coupled to the connection endpoints, i.e. s==0*/
							if (vs->isEmitterSupernode()) {
								/* If possible, convert 'vt' into an emitter sample */
								if (!vt->cast(scene, PathVertex::EEmitterSample) || vt->isDegenerate()){
									valuePdf[k] = *radiancePdfTmp;
									break;
								}

								Spectrum connectionParts = (k > 0 && t > vert_b + 1) ? connectionPartsBase : vs->eval(scene, vsPred, vt, EImportance)    * vt->eval(scene, vtPred, vs, ERadiance);
								if (k == 0) connectionPartsBase = connectionParts;

								value[k] = *radianceWeightTmp * connectionParts;
								valuePdf[k] = *radiancePdfTmp;
							}
							/* Accounts for direct hits of light-subpaths to sensor, i.e t==0. If this happens do not compute gradients but fall back to the T0 mapping */
							else if (vt->isSensorSupernode()) {
								if (!vs->cast(scene, PathVertex::ESensorSample) || vs->isDegenerate() || k > 0) //Note the k>0...
								{
									valuePdf[k] = *importancePdfTmp;
									break;
								}
								/* Make note of the changed pixel sample position */
								if (!vs->getSamplePosition(vsPred, samplePos)){
									valuePdf[k] = *importancePdfTmp;
									break;
								}
								value[k] = *importanceWeightTmp *  vs->eval(scene, vsPred, vt, EImportance)    * vt->eval(scene, vtPred, vs, ERadiance);
								valuePdf[k] = *importancePdfTmp;
							}
							else {
								if (!Path::isConnectable_GBDPT(vs, m_config.m_shiftThreshold)
									|| !Path::isConnectable_GBDPT(vt, m_config.m_shiftThreshold)
									|| vs->getType() == 0 || vt->getType() == 0){
									valuePdf[k] = *importancePdfTmp    * *radiancePdfTmp;
									break;
								}
								Spectrum connectionParts = (k>0 && t>vert_b + 1) ? connectionPartsBase : vs->eval(scene, vsPred, vt, EImportance)    * vt->eval(scene, vtPred, vs, ERadiance);
								if (k == 0) connectionPartsBase = connectionParts;
								value[k] = *importanceWeightTmp * *radianceWeightTmp * connectionParts;
								valuePdf[k] = *importancePdfTmp    * *radiancePdfTmp ;
								vs->measure = vt->measure = EArea;
							}

							if (value[k].isZero() || valuePdf[k]==0) //early exit
								break;

							/* Attempt to connect the two endpoints, which could result in the creation of additional vertices (index-matched boundaries etc.) */
							//only required when connection segment changed from base to offset path
							int interactions = remaining; // backup
							bool successConnect = (k>0 && t>vert_b) ? successConnectBase : connectionEdge.pathConnectAndCollapse(scene, vsEdge, vs, vt, vtEdge, interactions);
							if (k == 0)	successConnectBase = successConnect;

							if (!successConnect){ //early exit
								value[k] = Spectrum(0.f);
								break; 
							}

							//this is the missing geometry factor in c_{s,t} of f_j mentioned in veach thesis equation 10.8
							geomTerm = (k>0 && t>vert_b) ? geomTermBase : connectionEdge.evalCached(vs, vt, PathEdge::EGeneralizedGeometricTerm);
							value[k] *= geomTerm;
							valuePdf[k] *= (t < 2 ? genGeomTermLP[k] : pathData[k].genGeomTerm[t]);

							if (value[k].isZero() || valuePdf[k]==0) //early exit
								break;


							if (k == 0){
								connectionEdgeBase = connectionEdge;
								geomTermBase = geomTerm;

							// using the original routine Path::miWeight is more efficient, but we use the balance heuristic for numberical stability...
							//	miWeight[0] = Path::miWeight(scene, emitterSubpath, &connectionEdge, sensorSubpath[0], s, t, m_config.sampleDirect, m_config.lightImage) / valuePdf[0];
								miWeight[0] = Path::miWeightBaseNoSweep_GBDPT(scene, emitterSubpath, &connectionEdgeBase, sensorSubpath[0],
										*emitterSubpathTmp, &connectionEdge, *sensorSubpathTmp, s, t, m_config.sampleDirect, m_config.lightImage,
										1.0, 2.0, (t<2 ? genGeomTermLP[0] : pathData[0].genGeomTerm[t]), 0.f, vert_b, m_config.m_shiftThreshold) / valuePdf[0];
							}
							else {
								/* compute MIS weight for gradients: 1/sum(p_st(x)^n+p_st(y)^n)*/
								// Note: we use the balance heuristic, not the power heuristic! The latter may cause numerical errors with long paths (since we compute the pdf explicitly)
								// some smarter computation should be done at some point to handle this
								miWeight[k] = Path::miWeightGradNoSweep_GBDPT(scene, emitterSubpath, &connectionEdgeBase, sensorSubpath[0],
									*emitterSubpathTmp, &connectionEdge, *sensorSubpathTmp, s, t, m_config.sampleDirect,  m_config.lightImage,
									(t<2 ? jacobianLP[k - 1] : pathData[k].jacobianDet[t] ), 1.0, 
									(t<2 ? genGeomTermLP[0] : pathData[0].genGeomTerm[t]), (t<2 ? genGeomTermLP[k] : pathData[k].genGeomTerm[t]), 
									vert_b, m_config.m_shiftThreshold) / valuePdf[0];
							}

						}
					} while (false);

					/* release offset emitter path for light path if needed */
					if (t == 1 && k>0 && pathSuccess[k] && !value[0].isZero())
						offsetEmitterSubpath.release(muRec.l, muRec.m + 1, m_pool);


					/* use T0 if base or offset is occluded with at least one strategy */
					if (value[k].isZero() || value[0].isZero()){
						value[k] = Spectrum(0.f);
						miWeight[k] = miWeight[0];
						valuePdf[k] = valuePdf[0];
					}
				}

				if (t == 1)
					connectedBasePath.release(memPointer, memPointer + 2, m_pool);


				if (value[0].isZero()) //if basepath is zero everything is zero!
					continue;

				/* store primal paths contribution */
				Spectrum mainRad = valuePdf[0] * miWeight[0] * value[0];
				if (t >= 2)
					primal += mainRad;
				else
					wr->putLightSample(samplePos, mainRad, 0);

				/* compute and store gradients */
				Spectrum fx = value[0] * valuePdf[0];
				for (int n = 0; n<neighbourCount; n++)
				{
					Spectrum fy = value[n + 1] * valuePdf[n + 1] * (t<2 ? jacobianLP[n] : pathData[n+1].jacobianDet[t]);
					Spectrum gradVal = Float(2.f)*miWeight[n + 1] * (fy - fx);
					if (t >= 2)
							gradient[n] += gradVal;
					else
							wr->putLightSample(samplePos, gradVal, n + 1);
				}
			}
		}

		/* store accumulated primal and gradient samples with t>=2 */
		wr->putSample(initialSamplePos, primal, 0);
		for (int k = 0; k < neighbourCount; ++k)
			wr->putSample(initialSamplePos , gradient[k], k + 1);
	}


	ref<WorkProcessor> clone() const {
		return new GBDPTRenderer(m_config);
	}

private:

	/* Compute importance data from the emitter path and store it in a easy to access way */
	void combineImportanceData(const Path &path, Spectrum *weights, Float *pdf){
		weights[0] = Spectrum(1.0f);
		pdf[0] = Float(1.0);
		for (size_t i = 1; i<path.vertexCount(); ++i){
			weights[i] = weights[i - 1] * path.vertex(i - 1)->weight[EImportance] * path.vertex(i - 1)->rrWeight * path.edge(i - 1)->weight[EImportance];
			pdf[i] = pdf[i - 1] * path.vertex(i - 1)->pdf[EImportance] * path.vertex(i - 1)->rrWeight * path.edge(i - 1)->pdf[EImportance];
		}
	}

	/* Compute radiance data from the emitter path and store it in a easy to access way */
	void combineRadianceData(const std::vector<Path> path, std::vector<ShiftPathData> pathData, const int neighbours, Spectrum **weights, Float **pdf){
		for (int k = 0; k <= neighbours; k++){
			weights[k][0] = Spectrum(1.0f);
			pdf[k][0] = Float(1.0);
			for (size_t i = 1; i<path[0].vertexCount(); ++i){
				if (pathData[k].success && i<path[k].vertexCount()){
					weights[k][i] = weights[k][i - 1] * path[k].vertex(i - 1)->weight[ERadiance] * path[k].vertex(i - 1)->rrWeight * path[k].edge(i - 1)->weight[ERadiance];
					pdf[k][i] = pdf[k][i - 1] * path[k].vertex(i - 1)->pdf[ERadiance] * path[k].vertex(i - 1)->rrWeight * path[k].edge(i - 1)->pdf[ERadiance];
				}
			}
		}
	}

	/* Shift path method for light tracing paths: Shifts the base path and stores important data */
	void createShiftedLightPath(Path &base, Path &offset, double &jacobian, bool &pathSuccess, Spectrum &offsetWeight, Float &offsetPdf, MutationRecord &muRec,
		const Point2 samplePos, const Vector2 shift, const int s){
		jacobian = 1.0;
		//get offset shift position					
		int bufferIdx; bool symmetric;
		Vector2 shiftOffset = shift;
		bool couldConnectWithB;
		pathSuccess = m_offsetGenerator->generateOffsetPathGBDPT(base, offset, muRec, shiftOffset, couldConnectWithB, true);

		//we also need to recompute the jacobian determinant
		if (pathSuccess){
			jacobian = offset.halfJacobian_GBDPT(muRec.extra[0], muRec.extra[1], muRec.extra[2], m_offsetGenerator) /
				base.halfJacobian_GBDPT(muRec.extra[0], muRec.extra[1], muRec.extra[2], m_offsetGenerator);

			offsetPdf = Float(1.0);	// pdf of ERadiance of SensorSample ( =vertex(s+1) ) is always 1
			offsetWeight = Spectrum(1.f); 

			for (size_t i = 1; i <= s; ++i){
				offsetWeight = offsetWeight * offset.vertex(i - 1)->weight[EImportance] * offset.vertex(i - 1)->rrWeight * offset.edge(i - 1)->weight[EImportance];
				offsetPdf = offsetPdf * offset.vertex(i - 1)->pdf[EImportance] * offset.vertex(i - 1)->rrWeight * offset.edge(i - 1)->pdf[EImportance];
			}
		}
	}
	/**
	*This method connects a emitter and sensor subpath to one new path at the s,t position.
	*This is a hack to allow proper support of the perturbation  routines, since partial paths
	*cannot be shifted properly without breaking much of the manifold walk code.
	*
	*This method involves ray-tracing to determine the connection and is hence costy, but for light 
	*tracing paths this is needed anyway to ensure that we don't create base paths that are occluded 
	*from the eye. So the overhead is relatively small. 
	**/
	bool createShiftablePath(Path &connectedPath, Path &emitterSubpath, Path &sensorSubpath, int s, int t, int &memPointer){
		connectedPath.clear();

		//make sure we connect across connect able vertices only!
		while (!Path::isConnectable_GBDPT(sensorSubpath.vertex(t), m_config.m_shiftThreshold)){
			t--;
			//the tail of the path can then be removed
			sensorSubpath.removeAndReleaseLastElement(m_pool);
		}

		//if the sensor connection vertex is on the lightsource then we must connect to the emitterSuperSample

		if (sensorSubpath.vertex(t)->type == PathVertex::ESurfaceInteraction){
			const Intersection &its = sensorSubpath.vertex(t)->getIntersection();
			const Emitter *emitter = its.shape->getEmitter();
			if (emitter)
				s = 0;
		}

		//append unchanged part of emitter subpath				
		for (memPointer = 0; memPointer<s; memPointer++){
			connectedPath.append(emitterSubpath.vertex(memPointer));
			connectedPath.append(emitterSubpath.edge(memPointer));
		}

		//clone last vertex of emittersubpath part that is used in this connection
		connectedPath.append(emitterSubpath.vertex(memPointer));
		connectedPath.vertex(memPointer) = emitterSubpath.vertex(memPointer)->clone(m_pool);

		//add a new edge in between the connection
		connectedPath.append(m_pool.allocEdge());
		memset(connectedPath.edge(memPointer), 0, sizeof(PathEdge));

		//clone first vertex of sensorsubpath
		connectedPath.append(sensorSubpath.vertex(t));
		connectedPath.vertex(memPointer + 1) = sensorSubpath.vertex(t)->clone(m_pool);
		connectedPath.vertex(memPointer + 1)->cast(m_scene, PathVertex::EEmitterSample);

		//append unchanged part of sensor subpath
		for (int i = t - 1; i >= 0; i--){
			connectedPath.append(sensorSubpath.vertex(i));
			connectedPath.append(sensorSubpath.edge(i));
		}

		//make connection if we don't connect to the emitterSuperSample. note: if the connection is occluded it returns false!
		bool pathSuccess = true;
		pathSuccess = PathVertex::connect(m_scene,
			connectedPath.vertexOrNull(memPointer - 1),
			connectedPath.edgeOrNull(memPointer - 1),
			connectedPath.vertex(memPointer),
			connectedPath.edge(memPointer),
			connectedPath.vertex(memPointer + 1),
			connectedPath.edgeOrNull(memPointer + 1),
			connectedPath.vertexOrNull(memPointer + 2),
			connectedPath.vertex(memPointer)->isConnectable() ? EArea : EDiscrete,
			connectedPath.vertex(memPointer + 1)->isConnectable() ? EArea : EDiscrete);

		//update image-space position if its a light path connection
		if (t == 1)
			connectedPath.vertex(connectedPath.vertexCount() - 2)->updateSamplePosition(connectedPath.vertex(connectedPath.vertexCount() - 3));

		return pathSuccess;
	}


	MTS_DECLARE_CLASS()

private:
	ref<Scene> m_scene;
	ref<Sensor> m_sensor;
	ref<Sampler> m_sampler;
	ref<ReconstructionFilter> m_rfilter;
	MemoryPool m_pool;
	GBDPTConfiguration m_config;
	HilbertCurve2D<uint8_t> m_hilbertCurve;
	ref<ManifoldPerturbation> m_offsetGenerator;
	
};


/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

GBDPTProcess::GBDPTProcess(const RenderJob *parent, RenderQueue *queue,
	const GBDPTConfiguration &config) :
	BlockedRenderProcess(parent, queue, config.blockSize), m_config(config) {
	m_refreshTimer = new Timer();
}

ref<WorkProcessor> GBDPTProcess::createWorkProcessor() const {
	return new GBDPTRenderer(m_config);
}

void GBDPTProcess::develop() {
	if (!m_config.lightImage)
		return;

	LockGuard lock(m_resultMutex);
	for (int i = 0; i <= m_config.nNeighbours; ++i){
			m_film->setBitmapMulti(m_result->getImageBlock(i)->getBitmap()->crop(Point2i(m_config.extraBorder), m_film->getCropSize()), Float(1.0), i); 
			m_film->addBitmapMulti(m_result->getLightImage(i)->getBitmap(), 1.0f / m_config.sampleCount, i);
	}

	m_refreshTimer->reset();
	m_queue->signalRefresh(m_parent);
}

void GBDPTProcess::processResult(const WorkResult *wr, bool cancelled) {
	if (cancelled)
		return;

	const GBDPTWorkResult *result = static_cast<const GBDPTWorkResult *>(wr);
	LockGuard lock(m_resultMutex);
	m_progress->update(++m_resultCount);
	ImageBlock *block = NULL;

	if (m_config.lightImage) {
		block = const_cast<ImageBlock *>(result->getImageBlock(0));
		const ImageBlock *lightImage = m_result->getLightImage(0);
		m_result->put(result);	
		if (m_parent->isInteractive()) {
			/* Modify the finished image block so that it includes the light image contributions,
			which creates a more intuitive preview of the rendering process. This is
			not 100% correct but doesn't matter, as the shown image will be properly re-developed
			every 2 seconds and once more when the rendering process finishes */
			Float invSampleCount = 1.0f / m_config.sampleCount;
			const Bitmap *sourceBitmap = lightImage->getBitmap();
			Bitmap *destBitmap = block->getBitmap();
			int borderSize = block->getBorderSize();
			Point2i offset = block->getOffset();
			Vector2i size = block->getSize();

			for (int y = 0; y<size.y; ++y) {
				const Float *source = sourceBitmap->getFloatData() + (offset.x + (y + offset.y) * sourceBitmap->getWidth()) * SPECTRUM_SAMPLES;
				Float *dest = destBitmap->getFloatData() + (borderSize + (y + borderSize) * destBitmap->getWidth()) * (SPECTRUM_SAMPLES + 2);

				for (int x = 0; x<size.x; ++x) {
					Float weight = dest[SPECTRUM_SAMPLES + 1] * invSampleCount;
					for (int k = 0; k<SPECTRUM_SAMPLES; ++k)
						*dest++ += *source++ * weight;
					dest += 2;
				}
			}
		}
		m_film->put(block);	
	}
	else{
		for (int i = 0; i <= m_config.nNeighbours; ++i){
			block = const_cast<ImageBlock *>(result->getImageBlock(i));
			m_film->putMulti(block, i);	
		}
	}
	
	/* Re-develop the entire image every two seconds if partial results are
	visible (e.g. in a graphical user interface). This only applies when
	there is a light image. */
	bool developFilm = m_config.lightImage && (m_parent->isInteractive() && m_refreshTimer->getMilliseconds() > 2000);

	m_queue->signalWorkEnd(m_parent, result->getImageBlock(), false);

	if (developFilm)
		develop();
}

void GBDPTProcess::bindResource(const std::string &name, int id) {
	BlockedRenderProcess::bindResource(name, id);
	if (name == "sensor" && m_config.lightImage) {
		/* If needed, allocate memory for the light image */
		m_result = new GBDPTWorkResult(m_config, NULL, m_film->getCropSize(), m_config.nNeighbours, m_config.extraBorder);
		m_result->clear();
	}
}

MTS_IMPLEMENT_CLASS_S(GBDPTRenderer, false, WorkProcessor)
MTS_IMPLEMENT_CLASS(GBDPTProcess, false, BlockedRenderProcess)
MTS_NAMESPACE_END
