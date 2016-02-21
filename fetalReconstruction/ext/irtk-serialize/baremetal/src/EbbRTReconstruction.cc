//          Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#include "EbbRTReconstruction.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <utils.h>

#include <ebbrt/LocalIdMap.h>
#include <ebbrt/GlobalIdMap.h>
#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/IOBuf.h>
#include <ebbrt/Debug.h>
#include <ebbrt/Messenger.h>
#include <ebbrt/SpinBarrier.h>
#include <ebbrt/MulticoreEbb.h>
#include <ebbrt/EbbRef.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

EBBRT_PUBLISH_TYPE(, EbbRTReconstruction);

using namespace ebbrt;

EbbRTReconstruction& EbbRTReconstruction::HandleFault(ebbrt::EbbId id) {
  {
    ebbrt::LocalIdMap::ConstAccessor accessor;
    auto found = ebbrt::local_id_map->Find(accessor, id);
    if (found) {
      auto& pr = *boost::any_cast<EbbRTReconstruction*>(accessor->second);
      ebbrt::EbbRef<EbbRTReconstruction>::CacheRef(id, pr);
      return pr;
    }
  }

  ebbrt::EventManager::EventContext context;
  auto f = ebbrt::global_id_map->Get(id);
  EbbRTReconstruction* p;
  f.Then([&f, &context, &p, id](ebbrt::Future<std::string> inner) {
    p = new EbbRTReconstruction(ebbrt::Messenger::NetworkId(inner.Get()), id);
    ebbrt::event_manager->ActivateContext(std::move(context));
  });
  ebbrt::event_manager->SaveContext(context);
  auto inserted = ebbrt::local_id_map->Insert(std::make_pair(id, p));
  if (inserted) {
    ebbrt::EbbRef<EbbRTReconstruction>::CacheRef(id, *p);
    return *p;
  }

  delete p;
  // retry reading
  ebbrt::LocalIdMap::ConstAccessor accessor;
  ebbrt::local_id_map->Find(accessor, id);
  auto& pr = *boost::any_cast<EbbRTReconstruction*>(accessor->second);
  ebbrt::EbbRef<EbbRTReconstruction>::CacheRef(id, pr);
  return pr;
}

void EbbRTReconstruction::Print(const char* str) {
  auto len = strlen(str) + 1;
  auto buf = ebbrt::MakeUniqueIOBuf(len);
  snprintf(reinterpret_cast<char*>(buf->MutData()), len, "%s", str);

  ebbrt::kprintf("Sending %d bytes\n", buf->ComputeChainDataLength());

  SendMessage(remote_nid_, std::move(buf));
}

void ResetOrigin(irtkRealImage& image,
                 irtkRigidTransformation& transformation) {
  double ox, oy, oz;
  image.GetOrigin(ox, oy, oz);
  image.PutOrigin(0, 0, 0);
  transformation.PutTranslationX(ox);
  transformation.PutTranslationY(oy);
  transformation.PutTranslationZ(oz);
  transformation.PutRotationX(0);
  transformation.PutRotationY(0);
  transformation.PutRotationZ(0);
}

class MPMultiEbbCtr;
typedef EbbRef<MPMultiEbbCtr> MPMultiEbbCtrRef;

class MPMultiEbbCtrRoot {
 private:
  friend class MPMultiEbbCtr;
  EbbId _id;
  mutable MPMultiEbbCtr* repArray[ebbrt::Cpu::kMaxCpus];

  MPMultiEbbCtrRoot(EbbId id) : _id(id) { bzero(repArray, sizeof(repArray)); }

  void setRep(size_t i, MPMultiEbbCtr* rep) const { repArray[i] = rep; }

  int gatherVal(void) const;
  int otherGatherVal(void) const;
  void destroy() const;
};

class MPMultiEbbCtr : public MulticoreEbb<MPMultiEbbCtr, MPMultiEbbCtrRoot> {
  typedef MPMultiEbbCtrRoot Root;
  typedef MulticoreEbb<MPMultiEbbCtr, Root> Parent;
  const Root& _root;
  EbbId myId() { return _root._id; }

  int _val;
  MPMultiEbbCtr(const Root& root) : _root(root), _val(0) {
    _root.setRep(ebbrt::Cpu::GetMine(), this);
  }
  // give access to the constructor
  friend Parent;
  friend MPMultiEbbCtrRoot;

 public:
  void inc() { _val++; }
  void dec() { _val--; }
  void print() { ebbrt::kprintf("mycpu: %d\n", ebbrt::Cpu::GetMine()); }

  int val() { return _root.gatherVal(); }

  void destroy() { _root.destroy(); }

  static MPMultiEbbCtrRef Create(EbbId id = ebb_allocator->AllocateLocal()) {
    return Parent::Create(new Root(id), id);
  }
};

void MPMultiEbbCtrRoot::destroy(void) const {
  size_t numCores = ebbrt::Cpu::Count();
  for (size_t i = 0; numCores && i < ebbrt::Cpu::kMaxCpus; i++) {
    if (repArray[i]) {
      delete repArray[i];
      numCores--;
    }
  }
  delete this;
}

int MPMultiEbbCtrRoot::gatherVal(void) const {
  int gval = 0;
  size_t numCores = ebbrt::Cpu::Count();
  for (size_t i = 0; numCores && i < ebbrt::Cpu::kMaxCpus; i++) {
    if (repArray[i]) {
      gval = repArray[i]->_val;
      numCores--;
    }
  }
  return gval;
}

int MPMultiEbbCtrRoot::otherGatherVal(void) const {
  int gval = 0;
  LocalIdMap::ConstAccessor accessor;  // serves as a lock on the rep map
  auto found = local_id_map->Find(accessor, _id);
  if (!found)
    throw std::runtime_error("Failed to find root for MulticoreEbb");
  auto pair = boost::any_cast<std::pair<
      MPMultiEbbCtrRoot*, boost::container::flat_map<size_t, MPMultiEbbCtr*>>>(
      &accessor->second);
  const auto& rep_map = pair->second;
  for (auto it = rep_map.begin(); it != rep_map.end(); it++) {
    auto rep = boost::any_cast<const MPMultiEbbCtr*>(it->second);
    gval += rep->_val;
  }
  return gval;
};

inline double G(double x, double s, double _step)
{
  return _step*exp(-x*x / (2 * s)) / (sqrt(6.28*s));
}

inline double M(double m, double _step)
{
  return m*_step;
}

//Puts origin of the image into origin of world coordinates
void ResetOrigin(irtkGreyImage &image,
                                     irtkRigidTransformation &transformation) {
  double ox, oy, oz;
  image.GetOrigin(ox, oy, oz);
  image.PutOrigin(0, 0, 0);
  transformation.PutTranslationX(ox);
  transformation.PutTranslationY(oy);
  transformation.PutTranslationZ(oz);
  transformation.PutRotationX(0);
  transformation.PutRotationY(0);
  transformation.PutRotationZ(0);
}

void SetSmoothingParameters(double delta, double lambda, double& _delta, double& _lambda, double& _alpha)
{
    _delta = delta;
    _lambda = lambda*delta*delta;
    _alpha = 0.05 / lambda;
    if (_alpha > 1) _alpha = 1;
}

void ParallelAdaptiveRegularization2(int* _directions, irtkRealImage& _reconstructed, irtkRealImage& _confidence_map, double& _alpha, double& _lambda, double& _delta, vector<irtkRealImage>& _b, vector<double>& _factor, irtkRealImage& _original)
{
    int dx = _reconstructed.GetX();
    int dy = _reconstructed.GetY();
    int dz = _reconstructed.GetZ();
    for (size_t x = 0; x != _reconstructed.GetX(); ++x) {
	int xx, yy, zz;
	for (int y = 0; y < dy; y++)
	    for (int z = 0; z < dz; z++) {
		double val = 0;
		double valW = 0;
		double sum = 0;
		for (int i = 0; i < 13; i++) {
		    xx = x + _directions[i][0];
		    yy = y + _directions[i][1];
		    zz = z + _directions[i][2];
		    if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) &&
			(zz < dz)) {
			val += _b[i](x, y, z) * _original(xx, yy, zz) *
			    _confidence_map(xx, yy, zz);
			valW +=
			    _b[i](x, y, z) * _confidence_map(xx, yy, zz);
			sum += _b[i](x, y, z);
		    }
		}

		for (int i = 0; i < 13; i++) {
		    xx = x - _directions[i][0];
		    yy = y - _directions[i][1];
		    zz = z - _directions[i][2];
		    if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) &&
			(zz < dz)) {
			val += _b[i](xx, yy, zz) * _original(xx, yy, zz) *
			    _confidence_map(xx, yy, zz);
			valW +=
			    _b[i](xx, yy, zz) * _confidence_map(xx, yy, zz);
			sum += _b[i](xx, yy, zz);
		    }
		}

		val -=
		    sum * _original(x, y, z) * _confidence_map(x, y, z);
		valW -= sum * _confidence_map(x, y, z);
		val = _original(x, y, z) * _confidence_map(x, y, z) +
		    _alpha * _lambda /
                    (_delta * _delta) * val;
		valW = _confidence_map(x, y, z) +
		    _alpha * _lambda /
		    (_delta * _delta) * valW;

		if (valW > 0) {
		    _reconstructed(x, y, z) = val / valW;
		} else
		    _reconstructed(x, y, z) = 0;
	    }
    }
}

void ParallelAdaptiveRegularization1(int* _directions, irtkRealImage& _reconstructed, irtkRealImage& _confidence_map, double& _alpha, double& _lambda, double& _delta, vector<irtkRealImage>& _b, vector<double>& _factor, irtkRealImage& _original)
{
    int dx = _reconstructed.GetX();
    int dy = _reconstructed.GetY();
    int dz = _reconstructed.GetZ();
    for (size_t i = 0; i != 13; ++i) {
	int x, y, z, xx, yy, zz;
	double diff;
	for (x = 0; x < dx; x++)
	    for (y = 0; y < dy; y++)
		for (z = 0; z < dz; z++) {
		    xx = x + _directions[i][0];
		    yy = y + _directions[i][1];
		    zz = z + _directions[i][2];
		    if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) &&
			(zz < dz) && (_confidence_map(x, y, z) > 0) &&
			(_confidence_map(xx, yy, zz) > 0)) {
			diff = (_original(xx, yy, zz) - _original(x, y, z)) *
			    sqrt(_factor[i]) / _delta;
			b[i](x, y, z) = _factor[i] / sqrt(1 + diff * diff);

		    } else
			_b[i](x, y, z) = 0;
		}
    }
}
 
void AdaptiveRegularization(inter iter, irtkRealImage& _original, int* _directions, irtkRealImage& _reconstructed, irtkRealImage& _confidence_map, double& _alpha, double& _lambda, double& _delta)
{
    vector<double> factor(13, 0);
    for (int i = 0; i < 13; i++) {
	for (int j = 0; j < 3; j++)
	    factor[i] += fabs(double(_directions[i][j]));
	factor[i] = 1 / factor[i];
    }
    
    vector<irtkRealImage> b; //(13);
    for (int i = 0; i < 13; i++)
	b.push_back(_reconstructed);

    ParallelAdaptiveRegularization1(_directions, _reconstructed, _confidence_map, _alpha, _lambda, _delta, b, factor, _original);
	
    irtkRealImage _original2 = _reconstructed;
    ParallelAdaptiveRegularization2(_directions, _reconstructed, _confidence_map, _alpha, _lambda, _delta, b, factor, _original);

    if (_alpha * _lambda / (_delta * _delta) > 0.068) {
	ebbrt::kprintf("Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068.\n");
    }
}

void BiasCorrectVolume(irtkRealImage& _original, double& _low_intensity_cutoff, double& _min_intensity, double& _max_intensity, irtkRealImage& _reconstructed, irtkRealImage& _mask)
{
    // remove low-frequancy component in the reconstructed image which might have
    // accured due to overfitting of the biasfield
    irtkRealImage residual = _reconstructed;
    irtkRealImage weights = _mask;

    // calculate weighted residual
    irtkRealPixel *pr = residual.GetPointerToVoxels();
    irtkRealPixel *po = original.GetPointerToVoxels();
    irtkRealPixel *pw = weights.GetPointerToVoxels();
    for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
	// second and term to avoid numerical problems
	if ((*pw == 1) && (*po > _low_intensity_cutoff * _max_intensity) &&
	    (*pr > _low_intensity_cutoff * _max_intensity)) {
	    *pr /= *po;
	    *pr = log(*pr);
	} else {
	    *pw = 0;
	    *pr = 0;
	}
	pr++;
	po++;
	pw++;
    }
    // residual.Write("residual.nii.gz");
    // blurring needs to be same as for slices
    irtkGaussianBlurring<irtkRealPixel> gb(_sigma_bias);
    // blur weigted residual
    gb.SetInput(&residual);
    gb.SetOutput(&residual);
    gb.Run();
    // blur weight image
    gb.SetInput(&weights);
    gb.SetOutput(&weights);
    gb.Run();

    // calculate the bias field
    pr = residual.GetPointerToVoxels();
    pw = weights.GetPointerToVoxels();
    irtkRealPixel *pm = _mask.GetPointerToVoxels();
    irtkRealPixel *pi = _reconstructed.GetPointerToVoxels();
    for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {

	if (*pm == 1) {
	    // weighted gaussian smoothing
	    *pr /= *pw;
	    // exponential to recover multiplicative bias field
	    *pr = exp(*pr);
	    // bias correct reconstructed
	    *pi /= *pr;
	    // clamp intensities to allowed range
	    if (*pi < _min_intensity * 0.9)
		*pi = _min_intensity * 0.9;
	    if (*pi > _max_intensity * 1.1)
		*pi = _max_intensity * 1.1;
	} else {
	    *pr = 0;
	}
	pr++;
	pw++;
	pm++;
	pi++;
    }
}

void NormaliseBias(int iter, irtkRealImage& _volume_weights, irtkRealImage& _mask, double& _sigma_bias, irtkRealImage& _reconstructed, std::vector<irtkRealImage>& _slices, std::vector<irtkRealImage>& _bias, std::vector<double>& _scale_cpu, std::vector<SLICECOEFFS>& _volcoeffs, int start, int end)
{
    irtkRealImage bias;
    bias.Initialize(_reconstructed.GetImageAttributes());
    bias = 0;

    for (size_t inputIndex = (size_t)start; inputIndex < (size_t)end; ++inputIndex) {
	// alias the current slice
	irtkRealImage &slice = reconstructor->_slices[inputIndex];

	// read the current bias image
	irtkRealImage b = reconstructor->_bias[inputIndex];

	// read current scale factor
	double scale = reconstructor->_scale_cpu[inputIndex];

	irtkRealPixel *pi = slice.GetPointerToVoxels();
	irtkRealPixel *pb = b.GetPointerToVoxels();
	for (int i = 0; i < slice.GetNumberOfVoxels(); i++) {
	    if ((*pi > -1) && (scale > 0))
		*pb -= log(scale);
	    pb++;
	    pi++;
	}

	// Distribute slice intensities to the volume
	POINT3D p;
	for (int i = 0; i < slice.GetX(); i++)
	    for (int j = 0; j < slice.GetY(); j++)
		if (slice(i, j, 0) != -1) {
		    // number of volume voxels with non-zero coefficients for current
		    // slice voxel
		    int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
		    // add contribution of current slice voxel to all voxel volumes
		    // to which it contributes
		    for (int k = 0; k < n; k++) {
			p = reconstructor->_volcoeffs[inputIndex][i][j][k];
			bias(p.x, p.y, p.z) += p.value * b(i, j, 0);
		    }
		}
    }// end of loop for a slice inputIndex
    
    // normalize the volume by proportion of contributing slice voxels for each
    // volume voxel
    bias /= _volume_weights;

}

void Superresolution(int iter, double _min_intensity, double _max_intensity, irtkRealImage& _reconstructed, bool _global_bias_correction, std::vector<irtkRealImage>& _slices, std::vector<irtkRealImage>& _weights, std::vector<irtkRealImage>& _bias, std::vector<double>& _scale_cpu, std::vector<irtkRealImage>& _simulated_weights, std::vector<irtkRealImage>& _simulated_slices, int start, int end, irtkRealImage& _confidence_map, bool _adaptive, int* _directions, double _alpha, double _lambda, double _delta,  irtkRealImage& _mask, double _low_intensity_cutoff)
{
    int i, j, k;
    irtkRealImage addon, original;
    
    // Remember current reconstruction for edge-preserving smoothing
    original = _reconstructed;

    // Clear addon
    addon.Initialize(_reconstructed.GetImageAttributes());
    addon = 0;
    
    // Clear confidence map
    _confidence_map.Initialize(_reconstructed.GetImageAttributes());
    _confidence_map = 0;

    for (size_t inputIndex = (size_t)start; inputIndex < (size_t)end; ++inputIndex)
    {
	// read the current slice
	irtkRealImage slice = _slices[inputIndex];

	// read the current weight image
	irtkRealImage &w = _weights[inputIndex];

	// read the current bias image
	irtkRealImage &b = _bias[inputIndex];

	// identify scale factor
	double scale = _scale_cpu[inputIndex];

	// Update reconstructed volume using current slice

	// Distribute error to the volume
	POINT3D p;
	for (int i = 0; i < slice.GetX(); i++)
	    for (int j = 0; j < slice.GetY(); j++)
		if (slice(i, j, 0) != -1) {
		    // bias correct and scale the slice
		    slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

		    if (_simulated_slices[inputIndex](i, j, 0) > 0)
			slice(i, j, 0) -=
			    _simulated_slices[inputIndex](i, j, 0);
		    else
			slice(i, j, 0) = 0;

		    int n = _volcoeffs[inputIndex][i][j].size();
		    for (int k = 0; k < n; k++) {
			p = _volcoeffs[inputIndex][i][j][k];
			addon(p.x, p.y, p.z) +=
			    p.value * slice(i, j, 0) * w(i, j, 0) *
			    _slice_weight_cpu[inputIndex];
			confidence_map(p.x, p.y, p.z) +=
			    p.value * w(i, j, 0) *
			    _slice_weight_cpu[inputIndex];
		    }
		}
    } // end of loop for a slice inputIndex

    if (!_adaptive)
	for (i = 0; i < addon.GetX(); i++)
	    for (j = 0; j < addon.GetY(); j++)
		for (k = 0; k < addon.GetZ(); k++)
		    if (_confidence_map(i, j, k) > 0) {
			// ISSUES if _confidence_map(i, j, k) is too small leading
			// to bright pixels
			addon(i, j, k) /= _confidence_map(i, j, k);
			// this is to revert to normal (non-adaptive) regularisation
			_confidence_map(i, j, k) = 1;
		    }
    
    _reconstructed += addon * _alpha; //_average_volume_weight;
    
    // bound the intensities
    for (i = 0; i < _reconstructed.GetX(); i++)
	for (j = 0; j < _reconstructed.GetY(); j++)
	    for (k = 0; k < _reconstructed.GetZ(); k++) {
		if (_reconstructed(i, j, k) < _min_intensity * 0.9)
		    _reconstructed(i, j, k) = _min_intensity * 0.9;
		if (_reconstructed(i, j, k) > _max_intensity * 1.1)
		    _reconstructed(i, j, k) = _max_intensity * 1.1;
	    }

    // Smooth the reconstructed image
    AdaptiveRegularization(iter, original, _directions, _reconstructed, _confidence_map, _alpha, _lambda, _delta);
    
    // Remove the bias in the reconstructed volume compared to previous iteration
    if (_global_bias_correction)
	BiasCorrectVolume(original, _low_intensity_cutoff, _min_intensity, _max_intensity, _reconstructed, _mask);

}

void Scale(std::vector<irtkRealImage>& _slices, std::vector<irtkRealImage>& _weights, std::vector<irtkRealImage>& _bias, std::vector<double>& _scale_cpu, std::vector<irtkRealImage>& _simulated_weights, std::vector<irtkRealImage>& _simulated_slices, int start, int end)
{
    size_t inputIndex = 0;
    for(inputIndex = (size_t)start; inputIndex != (size_t)end; inputIndex++) {
	// alias the current slice
	irtkRealImage &slice = _slices[inputIndex];

	// alias the current weight image
	irtkRealImage &w = _weights[inputIndex];

	// alias the current bias image
	irtkRealImage &b = _bias[inputIndex];

	// initialise calculation of scale
	double scalenum = 0;
	double scaleden = 0;

	for (int i = 0; i < slice.GetX(); i++)
	    for (int j = 0; j < slice.GetY(); j++)
		if (slice(i, j, 0) != -1) {
		    if (_simulated_weights[inputIndex](i, j, 0) > 0.99) {
			// scale - intensity matching
			double eb = exp(-b(i, j, 0));
			scalenum += w(i, j, 0) * slice(i, j, 0) * eb *
			    _simulated_slices[inputIndex](i, j, 0);
			scaleden +=
			    w(i, j, 0) * slice(i, j, 0) * eb * slice(i, j, 0) * eb;
		    }
		}

	// calculate scale for this slice
	if (scaleden > 0)
	    _scale_cpu[inputIndex] = scalenum / scaleden;
	else
	    _scale_cpu[inputIndex] = 1;
    }
}

void Bias(std::vector<irtkRealImage>& _slices, std::vector<irtkRealImage>& _weights, std::vector<irtkRealImage>& _bias, std::vector<double>& _scale_cpu, std::vector<irtkRealImage>& _simulated_weights, std::vector<irtkRealImage>& _simulated_slices, bool _global_bias_correction, int start, int end)
{
    size_t inputIndex = 0;
    for(inputIndex = (size_t)start; inputIndex != (size_t)end; inputIndex++) {
	// read the current slice
	irtkRealImage slice = _slices[inputIndex];

	// alias the current weight image
	irtkRealImage &w = _weights[inputIndex];

	// alias the current bias image
	irtkRealImage b = _bias[inputIndex];

	// identify scale factor
	double scale = _scale_cpu[inputIndex];

	// prepare weight image for bias field
	irtkRealImage wb = w;

	// simulated slice
	irtkRealImage wresidual(slice.GetImageAttributes());
	wresidual = 0;

	for (int i = 0; i < slice.GetX(); i++)
	    for (int j = 0; j < slice.GetY(); j++)
		if (slice(i, j, 0) != -1) {
		    if (_simulated_weights[inputIndex](i, j, 0) > 0.99) {
			// bias-correct and scale current slice
			double eb = exp(-b(i, j, 0));
			slice(i, j, 0) *= (eb * scale);

			// calculate weight image
			wb(i, j, 0) = w(i, j, 0) * slice(i, j, 0);

			// calculate weighted residual image
			// make sure it is far from zero to avoid numerical instability
			if ((_simulated_slices[inputIndex](i, j, 0) > 1) &&
			    (slice(i, j, 0) > 1)) {
			    wresidual(i, j, 0) =
				log(slice(i, j, 0) /
				    _simulated_slices[inputIndex](i, j, 0)) *
				wb(i, j, 0);
			}
		    } else {
			// do not take into account this voxel when calculating bias field
			wresidual(i, j, 0) = 0;
			wb(i, j, 0) = 0;
		    }
		}

	// calculate bias field for this slice
	irtkGaussianBlurring<irtkRealPixel> gb(_sigma_bias);
	// smooth weighted residual
	gb.SetInput(&wresidual);
	gb.SetOutput(&wresidual);
	gb.Run();

	// smooth weight image
	gb.SetInput(&wb);
	gb.SetOutput(&wb);
	gb.Run();

	// update bias field
	double sum = 0;
	double num = 0;
	for (int i = 0; i < slice.GetX(); i++)
	    for (int j = 0; j < slice.GetY(); j++)
		if (slice(i, j, 0) != -1) {
		    if (wb(i, j, 0) > 0)
			b(i, j, 0) += wresidual(i, j, 0) / wb(i, j, 0);
		    sum += b(i, j, 0);
		    num++;
		}

	// normalize bias field to have zero mean
	if (!_global_bias_correction) {
	    double mean = 0;
	    if (num > 0)
		mean = sum / num;
	    for (int i = 0; i < slice.GetX(); i++)
		for (int j = 0; j < slice.GetY(); j++)
		    if ((slice(i, j, 0) != -1) && (num > 0)) {
			b(i, j, 0) -= mean;
		    }
	}

	_bias[inputIndex] = b;
    }	
}

void EStep(std::vector<irtkRealImage>& _slices, std::vector<irtkRealImage>& _weights, std::vector<irtkRealImage>& _bias, std::vector<double>& _scale_cpu, std::vector<SLICECOEFFS>& _volcoeffs, std::vector<irtkRealImage>& _simulated_slices, std::vector<irtkRealImage>& _simulated_weights, double& _sigma_cpu, double& _m_cpu, double& _mix_cpu, std::vector<int>& _small_slices, std::vector<double>& _slice_weight_cpu, double& _mean_s_cpu, double& _mean_s2_cpu, double& _sigma_s_cpu, double& _sigma_s2_cpu, double& _mix_s_cpu, int start, int end, double _step)
{
    size_t inputIndex;
    irtkRealImage slice, w, b, sim;
    int num = 0;
    vector<double> slice_potential(_slices.size(), 0);

    for(inputIndex = (size_t)start; inputIndex != (size_t)end; inputIndex++) {
	// read the current slice
	irtkRealImage slice = _slices[inputIndex];

	// read current weight image
	_weights[inputIndex] = 0;

	// alias the current bias image
	irtkRealImage &b = _bias[inputIndex];

	// identify scale factor
	double scale = _scale_cpu[inputIndex];

	double num = 0;
	// Calculate error, voxel weights, and slice potential
	for (int i = 0; i < slice.GetX(); i++)
	    for (int j = 0; j < slice.GetY(); j++)
		if (slice(i, j, 0) != -1) {
		    // bias correct and scale the slice
		    slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

		    // number of volumetric voxels to which
		    // current slice voxel contributes
		    int n = _volcoeffs[inputIndex][i][j].size();

		    // if n == 0, slice voxel has no overlap with volumetric ROI,
		    // do not process it

		    if ((n > 0) &&
			(_simulated_weights[inputIndex](i, j, 0) > 0)) {
			slice(i, j, 0) -=
			    _simulated_slices[inputIndex](i, j, 0);

			// calculate norm and voxel-wise weights

			// Gaussian distribution for inliers (likelihood)
			double g =
			    G(slice(i, j, 0), _sigma_cpu, _step);
			// Uniform distribution for outliers (likelihood)
			double m = M(_m_cpu, _step);

			// voxel_wise posterior
			double weight = g * _mix_cpu /
			    (g * _mix_cpu +
			     m * (1 - _mix_cpu));
			_weights[inputIndex].PutAsDouble(i, j, 0, weight);
			
			// calculate slice potentials
			if (_simulated_weights[inputIndex](i, j, 0) >
			    0.99) {
			    slice_potential[inputIndex] += (1.0 - weight) * (1.0 - weight);
			    num++;
			}
		    } else
			_weights[inputIndex].PutAsDouble(i, j, 0, 0);
		}
	
	// evaluate slice potential
	if (num > 0) {
	    slice_potential[inputIndex] = sqrt(slice_potential[inputIndex] / num);
	} else
	    slice_potential[inputIndex] = -1; // slice has no unpadded voxels
    }

    // exclude slices identified as having small overlap with ROI, set their
    // potentials to -1
    for (unsigned int i = 0; i < _small_slices.size(); i++)
	slice_potential[_small_slices[i]] = -1;

    // these are unrealistic scales pointing at misregistration - exclude the
    // corresponding slices
    for (inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
	if ((_scale_cpu[inputIndex] < 0.2) || (_scale_cpu[inputIndex] > 5)) {
	    slice_potential[inputIndex] = -1;
	}

    // Calulation of slice-wise robust statistics parameters.
    // This is theoretically M-step,
    // but we want to use latest estimate of slice potentials
    // to update the parameters

    // Calculate means of the inlier and outlier potentials
    double sum = 0, den = 0, sum2 = 0, den2 = 0, maxs = 0, mins = 1;
    for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
	if (slice_potential[inputIndex] >= 0) {
	    // calculate means
	    sum += slice_potential[inputIndex] * _slice_weight_cpu[inputIndex];
	    den += _slice_weight_cpu[inputIndex];
	    sum2 +=
		slice_potential[inputIndex] * (1 - _slice_weight_cpu[inputIndex]);
	    den2 += (1 - _slice_weight_cpu[inputIndex]);

	    // calculate min and max of potentials in case means need to be initalized
	    if (slice_potential[inputIndex] > maxs)
		maxs = slice_potential[inputIndex];
	    if (slice_potential[inputIndex] < mins)
		mins = slice_potential[inputIndex];
	}

    if (den > 0)
	_mean_s_cpu = sum / den;
    else
	_mean_s_cpu = mins;

    if (den2 > 0)
	_mean_s2_cpu = sum2 / den2;
    else
	_mean_s2_cpu = (maxs + _mean_s_cpu) / 2;

    // Calculate the variances of the potentials
    sum = 0;
    den = 0;
    sum2 = 0;
    den2 = 0;
    for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
	if (slice_potential[inputIndex] >= 0) {
	    sum += (slice_potential[inputIndex] - _mean_s_cpu) *
		(slice_potential[inputIndex] - _mean_s_cpu) *
		_slice_weight_cpu[inputIndex];
	    den += _slice_weight_cpu[inputIndex];

	    sum2 += (slice_potential[inputIndex] - _mean_s2_cpu) *
		(slice_potential[inputIndex] - _mean_s2_cpu) *
		(1 - _slice_weight_cpu[inputIndex]);
	    den2 += (1 - _slice_weight_cpu[inputIndex]);
	}

    //_sigma_s
    if ((sum > 0) && (den > 0)) {
	_sigma_s_cpu = sum / den;
	// do not allow too small sigma
	if (_sigma_s_cpu < _step * _step / 6.28)
	    _sigma_s_cpu = _step * _step / 6.28;
    } else {
	_sigma_s_cpu = 0.025;
    }

    // sigma_s2
    if ((sum2 > 0) && (den2 > 0)) {
	_sigma_s2_cpu = sum2 / den2;
	// do not allow too small sigma
	if (_sigma_s2_cpu < _step * _step / 6.28)
	    _sigma_s2_cpu = _step * _step / 6.28;
    } else {
	_sigma_s2_cpu =
	    (_mean_s2_cpu - _mean_s_cpu) * (_mean_s2_cpu - _mean_s_cpu) / 4;
	// do not allow too small sigma
	if (_sigma_s2_cpu < _step * _step / 6.28)
	    _sigma_s2_cpu = _step * _step / 6.28;
    }

    // Calculate slice weights
    double gs1, gs2;
    for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
	// Slice does not have any voxels in volumetric ROI
	if (slice_potential[inputIndex] == -1) {
	    _slice_weight_cpu[inputIndex] = 0;
	    continue;
	}

	// All slices are outliers or the means are not valid
	if ((den <= 0) || (_mean_s2_cpu <= _mean_s_cpu)) {
	    _slice_weight_cpu[inputIndex] = 1;
	    continue;
	}

	// likelihood for inliers
	if (slice_potential[inputIndex] < _mean_s2_cpu)
	    gs1 = G(slice_potential[inputIndex] - _mean_s_cpu, _sigma_s_cpu, _step);
	else
	    gs1 = 0;

	// likelihood for outliers
	if (slice_potential[inputIndex] > _mean_s_cpu)
	    gs2 = G(slice_potential[inputIndex] - _mean_s2_cpu, _sigma_s2_cpu, _step);
	else
	    gs2 = 0;

	// calculate slice weight
	double likelihood = gs1 * _mix_s_cpu + gs2 * (1 - _mix_s_cpu);
	if (likelihood > 0)
	    _slice_weight_cpu[inputIndex] = gs1 * _mix_s_cpu / likelihood;
	else {
	    if (slice_potential[inputIndex] <= _mean_s_cpu)
		_slice_weight_cpu[inputIndex] = 1;
	    if (slice_potential[inputIndex] >= _mean_s2_cpu)
		_slice_weight_cpu[inputIndex] = 0;
	    if ((slice_potential[inputIndex] < _mean_s2_cpu) &&
		(slice_potential[inputIndex] > _mean_s_cpu)) // should not happen
		_slice_weight_cpu[inputIndex] = 1;
	}
    }

    // Update _mix_s this should also be part of MStep
    sum = 0;
    num = 0;
    for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
	if (slice_potential[inputIndex] >= 0) {
	    sum += _slice_weight_cpu[inputIndex];
	    num++;
	}

    if (num > 0)
	_mix_s_cpu = sum / num;
    else {
	_mix_s_cpu = 0.9;
    }
}

void InitializeRobustStatistics(double& _sigma_cpu, double& _sigma_s_cpu, double& _mix_cpu, double& _mix_s_cpu, double& _m_cpu, int start, int end, std::vector<irtkRealImage>& _slices, std::vector<irtkRealImage>& _simulated_slices, std::vector<irtkRealImage>& _simulated_weights, std::vector<irtkRealImage>& _simulated_inside, std::vector<bool>& _slice_inside_cpu,  std::vector<double>& _slice_weight_cpu, double& _max_intensity, double& _min_intensity)
{
    // Initialise parameter of EM robust statistics
    int i, j;
    irtkRealImage slice, sim;
    double sigma = 0;
    int num = 0;

    // for each slice
    for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
	slice = _slices[inputIndex];

	// Voxel-wise sigma will be set to stdev of volumetric errors
	// For each slice voxel
	for (i = 0; i < slice.GetX(); i++)
	    for (j = 0; j < slice.GetY(); j++)
		if (slice(i, j, 0) != -1) {
		    // calculate stev of the errors
		    if ((_simulated_inside[inputIndex](i, j, 0) == 1) &&
			(_simulated_weights[inputIndex](i, j, 0) > 0.99)) {
			slice(i, j, 0) -= _simulated_slices[inputIndex](i, j, 0);
			sigma += slice(i, j, 0) * slice(i, j, 0);
			num++;
		    }
		}

	// if slice does not have an overlap with ROI, set its weight to zero
	if (!_slice_inside_cpu[inputIndex])
	    _slice_weight_cpu[inputIndex] = 0;
    }

    // Force exclusion of slices predefined by user
    //for (unsigned int i = 0; i < _force_excluded.size(); i++)
//	_slice_weight_cpu[_force_excluded[i]] = 0;

    // initialize sigma for voxelwise robust statistics
    _sigma_cpu = sigma / num;
    // initialize sigma for slice-wise robust statistics
    _sigma_s_cpu = 0.025;
    // initialize mixing proportion for inlier class in voxel-wise robust
    // statistics
    _mix_cpu = 0.9;
    // initialize mixing proportion for outlier class in slice-wise robust
    // statistics
    _mix_s_cpu = 0.9;
    // Initialise value for uniform distribution according to the range of
    // intensities
    _m_cpu = 1 / (2.1 * _max_intensity - 1.9 * _min_intensity);

}

void SimulateSlices(std::vector<irtkRealImage>& _slices, std::vector<irtkRealImage>& _simulated_slices, std::vector<irtkRealImage>& _simulated_weights, std::vector<irtkRealImage>& _simulated_inside, std::vector<bool>& _slice_inside_cpu, std::vector<SLICECOEFFS>& _volcoeffs, irtkRealImage& _reconstructed, irtkRealImage& _mask, int start, int end)
{
    size_t inputIndex = 0;
    for (inputIndex = (size_t)start; inputIndex != (size_t)end; inputIndex++) {
	// Calculate simulated slice
	_simulated_slices[inputIndex].Initialize(
	    _slices[inputIndex].GetImageAttributes());
	_simulated_slices[inputIndex] = 0;

	_simulated_weights[inputIndex].Initialize(
	    _slices[inputIndex].GetImageAttributes());
	_simulated_weights[inputIndex] = 0;

	_simulated_inside[inputIndex].Initialize(
	    _slices[inputIndex].GetImageAttributes());
	_simulated_inside[inputIndex] = 0;

	_slice_inside_cpu[inputIndex] = false;

	POINT3D p;
	for (unsigned int i = 0; i < _slices[inputIndex].GetX();
	     i++)
	    for (unsigned int j = 0; j < _slices[inputIndex].GetY();
		 j++)
		if (_slices[inputIndex](i, j, 0) != -1) {
		    double weight = 0;
		    int n = _volcoeffs[inputIndex][i][j].size();
		    for (unsigned int k = 0; k < n; k++) {
			p = _volcoeffs[inputIndex][i][j][k];
			_simulated_slices[inputIndex](i, j, 0) +=
			    p.value * _reconstructed(p.x, p.y, p.z);
			weight += p.value;
			if (_mask(p.x, p.y, p.z) == 1) {
			    _simulated_inside[inputIndex](i, j, 0) = 1;
			    _slice_inside_cpu[inputIndex] = true;
			}
		    }
		    if (weight > 0) {
			_simulated_slices[inputIndex](i, j, 0) /= weight;
			_simulated_weights[inputIndex](i, j, 0) = weight;
		    }
		}
    }
}


void GaussianReconstruction(irtkRealImage& _reconstructed, std::vector<int>& _small_slices, std::vector<irtkRealImage>& _slices, std::vector<irtkRealImage>& _bias, std::vector<double>& _scale_cpu, std::vector<SLICECOEFFS>& _volcoeffs, irtkRealImage& _volume_weights)
{
    unsigned int inputIndex;
    int i, j, k, n;
    irtkRealImage slice;
    double scale;
    POINT3D p;
    vector<int> voxel_num;
    int slice_vox_num;

    // clear _reconstructed image
    _reconstructed = 0;

    // CPU
    for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
	// copy the current slice
	slice = _slices[inputIndex];
	// alias the current bias image
	irtkRealImage &b = _bias[inputIndex];
	// read current scale factor
	scale = _scale_cpu[inputIndex];

	slice_vox_num = 0;

	// Distribute slice intensities to the volume
	for (i = 0; i < slice.GetX(); i++)
	    for (j = 0; j < slice.GetY(); j++)
		if (slice(i, j, 0) != -1) {
		    // biascorrect and scale the slice
		    slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

		    // number of volume voxels with non-zero coefficients
		    // for current slice voxel
		    n = _volcoeffs[inputIndex][i][j].size();

		    // if given voxel is not present in reconstructed volume at all,
		    // pad it
		    // calculate num of vox in a slice that have overlap with roi
		    if (n > 0)
			slice_vox_num++;

		    // add contribution of current slice voxel to all voxel volumes
		    // to which it contributes
		    for (k = 0; k < n; k++) {
			p = _volcoeffs[inputIndex][i][j][k];
			_reconstructed(p.x, p.y, p.z) += p.value * slice(i, j, 0);
		    }
		}
	voxel_num.push_back(slice_vox_num);
    }

    // normalize the volume by proportion of contributing slice voxels
    // for each volume voxe
    _reconstructed /= _volume_weights;

    // now find slices with small overlap with ROI and exclude them.
    vector<int> voxel_num_tmp;
    for (i = 0; i < voxel_num.size(); i++)
	voxel_num_tmp.push_back(voxel_num[i]);

    // find median
    sort(voxel_num_tmp.begin(), voxel_num_tmp.end());
    int median = voxel_num_tmp[round(voxel_num_tmp.size() * 0.5)];

    // remember slices with small overlap with ROI
    _small_slices.clear();
    for (i = 0; i < voxel_num.size(); i++)
	if (voxel_num[i] < 0.1 * median)
	    _small_slices.push_back(i);
}

void InitializeEM(std::vector<irtkRealImage>& _slices
		  , std::vector<irtkRealImage>& _weights
		  , std::vector<irtkRealImage>& _bias
		  , std::vector<double>& _slice_weight_cpu
		  , std::vector<double>& _scale_cpu
		  , double& _max_intensity
		  , double& _min_intensity) {

    _weights.clear();
    _bias.clear();
    _scale_cpu.clear();
    _slice_weight_cpu.clear();
    
    for (unsigned int i = 0; i < _slices.size(); i++) {
	// Create images for voxel weights and bias fields
	_weights.push_back(_slices[i]);
	_bias.push_back(_slices[i]);
	
	// Create and initialize scales
	_scale_cpu.push_back(1);
	
	// Create and initialize slice weights
	_slice_weight_cpu.push_back(1);
    }
    
    // TODO CUDA
    // Find the range of intensities
    _max_intensity = voxel_limits<irtkRealPixel>::min();
    _min_intensity = voxel_limits<irtkRealPixel>::max();
    for (unsigned int i = 0; i < _slices.size(); i++) {
	// to update minimum we need to exclude padding value
	irtkRealPixel *ptr = _slices[i].GetPointerToVoxels();
	for (int ind = 0; ind < _slices[i].GetNumberOfVoxels(); ind++) {
	    if (*ptr > 0) {
		if (*ptr > _max_intensity)
		    _max_intensity = *ptr;
		if (*ptr < _min_intensity)
		    _min_intensity = *ptr;
	    }
	    ptr++;
	}
    }
}

void InitializeEMValues(std::vector<irtkRealImage>& _slices
			, std::vector<irtkRealImage>& _weights
			, std::vector<irtkRealImage>& _bias
			, std::vector<double>& _slice_weight_cpu
			, std::vector<double>& _scale_cpu) {
    for (unsigned int i = 0; i < _slices.size(); i++) {
	// Initialise voxel weights and bias values
	irtkRealPixel *pw = _weights[i].GetPointerToVoxels();
	irtkRealPixel *pb = _bias[i].GetPointerToVoxels();
	irtkRealPixel *pi = _slices[i].GetPointerToVoxels();
	for (int j = 0; j < _weights[i].GetNumberOfVoxels(); j++) {
	    if (*pi != -1) {
		*pw = 1;
		*pb = 0;
	    } else {
		*pw = 0;
		*pb = 0;
	    }
	    pi++;
	    pw++;
	    pb++;
	}
	
	// Initialise slice weights
	_slice_weight_cpu[i] = 1;
	
	// Initialise scaling factors for intensity matching
	_scale_cpu[i] = 1;
    }
}

void SliceToVolumeRegistration(std::vector<double>& _slices_regCertainty
			       , irtkRealImage& _reconstructed
			       ,std::vector<irtkRealImage>& _slices
			       , std::vector<irtkRigidTransformation>& _transformations
			       , int start
			       , int end)
{
    if (_slices_regCertainty.size() == 0) {
	_slices_regCertainty.resize(_slices.size());
    }

    irtkImageAttributes attr = _reconstructed.GetImageAttributes();
    
    size_t inputIndex = 0;
    for(inputIndex = (size_t)start; inputIndex != (size_t)end; inputIndex++) {
	irtkImageRigidRegistrationWithPadding registration;
	irtkGreyPixel smin, smax;
	irtkGreyImage target;
	irtkRealImage slice, w, b, t;
	irtkResamplingWithPadding<irtkRealPixel> resampling(attr._dx, attr._dx,
							    attr._dx, -1);
	
	t = _slices[inputIndex];
	resampling.SetInput(&_slices[inputIndex]);
	resampling.SetOutput(&t);
	resampling.Run();
	target = t;
	target.GetMinMax(&smin, &smax);
    
	if (smax > -1) {
	    // put origin to zero
	    irtkRigidTransformation offset;
	    ResetOrigin(target, offset);
	    irtkMatrix mo = offset.GetMatrix();
	    irtkMatrix m =_transformations[inputIndex].GetMatrix();
	    m = m * mo;
	    _transformations[inputIndex].PutMatrix(m);
	    
	    irtkGreyImage source = _reconstructed;
	    registration.SetInput(&target, &source);
	    registration.SetOutput(&_transformations[inputIndex]);
	    registration.GuessParameterSliceToVolume();
	    registration.SetTargetPadding(-1);
	    registration.Run();
	    
	    _slices_regCertainty[inputIndex] = registration.last_similarity;
	    // undo the offset
	    mo.Invert();
	    m = _transformations[inputIndex].GetMatrix();
	    m = m * mo;
	    _transformations[inputIndex].PutMatrix(m);
	}
    }
}

void CoeffInit(irtkRealImage& _mask, irtkRealImage& _reconstructed,
	       double& _quality_factor, size_t& _max_slices,
	       std::vector<irtkRealImage>& _slices,
	       std::vector<irtkRigidTransformation>& _transformations,
	       std::vector<SLICECOEFFS>& _volcoeffs,
	       std::vector<bool>& _slice_inside_cpu, int start, int end
	       , irtkRealImage& _volume_weights) {
    
    _volcoeffs.clear();
    _volcoeffs.resize(_slices.size());
    
    _slice_inside_cpu.clear();
    _slice_inside_cpu.resize(_slices.size());

    size_t inputIndex = 0;
    for (inputIndex = (size_t)start; inputIndex != (size_t)end; inputIndex++) {
	bool slice_inside;
	
	// get resolution of the volume
	double vx, vy, vz;
	_reconstructed.GetPixelSize(&vx, &vy, &vz);

	// volume is always isotropic
	double res = vx;

	// read the slice
	irtkRealImage& slice = _slices[inputIndex];

	// prepare structures for storage
	POINT3D p;
	VOXELCOEFFS empty;
	SLICECOEFFS slicecoeffs(slice.GetX(),
				std::vector<VOXELCOEFFS>(slice.GetY(), empty));

	// to check whether the slice has an overlap with mask ROI
	slice_inside = false;

	// PSF will be calculated in slice space in higher resolution
	// get slice voxel size to define PSF
	double dx, dy, dz;
	slice.GetPixelSize(&dx, &dy, &dz);

	// sigma of 3D Gaussian (sinc with FWHM=dx or dy in-plane, Gaussian with
	// FWHM = dz through-plane)
	double sigmax = 1.2 * dx / 2.3548;
	double sigmay = 1.2 * dy / 2.3548;
	double sigmaz = dz / 2.3548;

	// calculate discretized PSF

	// isotropic voxel size of PSF - derived from resolution of reconstructed
	// volume
	double size = res / _quality_factor;

	// number of voxels in each direction
	// the ROI is 2*voxel dimension

	int xDim = round(2 * dx / size);
	int yDim = round(2 * dy / size);
	int zDim = round(2 * dz / size);

	// image corresponding to PSF
	irtkImageAttributes attr;
	attr._x = xDim;
	attr._y = yDim;
	attr._z = zDim;
	attr._dx = size;
	attr._dy = size;
	attr._dz = size;
	irtkRealImage PSF(attr);

	// centre of PSF
	double cx, cy, cz;
	cx = 0.5 * (xDim - 1);
	cy = 0.5 * (yDim - 1);
	cz = 0.5 * (zDim - 1);
	PSF.ImageToWorld(cx, cy, cz);

	double x, y, z;
	double sum = 0;
	int i, j, k;
	for (i = 0; i < xDim; i++)
	    for (j = 0; j < yDim; j++)
		for (k = 0; k < zDim; k++) {
		    x = i;
		    y = j;
		    z = k;
		    PSF.ImageToWorld(x, y, z);
		    x -= cx;
		    y -= cy;
		    z -= cz;
		    // continuous PSF does not need to be normalized as discrete will be
		    PSF(i, j, k) = exp(-x * x / (2 * sigmax * sigmax) -
				       y * y / (2 * sigmay * sigmay) -
				       z * z / (2 * sigmaz * sigmaz));
		    sum += PSF(i, j, k);
		}
	PSF /= sum;

	// prepare storage for PSF transformed and resampled to the space of
	// reconstructed volume
	// maximum dim of rotated kernel - the next higher odd integer plus two to
	// accound for rounding error of tx,ty,tz.
	// Note conversion from PSF image coordinates to tPSF image coordinates
	// *size/res
	int dim =
	    (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) *
			size / res) /
		   2)) *
            2 +
	    1 + 2;
	// prepare image attributes. Voxel dimension will be taken from the
	// reconstructed volume
	attr._x = dim;
	attr._y = dim;
	attr._z = dim;
	attr._dx = res;
	attr._dy = res;
	attr._dz = res;
	// create matrix from transformed PSF
	irtkRealImage tPSF(attr);
	// calculate centre of tPSF in image coordinates
	int centre = (dim - 1) / 2;

	// for each voxel in current slice calculate matrix coefficients
	int ii, jj, kk;
	int tx, ty, tz;
	int nx, ny, nz;
	int l, m, n;
	double weight;
	for (i = 0; i < slice.GetX(); i++)
	    for (j = 0; j < slice.GetY(); j++)
		if (slice(i, j, 0) != -1) {
		    // calculate centrepoint of slice voxel in volume space (tx,ty,tz)
		    x = i;
		    y = j;
		    z = 0;
		    slice.ImageToWorld(x, y, z);
		    _transformations[inputIndex].Transform(x, y, z);
		    _reconstructed.WorldToImage(x, y, z);
		    tx = round(x);
		    ty = round(y);
		    tz = round(z);

		    // Clear the transformed PSF
		    for (ii = 0; ii < dim; ii++)
			for (jj = 0; jj < dim; jj++)
			    for (kk = 0; kk < dim; kk++)
				tPSF(ii, jj, kk) = 0;

		    // for each POINT3D of the PSF
		    for (ii = 0; ii < xDim; ii++)
			for (jj = 0; jj < yDim; jj++)
			    for (kk = 0; kk < zDim; kk++) {
				// Calculate the position of the POINT3D of
				// PSF centered over current slice voxel
				// This is a bit complicated because slices
				// can be oriented in any direction

				// PSF image coordinates
				x = ii;
				y = jj;
				z = kk;
				// change to PSF world coordinates - now real sizes in mm
				PSF.ImageToWorld(x, y, z);
				// centre around the centrepoint of the PSF
				x -= cx;
				y -= cy;
				z -= cz;

				// Need to convert (x,y,z) to slice image
				// coordinates because slices can have
				// transformations included in them (they are
				// nifti)  and those are not reflected in
				// PSF. In slice image coordinates we are
				// sure that z is through-plane

				// adjust according to voxel size
				x /= dx;
				y /= dy;
				z /= dz;
				// center over current voxel
				x += i;
				y += j;

				// convert from slice image coordinates to world coordinates
				slice.ImageToWorld(x, y, z);

				// x+=(vx-cx); y+=(vy-cy); z+=(vz-cz);
				// Transform to space of reconstructed volume
				_transformations[inputIndex].Transform(x, y, z);

				// Change to image coordinates
				_reconstructed.WorldToImage(x, y, z);

				// determine coefficients of volume voxels for position x,y,z
				// using linear interpolation

				// Find the 8 closest volume voxels

				// lowest corner of the cube
				nx = (int)floor(x);
				ny = (int)floor(y);
				nz = (int)floor(z);

				// not all neighbours might be in ROI, thus we need to
				// normalize
				//(l,m,n) are image coordinates of 8 neighbours in volume
				// space
				// for each we check whether it is in volume
				sum = 0;
				// to find wether the current slice voxel has overlap with ROI
				bool inside = false;
				for (l = nx; l <= nx + 1; l++)
				    if ((l >= 0) && (l < _reconstructed.GetX()))
					for (m = ny; m <= ny + 1; m++)
					    if ((m >= 0) && (m < _reconstructed.GetY()))
						for (n = nz; n <= nz + 1; n++)
						    if ((n >= 0) && (n < _reconstructed.GetZ())) {
							weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) *
							    (1 - fabs(n - z));
							sum += weight;
							if (_mask(l, m, n) == 1) {
							    inside = true;
							    slice_inside = true;
							}
						    }
				// if there were no voxels do nothing
				if ((sum <= 0) || (!inside))
				    continue;
				// now calculate the transformed PSF
				for (l = nx; l <= nx + 1; l++)
				    if ((l >= 0) && (l < _reconstructed.GetX()))
					for (m = ny; m <= ny + 1; m++)
					    if ((m >= 0) && (m < _reconstructed.GetY()))
						for (n = nz; n <= nz + 1; n++)
						    if ((n >= 0) && (n < _reconstructed.GetZ())) {
							weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) *
							    (1 - fabs(n - z));

							// image coordinates in tPSF
							//(centre,centre,centre) in tPSF is aligned with
							//(tx,ty,tz)
							int aa, bb, cc;
							aa = l - tx + centre;
							bb = m - ty + centre;
							cc = n - tz + centre;

							// resulting value
							double value = PSF(ii, jj, kk) * weight / sum;

							// Check that we are in tPSF
							if ((aa < 0) || (aa >= dim) || (bb < 0) ||
							    (bb >= dim) || (cc < 0) || (cc >= dim)) {
							    ebbrt::kprintf("Error while trying to populate tPSF\n");
							    exit(1);
							} else
							    // update transformed PSF
							    tPSF(aa, bb, cc) += value;
						    }
			    }  // end of the loop for PSF points

		    // store tPSF values
		    for (ii = 0; ii < dim; ii++)
			for (jj = 0; jj < dim; jj++)
			    for (kk = 0; kk < dim; kk++)
				if (tPSF(ii, jj, kk) > 0) {
				    p.x = ii + tx - centre;
				    p.y = jj + ty - centre;
				    p.z = kk + tz - centre;
				    p.value = tPSF(ii, jj, kk);
				    slicecoeffs[i][j].push_back(p);
				}
		    // cout << " n = " << slicecoeffs[i][j].size() << std::endl;
		}  // end of loop for slice voxels

	_volcoeffs[inputIndex] = slicecoeffs;
	_slice_inside_cpu[inputIndex] = slice_inside;
    }

    // prepare image for volume weights, will be needed for Gaussian
    // Reconstruction
    _volume_weights.Initialize(_reconstructed.GetImageAttributes());
    _volume_weights = 0;

    int i, j, n, k;
    POINT3D p;
    for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
	for (i = 0; i < _slices[inputIndex].GetX(); i++)
	    for (j = 0; j < _slices[inputIndex].GetY(); j++) {
		n = _volcoeffs[inputIndex][i][j].size();
		for (k = 0; k < n; k++) {
		    p = _volcoeffs[inputIndex][i][j][k];
		    _volume_weights(p.x, p.y, p.z) += p.value;
		}
	    }
    }
    
    // find average volume weight to modify alpha parameters accordingly
    irtkRealPixel *ptr = _volume_weights.GetPointerToVoxels();
    irtkRealPixel *pm = _mask.GetPointerToVoxels();
    double sum = 0;
    int num = 0;
    for (int i = 0; i < _volume_weights.GetNumberOfVoxels(); i++) {
	if (*pm == 1) {
	    sum += *ptr;
	    num++;
	}
	ptr++;
	pm++;
    }
}

void EbbRTReconstruction::doNothing() { ebbrt::kprintf("doNothing\n"); }

struct membuf : std::streambuf {
  membuf(char* begin, char* end) { this->setg(begin, begin, end); }
};

static size_t indexToCPU(size_t i) { return i; }

void EbbRTReconstruction::ReceiveMessage(ebbrt::Messenger::NetworkId nid,
                                    std::unique_ptr<ebbrt::IOBuf>&& buffer) {
    ebbrt::kprintf("******************ReceiveMessage: ");
    auto end = std::chrono::system_clock::now();
    auto end_time = std::chrono::system_clock::to_time_t(end);
    ebbrt::kprintf("%s ***********\n", std::ctime(&end_time));
    
  auto output = std::string(reinterpret_cast<const char*>(buffer->Data()),
                            buffer->Length());
  if (output[0] == 'E') {
      ebbrt::kprintf("Received msg length: %d bytes\n", buffer->Length());
      ebbrt::kprintf("Number chain elements: %d\n", buffer->CountChainElements());
      ebbrt::kprintf("Computed chain length: %d bytes\n",
		     buffer->ComputeChainDataLength());
      
      ebbrt::IOBuf::DataPointer dp = buffer->GetDataPointer();
      char* t = (char*)(dp.Get(buffer->ComputeChainDataLength()));
      membuf sb{t + 2, t + buffer->ComputeChainDataLength()};
      std::istream stream{&sb};
      
      ebbrt::kprintf("Begin deserialization...\n");
      boost::archive::text_iarchive ia(stream);
      
      int start, end, diff;
      ia& start& end;
      diff = end - start;
      
      std::vector<irtkRealImage> _slices(diff);
      std::vector<irtkRigidTransformation> _transformations(diff);
      std::vector<irtkRealImage> _simulated_slices(diff);
      std::vector<irtkRealImage> _simulated_weights(diff);
      std::vector<irtkRealImage> _simulated_inside(diff);

      irtkRealImage _slice, _mask, _reconstructed;
      double _quality_factor;
      size_t _max_slices;
      
      for (int k = 0; k < diff; k++) {
	  ia& _slices[k];
      }
      
      for (int k = 0; k < diff; k++) {
	  ia& _transformations[k];
      }
      
      for (int k = 0; k < diff; k++) {
	  ia& _simulated_slices[k];
      }

      for (int k = 0; k < diff; k++) {
	  ia& _simulated_weights[k];
      }

      for (int k = 0; k < diff; k++) {
	  ia& _simulated_inside[k];
      }

      bool _global_bias_correction;
      ia& _reconstructed& _mask&_max_slices&_global_bias_correction;
      
      ebbrt::kprintf("Deserialized...\n");
      
      ebbrt::kprintf("_slices: %d\n", _slices.size());
      ebbrt::kprintf("_transformations: %d\n", _transformations.size());
      ebbrt::kprintf("start: %d end: %d\n", start, end);
      int iterations = 9; // 9 //2 for Shepp-Logan is enough
      double sigma = 20;
      double resolution = 0.75;
      double lambda = 0.02;
      double delta = 150;
      int levels = 3;
      double lastIterLambda = 0.01;
      int rec_iterations;
      double averageValue = 700;
      double smooth_mask = 4;
      bool global_bias_correction = false;
      double low_intensity_cutoff = 0.01;
      // flag to swich the intensity matching on and off
      bool intensity_matching = true;
      unsigned int rec_iterations_first = 4;
      unsigned int rec_iterations_last = 13;
      std::vector<double> _slices_regCertainty;
      double _delta, _alpha, _lambda, _max_intensity, _min_intensity
	  , _sigma_cpu, _sigma_s_cpu, _mix_cpu, _mix_s_cpu, _m_cpu
	  , _mean_s_cpu, _mean_s2_cpu, _sigma_s2_cpu, _step, _sigma_bias
	  , _low_intensity_cutoff;
      int i;
      std::vector<irtkRealImage> _weights;
      std::vector<irtkRealImage> _bias;
      std::vector<double> _slice_weight_cpu;
      std::vector<double> _scale_cpu;
      std::vector<SLICECOEFFS> _volcoeffs;
      std::vector<bool> _slice_inside_cpu;
      irtkRealImage _volume_weights, _confidence_map;
      std::vector<int> _small_slices;
      bool _adaptive;
      
      _step = 0.0001;
      _quality_factor = 2;
      _sigma_bias = 12;
      _sigma_s_cpu = 0.025;
      _sigma_s2_cpu = 0.025;
      _mix_s_cpu = 0.9;
      _mix_cpu = 0.9;
      _delta = 1;
      _lambda = 0.1;
      _alpha = (0.05 / _lambda) * _delta * _delta;
      _low_intensity_cutoff = 0.01;
      _adaptive = false;
      
      int _directions[13][3] = { { 1, 0, -1 },
				 { 0, 1, -1 },
				 { 1, 1, -1 },
				 { 1, -1, -1 },
				 { 1, 0, 0 },
				 { 0, 1, 0 },
				 { 1, 1, 0 },
				 { 1, -1, 0 },
				 { 1, 0, 1 },
				 { 0, 1, 1 },
				 { 1, 1, 1 },
				 { 1, -1, 1 },
				 { 0, 0, 1 } };

      InitializeEM(_slices, _weights, _bias, _slice_weight_cpu, _scale_cpu, _max_intensity, _min_intensity);

      for(int iter = 0; iter < iterations; iter++) {
	  ebbrt::kprintf("iter = %d\n", iter);
	  if (iter > 0) {
	      SliceToVolumeRegistration(_slices_regCertainty, _reconstructed
					, _slices, _transformations, start, end);
	  }
	  
	  if (iter == (iterations - 1))
	      SetSmoothingParameters(delta, lastIterLambda, _lambda, _delta, _alpha);
	  else {
	      double l = lambda;
	      for (i = 0; i < levels; i++) {
		  if (iter == iterations * (levels - i - 1) / levels)
		      SetSmoothingParameters(delta, l, _lambda, _delta, _alpha);
		  l *= 2;
	      }
	  }

	  // Use faster reconstruction during iterations and slower for final
	  // reconstruction
	  if (iter < (iterations - 1)) {
	      _quality_factor = 1;
	  } else {
	      _quality_factor = 2;
	  }

	  // Initialise values of weights, scales and bias fields
	  InitializeEMValues(_slices, _weights, _bias, _slice_weight_cpu, _scale_cpu);
	  ebbrt::kprintf("InitializeEMValues\n");
	  
          // Calculate matrix of transformation between voxels of slices and volume
	  CoeffInit(_mask, _reconstructed, _quality_factor, _max_slices,
		    _slices, _transformations, _volcoeffs,
		    _slice_inside_cpu, start, end, _volume_weights);
	  ebbrt::kprintf("CoeffInit\n");

	  // Initialize reconstructed image with Gaussian weighted reconstruction
	  GaussianReconstruction(_reconstructed, _small_slices, _slices, _bias, _scale_cpu, _volcoeffs, _volume_weights);
	  ebbrt::kprintf("GaussianReconstruction\n");

	   // Simulate slices (needs to be done after Gaussian reconstruction)
	  SimulateSlices(_slices, _simulated_slices, _simulated_weights, _simulated_inside, _slice_inside_cpu, _volcoeffs, _reconstructed, _mask, start, end);    
	  ebbrt::kprintf("SimulateSlices\n");

	  InitializeRobustStatistics(_sigma_cpu, _sigma_s_cpu, _mix_cpu, _mix_s_cpu, _m_cpu, start, end, _slices, _simulated_slices, _simulated_weights, _simulated_inside, _slice_inside_cpu, _slice_weight_cpu, _max_intensity, _min_intensity);
	  ebbrt::kprintf("InitializeRobustStatistics\n");

	  EStep(_slices, _weights, _bias, _scale_cpu, _volcoeffs, _simulated_slices, _simulated_weights, _sigma_cpu, _m_cpu, _mix_cpu, _small_slices, _slice_weight_cpu, _mean_s_cpu, _mean_s2_cpu, _sigma_s_cpu, _sigma_s2_cpu, _mix_s_cpu, start, end, _step);
	  ebbrt::kprintf("EStep\n");

	  // number of reconstruction iterations
	  if (iter == (iterations - 1)) {
	      rec_iterations = rec_iterations_last;
	  } else {
	      rec_iterations = rec_iterations_first;
	  }

	  // reconstruction iterations
	  i = 0;
	  for (i = 0; i < rec_iterations; i++) {
	      if (intensity_matching) {
		  // calculate bias fields
		  if (sigma > 0) {
		      Bias(_slices, _weights, _bias, _scale_cpu, _simulated_weights, _simulated_slices, _global_bias_correction);
		  }
		  
		  // calculate scales
		  Scale(_slices, _weights, _bias, _scale_cpu, _simulated_weights, _simulated_slices, start, end, _simulated_slices, _volcoeffs, _slice_weight_cpu);
	      }

	      // MStep and update reconstructed volume
	      Superresolution(i + 1, _min_intensity, _max_intensity, _reconstructed, _global_bias_correction, _slices, _weights, _bias, _scale_cpu, _simulated_slices, _volcoeffs, _slice_weight_cpu, start, end, _confidence_map, _adaptive, _directions, _alpha, _lambda, _delta, _mask, _low_intensity_cutoff);
      
	      if (intensity_matching) {
		  if ((sigma > 0) && (!_global_bias_correction))
		      NormaliseBias(i, _volume_weights, _mask, _sigma_bias, _reconstructed, _slices, _bias, _scale_cpu, _volcoeffs, start, end);
	      }

	      // Simulate slices (needs to be done
	      // after the update of the reconstructed volume)
	      SimulateSlices(_slices, _simulated_slices, _simulated_weights, _simulated_inside, _slice_inside_cpu, _volcoeffs, _reconstructed, _mask, start, end);

	      reconstruction.MStep(i + 1);
	      reconstruction.EStep();
	  } // end of reconstruction iterations

      }
            
      std::string ts = "E";
      ebbrt::kprintf("ts length: %d\n", ts.length());
      
      Print(ts.c_str());
  }
}
