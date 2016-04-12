/*=========================================================================
  Library   : Image Registration Toolkit (IRTK)
  Copyright : Imperial College, Department of Computing
  Visual Information Processing (VIP), 2011 onwards
  Date      : $Date: 2013-11-15 14:36:30 +0100 (Fri, 15 Nov 2013) $
  Version   : $Revision: 1 $
  Changes   : $Author: bkainz $

  Copyright (c) 2014, Bernhard Kainz, Markus Steinberger,
  Maria Murgasova, Kevin Keraudren
  All rights reserved.

  If you use this work for research we would very much appreciate if you cite
  Bernhard Kainz, Markus Steinberger, Maria Kuklisova-Murgasova, Christina
  Malamateniou,
  Wolfgang Wein, Thomas Torsney-Weir, Torsten Moeller, Mary Rutherford,
  Joseph V. Hajnal and Daniel Rueckert:
  Fast Volume Reconstruction from Motion Corrupted 2D Slices.
  IEEE Transactions on Medical Imaging, in press, 2015

  IRTK IS PROVIDED UNDER THE TERMS OF THIS CREATIVE
  COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY
  COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN
  AS AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.

  BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE
  TO BE BOUND BY THE TERMS OF THIS LICENSE. TO THE EXTENT THIS LICENSE MAY BE
  CONSIDERED TO BE A CONTRACT, THE LICENSOR GRANTS YOU THE RIGHTS CONTAINED
  HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND CONDITIONS.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
  FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  =========================================================================*/

#define NOMINMAX
#define _USE_MATH_DEFINES

#include "irtkReconstructionEbb.h"
#include <irtkResampling.h>
#include <irtkRegistration.h>
#include <irtkImageRigidRegistration.h>
#include <irtkImageRigidRegistrationWithPadding.h>
#include <irtkImageFunction.h>
#include <irtkTransformation.h>
#include <math.h>
#include <stdlib.h>
#include "utils.h"
#include <thread>

#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/LocalIdMap.h>
#include <ebbrt/EbbRef.h>
#include <ebbrt/SharedEbb.h>
#include <ebbrt/Message.h>
#include <ebbrt/IOBuf.h>
#include <ebbrt/Clock.h>
#include <ebbrt/SpinBarrier.h>
//#include <ebbrt/MulticoreEbb.h>
#ifdef __EBBRT_BM__
#include <ebbrt/SpinLock.h>
#endif

#include <time.h>
#include <sys/time.h>
#include <unistd.h>

// This is *IMPORTANT*, it allows the messenger to resolve remote HandleFaults
EBBRT_PUBLISH_TYPE(, irtkReconstructionEbb);

using namespace ebbrt;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wreturn-type"

#ifdef __EBBRT_BM__
#define PRINTF ebbrt::kprintf
#define FORPRINTF ebbrt::force_kprintf
static size_t indexToCPU(size_t i) { return i; }
ebbrt::SpinLock spinlock;
#else
#define PRINTF std::printf
#define FORPRINTF std::printf
#endif

struct membuf : std::streambuf {
  membuf(char* begin, char* end) { this->setg(begin, begin, end); }
};

void irtkReconstructionEbb::Print(ebbrt::Messenger::NetworkId nid, const char* str) {
  auto len = strlen(str) + 1;
  auto buf = ebbrt::MakeUniqueIOBuf(len);
  snprintf(reinterpret_cast<char*>(buf->MutData()), len, "%s", str);

#ifndef __EBBRT_BM__
  std::cout << "EbbRTReconstruction length of sent iobuf: "
            << buf->ComputeChainDataLength() << " bytes" << std::endl;
#else
  ebbrt::kprintf("EbbRTReconstruction length of sent iobuf: %ld bytes \n", buf->ComputeChainDataLength());
#endif

  SendMessage(nid, std::move(buf));
}

void irtkReconstructionEbb::setNumNodes(int i)
{
    numNodes = i;
}
void irtkReconstructionEbb::setNumThreads(int i)
{
    _numThreads = i;
}

EbbRef<irtkReconstructionEbb> irtkReconstructionEbb::Create(EbbId id) 
{ 
    return EbbRef<irtkReconstructionEbb>(id); 
}

// This Ebb is implemented with one representative per machine
irtkReconstructionEbb &irtkReconstructionEbb::HandleFault(EbbId id) {
  {
    // First we check if the representative is in the LocalIdMap (using a
    // read-lock)
    LocalIdMap::ConstAccessor accessor;
    auto found = local_id_map->Find(accessor, id);
    if (found) {
      auto &rep = *boost::any_cast<irtkReconstructionEbb *>(accessor->second);
      EbbRef<irtkReconstructionEbb>::CacheRef(id, rep);
      return rep;
    }
  }

  irtkReconstructionEbb *rep;
  {
    // Try to insert an entry into the LocalIdMap while holding an exclusive
    // (write) lock
    LocalIdMap::Accessor accessor;
    auto created = local_id_map->Insert(accessor, id);
    if (unlikely(!created)) {
      // We raced with another writer, use the rep it created and return
      rep = boost::any_cast<irtkReconstructionEbb *>(accessor->second);
    } else {
      // Create a new rep and insert it into the LocalIdMap
      rep = new irtkReconstructionEbb(id);
      accessor->second = rep;
    }
  }
  // Cache the reference to the rep in the local translation table
  EbbRef<irtkReconstructionEbb>::CacheRef(id, *rep);
  return *rep;
}

irtkReconstructionEbb::irtkReconstructionEbb(EbbId ebbid) : Messagable<irtkReconstructionEbb>(ebbid) 
{
    _step = 0.0001;
    _debug = false;
    _quality_factor = 2;
    _sigma_bias = 12;
    _sigma_s_cpu = 0.025;
    _sigma_s_gpu = 0.025;
    _sigma_s2_cpu = 0.025;
    _sigma_s2_gpu = 0.025;
    _mix_s_cpu = 0.9;
    _mix_s_gpu = 0.9;
    _mix_cpu = 0.9;
    _mix_gpu = 0.9;
    _delta = 1;
    _lambda = 0.1;
    _alpha = (0.05 / _lambda) * _delta * _delta;
    _template_created = false;
    _have_mask = false;
    _low_intensity_cutoff = 0.01;
    _global_bias_correction = false;
    _adaptive = false;
    _use_SINC = false;

    int directions[13][3] = { { 1, 0, -1 },
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
    for (int i = 0; i < 13; i++)
	for (int j = 0; j < 3; j++)
	    _directions[i][j] = directions[i][j];

    _useCPUReg = true;
    _useCPU = true;

    nids.clear();
    
}

ebbrt::Future<void> irtkReconstructionEbb::Ping(Messenger::NetworkId nid) {
    uint32_t id;
    Promise<void> promise;
    auto ret = promise.GetFuture();
    {
	std::lock_guard<std::mutex> guard(m_);
	id = id_; // Get a new id (always even)
	id_ += 2;

	bool inserted;
	// insert our promise into the hash table
	std::tie(std::ignore, inserted) =
	    promise_map_.emplace(id, std::move(promise));
	assert(inserted);
    }
    // Construct and send the ping message
    auto buf = MakeUniqueIOBuf(sizeof(uint32_t));
    auto dp = buf->GetMutDataPointer();
    dp.Get<uint32_t>() = id + 1; // Ping messages are odd
    SendMessage(nid, std::move(buf));
    std::printf("Ping SetMessage\n");
    return ret;
}

void irtkReconstructionEbb::SendRecon(int iterations)
{
    // get the event manager context and save it
    ebbrt::EventManager::EventContext context;
    
    size_t max_slices = _slices.size();
    
    int start, end, factor;

    // all sizes are the same
    std::cout << "_slices.size() " << _slices.size() << std::endl;
    std::cout << "_transformations.size() "
	      << _transformations.size() << std::endl;
    
    for (int i = 0; i < (int)nids.size(); i++) {
	// serialization
	std::ostringstream ofs;
	boost::archive::text_oarchive oa(ofs);
	
	std::cout << "Before serialize" << std::endl;
	
	//_slices
	factor = (int)ceil(max_slices / (float)numNodes);
	start = i * factor;
	end = i * factor + factor;
	end = ((size_t)end > max_slices) ? max_slices : end;

	std::cout << "start = " << start << " end = " << end << std::endl;
	
	oa& start& end;
	
	for (int j = start; j < end; j++) {
	    oa& _slices[j];
	}
	
	for (int j = start; j < end; j++) {
	    oa& _transformations[j];
	}

	for (int j = start; j < end; j++) {
	    oa& _simulated_slices[j];
	}

	for (int j = start; j < end; j++) {
	    oa& _simulated_weights[j];
	}

	for (int j = start; j < end; j++) {
	    oa& _simulated_inside[j];
	}

	for (int j = start; j < end; j++) {
	    oa& _stack_index[j];
	}
	
	size_t mslices = _stack_factor.size();
	oa&mslices;
	for(unsigned int j = 0; j < mslices; j++)
	{
	    oa& _stack_factor[j];
	}

	oa& _reconstructed& _mask &max_slices & _global_bias_correction & iterations;
	
	std::string ts = "E " + ofs.str();
	
	std::cout << "Sending to .. " /*<< nids[i].ToString()*/
		  << " size: " << ts.length() << std::endl;
	Print(nids[i], ts.c_str());
    }

    std::cout << "Saving context " << std::endl;

    emec = &context;
    ebbrt::event_manager->SaveContext(*emec);
    
    std::cout << "Received back " << std::endl;
    PRINTF("hosted checksum _reconstructed = %lf\n",
	   SumRecon());
    std::cout << "EbbRTReconstruction done " << std::endl;
    
    mypromise.SetValue();;
}

void irtkReconstructionEbb::ReceiveMessage(Messenger::NetworkId nid,
                              std::unique_ptr<IOBuf> &&buffer) 
{
    size_t _max_slices;
    
#ifndef __EBBRT_BM__
    auto output = std::string(reinterpret_cast<const char*>(buffer->Data()));
    std::cout << "Received ip: " << nid.ToString() << std::endl;
    
    if(output[0] == 'E') {
	ebbrt::IOBuf::DataPointer dp = buffer->GetDataPointer();
	char* t = (char*)(dp.Get(buffer->ComputeChainDataLength()));
	membuf sb{t + 2, t + buffer->ComputeChainDataLength()};
	std::istream stream{&sb};
	boost::archive::text_iarchive ia(stream);
	
	std::cout << "Parsing it back, received: "
		  << buffer->ComputeChainDataLength() << " bytes" << std::endl;
	
	ia& _reconstructed;
	ebbrt::event_manager->ActivateContext(std::move(*emec));
    }
#else
    
    auto output = std::string(reinterpret_cast<const char *>(buffer->Data()), buffer->Length());

    if (output[0] == 'E') {
	ebbrt::kprintf("Received msg length: %d bytes\n", buffer->Length());
	ebbrt::kprintf("Number chain elements: %d\n", buffer->CountChainElements());
	ebbrt::kprintf("Computed chain length: %d bytes\n",
	           buffer->ComputeChainDataLength());

	/****** copy sub buffers into one buffer *****/
	ebbrt::IOBuf::DataPointer dp = buffer->GetDataPointer();
	char *t = (char *)(dp.Get(buffer->ComputeChainDataLength()));
	membuf sb{ t + 2, t + buffer->ComputeChainDataLength() };
	std::istream stream{ &sb };
	/********* ******************************/

	ebbrt::kprintf("Begin deserialization...\n");
	/********** deserialize ******************/
	boost::archive::text_iarchive ia(stream);

	int iterations = 1; // 9 //2 for Shepp-Logan is enough
	ia & _start & _end;
	_diff = _end - _start;

	//ebbrt::kprintf("start:%d end:%d diff:%d\n", start, end, end);

	_slices.resize(_diff);
	_transformations.resize(_diff);
	_simulated_slices.resize(_diff);
	_simulated_weights.resize(_diff);
	_simulated_inside.resize(_diff);
	_stack_index.resize(_diff);

	for (int k = 0; k < _diff; k++) {
	    ia &_slices[k];
	}

	//ebbrt::kprintf("_slices deserialized\n");

	for (int k = 0; k < _diff; k++) {
	    ia &_transformations[k];
	}

	for (int k = 0; k < _diff; k++) {
	    ia &_simulated_slices[k];
	}

	for (int k = 0; k < _diff; k++) {
	    ia &_simulated_weights[k];
	}

	for (int k = 0; k < _diff; k++) {
	    ia &_simulated_inside[k];
	}

	for (int k = 0; k < _diff; k++) {
	    ia &_stack_index[k];
	}

	int ssend;
	ia &ssend;
	_stack_factor.resize(ssend);
	for (int k = 0; k < ssend; k++) {
	    ia &_stack_factor[k];
	}
	ia &_reconstructed &_mask &_max_slices &_global_bias_correction & iterations;

	/********* ******************************/
	
	//ebbrt::kprintf("Deserialized...\n");
	
	//ebbrt::kprintf("_slices: %d\n", _slices.size());
	//ebbrt::kprintf("_transformations: %d\n", _transformations.size());

	double sigma = 12;
	double lambda = 0.02;
	double delta = 150;
	int levels = 3;
	double lastIterLambda = 0.01;
	int rec_iterations;
	bool global_bias_correction = false;
	_global_bias_correction = false;
	// flag to swich the intensity matching on and off
	bool intensity_matching = true;
	bool useCPU = true;
	bool disableBiasCorr = true;

	int rec_iterations_first = 4;
	int rec_iterations_last = 13;

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

	int directions[13][3] = { { 1, 0, -1 },
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
	for (int i = 0; i < 13; i++)
	    for (int j = 0; j < 3; j++)
		_directions[i][j] = directions[i][j];
	
	InitializeEM();

	RunRecon(iterations, delta, lastIterLambda, rec_iterations_first, rec_iterations_last, intensity_matching, lambda, levels);
	
	std::ostringstream ofs;
	boost::archive::text_oarchive oa(ofs);
	oa &_reconstructed;
	
	std::string ts = "E " + ofs.str();
	//ebbrt::kprintf("ts length: %d\n", ts.length());
	
	Print(nid, ts.c_str());
      }
#endif
}

void irtkReconstructionEbb::addNid(ebbrt::Messenger::NetworkId nid) {
      nids.push_back(nid);
      if ((int)nids.size() == numNodes) {
	  nodesinit.SetValue();
      }
}

ebbrt::Future<void> irtkReconstructionEbb::waitNodes() { return std::move(nodesinit.GetFuture()); }

ebbrt::Future<void> irtkReconstructionEbb::waitReceive() { return std::move(mypromise.GetFuture()); }

void bbox(irtkRealImage &stack, irtkRigidTransformation &transformation,
          double &min_x, double &min_y, double &min_z, double &max_x,
          double &max_y, double &max_z) {

  cout << "bbox" << endl;

  min_x = voxel_limits<irtkRealPixel>::max();
  min_y = voxel_limits<irtkRealPixel>::max();
  min_z = voxel_limits<irtkRealPixel>::max();
  max_x = voxel_limits<irtkRealPixel>::min();
  max_y = voxel_limits<irtkRealPixel>::min();
  max_z = voxel_limits<irtkRealPixel>::min();
  double x, y, z;
  for (int i = 0; i <= stack.GetX(); i += stack.GetX())
    for (int j = 0; j <= stack.GetY(); j += stack.GetY())
      for (int k = 0; k <= stack.GetZ(); k += stack.GetZ()) {
        x = i;
        y = j;
        z = k;
        stack.ImageToWorld(x, y, z);
        // FIXME!!!
        transformation.Transform(x, y, z);
        // transformation.Inverse( x, y, z );
        if (x < min_x)
          min_x = x;
        if (y < min_y)
          min_y = y;
        if (z < min_z)
          min_z = z;
        if (x > max_x)
          max_x = x;
        if (y > max_y)
          max_y = y;
        if (z > max_z)
          max_z = z;
      }
}

void bboxCrop(irtkRealImage &image) {
  int min_x, min_y, min_z, max_x, max_y, max_z;
  min_x = image.GetX() - 1;
  min_y = image.GetY() - 1;
  min_z = image.GetZ() - 1;
  max_x = 0;
  max_y = 0;
  max_z = 0;
  for (int i = 0; i < image.GetX(); i++)
    for (int j = 0; j < image.GetY(); j++)
      for (int k = 0; k < image.GetZ(); k++) {
        if (image.Get(i, j, k) > 0) {
          if (i < min_x)
            min_x = i;
          if (j < min_y)
            min_y = j;
          if (k < min_z)
            min_z = k;
          if (i > max_x)
            max_x = i;
          if (j > max_y)
            max_y = j;
          if (k > max_z)
            max_z = k;
        }
      }

  // Cut region of interest
  image = image.GetRegion(min_x, min_y, min_z, max_x + 1, max_y + 1, max_z + 1);
}

void centroid(irtkRealImage &image, double &x, double &y, double &z) {
  double sum_x = 0;
  double sum_y = 0;
  double sum_z = 0;
  double norm = 0;
  double v;
  for (int i = 0; i < image.GetX(); i++)
    for (int j = 0; j < image.GetY(); j++)
      for (int k = 0; k < image.GetZ(); k++) {
        v = image.Get(i, j, k);
        if (v <= 0)
          continue;
        sum_x += v * i;
        sum_y += v * j;
        sum_z += v * k;
        norm += v;
      }

  x = sum_x / norm;
  y = sum_y / norm;
  z = sum_z / norm;

  image.ImageToWorld(x, y, z);

  //std::cout << "CENTROID:" << x << "," << y << "," << z << "\n\n";
}

irtkReconstructionEbb::~irtkReconstructionEbb() {}

void irtkReconstructionEbb::CenterStacks(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stack_transformations,
    int templateNumber) {
  // template center
  double x0, y0, z0;
  irtkRealImage mask;
  mask = stacks[templateNumber] != -1;
  centroid(mask, x0, y0, z0);

  double x, y, z;
  irtkMatrix m1, m2;
  for (unsigned int i = 0; i < stacks.size(); i++) {
    if (i == templateNumber)
      continue;

    mask = stacks[i] != -1;
    centroid(mask, x, y, z);

    irtkRigidTransformation translation;
    translation.PutTranslationX(x0 - x);
    translation.PutTranslationY(y0 - y);
    translation.PutTranslationZ(z0 - z);

    m1 = stack_transformations[i].GetMatrix();
    m2 = translation.GetMatrix();
    stack_transformations[i].PutMatrix(m2 * m1);
  }
}

class ParallelAverage {
  irtkReconstructionEbb *reconstructor;
  vector<irtkRealImage> &stacks;
  vector<irtkRigidTransformation> &stack_transformations;

  /// Padding value in target (voxels in the target image with this
  /// value will be ignored)
  double targetPadding;

  /// Padding value in source (voxels outside the source image will
  /// be set to this value)
  double sourcePadding;

  double background;

  // Volumetric registrations are stack-to-template while slice-to-volume
  // registrations are actually performed as volume-to-slice
  // (reasons: technicalities of implementation)
  // so transformations need to be inverted beforehand.

  bool linear;

    int nt;
public:
  irtkRealImage average;
  irtkRealImage weights;

  void operator()(const blocked_range<size_t> &r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      irtkImageTransformation imagetransformation;
      irtkImageFunction *interpolator;
      if (linear)
        interpolator = new irtkLinearInterpolateImageFunction;
      else
        interpolator = new irtkNearestNeighborInterpolateImageFunction;

      irtkRealImage s = stacks[i];
      irtkRigidTransformation t = stack_transformations[i];
      imagetransformation.SetInput(&s, &t);
      irtkRealImage image(reconstructor->_reconstructed.GetImageAttributes());
      image = 0;

      imagetransformation.SetOutput(&image);
      imagetransformation.PutTargetPaddingValue(targetPadding);
      imagetransformation.PutSourcePaddingValue(sourcePadding);
      imagetransformation.PutInterpolator(interpolator);
      imagetransformation.Run();

      irtkRealPixel *pa = average.GetPointerToVoxels();
      irtkRealPixel *pi = image.GetPointerToVoxels();
      irtkRealPixel *pw = weights.GetPointerToVoxels();
      for (int p = 0; p < average.GetNumberOfVoxels(); p++) {
        if (*pi != background) {
          *pa += *pi;
          *pw += 1;
        }
        pa++;
        pi++;
        pw++;
      }
      delete interpolator;
    }
  }

  ParallelAverage(ParallelAverage &x, split)
      : reconstructor(x.reconstructor), stacks(x.stacks),
        stack_transformations(x.stack_transformations) {
    average.Initialize(reconstructor->_reconstructed.GetImageAttributes());
    average = 0;
    weights.Initialize(reconstructor->_reconstructed.GetImageAttributes());
    weights = 0;
    targetPadding = x.targetPadding;
    sourcePadding = x.sourcePadding;
    background = x.background;
    linear = x.linear;
  }

  void join(const ParallelAverage &y) {
    average += y.average;
    weights += y.weights;
  }

  ParallelAverage(irtkReconstructionEbb *reconstructor,
                  vector<irtkRealImage> &_stacks,
                  vector<irtkRigidTransformation> &_stack_transformations,
                  double _targetPadding, double _sourcePadding,
                  double _background, int _nt, bool _linear = false)
      : reconstructor(reconstructor), stacks(_stacks),
        stack_transformations(_stack_transformations) {
    average.Initialize(reconstructor->_reconstructed.GetImageAttributes());
    average = 0;
    weights.Initialize(reconstructor->_reconstructed.GetImageAttributes());
    weights = 0;
    targetPadding = _targetPadding;
    sourcePadding = _sourcePadding;
    background = _background;
    linear = _linear;
    nt = _nt;
  }

  // execute
  void operator()() {
    task_scheduler_init init(nt);
    parallel_reduce(blocked_range<size_t>(0, stacks.size()), *this);
    init.terminate();
  }
};

irtkRealImage irtkReconstructionEbb::CreateAverage(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stack_transformations) {
  if (!_template_created) {
    cerr << "Please create the template before calculating the average of the "
            "stacks." << endl;
    exit(1);
  }

  InvertStackTransformations(stack_transformations);
  ParallelAverage parallelAverage(this, stacks, stack_transformations, -1, 0,
                                  0, // target/source/background
                                  true, _numThreads);
  parallelAverage();
  irtkRealImage average = parallelAverage.average;
  irtkRealImage weights = parallelAverage.weights;
  average /= weights;
  InvertStackTransformations(stack_transformations);
  return average;
}

double irtkReconstructionEbb::CreateTemplate(irtkRealImage stack,
                                          double resolution) {
  double dx, dy, dz, d;

  // Get image attributes - image size and voxel size
  irtkImageAttributes attr = stack.GetImageAttributes();

  // enlarge stack in z-direction in case top of the head is cut off
  attr._z += 2;

  // create enlarged image
  irtkRealImage enlarged(attr);

  // determine resolution of volume to reconstruct
  if (resolution <= 0) {
    // resolution was not given by user
    // set it to min of res in x or y direction
    stack.GetPixelSize(&dx, &dy, &dz);
    if ((dx <= dy) && (dx <= dz))
      d = dx;
    else if (dy <= dz)
      d = dy;
    else
      d = dz;
  } else
    d = resolution;

  //cout << "Constructing volume with isotropic voxel size " << d << endl;

  // resample "enlarged" to resolution "d"
  irtkNearestNeighborInterpolateImageFunction interpolator;
  irtkResampling<irtkRealPixel> resampling(d, d, d);
  resampling.SetInput(&enlarged);
  resampling.SetOutput(&enlarged);
  resampling.SetInterpolator(&interpolator);
  resampling.Run();

  // initialize recontructed volume
  _reconstructed = enlarged;

  _reconstructed_gpu.Initialize(_reconstructed.GetImageAttributes());
  _template_created = true;

  // return resulting resolution of the template image
  return d;
}

irtkRealImage irtkReconstructionEbb::CreateMask(irtkRealImage image) {
  // binarize mask
  irtkRealPixel *ptr = image.GetPointerToVoxels();
  for (int i = 0; i < image.GetNumberOfVoxels(); i++) {
    if (*ptr > 0.0)
      *ptr = 1;
    else
      *ptr = 0;
    ptr++;
  }
  return image;
}

void irtkReconstructionEbb::SetMask(irtkRealImage *mask, double sigma,
                                 double threshold) {
  if (!_template_created) {
    cerr << "Please create the template before setting the mask, so that the "
            "mask can be resampled to the correct dimensions." << endl;
    exit(1);
  }

  _mask = _reconstructed;

  if (mask != NULL) {
    // if sigma is nonzero first smooth the mask
    if (sigma > 0) {
      // blur mask
      irtkGaussianBlurring<irtkRealPixel> gb(sigma);
      gb.SetInput(mask);
      gb.SetOutput(mask);
      gb.Run();

      // binarize mask
      irtkRealPixel *ptr = mask->GetPointerToVoxels();
      for (int i = 0; i < mask->GetNumberOfVoxels(); i++) {
        if (*ptr > threshold)
          *ptr = 1;
        else
          *ptr = 0;
        ptr++;
      }
    }

    // resample the mask according to the template volume using identity
    // transformation
    irtkRigidTransformation transformation;
    irtkImageTransformation imagetransformation;
    irtkNearestNeighborInterpolateImageFunction interpolator;
    imagetransformation.SetInput(mask, &transformation);
    imagetransformation.SetOutput(&_mask);
    // target is zero image, need padding -1
    imagetransformation.PutTargetPaddingValue(-1);
    // need to fill voxels in target where there is no info from source with
    // zeroes
    imagetransformation.PutSourcePaddingValue(0);
    imagetransformation.PutInterpolator(&interpolator);
    imagetransformation.Run();
  } else {
    // fill the mask with ones
    _mask = 1;
  }
  // set flag that mask was created
  _have_mask = true;

  if (_debug)
    _mask.Write("mask.nii");
}

void
irtkReconstructionEbb::TransformMask(irtkRealImage &image, irtkRealImage &mask,
                                  irtkRigidTransformation &transformation) {
  // transform mask to the space of image
  irtkImageTransformation imagetransformation;
  irtkNearestNeighborInterpolateImageFunction interpolator;
  imagetransformation.SetInput(&mask, &transformation);
  irtkRealImage m = image;
  imagetransformation.SetOutput(&m);
  // target contains zeros and ones image, need padding -1
  imagetransformation.PutTargetPaddingValue(-1);
  // need to fill voxels in target where there is no info from source with
  // zeroes
  imagetransformation.PutSourcePaddingValue(0);
  imagetransformation.PutInterpolator(&interpolator);
  imagetransformation.Run();
  mask = m;
}

void irtkReconstructionEbb::ResetOrigin(irtkGreyImage &image,
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

void irtkReconstructionEbb::ResetOrigin(irtkRealImage &image,
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

class ParallelStackRegistrations {
  irtkReconstructionEbb *reconstructor;
  vector<irtkRealImage> &stacks;
  vector<irtkRigidTransformation> &stack_transformations;
  int templateNumber;
  irtkGreyImage &target;
  irtkRigidTransformation &offset;
  bool _externalTemplate;
    int nt;
public:
    ParallelStackRegistrations(
      irtkReconstructionEbb *_reconstructor, vector<irtkRealImage> &_stacks,
      vector<irtkRigidTransformation> &_stack_transformations,
      int _templateNumber, irtkGreyImage &_target,
      irtkRigidTransformation &_offset, int _nt, bool externalTemplate = false)
      : reconstructor(_reconstructor), stacks(_stacks),
        stack_transformations(_stack_transformations), target(_target),
        offset(_offset) {
	templateNumber = _templateNumber, _externalTemplate = externalTemplate,
	    nt = _nt;
  }

  void operator()(const blocked_range<size_t> &r) const {
    for (size_t i = r.begin(); i != r.end(); ++i) {

      // do not perform registration for template
      if (i == templateNumber)
        continue;

      // rigid registration object
      irtkImageRigidRegistrationWithPadding registration;
      // irtkRigidTransformation transformation = stack_transformations[i];

      // set target and source (need to be converted to irtkGreyImage)
      irtkGreyImage source = stacks[i];

      // include offset in trasformation
      irtkMatrix mo = offset.GetMatrix();
      irtkMatrix m = stack_transformations[i].GetMatrix();
      m = m * mo;
      stack_transformations[i].PutMatrix(m);

      // perform rigid registration
      registration.SetInput(&target, &source);
      registration.SetOutput(&stack_transformations[i]);
      if (_externalTemplate) {
        registration.GuessParameterThickSlicesNMI();
      } else {
        registration.GuessParameterThickSlices();
      }
      registration.SetTargetPadding(0);
      registration.Run();

      mo.Invert();
      m = stack_transformations[i].GetMatrix();
      m = m * mo;
      stack_transformations[i].PutMatrix(m);
    }
  }

  // execute
  void operator()() const {
    task_scheduler_init init(nt);
    parallel_for(blocked_range<size_t>(0, stacks.size()), *this);
    init.terminate();
  }
};

void irtkReconstructionEbb::StackRegistrations(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stack_transformations, int templateNumber,
    bool useExternalTarget) {
  if (_debug)
    cout << "StackRegistrations" << endl;

  InvertStackTransformations(stack_transformations);

  // template is set as the target
  irtkGreyImage target;
  if (!useExternalTarget) {
    target = stacks[templateNumber];
  } else {
    target = externalRegistrationTargetImage;
  }

  // target needs to be masked before registration
  if (_have_mask) {
    double x, y, z;
    for (int i = 0; i < target.GetX(); i++)
      for (int j = 0; j < target.GetY(); j++)
        for (int k = 0; k < target.GetZ(); k++) {
          // image coordinates of the target
          x = i;
          y = j;
          z = k;
          // change to world coordinates
          target.ImageToWorld(x, y, z);
          // change to mask image coordinates - mask is aligned with target
          _mask.WorldToImage(x, y, z);
          x = round(x);
          y = round(y);
          z = round(z);
          // if the voxel is outside mask ROI set it to -1 (padding value)
          if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) &&
              (y < _mask.GetY()) && (z >= 0) && (z < _mask.GetZ())) {
            if (_mask(x, y, z) == 0)
              target(i, j, k) = 0;
          } else
            target(i, j, k) = 0;
        }
  }

  irtkRigidTransformation offset;
  ResetOrigin(target, offset);

  // register all stacks to the target
  ParallelStackRegistrations registration(this, stacks, stack_transformations,
                                          templateNumber, target, offset, 
					  _numThreads, useExternalTarget);
  registration();

  InvertStackTransformations(stack_transformations);
}

void irtkReconstructionEbb::RestoreSliceIntensities() {
  if (_debug)
    cout << "Restoring the intensities of the slices. " << endl;

  unsigned int inputIndex;
  int i;
  double factor;
  irtkRealPixel *p;

  //std::cout << "factor = ";
  for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    // calculate scaling factor
    factor = _stack_factor[_stack_index[inputIndex]]; //_average_value;
    //std::printf(" %f ", factor);
    // read the pointer to current slice
    p = _slices[inputIndex].GetPointerToVoxels();
    for (i = 0; i < _slices[inputIndex].GetNumberOfVoxels(); i++) {
      if (*p > 0)
        *p = *p / factor;
      p++;
    }
  }
}

void irtkReconstructionEbb::ScaleVolume() {
  unsigned int inputIndex;
  int i, j;
  double scalenum = 0, scaleden = 0;

  for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    // alias for the current slice
    irtkRealImage &slice = _slices[inputIndex];

    // alias for the current weight image
    irtkRealImage &w = _weights[inputIndex];

    // alias for the current simulated slice
    irtkRealImage &sim = _simulated_slices[inputIndex];

    for (i = 0; i < slice.GetX(); i++)
      for (j = 0; j < slice.GetY(); j++)
        if (slice(i, j, 0) != -1) {
          // scale - intensity matching
          if (_simulated_weights[inputIndex](i, j, 0) > 0.99) {
            scalenum += w(i, j, 0) * _slice_weight_cpu[inputIndex] *
                        slice(i, j, 0) * sim(i, j, 0);
            scaleden += w(i, j, 0) * _slice_weight_cpu[inputIndex] *
                        sim(i, j, 0) * sim(i, j, 0);
          }
        }
  } // end of loop for a slice inputIndex

  // calculate scale for the volume
  double scale = scalenum / scaleden;

  irtkRealPixel *ptr = _reconstructed.GetPointerToVoxels();
  for (i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
    if (*ptr > 0)
      *ptr = *ptr * scale;
    ptr++;
  }
}

void runSimulateSlices(irtkReconstructionEbb *reconstructor, int start, int end)
{
    for (int inputIndex = start; inputIndex != end; ++inputIndex) {
	// Calculate simulated slice
	reconstructor->_simulated_slices[inputIndex].Initialize(
	    reconstructor->_slices[inputIndex].GetImageAttributes());
	reconstructor->_simulated_slices[inputIndex] = 0;

	reconstructor->_simulated_weights[inputIndex].Initialize(
	    reconstructor->_slices[inputIndex].GetImageAttributes());
	reconstructor->_simulated_weights[inputIndex] = 0;

	reconstructor->_simulated_inside[inputIndex].Initialize(
	    reconstructor->_slices[inputIndex].GetImageAttributes());
	reconstructor->_simulated_inside[inputIndex] = 0;

	reconstructor->_slice_inside_cpu[inputIndex] = false;

	POINT3D p;
	for (unsigned int i = 0; i < reconstructor->_slices[inputIndex].GetX();
	     i++)
	    for (unsigned int j = 0; j < reconstructor->_slices[inputIndex].GetY();
		 j++)
		if (reconstructor->_slices[inputIndex](i, j, 0) != -1) {
		    double weight = 0;
		    int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
		    for (unsigned int k = 0; k < n; k++) {
			p = reconstructor->_volcoeffs[inputIndex][i][j][k];
			reconstructor->_simulated_slices[inputIndex](i, j, 0) +=
			    p.value * reconstructor->_reconstructed(p.x, p.y, p.z);
			weight += p.value;
			if (reconstructor->_mask(p.x, p.y, p.z) == 1) {
			    reconstructor->_simulated_inside[inputIndex](i, j, 0) = 1;
			    reconstructor->_slice_inside_cpu[inputIndex] = true;
			}
		    }
		    if (weight > 0) {
			reconstructor->_simulated_slices[inputIndex](i, j, 0) /= weight;
			reconstructor->_simulated_weights[inputIndex](i, j, 0) = weight;
		    }
		}
    }
}

class ParallelSimulateSlices {
    irtkReconstructionEbb *reconstructor;
    int nt;
public:
    ParallelSimulateSlices(irtkReconstructionEbb *_reconstructor, int _nt)
	: reconstructor(_reconstructor), nt(_nt) {}

    void operator()(const blocked_range<int> &r) const {
	runSimulateSlices(reconstructor, r.begin(), r.end());
    }

    // execute
    void operator()() const {
	task_scheduler_init init(nt);
	parallel_for(blocked_range<int>(0, reconstructor->_slices.size()),
		     *this);
	init.terminate();
    }
};
void irtkReconstructionEbb::SimulateSlices() {
#ifndef __EBBRT_BM__
    ParallelSimulateSlices parallelSimulateSlices(this, _numThreads);
    parallelSimulateSlices();
#else
    size_t ncpus = ebbrt::Cpu::Count();
    if(ncpus > 1)
    {
	static ebbrt::SpinBarrier bar(ncpus);
	ebbrt::EventManager::EventContext context;
	std::atomic<size_t> count(0);
	size_t theCpu = ebbrt::Cpu::GetMine();
	int diff = this->_diff;
	for (size_t i = 0; i < ncpus; i++) {
	    // spawn jobs on each core using SpawnRemote
	    ebbrt::event_manager->SpawnRemote(
		[this, theCpu, ncpus, &count, &context, i, diff]() {
		    // get my cpu id
		    size_t mycpu = ebbrt::Cpu::GetMine();
		    int starte, ende, factor;
		    factor = (int)ceil(diff / (float)ncpus);
		    starte = i * factor;
		    ende = i * factor + factor;
		    ende = (ende > diff) ? diff : ende;
		    runSimulateSlices(this, starte, ende);
		    count++;
		    bar.Wait();
		    while (count < (size_t)ncpus);
		    if (mycpu == theCpu) {
			ebbrt::event_manager->ActivateContext(std::move(context));
		    }
		},
		    indexToCPU(
			i)); // if i don't add indexToCPU, one of the cores never run ? ?
	}
	ebbrt::event_manager->SaveContext(context);
    }
    else
    {
	runSimulateSlices(this, 0, _slices.size());
    }
#endif
}

void irtkReconstructionEbb::SimulateStacks(vector<irtkRealImage> &stacks) {
  unsigned int inputIndex;
  int i, j, k, n;
  irtkRealImage sim;
  POINT3D p;
  double weight;

  int z, current_stack;
  z = -1;             // this is the z coordinate of the stack
  current_stack = -1; // we need to know when to start a new stack

  for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {

    // read the current slice
    irtkRealImage &slice = _slices[inputIndex];

    // Calculate simulated slice
    sim.Initialize(slice.GetImageAttributes());
    sim = 0;

    // do not simulate excluded slice
    if (_slice_weight_cpu[inputIndex] > 0.5) {
      for (i = 0; i < slice.GetX(); i++)
        for (j = 0; j < slice.GetY(); j++)
          if (slice(i, j, 0) != -1) {
            weight = 0;
            n = _volcoeffs[inputIndex][i][j].size();
            for (k = 0; k < n; k++) {
              p = _volcoeffs[inputIndex][i][j][k];
              sim(i, j, 0) += p.value * _reconstructed(p.x, p.y, p.z);
              weight += p.value;
            }
            if (weight > 0)
              sim(i, j, 0) /= weight;
          }
    }

    if (_stack_index[inputIndex] == current_stack)
      z++;
    else {
      current_stack = _stack_index[inputIndex];
      z = 0;
    }

    for (i = 0; i < sim.GetX(); i++)
      for (j = 0; j < sim.GetY(); j++) {
        stacks[_stack_index[inputIndex]](i, j, z) = sim(i, j, 0);
      }
    // end of loop for a slice inputIndex
  }
}

void irtkReconstructionEbb::MatchStackIntensities(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stack_transformations, double averageValue,
    bool together) {
  if (_debug)
    cout << "Matching intensities of stacks. ";

  // Calculate the averages of intensities for all stacks
  double sum, num;
  char buffer[256];
  unsigned int ind;
  int i, j, k;
  double x, y, z;
  vector<double> stack_average;

  // remember the set average value
  _average_value = averageValue;

  // averages need to be calculated only in ROI
  for (ind = 0; ind < stacks.size(); ind++) {
    sum = 0;
    num = 0;
    for (i = 0; i < stacks[ind].GetX(); i++)
      for (j = 0; j < stacks[ind].GetY(); j++)
        for (k = 0; k < stacks[ind].GetZ(); k++) {
          // image coordinates of the stack voxel
          x = i;
          y = j;
          z = k;
          // change to world coordinates
          stacks[ind].ImageToWorld(x, y, z);
          // transform to template (and also _mask) space
          stack_transformations[ind].Transform(x, y, z);
          // change to mask image coordinates - mask is aligned with template
          _mask.WorldToImage(x, y, z);
          x = round(x);
          y = round(y);
          z = round(z);
          // if the voxel is inside mask ROI include it
          // if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y <
          // _mask.GetY()) && (z >= 0)
          //      && (z < _mask.GetZ()))
          //      {
          // if (_mask(x, y, z) == 1)
          if (stacks[ind](i, j, k) > 0) {
            sum += stacks[ind](i, j, k);
            num++;
          }
          //}
        }
    // calculate average for the stack
    if (num > 0)
      stack_average.push_back(sum / num);
    else {
      cerr << "Stack " << ind << " has no overlap with ROI" << endl;
      exit(1);
    }
  }

  double global_average;
  if (together) {
    global_average = 0;
    for (i = 0; i < stack_average.size(); i++)
      global_average += stack_average[i];
    global_average /= stack_average.size();
  }

  if (_debug) {
    cout << "Stack average intensities are ";
    for (ind = 0; ind < stack_average.size(); ind++)
      cout << stack_average[ind] << " ";
    cout << endl;
    cout << "The new average value is " << averageValue << endl;
  }

  // Rescale stacks
  irtkRealPixel *ptr;
  double factor;
  for (ind = 0; ind < stacks.size(); ind++) {
    if (together) {
      factor = averageValue / global_average;
      _stack_factor.push_back(factor);
    } else {
      factor = averageValue / stack_average[ind];
      _stack_factor.push_back(factor);
    }

    ptr = stacks[ind].GetPointerToVoxels();
    for (i = 0; i < stacks[ind].GetNumberOfVoxels(); i++) {
      if (*ptr > 0)
        *ptr *= factor;
      ptr++;
    }
  }
}

void irtkReconstructionEbb::MatchStackIntensitiesWithMasking(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stack_transformations, double averageValue,
    bool together) {
  if (_debug)
    cout << "Matching intensities of stacks. ";

  // Calculate the averages of intensities for all stacks
  double sum, num;
  char buffer[256];
  unsigned int ind;
  int i, j, k;
  double x, y, z;
  vector<double> stack_average;
  irtkRealImage m;

  // remember the set average value
  _average_value = averageValue;

  // averages need to be calculated only in ROI
  for (ind = 0; ind < stacks.size(); ind++) {
    m = stacks[ind];
    sum = 0;
    num = 0;
    for (i = 0; i < stacks[ind].GetX(); i++)
      for (j = 0; j < stacks[ind].GetY(); j++)
        for (k = 0; k < stacks[ind].GetZ(); k++) {
          // image coordinates of the stack voxel
          x = i;
          y = j;
          z = k;
          // change to world coordinates
          stacks[ind].ImageToWorld(x, y, z);
          // transform to template (and also _mask) space
          stack_transformations[ind].Transform(x, y, z);
          // change to mask image coordinates - mask is aligned with template
          _mask.WorldToImage(x, y, z);
          x = round(x);
          y = round(y);
          z = round(z);
          // if the voxel is inside mask ROI include it
          if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) &&
              (y < _mask.GetY()) && (z >= 0) && (z < _mask.GetZ())) {
            if (_mask(x, y, z) == 1) {
              m(i, j, k) = 1;
              sum += stacks[ind](i, j, k);
              num++;
            } else
              m(i, j, k) = 0;
          }
        }
    if (_debug) {
      sprintf(buffer, "mask-for-matching%i.nii.gz", ind);
      m.Write(buffer);
    }
    // calculate average for the stack
    if (num > 0)
      stack_average.push_back(sum / num);
    else {
      cerr << "Stack " << ind << " has no overlap with ROI" << endl;
      exit(1);
    }
  }

  double global_average;
  if (together) {
    global_average = 0;
    for (i = 0; i < stack_average.size(); i++)
      global_average += stack_average[i];
    global_average /= stack_average.size();
  }

  if (_debug) {
    cout << "Stack average intensities are ";
    for (ind = 0; ind < stack_average.size(); ind++)
      cout << stack_average[ind] << " ";
    cout << endl;
    cout << "The new average value is " << averageValue << endl;
  }

  // Rescale stacks
  irtkRealPixel *ptr;
  double factor;
  for (ind = 0; ind < stacks.size(); ind++) {
    if (together) {
      factor = averageValue / global_average;
      _stack_factor.push_back(factor);
    } else {
      factor = averageValue / stack_average[ind];
      _stack_factor.push_back(factor);
    }

    ptr = stacks[ind].GetPointerToVoxels();
    for (i = 0; i < stacks[ind].GetNumberOfVoxels(); i++) {
      if (*ptr > 0)
        *ptr *= factor;
      ptr++;
    }
  }

  if (_debug) {
    for (ind = 0; ind < stacks.size(); ind++) {
      sprintf(buffer, "rescaled-stack%i.nii.gz", ind);
      stacks[ind].Write(buffer);
    }

    cout << "Slice intensity factors are ";
    for (ind = 0; ind < stack_average.size(); ind++)
      cout << _stack_factor[ind] << " ";
    cout << endl;
    cout << "The new average value is " << averageValue << endl;
  }
}

void irtkReconstructionEbb::generatePSFVolume() {
  double dx, dy, dz;
  // currentlz just a test function
  _slices[_slices.size() - 1].GetPixelSize(&dx, &dy, &dz);

  // sigma of 3D Gaussian (sinc with FWHM=dx or dy in-plane, Gaussian with FWHM
  // = dz through-plane)
  // double psfSigma = 2.3548*((double)PSF_SIZE/2.0);
  double psfSigma = 2.3548;

  double size = _reconstructed.GetXSize() / _quality_factor;

  // TODO TUne until same as CPU
  // number of voxels in each direction
  // the ROI is 2*voxel dimension

  int xDim = PSF_SIZE; // round(2 * dx / size);
  int yDim = PSF_SIZE;
  int zDim = PSF_SIZE;

  double ldim;
  ldim = xDim;
  ldim = (ldim > yDim) ? ldim : yDim;
  ldim = (ldim > zDim) ? ldim : zDim;
  double xsize = xDim / ldim;
  double ysize = yDim / ldim;
  double zsize = zDim / ldim;

  // psfSigma = psfSigma + psfSigma/xsize;

  // printf("%f %f %f\n",xsize, ysize, zsize);

  double sigmax = (1.2) * (dx) / (psfSigma);
  double sigmay = (1.2) * (dy) / (psfSigma);
  double sigmaz = (1.0) * (dz) / (psfSigma);
  // printf("Sigmas orig: %f %f %f\n",sigmax, sigmay, sigmaz);

  // image corresponding to PSF
  irtkImageAttributes attr;
  attr._x = xDim;
  attr._y = yDim;
  attr._z = zDim;
  attr._dx = _reconstructed.GetXSize();
  attr._dy = _reconstructed.GetYSize();
  attr._dz = _reconstructed.GetZSize();
  // attr._dx = size;
  // attr._dy = size;
  // attr._dz = size;
  irtkGenericImage<float> PSF(attr);

  // centre of PSF
  double cx, cy, cz;
  cx = 0.5 * (attr._x - 1);
  cy = 0.5 * (attr._y - 1);
  cz = 0.5 * (attr._z - 1);
  PSF.ImageToWorld(cx, cy, cz);
  // PSF.GetImageToWorldMatrix().Print();
  // printf("%d %d %d \n\n", xDim, yDim, zDim);

  double x, y, z;
  double sum = 0;
  int i, j, k;
  for (i = 0; i < attr._x; i++)
    for (j = 0; j < attr._y; j++)
      for (k = 0; k < attr._z; k++) {
        x = i;
        y = j;
        z = k;

        // printf("%f %f %f \n", x, y, z);
        PSF.ImageToWorld(x, y, z);
        // printf("%f %f %f \n", x, y, z);

        x -= cx;
        y -= cy;
        z -= cz;

#if 0
	//Gauss
	PSF(i, j, k) = exp(-x * x / (2.0 * sigmax * sigmax) - y * y / (2.0 * sigmay * sigmay)
			   - z * z / (2.0 * sigmaz * sigmaz));
#endif
#if 0
	//sinc
	double R = sqrt(x * x / (2 * sigmax * sigmax) + y * y / (2 * sigmay * sigmay)
			+ z * z / (2 * sigmaz * sigmaz));
	PSF(i, j, k) = sin(R) / (R);
#endif
#if 1
        // sinc gauss
        double R =
            sqrt(x * x / (2 * sigmax * sigmax) + y * y / (2 * sigmay * sigmay) +
                 z * z / (2 * sigmaz * sigmaz));
        PSF(i, j, k) = sin(R) / (R) * exp(-x * x / (2 * sigmax * sigmax) -
                                          y * y / (2 * sigmay * sigmay) -
                                          z * z / (2 * sigmaz * sigmaz));
#endif
        sum += PSF(i, j, k);
      }
  PSF /= sum;

  // PSF.Write("PSFtest.nii");
  if (_debugGPU) {
    PSF.Write("PSFtest.nii");
    PSF.GetWorldToImageMatrix().Print();
    PSF.GetImageToWorldMatrix().Print();
  }
}

void irtkReconstructionEbb::CreateSlicesAndTransformations(
    vector<irtkRealImage> &stacks,
    vector<irtkRigidTransformation> &stack_transformations,
    vector<double> &thickness, const vector<irtkRealImage> &probability_maps) {

  std::vector<uint3> stack_sizes_;
  uint3 temp;

  // for each stack
  for (unsigned int i = 0; i < stacks.size(); i++) {
    // image attributes contain image and voxel size
    irtkImageAttributes attr = stacks[i].GetImageAttributes();
    // printf("stack sizes z: %d \n", attr._z);
    temp.x = attr._x;
    temp.y = attr._y;
    temp.z = attr._z;
    // stack_sizes_.push_back(make_uint3(attr._x, attr._y, attr._z));
    stack_sizes_.push_back(temp);
    // attr._z is number of slices in the stack
    for (int j = 0; j < attr._z; j++) {
      // create slice by selecting the appropreate region of the stack
      irtkRealImage slice =
          stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
      // set correct voxel size in the stack. Z size is equal to slice
      // thickness.
      slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
      // remember the slice
      _slices.push_back(slice);
      _simulated_slices.push_back(slice);
      _simulated_weights.push_back(slice);
      _simulated_inside.push_back(slice);
      // remeber stack index for this slice
      _stack_index.push_back(i);
      // initialize slice transformation with the stack transformation
      _transformations.push_back(stack_transformations[i]);
      _transformations_gpu.push_back(stack_transformations[i]);
    }
  }

  //cout << "Number of slices: " << _slices.size() << endl;
}

void irtkReconstructionEbb::ResetSlices(vector<irtkRealImage> &stacks,
                                     vector<double> &thickness) {
  _slices.clear();

  // for each stack
  for (unsigned int i = 0; i < stacks.size(); i++) {
    // image attributes contain image and voxel size
    irtkImageAttributes attr = stacks[i].GetImageAttributes();

    // attr._z is number of slices in the stack
    for (int j = 0; j < attr._z; j++) {
      // create slice by selecting the appropreate region of the stack
      irtkRealImage slice =
          stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
      // set correct voxel size in the stack. Z size is equal to slice
      // thickness.
      slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
      // remember the slice
      _slices.push_back(slice);
    }
  }
  //cout << "Number of slices: " << _slices.size() << endl;

  for (int i = 0; i < _slices.size(); i++) {
    _bias[i].Initialize(_slices[i].GetImageAttributes());
    _weights[i].Initialize(_slices[i].GetImageAttributes());
  }
}

void irtkReconstructionEbb::SetSlicesAndTransformations(
    vector<irtkRealImage> &slices,
    vector<irtkRigidTransformation> &slice_transformations,
    vector<int> &stack_ids, vector<double> &thickness) {
  _slices.clear();
  _stack_index.clear();
  _transformations.clear();
  _transformations_gpu.clear();
  _slices.clear();
  _simulated_slices.clear();
  _simulated_weights.clear();
  _simulated_inside.clear();

  // for each slice
  for (unsigned int i = 0; i < slices.size(); i++) {
    // get slice
    irtkRealImage slice = slices[i];
    // std::cout << "setting slice " << i << "\n";
    // slice.Print();
    // set correct voxel size in the stack. Z size is equal to slice thickness.
    slice.PutPixelSize(slice.GetXSize(), slice.GetYSize(), thickness[i]);
    // remember the slice
    _slices.push_back(slice);
    _simulated_slices.push_back(slice);
    _simulated_weights.push_back(slice);
    _simulated_inside.push_back(slice);
    // remember stack index for this slice
    _stack_index.push_back(stack_ids[i]);
    // get slice transformation
    _transformations.push_back(slice_transformations[i]);
    _transformations_gpu.push_back(slice_transformations[i]);
  }
}

void irtkReconstructionEbb::UpdateSlices(vector<irtkRealImage> &stacks,
                                      vector<double> &thickness) {
  _slices.clear();
  // for each stack
  for (unsigned int i = 0; i < stacks.size(); i++) {
    // image attributes contain image and voxel size
    irtkImageAttributes attr = stacks[i].GetImageAttributes();

    // attr._z is number of slices in the stack
    for (int j = 0; j < attr._z; j++) {
      // create slice by selecting the appropreate region of the stack
      irtkRealImage slice =
          stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
      // set correct voxel size in the stack. Z size is equal to slice
      // thickness.
      slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
      // remember the slice
      _slices.push_back(slice);
    }
  }
//  cout << "Number of slices: " << _slices.size() << endl;
}

void irtkReconstructionEbb::MaskSlices() {
//  cout << "Masking slices ... ";

  double x, y, z;
  int i, j;

  // Check whether we have a mask
  if (!_have_mask) {
    cout << "Could not mask slices because no mask has been set." << endl;
    return;
  }
  //printf("%ld %ld \n", _slices.size(), _transformations.size());
  //_mask.Write("mask.nii");

  // mask slices
  for (int unsigned inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    irtkRealImage &slice = _slices[inputIndex];
    for (i = 0; i < slice.GetX(); i++)
      for (j = 0; j < slice.GetY(); j++) {
        // if the value is smaller than 1 assume it is padding
        if (slice(i, j, 0) < 0.01)
          slice(i, j, 0) = -1;
        // image coordinates of a slice voxel
        x = i;
        y = j;
        z = 0;
        // change to world coordinates in slice space
        slice.ImageToWorld(x, y, z);
        // world coordinates in volume space
        _transformations[inputIndex].Transform(x, y, z);
        // image coordinates in volume space
        _mask.WorldToImage(x, y, z);
        x = round(x);
        y = round(y);
        z = round(z);
        // if the voxel is outside mask ROI set it to -1 (padding value)
        if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) &&
            (z >= 0) && (z < _mask.GetZ())) {
          if (_mask(x, y, z) == 0)
            slice(i, j, 0) = -1;
        } else
          slice(i, j, 0) = -1;
      }
  }

//  cout << "done." << endl;
}

void runSliceToVolumeRegistration(irtkReconstructionEbb *reconstructor, int start, int end)
{
    irtkImageAttributes attr = reconstructor->_reconstructed.GetImageAttributes();
    
    for(int inputIndex = start; inputIndex != end; inputIndex++) {
	irtkImageRigidRegistrationWithPadding registration;
	irtkGreyPixel smin, smax;
	irtkGreyImage target;
	irtkRealImage slice, w, b, t;
	irtkResamplingWithPadding<irtkRealPixel> resampling(attr._dx, attr._dx,
							    attr._dx, -1);
	
	t = reconstructor->_slices[inputIndex];
	resampling.SetInput(&reconstructor->_slices[inputIndex]);
	resampling.SetOutput(&t);
	resampling.Run();
	target = t;
	target.GetMinMax(&smin, &smax);
    
	if (smax > -1) {
	    // put origin to zero
	    irtkRigidTransformation offset;
	    reconstructor->ResetOrigin(target, offset);
	    irtkMatrix mo = offset.GetMatrix();
	    irtkMatrix m =reconstructor->_transformations[inputIndex].GetMatrix();
	    m = m * mo;
	    reconstructor->_transformations[inputIndex].PutMatrix(m);
	    
	    irtkGreyImage source = reconstructor->_reconstructed;
	    registration.SetInput(&target, &source);
	    registration.SetOutput(&reconstructor->_transformations[inputIndex]);
	    registration.GuessParameterSliceToVolume();
	    registration.SetTargetPadding(-1);
	    registration.Run();
	    
	    reconstructor->_slices_regCertainty[inputIndex] = registration.last_similarity;
	    // undo the offset
	    mo.Invert();
	    m = reconstructor->_transformations[inputIndex].GetMatrix();
	    m = m * mo;
	    reconstructor->_transformations[inputIndex].PutMatrix(m);
	}
    }
}

class ParallelSliceToVolumeRegistration {
public:
    irtkReconstructionEbb *reconstructor;
    int nt;

    ParallelSliceToVolumeRegistration(irtkReconstructionEbb *_reconstructor, int _nt)
	: reconstructor(_reconstructor), nt(_nt) {}

    void operator()(const blocked_range<int> &r) const {
	runSliceToVolumeRegistration(reconstructor, r.begin(), r.end());
    }

    // execute
    void operator()() const {
	task_scheduler_init init(nt);
	parallel_for(blocked_range<int>(0, reconstructor->_slices.size()),
		     *this);
	init.terminate();
    }
};

void irtkReconstructionEbb::SliceToVolumeRegistration() {
    if (_slices_regCertainty.size() == 0) {
	_slices_regCertainty.resize(_slices.size());
    }

#ifndef __EBBRT_BM__
    ParallelSliceToVolumeRegistration registration(this, _numThreads);
    registration();
#else
    size_t ncpus = ebbrt::Cpu::Count();
    if(ncpus > 1)
    {
	static ebbrt::SpinBarrier bar(ncpus);
	ebbrt::EventManager::EventContext context;
	std::atomic<size_t> count(0);
	size_t theCpu = ebbrt::Cpu::GetMine();
	int diff = this->_diff;
	for (size_t i = 0; i < ncpus; i++) {
	    // spawn jobs on each core using SpawnRemote
	    ebbrt::event_manager->SpawnRemote(
		[this, theCpu, ncpus, &count, &context, i, diff]() {
		    // get my cpu id
		    size_t mycpu = ebbrt::Cpu::GetMine();
		    int starte, ende, factor;
		    factor = (int)ceil(diff / (float)ncpus);
		    starte = i * factor;
		    ende = i * factor + factor;
		    ende = (ende > diff) ? diff : ende;
		    runSliceToVolumeRegistration(this, starte, ende);
		    count++;
		    bar.Wait();
		    while (count < (size_t)ncpus);
		    if (mycpu == theCpu) {
			ebbrt::event_manager->ActivateContext(std::move(context));
		    }
		},
		    indexToCPU(
			i)); // if i don't add indexToCPU, one of the cores never run ? ?
	}
	ebbrt::event_manager->SaveContext(context);
    }
    else
    {
	runSliceToVolumeRegistration(this, 0, _slices.size());
    }
#endif
}

void runCoeffInit(irtkReconstructionEbb *reconstructor, int start, int end)
{
    for (int inputIndex = start; inputIndex != end; ++inputIndex) {
	
	bool slice_inside;
	
	// current slice
	// irtkRealImage slice;

	// get resolution of the volume
	double vx, vy, vz;
	reconstructor->_reconstructed.GetPixelSize(&vx, &vy, &vz);
	// volume is always isotropic
	double res = vx;

	// start of a loop for a slice inputIndex
	//cout << inputIndex << " ";

	// read the slice
	irtkRealImage &slice = reconstructor->_slices[inputIndex];

	// prepare structures for storage
	POINT3D p;
	VOXELCOEFFS empty;
	SLICECOEFFS slicecoeffs(slice.GetX(),
				vector<VOXELCOEFFS>(slice.GetY(), empty));

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
	double size = res / reconstructor->_quality_factor;

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
		    reconstructor->_transformations[inputIndex].Transform(x, y, z);
		    reconstructor->_reconstructed.WorldToImage(x, y, z);
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
				reconstructor->_transformations[inputIndex].Transform(x, y,
										      z);
				// Change to image coordinates
				reconstructor->_reconstructed.WorldToImage(x, y, z);

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
				    if ((l >= 0) && (l < reconstructor->_reconstructed.GetX()))
					for (m = ny; m <= ny + 1; m++)
					    if ((m >= 0) &&
						(m < reconstructor->_reconstructed.GetY()))
						for (n = nz; n <= nz + 1; n++)
						    if ((n >= 0) &&
							(n < reconstructor->_reconstructed.GetZ())) {
							weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) *
							    (1 - fabs(n - z));
							sum += weight;
							if (reconstructor->_mask(l, m, n) == 1) {
							    inside = true;
							    slice_inside = true;
							}
						    }
				// if there were no voxels do nothing
				if ((sum <= 0) || (!inside))
				    continue;
				// now calculate the transformed PSF
				for (l = nx; l <= nx + 1; l++)
				    if ((l >= 0) && (l < reconstructor->_reconstructed.GetX()))
					for (m = ny; m <= ny + 1; m++)
					    if ((m >= 0) &&
						(m < reconstructor->_reconstructed.GetY()))
						for (n = nz; n <= nz + 1; n++)
						    if ((n >= 0) &&
							(n < reconstructor->_reconstructed.GetZ())) {
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
							    cerr << "Error while trying to populate tPSF. "
								 << aa << " " << bb << " " << cc << endl;
							    cerr << l << " " << m << " " << n << endl;
							    cerr << tx << " " << ty << " " << tz << endl;
							    cerr << centre << endl;
							    tPSF.Write("tPSF.nii");
							    exit(1);
							} else
							    // update transformed PSF
							    tPSF(aa, bb, cc) += value;
						    }
			    } // end of the loop for PSF points

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
		} // end of loop for slice voxels

	reconstructor->_volcoeffs[inputIndex] = slicecoeffs;
	reconstructor->_slice_inside_cpu[inputIndex] = slice_inside;

    } // end of loop through the slices
}

class ParallelCoeffInit {
public:
    irtkReconstructionEbb *reconstructor;
    int nt;

    ParallelCoeffInit(irtkReconstructionEbb *_reconstructor, int _nt)
	: reconstructor(_reconstructor), nt(_nt) {}

    void operator()(const blocked_range<int> &r) const 
    {
	runCoeffInit(reconstructor, r.begin(), r.end());
    }

    // execute
    void operator()() const {
	task_scheduler_init init(nt);
	parallel_for(blocked_range<int>(0, reconstructor->_slices.size()),
		     *this);
	init.terminate();
    }
};
void irtkReconstructionEbb::CoeffInit() {
    _volcoeffs.clear();
    _volcoeffs.resize(_slices.size());
    
    _slice_inside_cpu.clear();
    _slice_inside_cpu.resize(_slices.size());

#ifndef __EBBRT_BM__
    ParallelCoeffInit coeffinit(this, _numThreads);
    coeffinit();
#else
    size_t ncpus = ebbrt::Cpu::Count();
    if(ncpus > 1)
    {
	static ebbrt::SpinBarrier bar(ncpus);
	ebbrt::EventManager::EventContext context;
	std::atomic<size_t> count(0);
	size_t theCpu = ebbrt::Cpu::GetMine();
	int diff = this->_diff;
	for (size_t i = 0; i < ncpus; i++) {
	    // spawn jobs on each core using SpawnRemote
	    ebbrt::event_manager->SpawnRemote(
		[this, theCpu, ncpus, &count, &context, i, diff]() {
		    // get my cpu id
		    size_t mycpu = ebbrt::Cpu::GetMine();
		    int starte, ende, factor;
		    factor = (int)ceil(diff / (float)ncpus);
		    starte = i * factor;
		    ende = i * factor + factor;
		    ende = (ende > diff) ? diff : ende;
		    runCoeffInit(this, starte, ende);
		    count++;
		    bar.Wait();
		    while (count < (size_t)ncpus);
		    if (mycpu == theCpu) {
			ebbrt::event_manager->ActivateContext(std::move(context));
		    }
		},
		    indexToCPU(
			i)); // if i don't add indexToCPU, one of the cores never run ? ?
	}
	ebbrt::event_manager->SaveContext(context);
    }
    else
    {
	runCoeffInit(this, 0, _slices.size());
    }
#endif

    // prepare image for volume weights, will be needed for Gaussian
    // Reconstruction
    _volume_weights.Initialize(_reconstructed.GetImageAttributes());
    _volume_weights = 0;

    int i, j, n, k, inputIndex;
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

    _average_volume_weight = sum / num;

    //std::printf("_average_volume_weight = %lf\n", _average_volume_weight);
} // end of CoeffInit()


void irtkReconstructionEbb::GaussianReconstruction() {
    
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

		    // if (n == 0)
		    //_slices[inputIndex].PutAsDouble(i, j, 0, -1);
		    // calculate num of vox in a slice that have overlap with roi
		    if (n > 0)
			slice_vox_num++;

		    // add contribution of current slice voxel to all voxel volumes
		    // to which it contributes
		    for (k = 0; k < n; k++) {
			p = _volcoeffs[inputIndex][i][j][k];
			_reconstructed(p.x, p.y, p.z) += p.value * slice(i, j, 0);
		    }
		    // debug
		    // p = _volcoeffs[inputIndex][i][j][0];
		    //_reconstructed(p.x, p.y, p.z) += slice(i, j, 0);
		}
	voxel_num.push_back(slice_vox_num);
	// end of loop for a slice inputIndex
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

void irtkReconstructionEbb::InitializeEM() {
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
    //_scale_gpu.push_back(1);

    // Create and initialize slice weights
    _slice_weight_cpu.push_back(1);
    //_slice_weight_gpu.push_back(1);
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

void irtkReconstructionEbb::InitializeEMValues() {
    for (int i = 0; i < _slices.size(); i++) {
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

//  std::cout << "_weights = " << sumImage(_weights) << std::endl;
    //std::cout << "_bias = " << sumImage(_bias) << std::endl;
    //std::cout << "_slices = " << sumImage(_slices) << std::endl;
    //std::cout << "_slice_weight_cpu = " << sumVec(_slice_weight_cpu) << std::endl;
    //std::cout << "_scale_cpu = " << sumVec(_scale_cpu) << std::endl;

}

void irtkReconstructionEbb::InitializeRobustStatistics() {
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
  for (unsigned int i = 0; i < _force_excluded.size(); i++)
    _slice_weight_cpu[_force_excluded[i]] = 0;

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

void runEStep(vector<double>& slice_potential, irtkReconstructionEbb *reconstructor, int start, int end)
{
    for(int inputIndex = start; inputIndex < end; inputIndex++) {
	// read the current slice
	irtkRealImage slice = reconstructor->_slices[inputIndex];
	
	// read current weight image
	// read the current weight image
	//irtkRealImage &w = _weights[inputIndex];

	//_weights[inputIndex] = 0;

	// alias the current bias image
	irtkRealImage &b = reconstructor->_bias[inputIndex];

	// identify scale factor
	double scale = reconstructor->_scale_cpu[inputIndex];

	double num = 0;
	// Calculate error, voxel weights, and slice potential
	for (int i = 0; i < slice.GetX(); i++)
	    for (int j = 0; j < slice.GetY(); j++) {
		if (slice(i, j, 0) != -1) {
		    // bias correct and scale the slice
		    slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

		    // number of volumetric voxels to which
		    // current slice voxel contributes
		    int n = reconstructor->_volcoeffs[inputIndex][i][j].size();

		    // if n == 0, slice voxel has no overlap with volumetric ROI,
		    // do not process it

		    if ((n > 0) &&
			(reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0)) {
			slice(i, j, 0) -=
			    reconstructor->_simulated_slices[inputIndex](i, j, 0);

			// calculate norm and voxel-wise weights

			// Gaussian distribution for inliers (likelihood)
			double g =
			    reconstructor->G(slice(i, j, 0), reconstructor->_sigma_cpu);
			// Uniform distribution for outliers (likelihood)
			double m = reconstructor->M(reconstructor->_m_cpu);

			// voxel_wise posterior
			double weight = g * reconstructor->_mix_cpu /
			    (g * reconstructor->_mix_cpu +
			     m * (1 - reconstructor->_mix_cpu));
			//w.PutAsDouble(i, j, 0, weight);
			
			//_weights[inputIndex].PutAsDouble(i, j, 0, weight);

			reconstructor->_weights[inputIndex](i, j, 0) = weight;
			// calculate slice potentials
			if (reconstructor->_simulated_weights[inputIndex](i, j, 0) >
			    0.99) {
			    slice_potential[inputIndex] += (1.0 - weight) * (1.0 - weight);
			    num++;
			}
		    } else {
			//w.PutAsDouble(i, j, 0, 0);
			reconstructor->_weights[inputIndex](i, j, 0) = 0;
		    }
		}
	    }
	// evaluate slice potential
	if (num > 0) {
	    slice_potential[inputIndex] = sqrt(slice_potential[inputIndex] / num);
	} else
	    slice_potential[inputIndex] = -1; // slice has no unpadded voxels
    }
}

class ParallelEStep {
    irtkReconstructionEbb *reconstructor;
    vector<double> &slice_potential;
    int nt;
public:
    void operator()(const blocked_range<int> &r) const {
	runEStep(slice_potential, reconstructor, r.begin(), r.end());
    }

    ParallelEStep(irtkReconstructionEbb *reconstructor,
		  vector<double> &slice_potential, int _nt)
	: reconstructor(reconstructor), slice_potential(slice_potential), nt(_nt) {}
    
    // execute
    void operator()() const {
	task_scheduler_init init(nt);
	parallel_for(blocked_range<int>(0, reconstructor->_slices.size()),
		     *this);
	init.terminate();
    }
};

void irtkReconstructionEbb::EStep() {
    size_t inputIndex;
    irtkRealImage slice, w, b, sim;
    int num = 0;
    vector<double> slice_potential(_slices.size(), 0);

#ifndef __EBBRT_BM__
    ParallelEStep parallelEStep(this, slice_potential, _numThreads);
    parallelEStep();
#else
    size_t ncpus = ebbrt::Cpu::Count();
    if(ncpus > 1)
    {
	static ebbrt::SpinBarrier bar(ncpus);
	ebbrt::EventManager::EventContext context;
	std::atomic<size_t> count(0);
	size_t theCpu = ebbrt::Cpu::GetMine();
	int diff = this->_diff;
	for (size_t i = 0; i < ncpus; i++) {
	    // spawn jobs on each core using SpawnRemote
	    ebbrt::event_manager->SpawnRemote(
		[this, theCpu, ncpus, &count, &context, i, diff, &slice_potential]() {
		    // get my cpu id
		    size_t mycpu = ebbrt::Cpu::GetMine();
		    int starte, ende, factor;
		    factor = (int)ceil(diff / (float)ncpus);
		    starte = i * factor;
		    ende = i * factor + factor;
		    ende = (ende > diff) ? diff : ende;

		    runEStep(slice_potential, this, starte, ende); 

		    count++;
		    bar.Wait();
		    while (count < (size_t)ncpus);
		    if (mycpu == theCpu) {
			ebbrt::event_manager->ActivateContext(std::move(context));
		    }
		},
		    indexToCPU(
			i)); // if i don't add indexToCPU, one of the cores never run ? ?
	}
	ebbrt::event_manager->SaveContext(context);
    }
    else
    {
	runEStep(slice_potential, this, 0, _slices.size()); 
    }
#endif

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
	    gs1 = G(slice_potential[inputIndex] - _mean_s_cpu, _sigma_s_cpu);
	else
	    gs1 = 0;

	// likelihood for outliers
	if (slice_potential[inputIndex] > _mean_s_cpu)
	    gs2 = G(slice_potential[inputIndex] - _mean_s2_cpu, _sigma_s2_cpu);
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

void runScale(irtkReconstructionEbb *reconstructor, int start, int end)
{
    for(int inputIndex = start; inputIndex != end; inputIndex++) {
	// alias the current slice
	irtkRealImage &slice = reconstructor->_slices[inputIndex];

	// alias the current weight image
	irtkRealImage &w = reconstructor->_weights[inputIndex];

	// alias the current bias image
	irtkRealImage &b = reconstructor->_bias[inputIndex];

	// initialise calculation of scale
	double scalenum = 0;
	double scaleden = 0;

	for (int i = 0; i < slice.GetX(); i++)
	    for (int j = 0; j < slice.GetY(); j++)
		if (slice(i, j, 0) != -1) {
		    if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {
			// scale - intensity matching
			double eb = exp(-b(i, j, 0));
			scalenum += w(i, j, 0) * slice(i, j, 0) * eb *
			    reconstructor->_simulated_slices[inputIndex](i, j, 0);
			scaleden +=
			    w(i, j, 0) * slice(i, j, 0) * eb * slice(i, j, 0) * eb;
		    }
		}

	// calculate scale for this slice
	if (scaleden > 0)
	    reconstructor->_scale_cpu[inputIndex] = scalenum / scaleden;
	else
	    reconstructor->_scale_cpu[inputIndex] = 1;
    }
}

class ParallelScale {
  irtkReconstructionEbb *reconstructor;
    int nt;

public:
    ParallelScale(irtkReconstructionEbb *_reconstructor, int _nt)
	: reconstructor(_reconstructor), nt(_nt) {}

  void operator()(const blocked_range<int> &r) const {
      runScale(reconstructor, r.begin(), r.end());
  }

  // execute
  void operator()() const {
    task_scheduler_init init(nt);
    parallel_for(blocked_range<int>(0, reconstructor->_slices.size()),
                 *this);
    init.terminate();
  }
};

void irtkReconstructionEbb::Scale() { 
#ifndef __EBBRT_BM__
    ParallelScale scale(this, _numThreads);
    scale();
#else
    size_t ncpus = ebbrt::Cpu::Count();
    if(ncpus > 1)
    {
	static ebbrt::SpinBarrier bar(ncpus);
	ebbrt::EventManager::EventContext context;
	std::atomic<size_t> count(0);
	size_t theCpu = ebbrt::Cpu::GetMine();
	int diff = this->_diff;
	for (size_t i = 0; i < ncpus; i++) {
	    // spawn jobs on each core using SpawnRemote
	    ebbrt::event_manager->SpawnRemote(
		[this, theCpu, ncpus, &count, &context, i, diff]() {
		    // get my cpu id
		    size_t mycpu = ebbrt::Cpu::GetMine();
		    int starte, ende, factor;
		    factor = (int)ceil(diff / (float)ncpus);
		    starte = i * factor;
		    ende = i * factor + factor;
		    ende = (ende > diff) ? diff : ende;
		    runScale(this, starte, ende);
		    count++;
		    bar.Wait();
		    while (count < (size_t)ncpus);
		    if (mycpu == theCpu) {
			ebbrt::event_manager->ActivateContext(std::move(context));
		    }
		},
		    indexToCPU(
			i)); // if i don't add indexToCPU, one of the cores never run ? ?
	}
	ebbrt::event_manager->SaveContext(context);
    }
    else
    {
       runScale(this, 0, _slices.size());
    }
#endif
}

void irtkReconstructionEbb::Bias() {

    
    size_t inputIndex = 0;
    for(inputIndex = 0; inputIndex != _slices.size(); inputIndex++) {
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

void irtkReconstructionEbb::AdaptiveRegularization2(vector<irtkRealImage>& _b, vector<double>& _factor, irtkRealImage& _original)
{
    int dx = _reconstructed.GetX();
    int dy = _reconstructed.GetY();
    int dz = _reconstructed.GetZ();
    for (size_t x = 0; x != (size_t)_reconstructed.GetX(); ++x) {
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

void irtkReconstructionEbb::AdaptiveRegularization1(vector<irtkRealImage>& _b, vector<double>& _factor, irtkRealImage& _original)
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
			_b[i](x, y, z) = _factor[i] / sqrt(1 + diff * diff);

		    } else
			_b[i](x, y, z) = 0;
		}
    }
}
 
void irtkReconstructionEbb::AdaptiveRegularization(int iter, irtkRealImage& original)
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

    AdaptiveRegularization1(b, factor, original);
	
    irtkRealImage original2 = _reconstructed;
    AdaptiveRegularization2(b, factor, original2);

    if (_alpha * _lambda / (_delta * _delta) > 0.068) {
	//ebbrt::kprintf("Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068.\n");
    }
}

void runSuperresolution(irtkReconstructionEbb *reconstructor, int start, int end, irtkRealImage& addon, irtkRealImage& _confidence_map)
{
    for (int inputIndex = start; inputIndex < end; ++inputIndex)
    {
	// read the current slice
	irtkRealImage slice = reconstructor->_slices[inputIndex];

	// read the current weight image
	irtkRealImage &w = reconstructor->_weights[inputIndex];

	// read the current bias image
	irtkRealImage &b = reconstructor->_bias[inputIndex];

	// identify scale factor
	double scale = reconstructor->_scale_cpu[inputIndex];

	// Update reconstructed volume using current slice

	// Distribute error to the volume
	POINT3D p;
	for (int i = 0; i < slice.GetX(); i++)
	    for (int j = 0; j < slice.GetY(); j++)
		if (slice(i, j, 0) != -1) {
		    // bias correct and scale the slice
		    slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

		    if (reconstructor->_simulated_slices[inputIndex](i, j, 0) > 0)
			slice(i, j, 0) -=
			    reconstructor->_simulated_slices[inputIndex](i, j, 0);
		    else
			slice(i, j, 0) = 0;

		    int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
		    for (int k = 0; k < n; k++) {
			p = reconstructor->_volcoeffs[inputIndex][i][j][k];
			addon(p.x, p.y, p.z) +=
			    p.value * slice(i, j, 0) * w(i, j, 0) *
			    reconstructor->_slice_weight_cpu[inputIndex];
			_confidence_map(p.x, p.y, p.z) +=
			    p.value * w(i, j, 0) *
			    reconstructor->_slice_weight_cpu[inputIndex];
		    }
		}
    } // end of loop for a slice inputIndex
}

class ParallelSuperresolution {
  irtkReconstructionEbb *reconstructor;
    int nt;

public:
    irtkRealImage confidence_map;
    irtkRealImage addon;

    void operator()(const blocked_range<int> &r) {
	runSuperresolution(reconstructor, r.begin(), r.end(), addon, confidence_map);
    }
    
    ParallelSuperresolution(ParallelSuperresolution &x, split)
	: reconstructor(x.reconstructor), nt(x.nt) {
	// Clear addon
	addon.Initialize(reconstructor->_reconstructed.GetImageAttributes());
	addon = 0;
	
	// Clear confidence map
	confidence_map.Initialize(
	    reconstructor->_reconstructed.GetImageAttributes());
	confidence_map = 0;
    }


    void join(const ParallelSuperresolution &y) {
	addon += y.addon;
	confidence_map += y.confidence_map;
    }

    ParallelSuperresolution(irtkReconstructionEbb *reconstructor, int _nt)
	: reconstructor(reconstructor), nt(_nt) {
	// Clear addon
	addon.Initialize(reconstructor->_reconstructed.GetImageAttributes());
	addon = 0;
	
	// Clear confidence map
	confidence_map.Initialize(
	    reconstructor->_reconstructed.GetImageAttributes());
	confidence_map = 0;
    }

  // execute
    void operator()() {
	task_scheduler_init init(nt);
	parallel_reduce(blocked_range<int>(0, reconstructor->_slices.size()),
			*this);
	init.terminate();
    }
};

void irtkReconstructionEbb::Superresolution(int iter) {

    int i, j, k;
    irtkRealImage addon, original;
    
    // Remember current reconstruction for edge-preserving smoothing
    original = _reconstructed;

#ifndef __EBBRT_BM__
    ParallelSuperresolution parallelSuperresolution(this, _numThreads);
    parallelSuperresolution();

    addon = parallelSuperresolution.addon;
    _confidence_map = parallelSuperresolution.confidence_map;
#else
    // Clear addon
    addon.Initialize(_reconstructed.GetImageAttributes());
    addon = 0;
    
    // Clear confidence map
    _confidence_map.Initialize(_reconstructed.GetImageAttributes());
    _confidence_map = 0;
    
    size_t ncpus = ebbrt::Cpu::Count();
    if(ncpus > 1)
    {
	static ebbrt::SpinBarrier bar(ncpus);
	ebbrt::EventManager::EventContext context;
	std::atomic<size_t> count(0);
	size_t theCpu = ebbrt::Cpu::GetMine();
	int diff = this->_diff;
	for (size_t i = 0; i < ncpus; i++) {
	    // spawn jobs on each core using SpawnRemote
	    ebbrt::event_manager->SpawnRemote(
		[this, theCpu, ncpus, &count, &context, i, diff, &addon]() {
		    // get my cpu id
		    size_t mycpu = ebbrt::Cpu::GetMine();
		    int starte, ende, factor;
		    factor = (int)ceil(diff / (float)ncpus);
		    starte = i * factor;
		    ende = i * factor + factor;
		    ende = (ende > diff) ? diff : ende;

		    irtkRealImage maddon, m_confidence_map;
		    maddon.Initialize(_reconstructed.GetImageAttributes());
		    maddon = 0;
		    m_confidence_map.Initialize(_reconstructed.GetImageAttributes());
		    m_confidence_map = 0;
		    runSuperresolution(this, starte, ende, maddon, m_confidence_map);
		    
		    // braces for scoping
		    {
			// lock_guard auto unlocks end of scope, mutex doesn't work yet
			std::lock_guard<ebbrt::SpinLock> l(spinlock);
			addon += maddon;
			this->_confidence_map += m_confidence_map;
		    }

		    count++;
		    bar.Wait();
		    while (count < (size_t)ncpus);
		    if (mycpu == theCpu) {
			ebbrt::event_manager->ActivateContext(std::move(context));
		    }
		},
		    indexToCPU(
			i)); // if i don't add indexToCPU, one of the cores never run ? ?
	}
	ebbrt::event_manager->SaveContext(context);
    }
    else
    {
	runSuperresolution(this, 0, _slices.size(), addon, _confidence_map);
    }
#endif
    
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
    for (i = 0; i < (int)_reconstructed.GetX(); i++)
	for (j = 0; j < (int)_reconstructed.GetY(); j++)
	    for (k = 0; k < (int)_reconstructed.GetZ(); k++) {
		if (_reconstructed(i, j, k) < _min_intensity * 0.9)
		    _reconstructed(i, j, k) = _min_intensity * 0.9;
		if (_reconstructed(i, j, k) > _max_intensity * 1.1)
		    _reconstructed(i, j, k) = _max_intensity * 1.1;
	    }

    // Smooth the reconstructed image
    AdaptiveRegularization(iter, original);

    // Remove the bias in the reconstructed volume compared to previous iteration
    if (_global_bias_correction)
    {
	BiasCorrectVolume(original);
    }
}

void runMStep(irtkReconstructionEbb *reconstructor, int start, int end, double& sigma, double& mix, double& num, double& min, double& max)
{
    for (int inputIndex = start; inputIndex < end; ++inputIndex) {
	// read the current slice
	irtkRealImage slice = reconstructor->_slices[inputIndex];

	// alias the current weight image
	irtkRealImage &w = reconstructor->_weights[inputIndex];

	// alias the current bias image
	irtkRealImage &b = reconstructor->_bias[inputIndex];

	// identify scale factor
	double scale = reconstructor->_scale_cpu[inputIndex];

	// calculate error
	for (int i = 0; i < slice.GetX(); i++)
	{
	    for (int j = 0; j < slice.GetY(); j++)
	    {
		//std::cout << i << " " << j << " " << inputIndex << " " << slice(i,j,0) << " " << w(i,j,0) << " " << b(i,j,0) << " " << scale << std::endl;
		if (slice(i, j, 0) != -1) {
		    // bias correct and scale the slice
		    slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

		    // otherwise the error has no meaning - it is equal to slice
		    // intensity
		    if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {

			slice(i, j, 0) -=
			    reconstructor->_simulated_slices[inputIndex](i, j, 0);

			// sigma and mix
			double e = slice(i, j, 0);
			sigma += e * e * w(i, j, 0);
			mix += w(i, j, 0);

			//_m
			if (e < min)
			    min = e;
			if (e > max)
			    max = e;

			num++;
		    }
		}
	    }
	}
    } // end of loop for a slice inputIndex
}

class ParallelMStep {
    irtkReconstructionEbb *reconstructor;
    int nt;
    
public:
    double sigma;
    double mix;
    double num;
    double min;
    double max;

    void operator()(const blocked_range<int> &r) {
	runMStep(reconstructor, r.begin(), r.end(), sigma, mix, num, min, max);
    }
    
    ParallelMStep(ParallelMStep &x, split) : reconstructor(x.reconstructor), nt(x.nt) {
	sigma = 0;
	mix = 0;
	num = 0;
	min = 0;
	max = 0;
    }
    
    void join(const ParallelMStep &y) {
	if (y.min < min)
	    min = y.min;
	if (y.max > max)
	    max = y.max;
	
	sigma += y.sigma;
	mix += y.mix;
	num += y.num;
    }
    
    ParallelMStep(irtkReconstructionEbb *reconstructor, int _nt)
	: reconstructor(reconstructor), nt(_nt) {
	sigma = 0;
	mix = 0;
	num = 0;
	min = voxel_limits<irtkRealPixel>::max();
	max = voxel_limits<irtkRealPixel>::min();
    }
    
    // execute
    void operator()() {
	task_scheduler_init init(nt);
	parallel_reduce(blocked_range<int>(0, reconstructor->_slices.size()),
			*this);
	init.terminate();
    }
};

void irtkReconstructionEbb::MStep(int iter) {
#ifndef __EBBRT_BM__
    ParallelMStep parallelMStep(this, _numThreads);
    parallelMStep();
    double sigma = parallelMStep.sigma;
    double mix = parallelMStep.mix;
    double num = parallelMStep.num;
    double min = parallelMStep.min;
    double max = parallelMStep.max;
#else
    double sigma = 0;
    double mix = 0;
    double num = 0;
    double min = voxel_limits<irtkRealPixel>::max();
    double max = voxel_limits<irtkRealPixel>::min();
    
    size_t ncpus = ebbrt::Cpu::Count();
    if(ncpus > 1)
    {
	static ebbrt::SpinBarrier bar(ncpus);
	ebbrt::EventManager::EventContext context;
	std::atomic<size_t> count(0);
	size_t theCpu = ebbrt::Cpu::GetMine();
	int diff = this->_diff;
	for (size_t i = 0; i < ncpus; i++) {
	    // spawn jobs on each core using SpawnRemote
	    ebbrt::event_manager->SpawnRemote(
		[this, theCpu, ncpus, &count, &context, i, diff, &sigma, &mix, &num, &min, &max]() {
		    // get my cpu id
		    size_t mycpu = ebbrt::Cpu::GetMine();
		    int starte, ende, factor;
		    factor = (int)ceil(diff / (float)ncpus);
		    starte = i * factor;
		    ende = i * factor + factor;
		    ende = (ende > diff) ? diff : ende;

		    double msigma = 0;
		    double mmix = 0;
		    double mnum = 0;
		    double mmin = voxel_limits<irtkRealPixel>::max();
		    double mmax = voxel_limits<irtkRealPixel>::min();

		    runMStep(this, starte, ende, msigma, mmix, mnum, mmin, mmax);

		    // braces for scoping
		    {
			// lock_guard auto unlocks end of scope, mutex doesn't work yet
			std::lock_guard<ebbrt::SpinLock> l(spinlock);
			if (mmin < min)
			    min = mmin;
			if (mmax > max)
			    max = mmax;
			sigma += msigma;
			mix += mmix;
			num += mnum;
		    }
		    
		    count++;
		    bar.Wait();
		    while (count < (size_t)ncpus);
		    if (mycpu == theCpu) {
			ebbrt::event_manager->ActivateContext(std::move(context));
		    }
		},
		    indexToCPU(
			i)); // if i don't add indexToCPU, one of the cores never run ? ?
	}
	ebbrt::event_manager->SaveContext(context);
    }
    else
    {
	runMStep(this, 0, _slices.size(), sigma, mix, num, min, max);
    }
#endif
    
    if (mix > 0) {
	_sigma_cpu = sigma / mix;
    } else {
	//ebbrt::kprintf("Something went wrong: sigma= %fmix=%f\n", sigma, mix);
	exit(1);
    }
    if (_sigma_cpu < _step * _step / 6.28)
	_sigma_cpu = _step * _step / 6.28;
    
    if (iter > 1)
	_mix_cpu = mix / num;
    
    // Calculate m
    _m_cpu = 1 / (max - min);
}


void irtkReconstructionEbb::BiasCorrectVolume(irtkRealImage &original) {
  // remove low-frequancy component in the reconstructed image which might have
  // accured due to overfitting of the biasfield
  irtkRealImage residual = _reconstructed;
  irtkRealImage weights = _mask;

  //_reconstructed.Write("super-notbiascor.nii.gz");

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

void irtkReconstructionEbb::Evaluate(int iter) {
    //cout << "Iteration " << iter << ": " << endl;

//  cout << "Included slices CPU: ";
  int sum = 0;
  unsigned int i;
  for (i = 0; i < _slices.size(); i++) {
    if ((_slice_weight_cpu[i] >= 0.5) && (_slice_inside_cpu[i])) {
//	cout << i << " ";
      sum++;
    }
  }
  //cout << endl << "Total: " << sum << endl;

  //cout << "Excluded slices CPU: ";
  sum = 0;
  for (i = 0; i < _slices.size(); i++) {
    if ((_slice_weight_cpu[i] < 0.5) && (_slice_inside_cpu[i])) {
//	cout << i << " ";
      sum++;
    }
  }

  sum = 0;
  for (i = 0; i < _slices.size(); i++) {
    if (!(_slice_inside_cpu[i])) {
      sum++;
    }
  }
}

void irtkReconstructionEbb::NormaliseBias(int iter) {
    irtkRealImage bias;
    bias.Initialize(_reconstructed.GetImageAttributes());
    bias = 0;

    for (size_t inputIndex = 0; inputIndex < _slices.size(); ++inputIndex)
    {
	// alias the current slice
	irtkRealImage &slice = _slices[inputIndex];

	// read the current bias image
	irtkRealImage b = _bias[inputIndex];

	// read current scale factor
	double scale = _scale_cpu[inputIndex];

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
		    int n = _volcoeffs[inputIndex][i][j].size();
		    // add contribution of current slice voxel to all voxel volumes
		    // to which it contributes
		    for (int k = 0; k < n; k++) {
			p = _volcoeffs[inputIndex][i][j][k];
			bias(p.x, p.y, p.z) += p.value * b(i, j, 0);
		    }
		}
    }// end of loop for a slice inputIndex
    
    // normalize the volume by proportion of contributing slice voxels for each
    // volume voxel
    bias /= _volume_weights;

    MaskImage(bias, 0);
    irtkRealImage m = _mask;
    irtkGaussianBlurring<irtkRealPixel> gb(_sigma_bias);
    gb.SetInput(&bias);
    gb.SetOutput(&bias);
    gb.Run();
    gb.SetInput(&m);
    gb.SetOutput(&m);
    gb.Run();
    bias /= m;
    
    irtkRealPixel *pi, *pb;
    pi = _reconstructed.GetPointerToVoxels();
    pb = bias.GetPointerToVoxels();
    for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
	if (*pi != -1)
	    *pi /= exp(-(*pb));
	pi++;
	pb++;
    }
}

void irtkReconstructionEbb::ReadTransformation(char *folder) {
  int n = _slices.size();
  char name[256];
  char path[256];
  irtkTransformation *transformation;
  irtkRigidTransformation *rigidTransf;

  if (n == 0) {
    cerr << "Please create slices before reading transformations!" << endl;
    exit(1);
  }
  cout << "Reading transformations:" << endl;

  _transformations.clear();
  _transformations_gpu.clear();
  for (int i = 0; i < n; i++) {
    if (folder != NULL) {
      sprintf(name, "/transformation%i.dof", i);
      strcpy(path, folder);
      strcat(path, name);
    } else {
      sprintf(path, "transformation%i.dof", i);
    }
    transformation = irtkTransformation::New(path);
    rigidTransf = dynamic_cast<irtkRigidTransformation *>(transformation);
    _transformations.push_back(*rigidTransf);
    _transformations_gpu.push_back(*rigidTransf);
    delete transformation;
    cout << path << endl;
  }
}


void irtkReconstructionEbb::SetReconstructed(irtkRealImage &reconstructed) {
  _reconstructed = reconstructed;
  _template_created = true;
}

void irtkReconstructionEbb::SetTransformations(
    vector<irtkRigidTransformation> &transformations) {
  _transformations.clear();
  _transformations_gpu.clear();
  for (int i = 0; i < transformations.size(); i++) {
    _transformations.push_back(transformations[i]);
    _transformations_gpu.push_back(transformations[i]);
  }
}

void irtkReconstructionEbb::SaveBiasFields() {
  char buffer[256];
  for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    sprintf(buffer, "bias%i.nii.gz", inputIndex);
    _bias[inputIndex].Write(buffer);
  }
}

void irtkReconstructionEbb::SaveConfidenceMap() {
  _confidence_map.Write("confidence-map.nii.gz");
}

void irtkReconstructionEbb::SaveSlices() {
  char buffer[256];
  for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    sprintf(buffer, "slice%i.nii.gz", inputIndex);
    _slices[inputIndex].Write(buffer);
  }
}

void irtkReconstructionEbb::SaveWeights() {
  char buffer[256];
  for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    sprintf(buffer, "weights%i.nii.gz", inputIndex);
    _weights[inputIndex].Write(buffer);
  }
}

void irtkReconstructionEbb::SaveTransformations() {
  char buffer[256];
  for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    sprintf(buffer, "transformation%i.dof", inputIndex);
    _transformations[inputIndex].irtkTransformation::Write(buffer);
    sprintf(buffer, "transformation_gpu%i.dof", inputIndex);
    _transformations_gpu[inputIndex].irtkTransformation::Write(buffer);
  }
}

void irtkReconstructionEbb::GetTransformations(
    vector<irtkRigidTransformation> &transformations) {
  transformations.clear();
  for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
    transformations.push_back(_transformations[inputIndex]);
  }
}

void irtkReconstructionEbb::GetSlices(vector<irtkRealImage> &slices) {
  slices.clear();
  for (unsigned int i = 0; i < _slices.size(); i++)
    slices.push_back(_slices[i]);
}

void irtkReconstructionEbb::SlicesInfo(const char *filename) {
  ofstream info;
  info.open(filename);

  // header
  info << "stack_index"
       << "\t"
       << "included"
       << "\t" // Included slices
       << "excluded"
       << "\t" // Excluded slices
       << "outside"
       << "\t" // Outside slices
       << "weight"
       << "\t"
       << "scale"
       << "\t"
      // << _stack_factor[i] << "\t"
       << "TranslationX"
       << "\t"
       << "TranslationY"
       << "\t"
       << "TranslationZ"
       << "\t"
       << "RotationX"
       << "\t"
       << "RotationY"
       << "\t"
       << "RotationZ" << endl;

  for (int i = 0; i < _slices.size(); i++) {
    irtkRigidTransformation &t = _transformations[i];
    info << _stack_index[i] << "\t"
         << (((_slice_weight_cpu[i] >= 0.5) && (_slice_inside_cpu[i])) ? 1 : 0)
         << "\t" // Included slices
         << (((_slice_weight_cpu[i] < 0.5) && (_slice_inside_cpu[i])) ? 1 : 0)
         << "\t"                                        // Excluded slices
         << ((!(_slice_inside_cpu[i])) ? 1 : 0) << "\t" // Outside slices
         << _slice_weight_cpu[i] << "\t" << _scale_cpu[i] << "\t"
        // << _stack_factor[i] << "\t"
         << t.GetTranslationX() << "\t" << t.GetTranslationY() << "\t"
         << t.GetTranslationZ() << "\t" << t.GetRotationX() << "\t"
         << t.GetRotationY() << "\t" << t.GetRotationZ() << endl;
  }

  info.close();
}


void irtkReconstructionEbb::SplitImage(irtkRealImage image, int packages,
                                    vector<irtkRealImage> &stacks) {
  irtkImageAttributes attr = image.GetImageAttributes();

  // slices in package
  int pkg_z = attr._z / packages;
  double pkg_dz = attr._dz * packages;
  cout << "packages: " << packages << "; slices: " << attr._z
       << "; slices in package: " << pkg_z << endl;
  cout << "slice thickness " << attr._dz
       << "; slickess thickness in package: " << pkg_dz << endl;

  char buffer[256];
  int i, j, k, l;
  double x, y, z, sx, sy, sz, ox, oy, oz;
  for (l = 0; l < packages; l++) {
    attr = image.GetImageAttributes();
    if ((pkg_z * packages + l) < attr._z)
      attr._z = pkg_z + 1;
    else
      attr._z = pkg_z;
    attr._dz = pkg_dz;

    cout << "split image " << l << " has " << attr._z << " slices." << endl;

    // fill values in each stack
    irtkRealImage stack(attr);
    stack.GetOrigin(ox, oy, oz);

    cout << "Stack " << l << ":" << endl;
    for (k = 0; k < stack.GetZ(); k++)
      for (j = 0; j < stack.GetY(); j++)
        for (i = 0; i < stack.GetX(); i++)
          stack.Put(i, j, k, image(i, j, k * packages + l));

    // adjust origin

    // original image coordinates
    x = 0;
    y = 0;
    z = l;
    image.ImageToWorld(x, y, z);
    cout << "image: " << x << " " << y << " " << z << endl;
    // stack coordinates
    sx = 0;
    sy = 0;
    sz = 0;
    stack.PutOrigin(ox, oy, oz); // adjust to original value
    stack.ImageToWorld(sx, sy, sz);
    cout << "stack: " << sx << " " << sy << " " << sz << endl;
    // adjust origin
    cout << "adjustment needed: " << x - sx << " " << y - sy << " " << z - sz
         << endl;
    stack.PutOrigin(ox + (x - sx), oy + (y - sy), oz + (z - sz));
    sx = 0;
    sy = 0;
    sz = 0;
    stack.ImageToWorld(sx, sy, sz);
    cout << "adjusted: " << sx << " " << sy << " " << sz << endl;

    // sprintf(buffer,"stack%i.nii.gz",l);
    // stack.Write(buffer);
    stacks.push_back(stack);
  }
//  cout << "done." << endl;
}

void irtkReconstructionEbb::SplitImageEvenOdd(irtkRealImage image, int packages,
                                           vector<irtkRealImage> &stacks) {
  vector<irtkRealImage> packs;
  vector<irtkRealImage> packs2;
  cout << "Split Image Even Odd: " << packages << " packages." << endl;

  stacks.clear();
  SplitImage(image, packages, packs);
  for (int i = 0; i < packs.size(); i++) {
    cout << "Package " << i << ": " << endl;
    packs2.clear();
    SplitImage(packs[i], 2, packs2);
    stacks.push_back(packs2[0]);
    stacks.push_back(packs2[1]);
  }

//  cout << "done." << endl;
}

void irtkReconstructionEbb::SplitImageEvenOddHalf(irtkRealImage image,
                                               int packages,
                                               vector<irtkRealImage> &stacks,
                                               int iter) {
  vector<irtkRealImage> packs;
  vector<irtkRealImage> packs2;

  cout << "Split Image Even Odd Half " << iter << endl;
  stacks.clear();
  if (iter > 1)
    SplitImageEvenOddHalf(image, packages, packs, iter - 1);
  else
    SplitImageEvenOdd(image, packages, packs);
  for (int i = 0; i < packs.size(); i++) {
    packs2.clear();
    HalfImage(packs[i], packs2);
    for (int j = 0; j < packs2.size(); j++)
      stacks.push_back(packs2[j]);
  }
}

void irtkReconstructionEbb::HalfImage(irtkRealImage image,
                                   vector<irtkRealImage> &stacks) {
  irtkRealImage tmp;
  irtkImageAttributes attr = image.GetImageAttributes();
  stacks.clear();

  // We would not like single slices - that is reserved for slice-to-volume
  if (attr._z >= 4) {
    tmp = image.GetRegion(0, 0, 0, attr._x, attr._y, attr._z / 2);
    stacks.push_back(tmp);
    tmp = image.GetRegion(0, 0, attr._z / 2, attr._x, attr._y, attr._z);
    stacks.push_back(tmp);
  } else
    stacks.push_back(image);
}

void irtkReconstructionEbb::PackageToVolume(vector<irtkRealImage> &stacks,
                                         vector<int> &pack_num, bool evenodd,
                                         bool half, int half_iter) {
  irtkImageRigidRegistrationWithPadding rigidregistration;
  irtkGreyImage t, s;
  // irtkRigidTransformation transformation;
  vector<irtkRealImage> packages;
  char buffer[256];

  int firstSlice = 0;
  cout << "Package to volume: " << endl;
  for (unsigned int i = 0; i < stacks.size(); i++) {
    cout << "Stack " << i << ": First slice index is " << firstSlice << endl;

    packages.clear();
    if (evenodd) {
      if (half)
        SplitImageEvenOddHalf(stacks[i], pack_num[i], packages, half_iter);
      else
        SplitImageEvenOdd(stacks[i], pack_num[i], packages);
    } else
      SplitImage(stacks[i], pack_num[i], packages);

    for (unsigned int j = 0; j < packages.size(); j++) {
      cout << "Package " << j << " of stack " << i << endl;
      if (_debug) {
        sprintf(buffer, "package%i-%i.nii.gz", i, j);
        packages[j].Write(buffer);
      }

      t = packages[j];
      s = _reconstructed;

      // find existing transformation
      double x, y, z;
      x = 0;
      y = 0;
      z = 0;
      packages[j].ImageToWorld(x, y, z);
      stacks[i].WorldToImage(x, y, z);

      int firstSliceIndex = round(z) + firstSlice;
      cout << "First slice index for package " << j << " of stack " << i
           << " is " << firstSliceIndex << endl;
      // transformation = _transformations[sliceIndex];

      // put origin in target to zero
      irtkRigidTransformation offset;
      ResetOrigin(t, offset);
      irtkMatrix mo = offset.GetMatrix();
      irtkMatrix m = _transformations[firstSliceIndex].GetMatrix();
      m = m * mo;
      _transformations[firstSliceIndex].PutMatrix(m);

      rigidregistration.SetInput(&t, &s);
      rigidregistration.SetOutput(&_transformations[firstSliceIndex]);
      rigidregistration.GuessParameterSliceToVolume();
      if (_debug) {
        sprintf(buffer, "par-packages.rreg");
        rigidregistration.Write(buffer);
      }
      rigidregistration.Run();

      // undo the offset
      mo.Invert();
      m = _transformations[firstSliceIndex].GetMatrix();
      m = m * mo;
      _transformations[firstSliceIndex].PutMatrix(m);

      if (_debug) {
        sprintf(buffer, "transformation%i-%i.dof", i, j);
        _transformations[firstSliceIndex].irtkTransformation::Write(buffer);
      }

      // set the transformation to all slices of the package
      cout << "Slices of the package " << j << " of the stack " << i
           << " are: ";
      for (int k = 0; k < packages[j].GetZ(); k++) {
        x = 0;
        y = 0;
        z = k;
        packages[j].ImageToWorld(x, y, z);
        stacks[i].WorldToImage(x, y, z);
        int sliceIndex = round(z) + firstSlice;
        cout << sliceIndex << " " << endl;

        if (sliceIndex >= _transformations.size()) {
          cerr << "irtkRecnstruction::PackageToVolume: sliceIndex out of range."
               << endl;
          cerr << sliceIndex << " " << _transformations.size() << endl;
          exit(1);
        }

        if (sliceIndex != firstSliceIndex) {
          _transformations[sliceIndex].PutTranslationX(
              _transformations[firstSliceIndex].GetTranslationX());
          _transformations[sliceIndex].PutTranslationY(
              _transformations[firstSliceIndex].GetTranslationY());
          _transformations[sliceIndex].PutTranslationZ(
              _transformations[firstSliceIndex].GetTranslationZ());
          _transformations[sliceIndex].PutRotationX(
              _transformations[firstSliceIndex].GetRotationX());
          _transformations[sliceIndex].PutRotationY(
              _transformations[firstSliceIndex].GetRotationY());
          _transformations[sliceIndex].PutRotationZ(
              _transformations[firstSliceIndex].GetRotationZ());
          _transformations[sliceIndex].UpdateMatrix();
        }
      }
    }
    cout << "End of stack " << i << endl << endl;

    firstSlice += stacks[i].GetZ();
  }
}



void irtkReconstructionEbb::CropImage(irtkRealImage &image, irtkRealImage &mask) {
  // Crops the image according to the mask

  int i, j, k;
  // ROI boundaries
  int x1, x2, y1, y2, z1, z2;

  // Original ROI
  x1 = 0;
  y1 = 0;
  z1 = 0;
  x2 = image.GetX();
  y2 = image.GetY();
  z2 = image.GetZ();

  // upper boundary for z coordinate
  int sum = 0;
  for (k = image.GetZ() - 1; k >= 0; k--) {
    sum = 0;
    for (j = image.GetY() - 1; j >= 0; j--)
      for (i = image.GetX() - 1; i >= 0; i--)
        if (mask.Get(i, j, k) > 0)
          sum++;
    if (sum > 0)
      break;
  }
  z2 = k;

  // lower boundary for z coordinate
  sum = 0;
  for (k = 0; k <= image.GetZ() - 1; k++) {
    sum = 0;
    for (j = image.GetY() - 1; j >= 0; j--)
      for (i = image.GetX() - 1; i >= 0; i--)
        if (mask.Get(i, j, k) > 0)
          sum++;
    if (sum > 0)
      break;
  }
  z1 = k;

  // upper boundary for y coordinate
  sum = 0;
  for (j = image.GetY() - 1; j >= 0; j--) {
    sum = 0;
    for (k = image.GetZ() - 1; k >= 0; k--)
      for (i = image.GetX() - 1; i >= 0; i--)
        if (mask.Get(i, j, k) > 0)
          sum++;
    if (sum > 0)
      break;
  }
  y2 = j;

  // lower boundary for y coordinate
  sum = 0;
  for (j = 0; j <= image.GetY() - 1; j++) {
    sum = 0;
    for (k = image.GetZ() - 1; k >= 0; k--)
      for (i = image.GetX() - 1; i >= 0; i--)
        if (mask.Get(i, j, k) > 0)
          sum++;
    if (sum > 0)
      break;
  }
  y1 = j;

  // upper boundary for x coordinate
  sum = 0;
  for (i = image.GetX() - 1; i >= 0; i--) {
    sum = 0;
    for (k = image.GetZ() - 1; k >= 0; k--)
      for (j = image.GetY() - 1; j >= 0; j--)
        if (mask.Get(i, j, k) > 0)
          sum++;
    if (sum > 0)
      break;
  }
  x2 = i;

  // lower boundary for x coordinate
  sum = 0;
  for (i = 0; i <= image.GetX() - 1; i++) {
    sum = 0;
    for (k = image.GetZ() - 1; k >= 0; k--)
      for (j = image.GetY() - 1; j >= 0; j--)
        if (mask.Get(i, j, k) > 0)
          sum++;
    if (sum > 0)
      break;
  }

  x1 = i;

  if (_debug)
    cout << "Region of interest is " << x1 << " " << y1 << " " << z1 << " "
         << x2 << " " << y2 << " " << z2 << endl;

  // Cut region of interest
  image = image.GetRegion(x1, y1, z1, x2 + 1, y2 + 1, z2 + 1);
}

void irtkReconstructionEbb::InvertStackTransformations(
    vector<irtkRigidTransformation> &stack_transformations) {
  // for each stack
  for (unsigned int i = 0; i < stack_transformations.size(); i++) {
    // invert transformation for the stacks
    stack_transformations[i].Invert();
    stack_transformations[i].UpdateParameter();
  }
}

void irtkReconstructionEbb::MaskVolume() {
  irtkRealPixel *pr = _reconstructed.GetPointerToVoxels();
  irtkRealPixel *pm = _mask.GetPointerToVoxels();
  for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
    if (*pm == 0)
      *pr = -1;
    pm++;
    pr++;
  }
}

void irtkReconstructionEbb::MaskImage(irtkRealImage &image, double padding) {
  if (image.GetNumberOfVoxels() != _mask.GetNumberOfVoxels()) {
    cerr << "Cannot mask the image - different dimensions" << endl;
    exit(1);
  }
  irtkRealPixel *pr = image.GetPointerToVoxels();
  irtkRealPixel *pm = _mask.GetPointerToVoxels();
  for (int i = 0; i < image.GetNumberOfVoxels(); i++) {
    if (*pm == 0)
      *pr = padding;
    pm++;
    pr++;
  }
}

/// Like PutMinMax but ignoring negative values (mask)
void irtkReconstructionEbb::Rescale(irtkRealImage &img, double max) {
  int i, n;
  irtkRealPixel *ptr, min_val, max_val;

  // Get lower and upper bound
  img.GetMinMax(&min_val, &max_val);

  n = img.GetNumberOfVoxels();
  ptr = img.GetPointerToVoxels();
  for (i = 0; i < n; i++)
    if (ptr[i] > 0)
      ptr[i] = double(ptr[i]) / double(max_val) * max;
}

void irtkReconstructionEbb::RunRecon(int iterations, double delta, double lastIterLambda, int rec_iterations_first, int rec_iterations_last, bool intensity_matching, double lambda, int levels)
{
    int i;
    int rec_iterations;
    float timers[13] = { 0 };
    struct timeval tstart, tend;
    struct timeval lstart, lend;  
    float sumCompute = 0.0;
    float tempTime = 0.0;
    gettimeofday(&tstart, NULL);

    for (int iter = 0; iter < iterations; iter++) {
	// perform slice-to-volume registrations - skip the first iteration
	if (iter > 0) {
	    gettimeofday(&lstart, NULL);
	    SliceToVolumeRegistration();
	    gettimeofday(&lend, NULL);
	    tempTime = (lend.tv_sec - lstart.tv_sec) +
		((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	    timers[SLICETOVOLUMEREGISTRATION] += tempTime;
	    sumCompute += tempTime;
	}

	if (iter == (iterations - 1)) {
	    SetSmoothingParameters(delta, lastIterLambda);
	} else {
	    double l = lambda;
	    for (i = 0; i < levels; i++) {
		if (iter == iterations * (levels - i - 1) / levels) {
		    SetSmoothingParameters(delta, l);
		}
		l *= 2;
	    }
	}

	// Use faster reconstruction during iterations and slower for final
	// reconstruction
	if (iter < (iterations - 1)) {
	    SpeedupOn();
	} else {
	    SpeedupOff();
	}

	// Initialise values of weights, scales and bias fields
	gettimeofday(&lstart, NULL);
	InitializeEMValues();
	gettimeofday(&lend, NULL);
	tempTime = (lend.tv_sec - lstart.tv_sec) +
	    ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	timers[INITIALIZEEMVALUES] += tempTime;
	sumCompute += tempTime;

	// Calculate matrix of transformation between voxels of slices and volume
	gettimeofday(&lstart, NULL);
	CoeffInit();
	gettimeofday(&lend, NULL);
	tempTime = (lend.tv_sec - lstart.tv_sec) +
	    ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	timers[COEFFINIT] += tempTime;
	sumCompute += tempTime;

	// Initialize reconstructed image with Gaussian weighted reconstruction
	gettimeofday(&lstart, NULL);
	GaussianReconstruction();
	gettimeofday(&lend, NULL);
	tempTime = (lend.tv_sec - lstart.tv_sec) +
	    ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	timers[GAUSSIANRECONSTRUCTION] += tempTime;
	sumCompute += tempTime;

	// Simulate slices (needs to be done after Gaussian reconstruction)
	gettimeofday(&lstart, NULL);
	SimulateSlices();
	gettimeofday(&lend, NULL);
	tempTime = (lend.tv_sec - lstart.tv_sec) +
	    ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	timers[SIMULATESLICES] += tempTime;
	sumCompute += tempTime;

	gettimeofday(&lstart, NULL);
	InitializeRobustStatistics();
	gettimeofday(&lend, NULL);
	tempTime = (lend.tv_sec - lstart.tv_sec) +
	    ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	timers[INITIALIZEROBUSTSTATISTICS] += tempTime;
	sumCompute += tempTime;

	gettimeofday(&lstart, NULL);
	EStep();
	gettimeofday(&lend, NULL);
	tempTime = (lend.tv_sec - lstart.tv_sec) +
	    ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	timers[ESTEP] += tempTime;
	sumCompute += tempTime;

	// number of reconstruction iterations
	if (iter == (iterations - 1)) {
	    rec_iterations = rec_iterations_last;
	} else
	    rec_iterations = rec_iterations_first;

	// reconstruction iterations
	i = 0;
	for (i = 0; i < rec_iterations; i++) {
	    if (intensity_matching) {
		// calculate bias fields
		gettimeofday(&lstart, NULL);
		// calculate scales
		Scale();
		gettimeofday(&lend, NULL);
		tempTime = (lend.tv_sec - lstart.tv_sec) +
		    ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
		timers[SCALE] += tempTime;
		sumCompute += tempTime;
	    }
	
	    // MStep and update reconstructed volume
	    gettimeofday(&lstart, NULL);
	    Superresolution(i + 1);
	    gettimeofday(&lend, NULL);
	    tempTime = (lend.tv_sec - lstart.tv_sec) +
		((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	    timers[SUPERRESOLUTION] += tempTime;
	    sumCompute += tempTime;

	    // Simulate slices (needs to be done
	    // after the update of the reconstructed volume)
	    gettimeofday(&lstart, NULL);
	    SimulateSlices();
	    gettimeofday(&lend, NULL);
	    tempTime = (lend.tv_sec - lstart.tv_sec) +
		((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	    timers[SIMULATESLICES] += tempTime;
	    sumCompute += tempTime;
            
	    gettimeofday(&lstart, NULL);
	    MStep(i + 1);
	    gettimeofday(&lend, NULL);
	    tempTime = (lend.tv_sec - lstart.tv_sec) +
		((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	    timers[MSTEP] += tempTime;
	    sumCompute += tempTime;

	    gettimeofday(&lstart, NULL);
	    EStep();
	    gettimeofday(&lend, NULL);
	    tempTime = (lend.tv_sec - lstart.tv_sec) +
		((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	    timers[ESTEP] += tempTime;
	    sumCompute += tempTime;
	} // end of reconstruction iterations

	// Mask reconstructed image to ROI given by the mask
	gettimeofday(&lstart, NULL);
	MaskVolume();
	gettimeofday(&lend, NULL);
	tempTime = (lend.tv_sec - lstart.tv_sec) +
	    ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	timers[MASKVOLUME] += tempTime;
	sumCompute += tempTime;

	gettimeofday(&lstart, NULL);
	Evaluate(iter);
	gettimeofday(&lend, NULL);
	tempTime = (lend.tv_sec - lstart.tv_sec) +
	    ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	timers[EVALUATE] += tempTime;
	sumCompute += tempTime;
    } // end of interleaved registration-reconstruction iterations
  
    gettimeofday(&lstart, NULL);
    RestoreSliceIntensities();
    ScaleVolume();
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
	((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[RESTORESLICE] += tempTime;
    sumCompute += tempTime;

    gettimeofday(&tend, NULL);

    FORPRINTF("SliceToVolumeRegistration: %lf seconds\n", timers[SLICETOVOLUMEREGISTRATION]);
    FORPRINTF("InitializeEMValues: %lf seconds\n", timers[INITIALIZEEMVALUES]);
    FORPRINTF("CoeffInit: %lf seconds\n", timers[COEFFINIT]);
    FORPRINTF("GaussianReconstruction: %lf seconds\n", timers[GAUSSIANRECONSTRUCTION]);
    FORPRINTF("SimulateSlices: %lf seconds\n", timers[SIMULATESLICES]);
    FORPRINTF("InitializeRobustStatistics: %lf seconds\n", timers[INITIALIZEROBUSTSTATISTICS]);
    FORPRINTF("EStep: %lf seconds\n", timers[ESTEP]);
    FORPRINTF("Scale: %lf seconds\n", timers[SCALE]);
    FORPRINTF("Superresolution: %lf seconds\n", timers[SUPERRESOLUTION]);
    FORPRINTF("MStep: %lf seconds\n", timers[MSTEP]);
    FORPRINTF("MaskVolume: %lf seconds\n", timers[MASKVOLUME]);
    FORPRINTF("Evaluate: %lf seconds\n", timers[EVALUATE]);
    FORPRINTF("RestoreSliceIntensities and ScaleVolume: %lf seconds\n", timers[RESTORESLICE]);

    FORPRINTF("compute time: %lf seconds\n", (tend.tv_sec - tstart.tv_sec) + ((tend.tv_usec - tstart.tv_usec) / 1000000.0));
    FORPRINTF("sumCompute: %lf seconds\n", sumCompute);
  
    float temp2 = 0.0;
    for(i=0;i<13;i++)
    {
	temp2 += timers[i];
    }
    FORPRINTF("sumTimers: %lf seconds\n", temp2);
  
    FORPRINTF("checksum _reconstructed = %lf\n", SumRecon());

    // save final result
    //reconstructed = GetReconstructed();
    //reconstructed.Write(outputName.c_str());

}


#pragma GCC diagnostic pop
