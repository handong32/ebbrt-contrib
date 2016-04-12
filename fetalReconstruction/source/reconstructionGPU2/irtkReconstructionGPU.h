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
Bernhard Kainz, Markus Steinberger, Maria Kuklisova-Murgasova, Christina Malamateniou,
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

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=========================================================================*/

#ifndef APPS_IRTKRECONSTRUCTION_H_
#define APPS_IRTKRECONSTRUCTION_H_

#include <irtkImage.h>
#include <irtkTransformation.h>
#include <irtkGaussianBlurring.h>

#include <vector>

#include <unordered_map>

using namespace std;

#include "utils.h"

class irtkReconstruction : public irtkObject
{ 
public:
  vector<irtkRealImage> _slices_resampled;
  int _numThreads;
  int _start, _end, _diff;
  
  /// Transformations
  vector<irtkRigidTransformation> _transformations_gpu;
  /// Indicator whether slice has an overlap with volumetric mask

  vector<bool> _slice_inside_gpu;

  //VOLUME
  /// Reconstructed volume
  irtkGenericImage<float> _reconstructed_gpu;

  /// Flag to say whether the template volume has been created
  bool _template_created;


  /// Flag to say whether we have a mask
  bool _have_mask;
  /// Weights for Gaussian reconstruction
  irtkRealImage _volume_weights;
  /// Weights for regularization
  irtkRealImage _confidence_map;

  //EM algorithm
  /// Variance for inlier voxel errors
  double _sigma_cpu;
  float _sigma_gpu;
  /// Proportion of inlier voxels
  double _mix_cpu;
  float _mix_gpu;
  /// Uniform distribution for outlier voxels
  double _m_cpu;
  float _m_gpu;
  /// Mean for inlier slice errors
  double _mean_s_cpu;
  float _mean_s_gpu;
  /// Variance for inlier slice errors
  double _sigma_s_cpu;
  float _sigma_s_gpu;
  /// Mean for outlier slice errors
  double _mean_s2_cpu;
  float _mean_s2_gpu;
  /// Variance for outlier slice errors
  double _sigma_s2_cpu;
  float _sigma_s2_gpu;
  /// Proportion of inlier slices
  double _mix_s_cpu;
  float _mix_s_gpu;
  /// Step size for likelihood calculation
  double _step;
  /// Voxel posteriors
  vector<irtkRealImage> _weights;
  ///Slice posteriors
  vector<double> _slice_weight_cpu;
  vector<float> _slice_weight_gpu;

  //Bias field
  ///Variance for bias field
  double _sigma_bias;
  /// Slice-dependent bias fields
  vector<irtkRealImage> _bias;

  ///Slice-dependent scales
  vector<double> _scale_cpu;
  vector<float> _scale_gpu;


  //use sinc like function or not
  bool _use_SINC;

  ///Intensity min and max
  double _max_intensity;
  double _min_intensity;

  //Gradient descent and regulatization parameters
  ///Step for gradient descent
  double _alpha;
  ///Determine what is en edge in edge-preserving smoothing
  double _delta;
  ///Amount of smoothing
  double _lambda;
  ///Average voxel wights to modulate parameter alpha
  double _average_volume_weight;


  //global bias field correction
  ///low intensity cutoff for bias field estimation
  double _low_intensity_cutoff;

  //to restore original signal intensity of the MRI slices
  double _average_value;
  
  //forced excluded slices
  vector<int> _force_excluded;

  //slices identify as too small to be used
  vector<int> _small_slices;

  /// use adaptive or non-adaptive regularisation (default:false)
  bool _adaptive;

  //utility
  ///Debug mode
  bool _debug;


  //Probability density functions
  ///Zero-mean Gaussian PDF
  inline double G(double x, double s);
  ///Uniform PDF
  inline double M(double m);

  int _directions[13][3];

  //Reconstruction* reconstructionGPU;

  bool _useCPUReg;
  bool _useCPU;
  bool _debugGPU;

  void RunRecon(int iterations, double delta, double lastIterLambda, int rec_iterations_first, int rec_iterations_last, bool intensity_matching, double lambda, int levels);
  
    //Structures to store the matrix of transformation between volume and slices
  std::vector<SLICECOEFFS> _volcoeffs;
  vector<irtkRealImage> _slices;
  irtkRealImage _reconstructed;
  vector<irtkRigidTransformation> _transformations;
  vector<double> _slices_regCertainty;
  vector<bool> _slice_inside_cpu;
  //SLICES
  /// Slices
  vector<irtkRealImage> _simulated_slices;
  vector<irtkRealImage> _simulated_weights;
  vector<irtkRealImage> _simulated_inside;
  
  vector<float> _stack_factor;
  vector<int> _stack_index;

  
  ///global bias correction flag
  bool _global_bias_correction;

  /// Volume mask
  irtkRealImage _mask;

  ///Quality factor - higher means slower and better
  double _quality_factor;

  ///Constructor
  irtkReconstruction();
  ///Destructor
  //~irtkReconstruction();

  ///Create zero image as a template for reconstructed volume
  double CreateTemplate(irtkRealImage stack,
    double resolution = 0);

  ///If template image has been masked instead of creating the mask in separate
  ///file, this function can be used to create mask from the template image
  irtkRealImage CreateMask(irtkRealImage image);

  ///Remember volumetric mask and smooth it if necessary
  void SetMask(irtkRealImage * mask, double sigma, double threshold = 0.5);


  void SaveProbabilityMap(int i);

  void setNumThreads(int i);

  void CenterStacks(vector<irtkRealImage>& stacks,
    vector<irtkRigidTransformation>& stack_transformations,
    int templateNumber);

  //Create average image from the stacks and volumetric transformations
  irtkRealImage CreateAverage(vector<irtkRealImage>& stacks,
    vector<irtkRigidTransformation>& stack_transformations);

  ///Crop image according to the mask
  void CropImage(irtkRealImage& image,
    irtkRealImage& mask);

  /// Transform and resample mask to the space of the image
  void TransformMask(irtkRealImage& image,
    irtkRealImage& mask,
    irtkRigidTransformation& transformation);

  /// Rescale image ignoring negative values
  void Rescale(irtkRealImage &img, double max);

  ///Calculate initial registrations
  void StackRegistrations(vector<irtkRealImage>& stacks,
    vector<irtkRigidTransformation>& stack_transformations,
    int templateNumber, bool useExternalTarget = false);

  ///Create slices from the stacks and slice-dependent transformations from
  ///stack transformations
  void CreateSlicesAndTransformations(vector<irtkRealImage>& stacks,
    vector<irtkRigidTransformation>& stack_transformations,
    vector<double>& thickness,
    const vector<irtkRealImage> &probability_maps = vector<irtkRealImage>());
  void SetSlicesAndTransformations(vector<irtkRealImage>& slices,
    vector<irtkRigidTransformation>& slice_transformations,
    vector<int>& stack_ids,
    vector<double>& thickness);
  void ResetSlices(vector<irtkRealImage>& stacks,
    vector<double>& thickness);

  ///Update slices if stacks have changed
  void UpdateSlices(vector<irtkRealImage>& stacks, vector<double>& thickness);

  void GetSlices(vector<irtkRealImage>& second_stacks);

  ///Invert all stack transformation
  void InvertStackTransformations(vector<irtkRigidTransformation>& stack_transformations);

  ///Match stack intensities
  void MatchStackIntensities(vector<irtkRealImage>& stacks,
    vector<irtkRigidTransformation>& stack_transformations,
    double averageValue,
    bool together = false);

  ///Match stack intensities with masking
  void MatchStackIntensitiesWithMasking(vector<irtkRealImage>& stacks,
    vector<irtkRigidTransformation>& stack_transformations,
    double averageValue,
    bool together = false);

  ///Mask all slices
  void MaskSlices();

  ///Calculate transformation matrix between slices and voxels
  void CoeffInit();

  ///Reconstruction using weighted Gaussian PSF
  void GaussianReconstruction();

  ///Initialise variables and parameters for EM
  void InitializeEM();

  ///Initialise values of variables and parameters for EM
  void InitializeEMValues();

  ///Initalize robust statistics
  void InitializeRobustStatistics();

  ///Perform E-step 
  void EStep();

  ///Calculate slice-dependent scale
  void Scale();

  ///Calculate slice-dependent bias fields
  void Bias();
  void NormaliseBias(int iter);
  
  
  ///Superresolution
  void Superresolution(int iter);

  ///Calculation of voxel-vise robust statistics
  void MStep(int iter);

  ///Edge-preserving regularization
  void Regularization(int iter);

  void AdaptiveRegularization1(vector<irtkRealImage>& _b, vector<double>& _factor, irtkRealImage& _original);
  
  void AdaptiveRegularization2(vector<irtkRealImage>& _b, vector<double>& _factor, irtkRealImage& _original);
  
  ///Edge-preserving regularization with confidence map
  void AdaptiveRegularization(int iter, irtkRealImage& original);

  ///Slice to volume registrations
  void SliceToVolumeRegistration();

  ///Correct bias in the reconstructed volume
  void BiasCorrectVolume(irtkRealImage& original);

  ///Mask the volume
  void MaskVolume();
  void MaskImage(irtkRealImage& image, double padding = -1);


  ///Save slices
  void SaveSlices();
  void SlicesInfo(const char* filename);

  ///Save weights
  void SaveWeights();

  ///Save transformations
  void SaveTransformations();
  void GetTransformations(vector<irtkRigidTransformation> &transformations);
  void SetTransformations(vector<irtkRigidTransformation> &transformations);

  ///Save confidence map
  void SaveConfidenceMap();

  ///Save bias field
  void SaveBiasFields();

  ///Remember stdev for bias field
  inline void SetSigma(double sigma);

  ///Return reconstructed volume
  inline irtkRealImage GetReconstructed();
  void SetReconstructed(irtkRealImage &reconstructed);

  ///Return resampled mask
  inline irtkRealImage GetMask();

  ///Set smoothing parameters
  inline void SetSmoothingParameters(double delta, double lambda);

  inline float SumRecon();

  ///use in-plane sinc like PSF
  inline void useSINCPSF();

  ///Use faster lower quality reconstruction
  inline void SpeedupOn();

  ///Use slower better quality reconstruction
  inline void SpeedupOff();

  ///Switch on global bias correction
  inline void GlobalBiasCorrectionOn();

  ///Switch off global bias correction
  inline void GlobalBiasCorrectionOff();

  ///Set lower threshold for low intensity cutoff during bias estimation
  inline void SetLowIntensityCutoff(double cutoff);

  ///Set slices which need to be excluded by default
  inline void SetForceExcludedSlices(vector<int>& force_excluded);


  //utility
  ///Save intermediate results
  inline void DebugOn();
  ///Do not save intermediate results
  inline void DebugOff();

  inline void UseAdaptiveRegularisation();

  ///Write included/excluded/outside slices
  void Evaluate(int iter);

  /// Read Transformations
  void ReadTransformation(char* folder);

  /// Read and replace Slices
//  void replaceSlices(string folder);

  //To recover original scaling
  ///Restore slice intensities to their original values
  void RestoreSliceIntensities();
  ///Scale volume to match the slice intensities
  void ScaleVolume();

  ///To compare how simulation from the reconstructed volume matches the original stacks
  void SimulateStacks(vector<irtkRealImage>& stacks);

  void SimulateSlices();

  ///Puts origin of the image into origin of world coordinates
  static void ResetOrigin(irtkGreyImage &image,
    irtkRigidTransformation& transformation);

  ///Packages to volume registrations
  void PackageToVolume(vector<irtkRealImage>& stacks,
    vector<int> &pack_num,
    bool evenodd = false,
    bool half = false,
    int half_iter = 1);

  ///Splits stacks into packages
  void SplitImage(irtkRealImage image,
    int packages,
    vector<irtkRealImage>& stacks);
  ///Splits stacks into packages and each package into even and odd slices
  void SplitImageEvenOdd(irtkRealImage image,
    int packages,
    vector<irtkRealImage>& stacks);
  ///Splits image into top and bottom half roi according to z coordinate
  void HalfImage(irtkRealImage image,
    vector<irtkRealImage>& stacks);
  ///Splits stacks into packages and each package into even and odd slices and top and bottom roi
  void SplitImageEvenOddHalf(irtkRealImage image,
    int packages,
    vector<irtkRealImage>& stacks,
    int iter = 1);

  ///sync GPU with data
  //TODO distribute in documented functions above
  irtkRealImage externalRegistrationTargetImage;
  void generatePSFVolume();
  static void ResetOrigin(irtkRealImage &image, irtkRigidTransformation& transformation);

  void PrepareRegistrationSlices();
  //friend class ParallelAverage;
  //friend class ParallelSimulateSlices;
  //friend class ParallelStackRegistrations;
  //friend class ParallelScale;
};

inline double irtkReconstruction::G(double x, double s)
{
  return _step*exp(-x*x / (2 * s)) / (sqrt(6.28*s));
}

inline double irtkReconstruction::M(double m)
{
  return m*_step;
}

inline irtkRealImage irtkReconstruction::GetReconstructed()
{
  return _reconstructed;
}

inline irtkRealImage irtkReconstruction::GetMask()
{
  return _mask;
}

inline void irtkReconstruction::DebugOn()
{
  _debug = true;
//  cout << "Debug mode." << endl;
}

inline void irtkReconstruction::UseAdaptiveRegularisation()
{
  _adaptive = true;
}

inline void irtkReconstruction::DebugOff()
{
  _debug = false;
}

inline void irtkReconstruction::SetSigma(double sigma)
{
  _sigma_bias = sigma;
  //cout << "_sigma_bias = " << sigma << endl;
}

inline void irtkReconstruction::useSINCPSF()
{
  _use_SINC = true;
}


inline void irtkReconstruction::SpeedupOn()
{
  _quality_factor = 1;
}

inline void irtkReconstruction::SpeedupOff()
{
  _quality_factor = 2;
}

inline void irtkReconstruction::GlobalBiasCorrectionOn()
{
  _global_bias_correction = true;
//  cout << "_global_bias_correction = true " << endl;
}

inline void irtkReconstruction::GlobalBiasCorrectionOff()
{
  _global_bias_correction = false;
//  cout << "_global_bias_correction = false " << endl;
}

inline void irtkReconstruction::SetLowIntensityCutoff(double cutoff)
{
  if (cutoff > 1) cutoff = 1;
  if (cutoff < 0) cutoff = 0;
  _low_intensity_cutoff = cutoff;
  //cout<<"Setting low intensity cutoff for bias correction to "<<_low_intensity_cutoff<<" of the maximum intensity."<<endl;
}


inline void irtkReconstruction::SetSmoothingParameters(double delta, double lambda)
{
  _delta = delta;
  _lambda = lambda*delta*delta;
  _alpha = 0.05 / lambda;
  if (_alpha > 1) _alpha = 1;
  //cout << "_delta = " << _delta << " _lambda = " << _lambda << " _alpha = " << _alpha << endl;
}

inline void irtkReconstruction::SetForceExcludedSlices(vector<int>& force_excluded)
{
  _force_excluded = force_excluded;
}

inline float irtkReconstruction::SumRecon() {
  float sum = 0.0;
  irtkRealPixel *ap = _reconstructed.GetPointerToVoxels();

  for (int j = 0; j < _reconstructed.GetNumberOfVoxels(); j++) {
    sum += *ap;
    ap++;
  }
  return sum;
}

#endif
