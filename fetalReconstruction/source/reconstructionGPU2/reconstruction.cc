//#define EBB
#define NCPUS 2
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
#include <irtkImage.h>
#include <irtkTransformation.h>
#include <irtkReconstructionGPU.h>
#include <irtkResampling.h>
#include <vector>
#include <string>
//#include <perfstats.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <boost/program_options.hpp>

#include <utils.h>
//#include "utils.h"

#include <signal.h>
#include <thread>
#include <chrono>

#include <boost/filesystem.hpp>

#include <ebbrt/Context.h>
#include <ebbrt/ContextActivation.h>
#include <ebbrt/GlobalIdMap.h>
#include <ebbrt/StaticIds.h>
#include <ebbrt/NodeAllocator.h>
#include <ebbrt/Runtime.h>
#include <ebbrt/Clock.h>

#include <EbbRTReconstruction.h>

enum TTYPE {
  INITIALIZEEMVALUES,
  COEFFINIT,
  GAUSSIANRECONSTRUCTION,
  SIMULATESLICES,
  INITIALIZEROBUSTSTATISTICS,
  ESTEP,
  SCALE,
  SUPERRESOLUTION,
  MSTEP,
  MASKVOLUME,
  EVALUATE,
  SLICETOVOLUMEREGISTRATION,
  RESTORESLICE
};

//#include <boost/serialization/vector.hpp>
//#include <boost/archive/text_oarchive.hpp>
//#include <boost/archive/text_iarchive.hpp>

namespace po = boost::program_options;

#if HAVE_CULA
#include "stackMotionEstimator.h"
#endif

using namespace std;

const std::string currentDateTime() {
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);

  return buf;
}

float sumOneImage(irtkRealImage a) {
  float sum = 0.0;
  irtkRealPixel *ap = a.GetPointerToVoxels();

  for (int j = 0; j < a.GetNumberOfVoxels(); j++) {
    sum += *ap;
    ap++;
  }
  return sum;
}

int main(int argc, char **argv) {
    struct timeval totstart, totend;
    gettimeofday(&totstart, NULL);
    std::cout << "starting reconstruction on " << currentDateTime() << std::endl;
    
    float timers[13] = { 0 };
    int i, ok;
    char buffer[256];
    irtkRealImage stack;

    // declare variables for input
    /// Slice stacks
    vector<irtkRealImage> stacks;
    /// Stack transformation
    vector<irtkRigidTransformation> stack_transformations;
    /// Stack thickness
    vector<double> thickness;
    /// number of stacks
    int nStacks;
    /// number of packages for each stack
    vector<int> packages;

    vector<float> stackMotion;

    // Default values.
    int templateNumber = -1;
    irtkRealImage *mask = NULL;
    int iterations = 9; // 9 //2 for Shepp-Logan is enough
    bool debug = false;
    bool debug_gpu = false;
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
    // folder for slice-to-volume registrations, if given
    string tfolder;
    // folder to replace slices with registered slices, if given
    string sfolder;
    // flag to swich the intensity matching on and off
    bool intensity_matching = true;
    unsigned int rec_iterations_first = 4;
    unsigned int rec_iterations_last = 13;

    // number of threads
    int numThreads;

    bool useCPU = false;
    bool useCPUReg = true;
    bool useGPUReg = false;
    bool disableBiasCorr = false;
    bool useAutoTemplate = false;

    irtkRealImage average;

    string log_id;
    bool no_log = false;

    // forced exclusion of slices
    int number_of_force_excluded_slices = 0;
    vector<int> force_excluded;
    vector<int> devicesToUse;

    vector<string> inputStacks;
    vector<string> inputTransformations;
    string maskName;
    /// Name for output volume
    string outputName;
    unsigned int num_input_stacks_tuner = 0;
    string referenceVolumeName;
    unsigned int T1PackageSize = 0;
    unsigned int numDevicesToUse = UINT_MAX;
    bool useSINCPSF = false;

    try {
	    po::options_description desc("Options");
    desc.add_options()("help,h", "Print usage messages")(
        "output,o", po::value<string>(&outputName)->required(),
        "Name for the reconstructed volume. Nifti or Analyze format.")(
        "mask,m", po::value<string>(&maskName), "Binary mask to define the "
                                                "region od interest. Nifti or "
                                                "Analyze format.")(
        "input,i", po::value<vector<string> >(&inputStacks)->multitoken(),
        "[stack_1] .. [stack_N]  The input stacks. Nifti or Analyze format.")(
        "transformation,t",
        po::value<vector<string> >(&inputTransformations)->multitoken(),
        "The transformations of the input stack to template in \'dof\' format "
        "used in IRTK. Only rough alignment with correct orienation and some "
        "overlap is needed. Use \'id\' for an identity transformation for at "
        "least one stack. The first stack with \'id\' transformation  will be "
        "resampled as template.")(
        "thickness", po::value<vector<double> >(&thickness)->multitoken(),
        "[th_1] .. [th_N] Give slice thickness.[Default: twice voxel size in z "
        "direction]")(
        "packages,p", po::value<vector<int> >(&packages)->multitoken(),
        "Give number of packages used during acquisition for each stack. The "
        "stacks will be split into packages during registration iteration 1 "
        "and then into odd and even slices within each package during "
        "registration iteration 2. The method will then continue with slice to "
        " volume approach. [Default: slice to volume registration only]")(
        "iterations", po::value<int>(&iterations)->default_value(4),
        "Number of registration-reconstruction iterations.")(
        "sigma", po::value<double>(&sigma)->default_value(12.0),
        "Stdev for bias field. [Default: 12mm]")(
        "resolution", po::value<double>(&resolution)->default_value(0.75),
        "Isotropic resolution of the volume. [Default: 0.75mm]")(
        "multires", po::value<int>(&levels)->default_value(3),
        "Multiresolution smooting with given number of levels. [Default: 3]")(
        "average", po::value<double>(&averageValue)->default_value(700),
        "Average intensity value for stacks [Default: 700]")(
        "delta", po::value<double>(&delta)->default_value(150),
        " Parameter to define what is an edge. [Default: 150]")(
        "lambda", po::value<double>(&lambda)->default_value(0.02),
        "  Smoothing parameter. [Default: 0.02]")(
        "lastIterLambda",
        po::value<double>(&lastIterLambda)->default_value(0.01),
        "Smoothing parameter for last iteration. [Default: 0.01]")(
        "smooth_mask", po::value<double>(&smooth_mask)->default_value(4),
        "Smooth the mask to reduce artefacts of manual segmentation. [Default: "
        "4mm]")(
        "global_bias_correction",
        po::value<bool>(&global_bias_correction)->default_value(false),
        "Correct the bias in reconstructed image against previous estimation.")(
        "low_intensity_cutoff",
        po::value<double>(&low_intensity_cutoff)->default_value(0.01),
        "Lower intensity threshold for inclusion of voxels in global bias "
        "correction.")("force_exclude",
                       po::value<vector<int> >(&force_excluded)->multitoken(),
                       "force_exclude [number of slices] [ind1] ... [indN]  "
                       "Force exclusion of slices with these indices.")(
        "no_intensity_matching", po::value<bool>(&intensity_matching),
        "Switch off intensity matching.")(
        "log_prefix", po::value<string>(&log_id), "Prefix for the log file.")(
        "debug", po::value<bool>(&debug)->default_value(false),
        " Debug mode - save intermediate results.")(
        "debug_gpu", po::bool_switch(&debug_gpu)->default_value(false),
        " Debug only GPU results.")(
        "rec_iterations_first",
        po::value<unsigned int>(&rec_iterations_first)->default_value(4),
        " Set number of superresolution iterations")(
        "rec_iterations_last",
        po::value<unsigned int>(&rec_iterations_last)->default_value(13),
        " Set number of superresolution iterations for the last iteration")(
        "num_stacks_tuner",
        po::value<unsigned int>(&num_input_stacks_tuner)->default_value(0),
        "  Set number of input stacks that are really used (for tuner "
        "evaluation, use only first x)")(
        "no_log", po::value<bool>(&no_log)->default_value(false),
        "  Do not redirect cout and cerr to log files.")(
        "devices,d", po::value<vector<int> >(&devicesToUse)->multitoken(),
        "  Select the CP > 3.0 GPUs on which the reconstruction should be "
        "executed. Default: all devices > CP 3.0")(
        "tfolder", po::value<string>(&tfolder),
        "[folder] Use existing slice-to-volume transformations to initialize "
        "the reconstruction.")("sfolder", po::value<string>(&sfolder),
                               "[folder] Use existing registered slices and "
                               "replace loaded ones (have to be equally many "
                               "as loaded from stacks).")(
        "referenceVolume", po::value<string>(&referenceVolumeName),
        "Name for an optional reference volume. Will be used as inital "
        "reconstruction.")("T1PackageSize",
                           po::value<unsigned int>(&T1PackageSize),
                           "is a test if you can register T1 to T2 using NMI "
                           "and only one iteration")(
        "numDevicesToUse", po::value<unsigned int>(&numDevicesToUse),
        "sets how many GPU devices to use in case of automatic device "
        "selection. Default is as many as available.")(
        "useCPU", po::bool_switch(&useCPU)->default_value(false),
        "use CPU for reconstruction and registration; performs superresolution "
        "and robust statistics on CPU. Default is using the GPU")(
        "useCPUReg", po::bool_switch(&useCPUReg)->default_value(true),
        "use CPU for more flexible CPU registration; performs superresolution "
        "and robust statistics on GPU. [default, best result]")(
        "useGPUReg", po::bool_switch(&useGPUReg)->default_value(false),
        "use faster but less accurate and flexible GPU registration; performs "
        "superresolution and robust statistics on GPU.")(
        "useAutoTemplate",
        po::bool_switch(&useAutoTemplate)->default_value(false),
        "select 3D registration template stack automatically with matrix rank "
        "method.")("useSINCPSF",
                   po::bool_switch(&useSINCPSF)->default_value(false),
                   "use a more MRI like SINC point spread function (PSF) Will "
                   "be in plane sinc (Bartlett) and through plane Gaussian.")(
        "disableBiasCorrection",
        po::bool_switch(&disableBiasCorr)->default_value(false),
        "disable bias field correction for cases with little or no bias field "
        "inhomogenities (makes it faster but less reliable for stron intensity "
        "bias)")("numThreads", po::value<int>(&numThreads)->default_value(-1),
                 "Number of CPU threads to run for TBB");
    
    po::variables_map vm;

	try {
	    po::store(po::parse_command_line(argc, argv, desc), vm); // can throw

	    if (vm.count("help")) {
		std::cout << "Application to perform reconstruction of volumetric MRI "
		    "from thick slices." << std::endl << desc << std::endl;
		return EXIT_SUCCESS;
	    }

	    po::notify(vm);
	}
	catch (po::error &e) {
	    std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
	    std::cerr << desc << std::endl;
	    return EXIT_FAILURE;
	}
    }
    catch (std::exception &e) {
	std::cerr << "Unhandled exception while parsing arguments:  " << e.what()
		  << ", application will now exit" << std::endl;
	return EXIT_FAILURE;
    }
    
    if (useCPU) {
	// security measure for wrong input params
	useCPUReg = true;
	useGPUReg = false;
	
	// set CPU  threads
	if (numThreads > 0) {
	    cout << "PREV tbb_no_threads = " << tbb_no_threads << endl;
	    tbb_no_threads = numThreads;
	    cout << "NEW tbb_no_threads = " << tbb_no_threads << endl;
	} else {
	    cout << "Using task_scheduler_init::automatic number of threads" << endl;
	}
  }

    cout << "Reconstructed volume name ... " << outputName << endl;
    nStacks = inputStacks.size();
    cout << "Number of stacks ... " << nStacks << endl;

    float tmp_motionestimate = FLT_MAX;
    for (i = 0; i < nStacks; i++) {
	stack.Read(inputStacks[i].c_str());
	cout << "Reading stack ... " << inputStacks[i] << endl;
	stacks.push_back(stack);
    }

    for (i = 0; i < nStacks; i++) {
	irtkTransformation *transformation;
	if (!inputTransformations.empty()) {
	    try {
		transformation =
		    irtkTransformation::New((char *)(inputTransformations[i].c_str()));
	    }
	    catch (...) {
		transformation = new irtkRigidTransformation;
		if (templateNumber < 0)
		    templateNumber = 0;
	    }
	} else {
	    transformation = new irtkRigidTransformation;
	    if (templateNumber < 0)
		templateNumber = 0;
	}

	irtkRigidTransformation *rigidTransf =
	    dynamic_cast<irtkRigidTransformation *>(transformation);
	stack_transformations.push_back(*rigidTransf);
	delete rigidTransf;
    }

      // Create reconstruction object
    irtkReconstruction reconstruction(
	devicesToUse, useCPUReg,
	useCPU); // to emulate error for multi-threaded GPU
    
    if (useSINCPSF) {
	reconstruction.useSINCPSF();
    }

    reconstruction.InvertStackTransformations(stack_transformations);
    
    if (!maskName.empty()) {
	mask = new irtkRealImage((char *)(maskName.c_str()));
    }
    
    if (num_input_stacks_tuner > 0) {
	nStacks = num_input_stacks_tuner;
	cout << "actually used stacks for tuner test .... " << num_input_stacks_tuner << endl;
    }

    number_of_force_excluded_slices = force_excluded.size();
    
    // erase stacks for tuner evaluation
    if (num_input_stacks_tuner > 0) {
	stacks.erase(stacks.begin() + num_input_stacks_tuner, stacks.end());
	stack_transformations.erase(stack_transformations.begin() +
                                    num_input_stacks_tuner,
				    stack_transformations.end());
	std::cout << "stack sizes: " << nStacks << " " << stacks.size() << " "
		  << thickness.size() << " " << stack_transformations.size()
		  << std::endl;
    }

    // Initialise 2*slice thickness if not given by user
    if (thickness.size() == 0) {
	cout << "Slice thickness is ";
	for (i = 0; i < nStacks; i++) {
	    double dx, dy, dz;
	    stacks[i].GetPixelSize(&dx, &dy, &dz);
	    thickness.push_back(dz * 2);
	    cout << thickness[i] << " ";
	}
	cout << "." << endl;
    }

    // Output volume
    irtkRealImage reconstructed;
    irtkRealImage lastReconstructed;
    irtkRealImage reconstructedGPU;

    std::vector<double> samplingUcert;

    // Set debug mode
    if (debug)
	reconstruction.DebugOn();
    else
	reconstruction.DebugOff();

    // Set force excluded slices
    reconstruction.SetForceExcludedSlices(force_excluded);

    // Set low intensity cutoff for bias estimation
    reconstruction.SetLowIntensityCutoff(low_intensity_cutoff);

    // Check whether the template stack can be indentified
    if (templateNumber < 0) {
	cerr << "Please identify the template by assigning id transformation."
	     << endl;
	exit(1);
    }
    // If no mask was given  try to create mask from the template image in case it
    // was padded
    if ((mask == NULL) && (sfolder.empty())) {
	mask = new irtkRealImage(stacks[templateNumber]);
	*mask = reconstruction.CreateMask(*mask);
    }

    // copy to tmp stacks for template determination
    std::vector<irtkRealImage> tmpStacks;
    for (i = 0; i < stacks.size(); i++) {
	tmpStacks.push_back(stacks[i]);
    }

    // Before creating the template we will crop template stack according to the
    // given mask
    if (mask != NULL) {
	// first resample the mask to the space of the stack
	// for template stact the transformation is identity
	irtkRealImage m = *mask;

	// now do it really with best stack
	reconstruction.TransformMask(stacks[templateNumber], m,
				     stack_transformations[templateNumber]);
	// Crop template stack
	reconstruction.CropImage(stacks[templateNumber], m);

	if (debug) {
	    m.Write("maskTemplate.nii.gz");
	    stacks[templateNumber].Write("croppedTemplate.nii.gz");
	}
    }

    tmpStacks.erase(tmpStacks.begin(), tmpStacks.end());

    std::vector<uint3> stack_sizes;
    uint3 temp; // = (uint3) malloc(sizeof(uint3));
    for (int i = 0; i < stacks.size(); i++) {
	temp.x = stacks[i].GetX();
	temp.y = stacks[i].GetY();
	temp.z = stacks[i].GetZ();
	stack_sizes.push_back(temp);
    }

    // Create template volume with isotropic resolution
    // if resolution==0 it will be determined from in-plane resolution of the
    // image
    resolution =
	reconstruction.CreateTemplate(stacks[templateNumber], resolution);

    // Set mask to reconstruction object.
    reconstruction.SetMask(mask, smooth_mask);

    // to redirect output from screen to text files
    if (T1PackageSize == 0 && sfolder.empty()) {
	std::cout << "StackRegistrations start" << std::endl;
	// volumetric registration
	reconstruction.StackRegistrations(stacks, stack_transformations,
					  templateNumber);
    }

    cout << endl;

    std::cout << "reconstruction.CreateAverage" << std::endl;
    average = reconstruction.CreateAverage(stacks, stack_transformations);

    // Mask is transformed to the all other stacks and they are cropped
    for (i = 0; i < nStacks; i++) {
	// template stack has been cropped already
	if ((i == templateNumber))
	    continue;
	// transform the mask
	irtkRealImage m = reconstruction.GetMask();
	reconstruction.TransformMask(stacks[i], m, stack_transformations[i]);
	// Crop template stack
	reconstruction.CropImage(stacks[i], m);
    }

    if (T1PackageSize == 0 && sfolder.empty()) {
	// volumetric registration
	reconstruction.StackRegistrations(stacks, stack_transformations,
					  templateNumber);
	cout << endl;
    }

    // Rescale intensities of the stacks to have the same average
    if (intensity_matching)
	reconstruction.MatchStackIntensitiesWithMasking(
	    stacks, stack_transformations, averageValue);
    else
	reconstruction.MatchStackIntensitiesWithMasking(
	    stacks, stack_transformations, averageValue, true);
    average = reconstruction.CreateAverage(stacks, stack_transformations);

    // Create slices and slice-dependent transformations
    // resolution =
    // reconstruction.CreateTemplate(stacks[templateNumber],resolution);
    reconstruction.CreateSlicesAndTransformations(stacks, stack_transformations,
						  thickness);

    if (!sfolder.empty()) {
	// TODO replace slices for US experiment
	reconstruction.replaceSlices(sfolder);
    }

    // Mask all the slices
    reconstruction.MaskSlices();

    // Set sigma for the bias field smoothing
    if (sigma > 0)
	reconstruction.SetSigma(sigma);
    else {
	// cerr<<"Please set sigma larger than zero. Current value: "<<sigma<<endl;
	// exit(1);
	reconstruction.SetSigma(20);
    }

    // Set global bias correction flag
    if (global_bias_correction)
	reconstruction.GlobalBiasCorrectionOn();
    else
	reconstruction.GlobalBiasCorrectionOff();

    // if given read slice-to-volume registrations
    if (!tfolder.empty())
	reconstruction.ReadTransformation((char *)tfolder.c_str());


    // Initialise data structures for EM
    if (useCPU) {
	reconstruction.InitializeEM();
    } else {
	reconstruction.InitializeEMGPU();
    }

    std::cout << "*************** packages.size() " << packages.size() << std::endl;

/**************************************8START *****************/
// interleaved registration-reconstruction iterations
#ifdef EBB
    /*
  struct timeval ebbstart, ebbend;
  gettimeofday(&ebbstart, NULL);
  
  auto bindir =
      boost::filesystem::system_complete(argv[0]).parent_path() /
      "/../../ext/irtk-serialize/hosted/build/Release/bm/AppMain.elf32";

  static ebbrt::Runtime runtime;
  static ebbrt::Context c(runtime);
  ebbrt::ContextActivation activation(c);
  int numNodes = 1; // 4 seems to be max for vCPUs

  ebbrt::event_manager->Spawn([&reconstruction, bindir, numNodes,
                               iterations]() {
    EbbRTReconstruction::Create(&reconstruction, numNodes, iterations)
        .Then([bindir, numNodes](ebbrt::Future<EbbRTReconstructionEbbRef> f) {
          EbbRTReconstructionEbbRef ref = f.Get();

          std::cout << "EbbId: " << ref->getEbbId() << std::endl;

          for (int i = 0; i < numNodes; i++) {
            ebbrt::NodeAllocator::NodeDescriptor nd =
                ebbrt::node_allocator->AllocateNode(bindir.string(), NCPUS, 2, 8);

            nd.NetworkId().Then([ref](
                ebbrt::Future<ebbrt::Messenger::NetworkId> f) {
              ebbrt::Messenger::NetworkId nid = f.Get();
              std::cout << nid.ToString() << std::endl;
              ref->addNid(nid);
            });
          }

          // waiting for all nodes to be initialized
          ref->waitNodes().Then([ref](ebbrt::Future<void> f) {
            f.Get();
            std::cout << "all nodes initialized" << std::endl;
            ebbrt::event_manager->Spawn([ref]() { ref->runRecon(); });
          });
        });
  });
  std::cout << "######## WAITING FOR BM ############" << std::endl;

  c.Deactivate();
  c.Run();
  c.Reset();

  gettimeofday(&ebbend, NULL);
  std::printf("main ebbrt time: %lf seconds\n",
              (ebbend.tv_sec - ebbstart.tv_sec) +
                  ((ebbend.tv_usec - ebbstart.tv_usec) / 1000000.0));
  */
#else
    
  std::printf("lambda = %f delta = %f intensity_matching = %d useCPU = %d disableBiasCorr = %d sigma = %f global_bias_correction = %d lastIterLambda = %f iterations = %d\n", lambda, delta, intensity_matching, useCPU, disableBiasCorr, sigma, global_bias_correction, lastIterLambda, iterations);
  
  struct timeval tstart, tend;
  float sumCompute = 0.0;
  float tempTime = 0.0;
  gettimeofday(&tstart, NULL);

  struct timeval lstart, lend;

  iterations = 9;
  for (int iter = 0; iter < iterations; iter++) {
    // perform slice-to-volume registrations - skip the first iteration
    if (iter > 0) {
      gettimeofday(&lstart, NULL);
      reconstruction.SliceToVolumeRegistration();
      gettimeofday(&lend, NULL);
      tempTime = (lend.tv_sec - lstart.tv_sec) +
                 ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
      timers[SLICETOVOLUMEREGISTRATION] += tempTime;
      sumCompute += tempTime;
    }

    if (iter == (iterations - 1)) {
      reconstruction.SetSmoothingParameters(delta, lastIterLambda);
    } else {
      double l = lambda;
      for (i = 0; i < levels; i++) {
        if (iter == iterations * (levels - i - 1) / levels) {
          reconstruction.SetSmoothingParameters(delta, l);
        }
        l *= 2;
      }
    }

    // Use faster reconstruction during iterations and slower for final
    // reconstruction
    if (iter < (iterations - 1)) {
      reconstruction.SpeedupOn();
    } else {
      reconstruction.SpeedupOff();
    }

    // Initialise values of weights, scales and bias fields
    gettimeofday(&lstart, NULL);
    reconstruction.InitializeEMValues();
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[INITIALIZEEMVALUES] += tempTime;
    sumCompute += tempTime;

    // Calculate matrix of transformation between voxels of slices and volume
    gettimeofday(&lstart, NULL);
    reconstruction.CoeffInit();
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[COEFFINIT] += tempTime;
    sumCompute += tempTime;

    // Initialize reconstructed image with Gaussian weighted reconstruction
    gettimeofday(&lstart, NULL);
    reconstruction.GaussianReconstruction();
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[GAUSSIANRECONSTRUCTION] += tempTime;
    sumCompute += tempTime;

    // Simulate slices (needs to be done after Gaussian reconstruction)
    gettimeofday(&lstart, NULL);
    reconstruction.SimulateSlices();
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[SIMULATESLICES] += tempTime;
    sumCompute += tempTime;

    gettimeofday(&lstart, NULL);
    reconstruction.InitializeRobustStatistics();
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[INITIALIZEROBUSTSTATISTICS] += tempTime;
    sumCompute += tempTime;

    gettimeofday(&lstart, NULL);
    reconstruction.EStep();
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
        if (useCPU) {
          if (!disableBiasCorr) {
            if (sigma > 0) {
              gettimeofday(&lstart, NULL);
              reconstruction.Bias();
              gettimeofday(&lend, NULL);
              tempTime = (lend.tv_sec - lstart.tv_sec) +
                         ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
              std::printf("Bias: %lf seconds\n", tempTime);
              sumCompute += tempTime;
            }
          }
          gettimeofday(&lstart, NULL);
          // calculate scales
          reconstruction.Scale();
          gettimeofday(&lend, NULL);
          tempTime = (lend.tv_sec - lstart.tv_sec) +
                     ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
	  timers[SCALE] += tempTime;
          sumCompute += tempTime;
        }
      }

      // MStep and update reconstructed volume
      gettimeofday(&lstart, NULL);
      reconstruction.Superresolution(i + 1);
      gettimeofday(&lend, NULL);
      tempTime = (lend.tv_sec - lstart.tv_sec) +
                 ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
      timers[SUPERRESOLUTION] += tempTime;
      sumCompute += tempTime;
	    
      if (intensity_matching) {
        if (!disableBiasCorr) {
          if ((sigma > 0) && (!global_bias_correction)) {
            gettimeofday(&lstart, NULL);
            reconstruction.NormaliseBias(i);
            gettimeofday(&lend, NULL);
            tempTime = (lend.tv_sec - lstart.tv_sec) +
                       ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
            std::printf("NormaliseBias: %lf seconds\n", tempTime);
            sumCompute += tempTime;
          }
        }
      }

      // Simulate slices (needs to be done
      // after the update of the reconstructed volume)
      gettimeofday(&lstart, NULL);
      reconstruction.SimulateSlices();
      gettimeofday(&lend, NULL);
      tempTime = (lend.tv_sec - lstart.tv_sec) +
                 ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
      timers[SIMULATESLICES] += tempTime;
      sumCompute += tempTime;
            
      gettimeofday(&lstart, NULL);
      reconstruction.MStep(i + 1);
      gettimeofday(&lend, NULL);
      tempTime = (lend.tv_sec - lstart.tv_sec) +
                 ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
      timers[MSTEP] += tempTime;
      sumCompute += tempTime;

      gettimeofday(&lstart, NULL);
      reconstruction.EStep();
      gettimeofday(&lend, NULL);
      tempTime = (lend.tv_sec - lstart.tv_sec) +
                 ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
      timers[ESTEP] += tempTime;
      sumCompute += tempTime;

    } // end of reconstruction iterations

    // Mask reconstructed image to ROI given by the mask
    gettimeofday(&lstart, NULL);
    reconstruction.MaskVolume();
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[MASKVOLUME] += tempTime;
    sumCompute += tempTime;

    gettimeofday(&lstart, NULL);
    reconstruction.Evaluate(iter);
    gettimeofday(&lend, NULL);
    tempTime = (lend.tv_sec - lstart.tv_sec) +
               ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
    timers[EVALUATE] += tempTime;
    sumCompute += tempTime;
  } // end of interleaved registration-reconstruction iterations
  
  gettimeofday(&lstart, NULL);
  reconstruction.RestoreSliceIntensities();
  reconstruction.ScaleVolume();
  gettimeofday(&lend, NULL);
  tempTime = (lend.tv_sec - lstart.tv_sec) +
             ((lend.tv_usec - lstart.tv_usec) / 1000000.0);
  timers[RESTORESLICE] += tempTime;
  sumCompute += tempTime;

  gettimeofday(&tend, NULL);

  std::printf("SliceToVolumeRegistration: %lf seconds\n",
              timers[SLICETOVOLUMEREGISTRATION]);
  std::printf("InitializeEMValues: %lf seconds\n",
	      timers[INITIALIZEEMVALUES]);
  std::printf("CoeffInit: %lf seconds\n", timers[COEFFINIT]);
  std::printf("GaussianReconstruction: %lf seconds\n",
	      timers[GAUSSIANRECONSTRUCTION]);
  std::printf("SimulateSlices: %lf seconds\n", timers[SIMULATESLICES]);
  std::printf("InitializeRobustStatistics: %lf seconds\n",
	      timers[INITIALIZEROBUSTSTATISTICS]);
  std::printf("EStep: %lf seconds\n", timers[ESTEP]);
  
  std::printf("Scale: %lf seconds\n", timers[SCALE]);
  std::printf("Superresolution: %lf seconds\n", timers[SUPERRESOLUTION]);
  std::printf("MStep: %lf seconds\n", timers[MSTEP]);
  std::printf("MaskVolume: %lf seconds\n", timers[MASKVOLUME]);
  std::printf("Evaluate: %lf seconds\n", timers[EVALUATE]);
  std::printf("RestoreSliceIntensities and ScaleVolume: %lf seconds\n",
              timers[RESTORESLICE]);

  std::printf("compute time: %lf seconds\n",
              (tend.tv_sec - tstart.tv_sec) +
                  ((tend.tv_usec - tstart.tv_usec) / 1000000.0));
  std::printf("sum compute time: %lf seconds\n", sumCompute);

  std::printf("checksum _reconstructed = %lf\n",
              sumOneImage(reconstruction._reconstructed));

  // save final result
  reconstructed = reconstruction.GetReconstructed();
  reconstructed.Write(outputName.c_str());

  gettimeofday(&totend, NULL);
  std::printf("total time: %lf seconds\n",
              (totend.tv_sec - totstart.tv_sec) +
                  ((totend.tv_usec - totstart.tv_usec) / 1000000.0));
  
#endif
  /************************8 END ********************************/
  std::printf("checksum _reconstructed = %lf\n",
              sumOneImage(reconstruction._reconstructed));

  // save final result
  reconstructed = reconstruction.GetReconstructed();
  reconstructed.Write(outputName.c_str());

  gettimeofday(&totend, NULL);
  std::printf("total time: %lf seconds\n",
              (totend.tv_sec - totstart.tv_sec) +
	      ((totend.tv_usec - totstart.tv_usec) / 1000000.0));

  // The end of main()
}
