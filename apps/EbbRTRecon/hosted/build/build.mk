MYDIR := $(dir $(lastword $(MAKEFILE_LIST)))

FETAL_RECON_DIR := $(abspath $(MYDIR)/../../../../fetalReconstruction)

app_sources := irtkReconstructionEbb.cc reconstruction.cc irtkMatrix.cc irtkGenericImage.cc irtkImageRigidRegistrationWithPadding.cc irtkObject.cc irtkHomogeneousTransformation.cc irtkTransformation.cc irtkTemporalHomogeneousTransformation.cc irtkHistogram_2D.cc irtkImageRegistrationWithPadding.cc irtkImageRegistration.cc irtkPointSet.cc irtkVector.cc irtkBaseImage.cc irtkFileToImage.cc irtkFileNIFTIToImage.cc irtkImageToFileNIFTI.cc irtkImageToFile.cc irtkUtil.cc irtkCifstream.cc irtkCofstream.cc irtkRigidTransformation.cc irtkAffineTransformation.cc basename.cc irtkTemporalRigidTransformation.cc irtkTemporalAffineTransformation.cc irtkHistogram_1D.cc irtkGaussianBlurringWithPadding.cc irtkResamplingWithPadding.cc irtkInterpolateImageFunction.cc irtkGradientDescentConstrainedOptimizer.cc irtkOptimizer.cc irtkDownhillDescentOptimizer.cc irtkGradientDescentOptimizer.cc irtkSteepestGradientDescentOptimizer.cc irtkConjugateGradientDescentOptimizer.cc irtkImageToImage.cc irtkResampling.cc irtkGaussianBlurring.cc irtkFileGIPLToImage.cc irtkFileANALYZEToImage.cc irtkFileVTKToImage.cc irtkFilePGMToImage.cc irtkImageToFileANALYZE.cc irtkImageToFileGIPL.cc irtkImageToFileVTK.cc irtkImageToFilePGM.cc irtkScalarGaussian.cc irtkScalarFunctionToImage.cc irtkConvolutionWithPadding_1D.cc irtkConvolution_1D.cc irtkImageFunction.cc irtkNearestNeighborInterpolateImageFunction2D.cc irtkLinearInterpolateImageFunction2D.cc irtkCSplineInterpolateImageFunction2D.cc irtkBSplineInterpolateImageFunction2D.cc irtkSincInterpolateImageFunction2D.cc irtkGaussianInterpolateImageFunction2D.cc irtkNearestNeighborInterpolateImageFunction.cc irtkLinearInterpolateImageFunction.cc irtkCSplineInterpolateImageFunction.cc irtkBSplineInterpolateImageFunction.cc irtkShapeBasedInterpolateImageFunction.cc irtkSincInterpolateImageFunction.cc irtkGaussianInterpolateImageFunction.cc irtkScalarFunction.cc irtkConvolution.cc swap.cc irtkImageTransformation.cc irtkImageHomogeneousTransformation.cc

target := reconstruction

EBBRT_APP_VPATH := $(abspath $(MYDIR)../../src):$(abspath $(MYDIR)../src):$(FETAL_RECON_DIR)/source/IRTKSimple2/geometry++/src:$(FETAL_RECON_DIR)/source/IRTKSimple2/image++/src:$(FETAL_RECON_DIR)/source/IRTKSimple2/packages/registration/src:$(FETAL_RECON_DIR)/source/IRTKSimple2/common++/src:$(FETAL_RECON_DIR)/source/IRTKSimple2/packages/transformation/src

EBBRT_APP_LINK := -L $(MYDIR)lib -lniftiio -lznz -lm -lboost_system -lboost_serialization -lboost_program_options -lgsl -lgslcblas -L /usr/lib/x86_64-linux-gnu/ -lz

include $(abspath ../../../ebbrthosted.mk)
include $(abspath ../../../irtk.mk)
