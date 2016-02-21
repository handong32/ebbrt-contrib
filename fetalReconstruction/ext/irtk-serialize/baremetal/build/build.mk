MYDIR := $(dir $(lastword $(MAKEFILE_LIST)))

FETAL_RECON_DIR := $(abspath $(MYDIR)/../../../../)

EBBRT_APP_OBJECTS := AppMain.o EbbRTReconstruction.o irtkMatrix.o irtkGenericImage.o irtkImageRigidRegistrationWithPadding.o irtkObject.o irtkHomogeneousTransformation.o irtkTransformation.o irtkTemporalHomogeneousTransformation.o irtkHistogram_2D.o irtkImageRegistrationWithPadding.o irtkImageRegistration.o irtkPointSet.o irtkGenericHistogramSimilarityMetric.o irtkVector.o irtkBaseImage.o irtkFileToImage.o irtkImageToFile.o irtkUtil.o irtkCifstream.o irtkCofstream.o irtkRigidTransformation.o irtkAffineTransformation.o basename.o irtkTemporalRigidTransformation.o irtkTemporalAffineTransformation.o irtkHistogram_1D.o irtkGaussianBlurringWithPadding.o irtkResamplingWithPadding.o irtkInterpolateImageFunction.o irtkGradientDescentConstrainedOptimizer.o irtkOptimizer.o irtkDownhillDescentOptimizer.o irtkGradientDescentOptimizer.o irtkSteepestGradientDescentOptimizer.o irtkConjugateGradientDescentOptimizer.o irtkImageToImage.o irtkResampling.o irtkGaussianBlurring.o irtkFileGIPLToImage.o irtkFileANALYZEToImage.o irtkFileVTKToImage.o irtkFilePGMToImage.o irtkImageToFileANALYZE.o irtkImageToFileGIPL.o irtkImageToFileVTK.o irtkImageToFilePGM.o irtkScalarGaussian.o irtkScalarFunctionToImage.o irtkConvolutionWithPadding_1D.o irtkConvolution_1D.o irtkImageFunction.o irtkNearestNeighborInterpolateImageFunction2D.o irtkLinearInterpolateImageFunction2D.o irtkCSplineInterpolateImageFunction2D.o irtkBSplineInterpolateImageFunction2D.o irtkSincInterpolateImageFunction2D.o irtkGaussianInterpolateImageFunction2D.o irtkNearestNeighborInterpolateImageFunction.o irtkLinearInterpolateImageFunction.o irtkCSplineInterpolateImageFunction.o irtkBSplineInterpolateImageFunction.o irtkShapeBasedInterpolateImageFunction.o irtkSincInterpolateImageFunction.o irtkGaussianInterpolateImageFunction.o irtkScalarFunction.o irtkConvolution.o swap.o

#EbbRTStackRegistrations.o

EBBRT_TARGET := AppMain
EBBRT_APP_VPATH := $(abspath $(MYDIR)../src):$(FETAL_RECON_DIR)/source/IRTKSimple2/geometry++/src:$(FETAL_RECON_DIR)/source/IRTKSimple2/image++/src:$(FETAL_RECON_DIR)/source/IRTKSimple2/packages/registration/src:$(FETAL_RECON_DIR)/source/IRTKSimple2/common++/src:$(FETAL_RECON_DIR)/source/IRTKSimple2/packages/transformation/src

EBBRT_CONFIG := $(abspath $(MYDIR)../src/ebbrtcfg.h)

###EBBRT_APP_INCLUDES += /home/jappavoo/Work/EbbRT-ext-libs/include
EBBRT_APP_INCLUDES := -I $(abspath $(MYDIR)../ext)

EBBRT_APP_LINK := -L $(MYDIR)lib -lgsl -lgslcblas -lboost_system-gcc-1_54 -lboost_serialization-gcc-1_54

# FIXME:  For the moment we are using EBBRT_OPTFLAGS probably want to look at an APP secific variable
#EBBRT_OPTFLAGS = -Wno-unused-local-typedefs -O2

EBBRT_APP_CPPFLAGS = -Wno-unused-local-typedefs -O2

include $(abspath ../../../irtk.mk)

include $(abspath $(EBBRT_SRCDIR)/apps/ebbrtbaremetal.mk)

-include $(shell find -name '*.d')
