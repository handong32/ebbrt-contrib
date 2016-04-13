MYDIR := $(dir $(lastword $(MAKEFILE_LIST)))

FETAL_RECON_DIR := $(abspath $(MYDIR)/../../fetalReconstruction/)

CXXFLAGS += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/common++/include
CXXFLAGS += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/contrib++/include
CXXFLAGS += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/geometry++/include
CXXFLAGS += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/image++/include
CXXFLAGS += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/packages/registration/include
CXXFLAGS += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/packages/transformation/include
CXXFLAGS += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/nifti/niftilib
CXXFLAGS += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/nifti/znzlib

EBBRT_APP_INCLUDES += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/common++/include
EBBRT_APP_INCLUDES += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/contrib++/include
EBBRT_APP_INCLUDES += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/geometry++/include
EBBRT_APP_INCLUDES += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/image++/include
EBBRT_APP_INCLUDES += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/packages/registration/include
EBBRT_APP_INCLUDES += -I $(FETAL_RECON_DIR)/source/IRTKSimple2/packages/transformation/include
