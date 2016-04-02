MYDIR := $(dir $(lastword $(MAKEFILE_LIST)))

FETAL_RECON_DIR := $(abspath $(MYDIR)/../../../../fetalReconstruction)

app_sources := AppMain.cc $(FETAL_RECON_DIR)/source/reconstructionGPU2/irtkReconstructionGPU.cc

target := AppMain

EBBRT_APP_LINK := -L $(MYDIR)lib $(MYDIR)lib/libzlib.so.1.2.7 -lgeometry++ -lcontrib++ -lcommon++ -limage++ -ltransformation++ -lregistration++ -lniftiio -lznz -lm -lboost_system -lboost_serialization -lboost_program_options -lgsl -lgslcblas -L /usr/lib/x86_64-linux-gnu/ -lz

include $(abspath ../../..//ebbrthosted.mk)
include $(abspath ../../../irtk.mk)

-include $(shell find -name '*.d')
