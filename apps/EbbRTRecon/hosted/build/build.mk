MYDIR := $(dir $(lastword $(MAKEFILE_LIST)))

app_sources := irtkReconstructionEbb.cc reconstruction.cc

target := reconstruction

EBBRT_APP_LINK := -L $(MYDIR)lib $(MYDIR)lib/libzlib.so.1.2.7 -lgeometry++ -lcontrib++ -lcommon++ -limage++ -ltransformation++ -lregistration++ -lniftiio -lznz -lm -lboost_system -lboost_serialization -lboost_program_options -lgsl -lgslcblas -L /usr/lib/x86_64-linux-gnu/ -lz

EBBRT_APP_VPATH := $(abspath $(MYDIR)../../src)

#CPPFLAGS='-D__EBB__ -D__TEST__ -D__ITERS__=2'

include $(abspath ../../../ebbrthosted.mk)
include $(abspath ../../../irtk.mk)
