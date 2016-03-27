MYDIR := $(dir $(lastword $(MAKEFILE_LIST)))

app_sources := \
	RandPing.cc \
	randping.cc \
	pcg_basic.cc

target := randping

EBBRT_APP_VPATH := $(abspath $(MYDIR)../../src)

include $(abspath $(EBBRT_SRCDIR)/apps/ebbrthosted.mk)
