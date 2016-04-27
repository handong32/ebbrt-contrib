MYDIR := $(dir $(lastword $(MAKEFILE_LIST)))

app_sources := MultiEbb.cc multiebb.cc

target := multiebb

EBBRT_APP_VPATH := $(abspath $(MYDIR)../../src):$(abspath $(MYDIR)../src)

EBBRT_APP_LINK := -lm -lboost_system -lboost_serialization

include $(abspath ../../../ebbrthosted.mk)
