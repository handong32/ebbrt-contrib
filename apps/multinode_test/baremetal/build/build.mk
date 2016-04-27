MYDIR := $(dir $(lastword $(MAKEFILE_LIST)))

EBBRT_TARGET := multiebb
EBBRT_APP_OBJECTS := MultiEbb.o
EBBRT_APP_VPATH := $(abspath $(MYDIR)../src):$(abspath $(MYDIR)../../src)
EBBRT_CONFIG := $(abspath $(MYDIR)../src/ebbrtcfg.h)

EBBRT_APP_INCLUDES := -I $(abspath $(MYDIR)../ext)
EBBRT_APP_LINK := -L $(MYDIR)lib -lboost_serialization -lboost_wserialization -lboost_system

include $(abspath $(EBBRT_SRCDIR)/apps/ebbrtbaremetal.mk)
