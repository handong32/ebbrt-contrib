MYDIR := $(dir $(lastword $(MAKEFILE_LIST)))

EBBRT_TARGET := randping
EBBRT_APP_OBJECTS := RandPing.o pcg_basic.o
EBBRT_APP_VPATH := $(abspath $(MYDIR)../src):$(abspath $(MYDIR)../../src)
EBBRT_CONFIG := $(abspath $(MYDIR)../src/ebbrtcfg.h)

include $(abspath $(EBBRT_SRCDIR)/apps/ebbrtbaremetal.mk)

-include $(shell find -name '*.d')
