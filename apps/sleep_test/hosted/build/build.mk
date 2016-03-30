MYDIR := $(dir $(lastword $(MAKEFILE_LIST)))

app_sources := \
	SleepPing.cc \
	sleepping.cc

target := sleepping

EBBRT_APP_VPATH := $(abspath $(MYDIR)../../src)

include $(abspath $(EBBRT_SRCDIR)/apps/ebbrthosted.mk)
