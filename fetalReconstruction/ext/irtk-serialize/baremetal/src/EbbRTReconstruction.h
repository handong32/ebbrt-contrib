//          Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#ifndef APPS_BAREMETAL_SRC_EBBRTRECONSTRUCTION_H_
#define APPS_BAREMETAL_SRC_EBBRTRECONSTRUCTION_H_

#include <string>
#include <string.h>

#include <ebbrt/Message.h>
#include <ebbrt/Clock.h>
#include <time.h>

#include <irtkReconstructionGPU.h>
#include <irtkResampling.h>
#include <irtkRegistration.h>
#include <irtkImageRigidRegistration.h>
#include <irtkImageRigidRegistrationWithPadding.h>
#include <irtkImageFunction.h>
#include <irtkTransformation.h>

using namespace ebbrt;

class EbbRTReconstruction : public ebbrt::Messagable<EbbRTReconstruction> {
    EbbId ebbid;

 public:
    explicit EbbRTReconstruction(ebbrt::Messenger::NetworkId nid, EbbId id)
    : Messagable<EbbRTReconstruction>(id), remote_nid_(std::move(nid)) 
    {
	ebbid = id;
    }

  static EbbRTReconstruction& HandleFault(ebbrt::EbbId id);

  void doNothing();
  void Print(const char* string);
  void ReceiveMessage(ebbrt::Messenger::NetworkId nid,
                      std::unique_ptr<ebbrt::IOBuf>&& buffer);

  typedef ebbrt::clock::HighResTimer MyTimer;
  static inline void ns_start(MyTimer &t) {t.tick();}
  static inline uint64_t ns_stop(MyTimer &t) { return t.tock().count(); }
 private:
  ebbrt::Messenger::NetworkId remote_nid_;
  MyTimer tmr;
  time_t tmr_start;
};

#endif
