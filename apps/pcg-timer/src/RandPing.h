//          Copyright Boston University SESA Group 2013 - 2015.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#ifndef APPS_RANDPING_H_
#define APPS_RANDPING_H_

#include <unordered_map>

#include <ebbrt/EbbAllocator.h>
#include <ebbrt/Future.h>
#include <ebbrt/Message.h>

/* The RandPing(PingPong) Ebb allows both hosted and native implementations to send "PING"
 * messages via the messenger to another node (machine). The remote node
 * responds back and the initiator fulfills its Promise. */

class RandPing : public ebbrt::Messagable<RandPing> {
 public:
  static ebbrt::EbbRef<RandPing>
  Create(ebbrt::EbbId id = ebbrt::ebb_allocator->Allocate());

  static RandPing& HandleFault(ebbrt::EbbId id);

  RandPing(ebbrt::EbbId ebbid);

  ebbrt::Future<void> Ping(ebbrt::Messenger::NetworkId nid);

  void ReceiveMessage(ebbrt::Messenger::NetworkId nid,
                      std::unique_ptr<ebbrt::IOBuf>&& buffer);

 private:
  std::mutex m_;
  std::unordered_map<uint32_t, ebbrt::Promise<void>> promise_map_;
  uint32_t id_{0};
};

#endif  // APPS_RANDPING_H_
