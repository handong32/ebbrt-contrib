//          Copyright Boston University SESA Group 2013 - 2015.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#ifndef APPS_MULTIEBB_H_
#define APPS_MULTIEBB_H_

#include <vector>

#include <unordered_map>

#include <ebbrt/EbbAllocator.h>
#include <ebbrt/Future.h>
#include <ebbrt/Message.h>

#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

/* Thi Ebb allows both hosted and native implementations to send "PING"
 * messages via the messenger to another node (machine). The remote node
 * responds back and the initiator fulfills its Promise. */

class MultiEbb : public ebbrt::Messagable<MultiEbb> {
 public:
  static ebbrt::EbbRef<MultiEbb>
  Create(ebbrt::EbbId id = ebbrt::ebb_allocator->Allocate());

  static MultiEbb& HandleFault(ebbrt::EbbId id);

  MultiEbb(ebbrt::EbbId ebbid);


  void ReceiveMessage(ebbrt::Messenger::NetworkId nid,
                      std::unique_ptr<ebbrt::IOBuf>&& buffer);

  void setNumNodes(int i);
  void setNumThreads(int i);
  void addNid(ebbrt::Messenger::NetworkId nid);
  ebbrt::Future<void> waitNodes();
  ebbrt::Future<void> waitReceive();

  void Send();
  void Run();
  
  void Print(ebbrt::Messenger::NetworkId nid, const char* str);
  
  int sum;
  int recvNodes;
  int recvNodes2;
  int tempVal;
  
 private:
  std::mutex m_;
  std::unordered_map<uint32_t, ebbrt::Promise<void>> promise_map_;
  uint32_t id_{0};
  std::vector<ebbrt::Messenger::NetworkId> nids;
  ebbrt::Promise<void> nodesinit;
  ebbrt::Promise<void> mypromise;
  // this is used to save and load context
  ebbrt::EventManager::EventContext* emec{nullptr};
  ebbrt::EventManager::EventContext* emec2{nullptr};
  int numNodes;
  int numThreads;
  
};

#endif  // APPS_MultiEbb_H_
