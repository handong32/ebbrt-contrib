//          Copyright Boston University SESA Group 2013 - 2015.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#include "SleepPing.h"

#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/LocalIdMap.h>
#include <ebbrt/EbbRef.h>
#include <ebbrt/SharedEbb.h>
#include <ebbrt/Message.h>
#include <ebbrt/IOBuf.h>
#include <ebbrt/Clock.h>

#include <time.h>
#include <sys/time.h>
#include <unistd.h>

// This is *IMPORTANT*, it allows the messenger to resolve remote HandleFaults
EBBRT_PUBLISH_TYPE(, SleepPing);

using namespace ebbrt;

// We don't store anything in the GlobalIdMap, just return the EbbRef
EbbRef<SleepPing> SleepPing::Create(EbbId id) { return EbbRef<SleepPing>(id); }

// This Ebb is implemented with one representative per machine
SleepPing &SleepPing::HandleFault(EbbId id) {
  {
    // First we check if the representative is in the LocalIdMap (using a
    // read-lock)
    LocalIdMap::ConstAccessor accessor;
    auto found = local_id_map->Find(accessor, id);
    if (found) {
      auto &rep = *boost::any_cast<SleepPing *>(accessor->second);
      EbbRef<SleepPing>::CacheRef(id, rep);
      return rep;
    }
  }

  SleepPing *rep;
  {
    // Try to insert an entry into the LocalIdMap while holding an exclusive
    // (write) lock
    LocalIdMap::Accessor accessor;
    auto created = local_id_map->Insert(accessor, id);
    if (unlikely(!created)) {
      // We raced with another writer, use the rep it created and return
      rep = boost::any_cast<SleepPing *>(accessor->second);
    } else {
      // Create a new rep and insert it into the LocalIdMap
      rep = new SleepPing(id);
      accessor->second = rep;
    }
  }
  // Cache the reference to the rep in the local translation table
  EbbRef<SleepPing>::CacheRef(id, *rep);
  return *rep;
}

SleepPing::SleepPing(EbbId ebbid) : Messagable<SleepPing>(ebbid) {}

// We need to store promises for each Ping and therefore also need an identifier
// for each Ping
ebbrt::Future<void> SleepPing::Ping(Messenger::NetworkId nid) {
  uint32_t id;
  Promise<void> promise;
  auto ret = promise.GetFuture();
  {
    std::lock_guard<std::mutex> guard(m_);
    id = id_; // Get a new id (always even)
    id_ += 2;
    bool inserted;
    // insert our promise into the hash table
    std::tie(std::ignore, inserted) =
        promise_map_.emplace(id, std::move(promise));
    assert(inserted);
  }
  // Construct and send the ping message
  auto buf = MakeUniqueIOBuf(sizeof(uint32_t));
  auto dp = buf->GetMutDataPointer();
  dp.Get<uint32_t>() = id + 1; // Ping messages are odd
  SendMessage(nid, std::move(buf));
  return ret;
}

void SleepPing::ReceiveMessage(Messenger::NetworkId nid,
                              std::unique_ptr<IOBuf> &&iobuf) {
  auto dp = iobuf->GetDataPointer();
  auto id = dp.Get<uint32_t>();

  struct timeval start, end;
  
  // Ping messages use id + 1, so they are always odd
  if (id & 1) {

#ifdef __EBBRT_BM__
    ebbrt::kprintf("***************************\n");
    for(int i = 0; i < 10; i ++) {
	auto t1 = ebbrt::clock::Wall::Now();
	gettimeofday(&start, NULL); //gettimeofday start
	while ((ebbrt::clock::Wall::Now() - t1) < std::chrono::seconds(1)) {
	}
	gettimeofday(&end, NULL); //gettimeofday end
	ebbrt::kprintf("ebbrt sleep diff time: %lf seconds\n",
		       (end.tv_sec - start.tv_sec) +
		       ((end.tv_usec - start.tv_usec) / 1000000.0));
    }
    ebbrt::kprintf("***************************\n");
#endif
    // Received Ping
    auto buf = MakeUniqueIOBuf(sizeof(uint32_t));
    auto dp = buf->GetMutDataPointer();
    dp.Get<uint32_t>() = id - 1; // Send back with the original id
    SendMessage(nid, std::move(buf));
  } else {
      std::printf("***************************\n");
#ifndef __EBBRT_BM__
      for(int i = 0; i < 10; i ++) {
	  gettimeofday(&start, NULL); //gettimeofday start
	  sleep(1);
	  gettimeofday(&end, NULL); //gettimeofday end
	  std::printf("linux sleep diff time: %lf seconds\n",
		      (end.tv_sec - start.tv_sec) +
		      ((end.tv_usec - start.tv_usec) / 1000000.0));
      }
      std::printf("***************************\n");
#endif      
      // Received Pong, lookup in the hash table for our promise and set it
      std::lock_guard<std::mutex> guard(m_);
      auto it = promise_map_.find(id);
      assert(it != promise_map_.end());
      it->second.SetValue();
      promise_map_.erase(it);
  }
}
