//          Copyright Boston University SESA Group 2013 - 2015.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#include "MultiEbb.h"

#include <ebbrt/EbbRef.h>
#include <ebbrt/IOBuf.h>
#include <ebbrt/LocalIdMap.h>
#include <ebbrt/Message.h>
#include <ebbrt/SharedEbb.h>
#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/Clock.h>
#include <ebbrt/SpinBarrier.h>

#ifdef __EBBRT_BM__
#include <ebbrt/SpinLock.h>
#endif

// This is *IMPORTANT*, it allows the messenger to resolve remote HandleFaults
EBBRT_PUBLISH_TYPE(, MultiEbb);

using namespace ebbrt;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wreturn-type"

#ifdef __EBBRT_BM__
#define PRINTF ebbrt::kprintf
#define FORPRINTF ebbrt::force_kprintf
static size_t indexToCPU(size_t i) { return i; }
ebbrt::SpinLock spinlock;
#else
#define PRINTF std::printf
#define FORPRINTF std::printf
#endif

struct membuf : std::streambuf {
  membuf(char *begin, char *end) { this->setg(begin, begin, end); }
};

void MultiEbb::Print(ebbrt::Messenger::NetworkId nid, const char *str) {
  auto len = strlen(str) + 1;
  auto buf = ebbrt::MakeUniqueIOBuf(len);
  snprintf(reinterpret_cast<char *>(buf->MutData()), len, "%s", str);

#ifndef __EBBRT_BM__
  std::cout << "MultiEbb length of sent iobuf: "
            << buf->ComputeChainDataLength() << " bytes" << std::endl;
#else
  ebbrt::kprintf("MultiEbb length of sent iobuf: %ld bytes \n",
                 buf->ComputeChainDataLength());
#endif

  SendMessage(nid, std::move(buf));
}

// We don't store anything in the GlobalIdMap, just return the EbbRef
EbbRef<MultiEbb> MultiEbb::Create(EbbId id) { return EbbRef<MultiEbb>(id); }

// This Ebb is implemented with one representative per machine
MultiEbb &MultiEbb::HandleFault(EbbId id) {
  {
    // First we check if the representative is in the LocalIdMap (using a
    // read-lock)
    LocalIdMap::ConstAccessor accessor;
    auto found = local_id_map->Find(accessor, id);
    if (found) {
      auto &rep = *boost::any_cast<MultiEbb *>(accessor->second);
      EbbRef<MultiEbb>::CacheRef(id, rep);
      return rep;
    }
  }

  MultiEbb *rep;
  {
    // Try to insert an entry into the LocalIdMap while holding an exclusive
    // (write) lock
    LocalIdMap::Accessor accessor;
    auto created = local_id_map->Insert(accessor, id);
    if (unlikely(!created)) {
      // We raced with another writer, use the rep it created and return
      rep = boost::any_cast<MultiEbb *>(accessor->second);
    } else {
      // Create a new rep and insert it into the LocalIdMap
      rep = new MultiEbb(id);
      accessor->second = rep;
    }
  }
  // Cache the reference to the rep in the local translation table
  EbbRef<MultiEbb>::CacheRef(id, *rep);
  return *rep;
}

MultiEbb::MultiEbb(EbbId ebbid) : Messagable<MultiEbb>(ebbid) {
    sum = 0;
    recvNodes = 0;
    recvNodes2 = 0;
    tempVal = 0;
    nids.clear();
}

void MultiEbb::Send() {
  // get the event manager context and save it
  ebbrt::EventManager::EventContext context;

  int start, end, factor;
  int vecSize = 100;
  std::vector<int> v;

  for (int j = 0; j < vecSize; j++) {
      v.push_back(1);
  }

  for (int i = 0; i < (int)nids.size(); i++) {
    // serialization
      std::ostringstream ofs;
    boost::archive::text_oarchive oa(ofs);

    factor = (int)ceil(vecSize / (float)numNodes);
    start = i * factor;
    end = i * factor + factor;
    end = ((int)end > vecSize) ? vecSize : end;

    oa &vecSize &start & end &v;
       
    std::string ts = "A " + ofs.str();
    
    std::cout << "Sending to " << nids[i].ToString() << " size: " << ts.length()
              << std::endl;

    Print(nids[i], ts.c_str());
  }

  std::cout << "Saving context " << std::endl;

  emec = &context;
  ebbrt::event_manager->SaveContext(*emec);
  std::cout << "Received back " << std::endl;
  mypromise.SetValue();
}

void MultiEbb::ReceiveMessage(Messenger::NetworkId nid,
                              std::unique_ptr<IOBuf> &&buffer) {
#ifndef __EBBRT_BM__
  auto output = std::string(reinterpret_cast<const char *>(buffer->Data()));
  std::cout << "Received ip: " << nid.ToString() << std::endl;

  if (output[0] == 'A') {
    ebbrt::IOBuf::DataPointer dp = buffer->GetDataPointer();
    char *t = (char *)(dp.Get(buffer->ComputeChainDataLength()));
    membuf sb{t + 2, t + buffer->ComputeChainDataLength()};
    std::istream stream{&sb};
    boost::archive::text_iarchive ia(stream);

    std::cout << "Parsing it back, received: "
              << buffer->ComputeChainDataLength() << " bytes" << std::endl;

    int temp;
    ia & temp;

    sum += temp;
    recvNodes ++;
    
    if(recvNodes == numNodes) {
	PRINTF("sum = %d\n", sum);
	ebbrt::event_manager->ActivateContext(std::move(*emec));
    }
  } 
  else if(output[0] == 'B'){

      recvNodes2 ++;
    
      if(recvNodes2 == numNodes) {
	  std::ostringstream ofs;
	  boost::archive::text_oarchive oa(ofs);
	  int val = 3;
	  oa & val;
	  std::string ts = "B " + ofs.str();
      
	  for (int i = 0; i < (int)nids.size(); i++) {
	      Print(nids[i], ts.c_str());
	  }
      }
  }
  else {
    PRINTF("Did not receive A\n");
  }
#else

  auto output = std::string(reinterpret_cast<const char *>(buffer->Data()),
                            buffer->Length());

  if (output[0] == 'A') {
      ebbrt::kprintf("Received msg length: %d bytes\n", buffer->Length());
      ebbrt::kprintf("Number chain elements: %d\n", buffer->CountChainElements());
      ebbrt::kprintf("Computed chain length: %d bytes\n",
		     buffer->ComputeChainDataLength());

      /****** copy sub buffers into one buffer *****/
	ebbrt::IOBuf::DataPointer dp = buffer->GetDataPointer();
	char *t = (char *)(dp.Get(buffer->ComputeChainDataLength()));
	membuf sb{ t + 2, t + buffer->ComputeChainDataLength() };
	std::istream stream{ &sb };
	/********* ******************************/

	ebbrt::kprintf("Begin deserialization...\n");
	boost::archive::text_iarchive ia(stream);

	int vecSize, sum, start, end;
	std::vector<int> v;
	ia & vecSize & start & end & v;

	sum = 0;
	for(int i = start; i < end; i ++)
	{
	    sum += v[i];
	}
	
	PRINTF("sum = %d\n", sum);

	// get the event manager context and save it
	PRINTF("requesting for data from host\n");
	ebbrt::EventManager::EventContext context2;
	std::string ts = "B";
	Print(nid, ts.c_str());
	emec2 = &context2;
	ebbrt::event_manager->SaveContext(*emec2);
	PRINTF("got request from host, tempVal = %d\n", tempVal);

	sum += tempVal;
	std::ostringstream ofs;
	boost::archive::text_oarchive oa(ofs);
	oa & sum;
	
	ts = "A " + ofs.str();
	
	Print(nid, ts.c_str());

  } 
  else if (output[0] == 'B')
  {
      /****** copy sub buffers into one buffer *****/
	ebbrt::IOBuf::DataPointer dp = buffer->GetDataPointer();
	char *t = (char *)(dp.Get(buffer->ComputeChainDataLength()));
	membuf sb{ t + 2, t + buffer->ComputeChainDataLength() };
	std::istream stream{ &sb };
	/********* ******************************/
	PRINTF("received B\n");
	boost::archive::text_iarchive ia(stream);
	ia & tempVal;
	ebbrt::event_manager->ActivateContext(std::move(*emec2));
  }
  else {
      PRINTF("Unknown command\n");
  }

#endif
}

void MultiEbb::setNumNodes(int i) { numNodes = i; }

void MultiEbb::setNumThreads(int i) { numThreads = i; }

ebbrt::Future<void> MultiEbb::waitNodes() {
  return std::move(nodesinit.GetFuture());
}

ebbrt::Future<void> MultiEbb::waitReceive() {
  return std::move(mypromise.GetFuture());
}

void MultiEbb::addNid(ebbrt::Messenger::NetworkId nid) {
  nids.push_back(nid);
  if ((int)nids.size() == numNodes) {
    nodesinit.SetValue();
  }
}

#pragma GCC diagnostic pop
