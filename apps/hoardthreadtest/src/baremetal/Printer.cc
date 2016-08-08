//          Copyright Boston University SESA Group 2013 - 2016.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#include "Printer.h"

#include <ebbrt/Clock.h>
#include <ebbrt/Cpu.h>
#include <ebbrt/EventManager.h>
#include <ebbrt/GlobalIdMap.h>
#include <ebbrt/LocalIdMap.h>
#include <ebbrt/SpinBarrier.h>
#include <ebbrt/UniqueIOBuf.h>
#include <ebbrt/Multiboot.h>

int niterations = 50;  // Default number of iterations.
int nobjects = 30000;  // Default number of objects.
int nthreads = 1;  // Default number of threads.
int work = 0;  // Default number of loop iterations.
int objSize = 1;

EBBRT_PUBLISH_TYPE(, Printer);

static size_t indexToCPU(size_t i) { return i; }

int ParseInt(std::string str, std::string mstr)
{
    auto test = str.find(mstr);
    if(test == std::string::npos) return -1;
    auto test2 = str.substr(test);
    
    auto test3 = test2.find(";");
    if(test3 == std::string::npos) return -1;
    auto test4 = test2.substr(0, test3);
    
    auto test5 = test4.find("=");
    if(test5 == std::string::npos) return -1;
    auto test6 = test4.substr(test5+1);
    //ebbrt::kprintf("%d\n", atoi(test6.c_str()));
    return atoi(test6.c_str());
}

class Foo {
 public:
  Foo(void) : x(14), y(29) {}
  int x;
  int y;
};

void worker(void) {
  int i, j;
  Foo** a;
  a = new Foo*[nobjects / nthreads];

  for (j = 0; j < niterations; j++) {

    for (i = 0; i < (nobjects / nthreads); i++) {
      a[i] = new Foo[objSize];
      for (volatile int d = 0; d < work; d++) {
        volatile int f = 1;
        f = f + f;
        f = f * f;
        f = f + f;
        f = f * f;
      }
      assert(a[i]);
    }

    for (i = 0; i < (nobjects / nthreads); i++) {
      delete[] a[i];
      for (volatile int d = 0; d < work; d++) {
        volatile int f = 1;
        f = f + f;
        f = f * f;
        f = f + f;
        f = f * f;
      }
    }
  }

  delete[] a;
}

Printer::Printer(ebbrt::Messenger::NetworkId nid)
    : Messagable<Printer>(kPrinterEbbId), remote_nid_(std::move(nid)) {}

Printer& Printer::HandleFault(ebbrt::EbbId id) {
  {
    ebbrt::LocalIdMap::ConstAccessor accessor;
    auto found = ebbrt::local_id_map->Find(accessor, id);
    if (found) {
      auto& pr = *boost::any_cast<Printer*>(accessor->second);
      ebbrt::EbbRef<Printer>::CacheRef(id, pr);
      return pr;
    }
  }

  ebbrt::EventManager::EventContext context;
  auto f = ebbrt::global_id_map->Get(id);
  Printer* p;
  f.Then([&f, &context, &p](ebbrt::Future<std::string> inner) {
    p = new Printer(ebbrt::Messenger::NetworkId(inner.Get()));
    ebbrt::event_manager->ActivateContext(std::move(context));
  });
  ebbrt::event_manager->SaveContext(context);
  auto inserted = ebbrt::local_id_map->Insert(std::make_pair(id, p));
  if (inserted) {
    ebbrt::EbbRef<Printer>::CacheRef(id, *p);
    return *p;
  }

  delete p;
  // retry reading
  ebbrt::LocalIdMap::ConstAccessor accessor;
  ebbrt::local_id_map->Find(accessor, id);
  auto& pr = *boost::any_cast<Printer*>(accessor->second);
  ebbrt::EbbRef<Printer>::CacheRef(id, pr);
  return pr;
}

void Printer::Run() {
    int ret = 0;
  auto len = 5;
  auto buf = ebbrt::MakeUniqueIOBuf(len);
  snprintf(reinterpret_cast<char*>(buf->MutData()), len, "%s", "Done");
  
  auto cmdline = std::string(ebbrt::multiboot::CmdLine());
  ret = ParseInt(cmdline, "niterations=");
  if(ret != -1) niterations = ret;
  
  ret = ParseInt(cmdline, "nobjects=");
  if(ret != -1) nobjects = ret;

  ret = ParseInt(cmdline, "nthreads=");
  if(ret != -1) nthreads = ret;
  
  ret = ParseInt(cmdline, "work=");
  if(ret != -1) work = ret;

  ret = ParseInt(cmdline, "objSize=");
  if(ret != -1) objSize = ret;
  
  uint64_t ns;
  MyTimer tmr;

  ebbrt::kprintf("Running threadtest for %d threads, %d iterations,"
                 " %d objects, %d work and %d objSize...\n",
                 nthreads, niterations, nobjects, work, objSize);

  ns_start(tmr);

  size_t ncpus = nthreads; //ebbrt::Cpu::Count();
  static ebbrt::SpinBarrier bar(ncpus);
  ebbrt::EventManager::EventContext context;
  std::atomic<size_t> count(0);
  size_t theCpu = ebbrt::Cpu::GetMine();

  for (size_t i = 0; i < ncpus; i++) {
    // spawn jobs on each core using SpawnRemote
    ebbrt::event_manager->SpawnRemote(
        [theCpu, ncpus, &count, &context, i]() {
          ebbrt::kprintf("running core %d of %d\n", (int)i, (int)ncpus);
          worker();
          count++;
          bar.Wait();
          while (count < (size_t)ncpus)
            ;
          if (ebbrt::Cpu::GetMine() == theCpu) {
            ebbrt::event_manager->ActivateContext(std::move(context));
          }
        },
        indexToCPU(i));
  }
  ebbrt::event_manager->SaveContext(context);
  
  ns = ns_stop(tmr);

  ebbrt::kprintf("hoardthreadtest time elapsed = %" PRIu64 " ~ns\n", ns);

  SendMessage(remote_nid_, std::move(buf));
}

void Printer::Print(const char* str) {
  auto len = strlen(str) + 1;
  auto buf = ebbrt::MakeUniqueIOBuf(len);
  snprintf(reinterpret_cast<char*>(buf->MutData()), len, "%s", str);
  SendMessage(remote_nid_, std::move(buf));
}

void Printer::ReceiveMessage(ebbrt::Messenger::NetworkId nid,
                             std::unique_ptr<ebbrt::IOBuf>&& buffer) {
  throw std::runtime_error("Printer: Received message unexpectedly!");
}
