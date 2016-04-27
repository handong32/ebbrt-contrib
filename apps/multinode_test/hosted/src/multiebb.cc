//          Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <signal.h>

#include <boost/filesystem.hpp>

#include <ebbrt/Context.h>
#include <ebbrt/ContextActivation.h>
#include <ebbrt/GlobalIdMap.h>
#include <ebbrt/NodeAllocator.h>
#include <ebbrt/Runtime.h>
#include <ebbrt/StaticIds.h>

#include "../../src/MultiEbb.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

using namespace ebbrt;

int main(int argc, char **argv) {
  auto bindir = boost::filesystem::system_complete(argv[0]).parent_path() /
                "/bm/multiebb.elf32";

  Runtime runtime;
  Context c(runtime);
  ContextActivation activation(c);

  auto mebb = MultiEbb::Create();
  int numNodes = 2;
  int numThreads = 1;
  mebb->setNumNodes(numNodes);
  mebb->setNumThreads(numThreads);

  /********************* ebbrt ******************/
  for (int i = 0; i < numNodes; i++) {
    auto node_desc =
        node_allocator->AllocateNode(bindir.string(), numThreads, 1, 2);

    node_desc.NetworkId().Then(
        [mebb, &c](Future<Messenger::NetworkId> f) {
          // pass context c
          auto nid = f.Get();
          mebb->addNid(nid);
        });
  }

  mebb->waitNodes().Then(
      [mebb](ebbrt::Future<void> f) {
        f.Get();
        std::cout << "all nodes initialized" << std::endl;
        ebbrt::event_manager->Spawn([mebb]() {
          mebb->Send();
        });
      });

  mebb->waitReceive().Then([&c](ebbrt::Future<void> f) {
    f.Get();
    c.io_service_.stop();
  });
  
  c.Run();

  printf("EBBRT ends\n");

  return 0;
}

#pragma GCC diagnostic pop
