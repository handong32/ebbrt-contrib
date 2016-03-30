//          Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <signal.h>

#include <boost/filesystem.hpp>

#include <ebbrt/Context.h>
#include <ebbrt/ContextActivation.h>
#include <ebbrt/GlobalIdMap.h>
#include <ebbrt/StaticIds.h>
#include <ebbrt/NodeAllocator.h>
#include <ebbrt/Runtime.h>

#include "../../src/SleepPing.h"

using namespace ebbrt;

int main(int argc, char **argv) {
  auto bindir = boost::filesystem::system_complete(argv[0]).parent_path() /
                "/bm/sleepping.elf32";

  Runtime runtime;
  Context c(runtime);
  ContextActivation activation(c);

  auto ping_pong_ebb = SleepPing::Create();
  
  auto node_desc = node_allocator->AllocateNode(bindir.string());
  node_desc.NetworkId()
      //pass context c
      .Then([ping_pong_ebb, &c](Future<Messenger::NetworkId> f) {
        auto nid = f.Get();
        return ping_pong_ebb->Ping(nid);
      })
      .Then([&c](Future<void> f) {
        f.Get();
	//cleanly exits using context c
        c.io_service_.stop();
      });

  c.Run();

  printf("EBBRT ends\n");
  
  return 0;
}
