//          Copyright Boston University SESA Group 2013 - 2016.
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

#include "Printer.h"

int main(int argc, char** argv) {
  auto bindir = boost::filesystem::system_complete(argv[0]).parent_path() /
                "/bm/hoardthreadtest.elf32";

  ebbrt::Runtime runtime;
  ebbrt::Context c(runtime);
  
  boost::asio::signal_set sig(c.io_service_, SIGINT);
  {
    ebbrt::ContextActivation activation(c);

    // ensure clean quit on ctrl-c
    sig.async_wait([&c](const boost::system::error_code& ec,
                        int signal_number) { c.io_service_.stop(); });
    Printer::Init().Then([bindir](ebbrt::Future<void> f) {
      f.Get();
      ebbrt::node_allocator->AppendArgs("niterations=50");
      ebbrt::node_allocator->AppendArgs("nobjects=30000");
      ebbrt::node_allocator->AppendArgs("nthreads=2");
      ebbrt::node_allocator->AppendArgs("work=0");
      ebbrt::node_allocator->AppendArgs("objSize=1");
      ebbrt::node_allocator->AllocateNode(bindir.string());
    });
  }
  c.Run();

  /*ebbrt::ContextActivation activation(c);

  Printer::Init().Then([bindir](ebbrt::Future<void> f) {
    f.Get();
    ebbrt::node_allocator->AppendArgs("abc=1");
    ebbrt::node_allocator->AllocateNode(bindir.string());
  });

  Printer::Wait().Then([&c](ebbrt::Future<void> f) {
    f.Get();
    c.io_service_.stop();
  });

  c.Run();*/

  return 0;
}
