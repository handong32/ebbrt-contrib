//          Copyright Boston University SESA Group 2013 - 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <signal.h>
#include <thread>
#include <chrono>

#include <boost/filesystem.hpp>

#include <ebbrt/Context.h>
#include <ebbrt/ContextActivation.h>
#include <ebbrt/GlobalIdMap.h>
#include <ebbrt/StaticIds.h>
#include <ebbrt/NodeAllocator.h>
#include <ebbrt/Runtime.h>

#include "EbbRTStackRegistrations.h"
#include "EbbRTSliceToVolumeRegistration.h"
#include "EbbRTCoeffInit.h"

void func1(char** argv) {
  auto bindir = boost::filesystem::system_complete(argv[0]).parent_path() /
                "/bm/AppMain.elf32";

  static ebbrt::Runtime runtime;
  static ebbrt::Context c(runtime);
  ebbrt::ContextActivation activation(c);

  irtkReconstruction* reconstructor;
  int numNodes = 2;

  ebbrt::event_manager->Spawn([&reconstructor, bindir, numNodes]() {
    EbbRTCoeffInit::Create(reconstructor, numNodes)
        .Then([bindir, numNodes](ebbrt::Future<EbbRTCoeffInitEbbRef> f) {
          EbbRTCoeffInitEbbRef ref = f.Get();

          std::cout << "#######################################EbbId: "
                    << ref->getEbbId() << std::endl;

          for (int i = 0; i < numNodes; i++) {
            std::cout << bindir.string() << std::endl;
            ebbrt::NodeAllocator::NodeDescriptor nd =
                ebbrt::node_allocator->AllocateNode(bindir.string(), 1, 1, 16);

            nd.NetworkId().Then([ref](
                ebbrt::Future<ebbrt::Messenger::NetworkId> f) {
              ebbrt::Messenger::NetworkId nid = f.Get();
              std::cout << nid.ToString() << std::endl;
              ref->addNid(nid);
            });
          }

          // waiting for all nodes to be initialized
          ref->waitNodes().Then([ref](ebbrt::Future<void> f) {
            f.Get();
            std::cout << "all nodes initialized" << std::endl;
            ebbrt::event_manager->Spawn([ref]() { ref->runJob(1000000); });
          });
        });
  });

  c.Deactivate();
  c.Run();
  c.Reset();

  std::cout << "Finished" << std::endl;
}

int main(int argc, char** argv) {

    func1(argv);
    func1(argv);
    func1(argv);
    func1(argv);
    func1(argv);
    func1(argv);
    
  // vector<irtkRealImage> stacks;
  // vector<irtkRigidTransformation> stack_transformation;
  // int templateNumber = 0;
  // irtkGreyImage target;
  // irtkRigidTransformation offset;

  /*irtkRealImage slice;
  irtkRealImage reconstructed;
  irtkRigidTransformation transformations;
  irtkRealImage mask;

  std::istringstream ifs;
  ifs.str(maskstr);
  boost::archive::text_iarchive ia(ifs);

  ia & mask;

  mask.Print();
  std::cout << ifs.str() <<  std::endl;
  std::cout << "test" << std::endl;*/

  // bool useExternalTarget = false;

  /*
   * event_manager -> Spawn(..) puts the code block on the event queue
   * to be executed after resolving the Future
   *
   */

  /*  ebbrt::event_manager
        ->Spawn([&reconstructor, &stacks, &stack_transformation, templateNumber,
                 &target, &offset, useExternalTarget, bindir]() {

          // EbbRTStackRegistrations::Create(..) returns a
          // Future<EbbRTStackRegistrationsEbbRef>
          // that gets accessed on the Then(..) call, f.Get() ensures the EbbRef
          // was created successfully, then we can allocate the baremetal node
     on
          // the backend.
          EbbRTStackRegistrations::Create(reconstructor, stacks,
                                          stack_transformation, templateNumber,
                                          target, offset, useExternalTarget)
              // Then(...) gets the EbbRef
              .Then([bindir](ebbrt::Future<EbbRTStackRegistrationsEbbRef> f) {
                // ensures it was created
                EbbRTStackRegistrationsEbbRef ref = f.Get();

                // allocated baremetal AppMain.elf32
                ebbrt::node_allocator->AllocateNode(bindir.string());

                // test code to get EbbId
                std::cout << "EbbId: " << ref->getEbbId() << std::endl;

                // std::this_thread::sleep_for(std::chrono::seconds(5));

                // waitReceive essentially blocks until the void promise
                // is fulfilled, in which it will then start the actual
                // job to send data to the backend node(s)
                ref->waitReceive().Then([ref](ebbrt::Future<void> f) {
                  std::cout << "Running job2 " << std::endl;
                  ebbrt::event_manager->Spawn([ref]() { ref->runJob2(); });
                });
              });
        });

  */

  return 0;
}
