//          Copyright Boston University SESA Group 2013 - 2016.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#ifndef APPS_HOARD_BAREMETAL_SRC_PRINTER_H_
#define APPS_HOARD_BAREMETAL_SRC_PRINTER_H_

#include <string>

#include <ebbrt/Message.h>

#include "../../src/StaticEbbIds.h"

class Printer : public ebbrt::Messagable<Printer> {
 public:
  explicit Printer(ebbrt::Messenger::NetworkId nid);

  static Printer& HandleFault(ebbrt::EbbId id);

  void Print(const char* string);
  void ReceiveMessage(ebbrt::Messenger::NetworkId nid,
                      std::unique_ptr<ebbrt::IOBuf>&& buffer);
  void Run();

 private:
  ebbrt::Messenger::NetworkId remote_nid_;
};

constexpr auto printer = ebbrt::EbbRef<Printer>(kPrinterEbbId);

typedef ebbrt::clock::HighResTimer MyTimer;
static inline void ns_start(MyTimer &t) { t.tick(); }
static inline uint64_t ns_stop(MyTimer &t) { return t.tock().count(); }

#endif  // APPS_HOARD_BAREMETAL_SRC_PRINTER_H_
