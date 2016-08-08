cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g3")

project("hoardthreadtest-ebbrt" CXX)

set(HOSTED_SOURCES 
      src/hosted/hoardthreadtest.cc 
      src/hosted/Printer.cc)

set(BAREMETAL_SOURCES 
      src/baremetal/hoardthreadtest.cc 
      src/baremetal/Printer.cc)

# Baremetal  ========================================================

if( ${CMAKE_SYSTEM_NAME} STREQUAL "EbbRT")

  message(STATUS "Building baremetal image...")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -Wall -Werror")
  add_executable(hoardthreadtest.elf ${BAREMETAL_SOURCES})
  add_custom_command(TARGET hoardthreadtest.elf POST_BUILD 
    COMMAND objcopy -O elf32-i386 hoardthreadtest.elf hoardthreadtest.elf32 )

# Hosted  ===========================================================

elseif( ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" )
  message(STATUS "Building Linux executable...")

  # TODO(jmcadden): remove source-dir requirement 
  if(DEFINED ENV{EBBRT_SRCDIR})
    list(APPEND CMAKE_MODULE_PATH 
      "$ENV{EBBRT_SRCDIR}/hosted/cmake/Modules")
  endif()

  find_package(EbbRT REQUIRED)
  find_package(Boost 1.53.0 REQUIRED COMPONENTS 
    filesystem system coroutine context )
  find_package(Capnp REQUIRED)
  find_package(TBB REQUIRED)
  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -Wall -Werror")
  include_directories(${EBBRT_INCLUDE_DIRS})
  add_executable(hoardthreadtest ${HOSTED_SOURCES})
  target_link_libraries(hoardthreadtest -Wl,--whole-archive ${EBBRT_LIBRARIES}
    -Wl,--no-whole-archive -lpthread
    ${CAPNP_LIBRARIES_LITE} ${CMAKE_THREAD_LIBS_INIT} 
    ${Boost_LIBRARIES} ${TBB_LIBRARIES} 
    )

else()
  message(FATAL_ERROR "System name unsupported: ${CMAKE_SYSTEM_NAME}")
endif()
