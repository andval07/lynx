cmake_minimum_required(VERSION 3.20)
project(lynx CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8 /wd4127)
  add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
endif()

include(FetchContent)

# Eigen is header only: download it, do not run its CMake project.
FetchContent_Declare(eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  SOURCE_SUBDIR none)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen googletest)

add_library(eigen INTERFACE)
target_include_directories(eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})

# Geometry and optimization, Eigen only.
add_library(lynx_core
  src/se3.cpp
  src/geometry.cpp
  src/epnp.cpp
  src/pnp.cpp
  src/ba.cpp
  src/trajectory.cpp
  src/tum.cpp)
target_include_directories(lynx_core PUBLIC include)
target_link_libraries(lynx_core PUBLIC eigen)

# Front-end and odometry, needs OpenCV for ORB and image IO.
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/third_party/opencv/build CACHE PATH "")
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d)

add_library(lynx_vision
  src/frontend.cpp
  src/odometry.cpp)
target_link_libraries(lynx_vision PUBLIC lynx_core ${OpenCV_LIBS})
target_include_directories(lynx_vision SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})

add_executable(lynx_vo apps/lynx_vo.cpp)
target_link_libraries(lynx_vo PRIVATE lynx_vision)

# The prebuilt Windows OpenCV is one DLL, copy it next to the executables.
if(WIN32)
  file(GLOB OPENCV_DLL ${OpenCV_DIR}/x64/vc16/bin/opencv_world*[0-9].dll)
endif()

enable_testing()
add_executable(lynx_tests
  tests/test_se3.cpp
  tests/test_geometry.cpp
  tests/test_pnp.cpp
  tests/test_ba.cpp
  tests/test_trajectory.cpp
  tests/test_e2e.cpp)
target_link_libraries(lynx_tests PRIVATE lynx_vision GTest::gtest_main)
target_compile_definitions(lynx_tests PRIVATE
  LYNX_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
include(GoogleTest)
gtest_discover_tests(lynx_tests DISCOVERY_MODE PRE_TEST)

foreach(exe lynx_vo lynx_tests)
  if(OPENCV_DLL)
    add_custom_command(TARGET ${exe} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DLL} $<TARGET_FILE_DIR:${exe}>)
  endif()
endforeach()
