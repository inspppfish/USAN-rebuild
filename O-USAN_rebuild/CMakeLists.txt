cmake_minimum_required(VERSION 3.5)

project(demo)

SET(BIN_NAME "run")

find_package(OpenCV 3.4.7 REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
#message(STATUS "    libraries: ${OpenCV_DIR}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE sourcefiles "src/*.cpp")

ADD_EXECUTABLE(${BIN_NAME} main.cpp ${sourcefiles} include/O-USAN.h src/O-USAN.cpp include/kernel_gen.h src/kernel_gen.cpp include/Arrow.h src/Arrow.cpp)
add_definitions(-std=c++11)

target_link_libraries(${BIN_NAME} ${OpenCV_LIBS})


