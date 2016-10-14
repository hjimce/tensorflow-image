cmake_minimum_required(VERSION 3.5)
project(c_predict)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(.)

set(SOURCE_FILES main.cpp)
add_executable(c_predict ${SOURCE_FILES})