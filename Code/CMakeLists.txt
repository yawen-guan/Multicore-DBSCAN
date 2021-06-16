project(DBSCAN)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp -O3")

include_directories("${PROJECT_SOURCE_DIR}/include")

link_directories("${PROJECT_SOURCE_DIR}/lib")

add_subdirectory("${PROJECT_SOURCE_DIR}/lib")

add_executable(DBSCAN "${PROJECT_SOURCE_DIR}/src/main.cpp")

target_link_libraries(DBSCAN libcudaDBSCAN)
