cmake_minimum_required(VERSION 3.12)

project(playback)

# Use C++14 (VS2019/2022 supports this)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PCL from vcpkg
find_package(PCL CONFIG REQUIRED)

add_executable(environment 
    src/environment.cpp 
    src/render/render.cpp 
    src/processPointClouds.cpp
)

# Link full PCL bundle (similar to your old code)
target_link_libraries(environment PRIVATE ${PCL_LIBRARIES})





