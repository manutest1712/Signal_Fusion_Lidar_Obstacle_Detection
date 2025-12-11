cmake_minimum_required(VERSION 3.10)
project(playback)

# Use C++14 (VS2019/2022 supports this)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional warnings (MSVC != GCC)
if (MSVC)
    add_compile_options(/W3)
else()
    add_compile_options(-Wall)
endif()


# Find PCL from vcpkg
find_package(PCL CONFIG REQUIRED)

# Executable
add_executable(quizCluster 
    cluster.cpp
    ../../render/render.cpp
)

target_link_libraries (quizCluster PRIVATE ${PCL_LIBRARIES})




