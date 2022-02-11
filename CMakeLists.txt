cmake_minimum_required(VERSION 3.1...3.21)

set(PROJECT_NAME TreeSAT)

project(${PROJECT_NAME} VERSION 1.0 LANGUAGES CXX)

option(${PROJECT_NAME}_BUILD_PYTHON "Build the python bindings" ON)


#--------------------------------------------------
# dependencies pybind11 and Eigen (header only)
if(${PROJECT_NAME}_BUILD_PYTHON)
    add_subdirectory(ext/pybind11)
endif()

add_library(Eigen INTERFACE)
target_include_directories(Eigen
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/ext/eigen
)

find_package(OpenMP)
#--------------------------------------------------

set(TARGET_NAME sat_tile_tree)

if(${PROJECT_NAME}_BUILD_PYTHON)
    pybind11_add_module(${TARGET_NAME}
        MODULE
        src/sat_tile_tree.cpp
    )

    set_property(TARGET ${TARGET_NAME} PROPERTY COMPILE_DEFINITIONS SAT_TILE_TREE_BUILD_PYTHON APPEND)
endif()

target_include_directories(${TARGET_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)

target_link_libraries(${TARGET_NAME}
    PUBLIC 
    Eigen
    OpenMP::OpenMP_CXX
)