cmake_minimum_required(VERSION 3.1...3.21)

set(PROJECT_NAME TreeSAT)
set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

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

set(TARGET_NAME ${PROJECT_NAME})

set(SOURCE_FILES src/sat_tile_tree.cpp)

if(${PROJECT_NAME}_BUILD_PYTHON)
    pybind11_add_module(${TARGET_NAME}
        MODULE
        ${SOURCE_FILES}
    )

    set_property(TARGET ${TARGET_NAME} PROPERTY COMPILE_DEFINITIONS SAT_TILE_TREE_BUILD_PYTHON APPEND)
    set_property(TARGET ${TARGET_NAME} PROPERTY COMPILE_DEFINITIONS TARGET_NAME=${TARGET_NAME} APPEND)
else()
    add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})
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