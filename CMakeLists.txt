cmake_minimum_required(VERSION 3.18)
set(CMAKE_BUILD_TYPE Debug)

project(lightZK LANGUAGES CXX CUDA)
# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

# Set CUDA Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --extended-lambda -lineinfo")

# Set Searching Path for Header Files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Collecting source codes
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
	"${PROJECT_SOURCE_DIR}/src/*.c"
	"${PROJECT_SOURCE_DIR}/src/*.cpp"
	"${PROJECT_SOURCE_DIR}/src/*.cu"
)

# Add Core Library
add_library(libLightZK STATIC ${SRC_FILES})

set_target_properties(libLightZK PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	POSITION_INDEPENDENT_CODE ON
)

target_include_directories(libLightZK PUBLIC ${PROJECT_SOURCE_DIR}/include)

set(CURVE "ALT_BN128" CACHE STRING "Default curve: one of ALT_BN128, BN128, EDWARDS, MNT4, MNT6")
add_definitions(-DCURVE_${CURVE})
add_subdirectory(depends/libsnark)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fopenmp")

# 编译测试程序                                                                                                                                                                                  
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/test/*.cu")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE 
		libLightZK
		snark
	)
    target_include_directories(${test_name} PRIVATE 
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_SOURCE_DIR}/depends/libsnark
		${PROJECT_SOURCE_DIR}/depends/cgbn
	)
endforeach()

# 数据文件目录 (app 程序读写 R1CS 文件)
set(APP_DATA_DIR "${PROJECT_SOURCE_DIR}/app/data" CACHE PATH "Directory for app data files")

# 编译 app 程序
file(GLOB APP_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/app/*.cu")
foreach(app_src ${APP_SOURCES})
    get_filename_component(app_name ${app_src} NAME_WE)
    add_executable(${app_name} ${app_src})
    target_link_libraries(${app_name} PRIVATE 
		libLightZK
		snark
	)
    target_include_directories(${app_name} PRIVATE 
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_SOURCE_DIR}/depends/libsnark
		${PROJECT_SOURCE_DIR}/depends/cgbn
	)
    target_compile_definitions(${app_name} PRIVATE APP_DATA_DIR="${APP_DATA_DIR}")
endforeach()
