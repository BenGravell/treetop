cmake_minimum_required(VERSION 3.23)
project(treetop)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Enable compile_commands.json (required by IWYU)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find packages
find_package(Eigen3 REQUIRED)
find_package(raylib REQUIRED)

# Add all source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
message(STATUS "Source files: ${SOURCES}")

# Manually include main.cpp from the repo root
set(MAIN_FILE ${CMAKE_SOURCE_DIR}/src/main.cpp)

# Add the executable, combining main.cpp and the other source files
add_executable(${PROJECT_NAME} ${MAIN_FILE} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Link external libraries to the target
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
        Eigen3::Eigen
        raylib
)