cmake_minimum_required(VERSION 3.23)
project(treetop)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Gather source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
set(MAIN_FILE ${CMAKE_SOURCE_DIR}/src/main.cpp)
add_executable(${PROJECT_NAME} ${MAIN_FILE} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)

# --- Platform-specific configuration ---
if(PLATFORM_WEB)
    message(STATUS "Configuring for Web (Emscripten) target")

    # Rename output to index.html (and automatically index.js, index.wasm)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_NAME "index"
    )

    # Paths to prebuilt raylib and system include dirs
    set(RAYLIB_INCLUDE_DIR "$ENV{HOME}/emsdk/upstream/emscripten/cache/sysroot/include")
    set(RAYLIB_LIB "$ENV{HOME}/emsdk/upstream/emscripten/cache/sysroot/lib/libraylib.a")

    # Add Eigen manually for Web builds
    include_directories(${CMAKE_SOURCE_DIR}/external/eigen)

    # Add raylib includes and link
    include_directories(${RAYLIB_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${RAYLIB_LIB})

    # Emscripten linker flags
    # NOTE: TOTAL_STACK and INITIAL_MEMORY must be high enough to support performance requirements of this app.
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LINK_FLAGS "-O3 -Wall \
                    -s USE_GLFW=3 \
                    -s ASYNCIFY \
                    -s TOTAL_STACK=64MB \
                    -s INITIAL_MEMORY=128MB \
                    --shell-file ${CMAKE_SOURCE_DIR}/shell.html \
                    --preload-file ${CMAKE_SOURCE_DIR}/assets"
        SUFFIX ".html"
    )

    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
        RUNTIME_OUTPUT_NAME "index"
    )


    # DEBUG
    # set_target_properties(${PROJECT_NAME} PROPERTIES
    #     LINK_FLAGS "-O0 -gsource-map \
    #                 -s USE_GLFW=3 \
    #                 -s ASYNCIFY \
    #                 -s ASSERTIONS=2 \
    #                 -s SAFE_HEAP=1 \
    #                 -s STACK_OVERFLOW_CHECK=2 \
    #                 -s TOTAL_STACK=64MB \
    #                 -s INITIAL_MEMORY=128MB \
    #                 --shell-file ${CMAKE_SOURCE_DIR}/shell.html \
    #                 --preload-file ${CMAKE_SOURCE_DIR}/assets"
    #     SUFFIX ".html"
    # )


else()
    message(STATUS "Configuring for native desktop build")

    find_package(Eigen3 REQUIRED)
    find_package(raylib REQUIRED)

    target_link_libraries(${PROJECT_NAME} 
        PRIVATE 
            Eigen3::Eigen
            raylib
    )
endif()
