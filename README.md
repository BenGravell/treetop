# TreeTop

Tree-initialized trajectory-optimizing planner.

The approach is to use an ego motion sampling tree + iLQR.

The ego motion sampling tree provides a strong initial guess at a good path to the goal that avoids obstacles.

Iterative Linear Quadratic Regulator (iLQR) optimizes the trajectory.

The optimized trajectory is fed into the tree to warm-start it and re-use computations from previous iterations.

The tree and iLQR work together to provide rapid replanning and iterative optimization.

This planner runs extremely quickly, at rates as fast as 50 Hz.

The planner is interactive via a Raylib app.

## Local app

### Build

```bash
conan install . --build=missing -of=build/conan --settings=build_type=Release

cmake -B build/release -S . -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="build/conan/conan_toolchain.cmake" -DCMAKE_CXX_FLAGS="-march=native -ffast-math -flto=auto" -DCMAKE_C_FLAGS="-march=native -ffast-math -flto=auto"

cmake --build build/release --config Release
```

### Build Debug

```bash
conan install . --build=missing -of=build/conan --settings=build_type=Debug

cmake -B build/debug -S . -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE="build/conan/conan_toolchain.cmake"

cmake --build build/debug --config Debug
```

### Run

```pwsh
build/release/treetop
```

### Run debug

```pwsh
build/debug/treetop
```

## Web app

Follow the guides

- <https://anguscheng.com/post/2023-12-12-wasm-game-in-c-raylib/>
- <https://dev.to/marcosplusplus/how-to-install-raylib-with-web-support-l71>

### Build

#### Get EMSDK

```bash
# Change to home dir
cd

# Clone the emsdk repo
git clone https://github.com/emscripten-core/emsdk

# Enter the repo directory
cd emsdk

# Download and install the latest SDK tools.
./emsdk install latest
```

#### Activate EMSDK

```bash
# Change to emsdk dir
cd ~/emsdk

# Make the "latest" SDK "active" for the current user. (writes .emscripten file)
./emsdk activate latest

# Activate PATH and other environment variables in the current terminal
source ./emsdk_env.sh
```

#### build raylib for web

```bash
# Change to home dir
cd

# Clone the raylib repo
git clone https://github.com/raysan5/raylib

# Enter the repo directory
cd raylib

emcmake cmake . -DPLATFORM=Web -DSUPPORT_TRACELOG=OFF

emmake make

sudo make install 
```

#### build raylib for desktop

```bash
cmake -B build -DPLATFORM=PLATFORM_DESKTOP -DPLATFORM=Desktop;Web -DSUPPORT_TRACELOG=OFF
cmake --build build
sudo cmake --install build/
```

#### Get HTML base file

```bash
# Change directory up out of emsdk
cd ~/treetop

# Download base shell.html from raylib
wget https://raw.githubusercontent.com/raysan5/raylib/refs/heads/master/src/shell.html
```

#### Build to Web Assembly

```bash
cd ~/emsdk
source emsdk_env.sh
cd ~/treetop

em++ -o index.html src/main.cpp -O3 -Wall \
-I src \
-I ~/emsdk/upstream/emscripten/cache/sysroot/include \
-L ~/emsdk/upstream/emscripten/cache/sysroot/lib/libraylib.a \
-s USE_GLFW=3 -s ASYNCIFY \
--preload-file assets \
--shell-file shell.html \
-DPLATFORM_WEB \
~/emsdk/upstream/emscripten/cache/sysroot/lib/libraylib.a
```

#### Run

```bash
cd ~/emsdk
source emsdk_env.sh
cd ~/treetop

emrun index.html
```
