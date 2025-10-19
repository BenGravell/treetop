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
conan install . --build=missing -of=build/conan/release --settings=build_type=Release

cmake -B build/release -S . -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="build/conan/release/conan_toolchain.cmake" -DCMAKE_CXX_FLAGS="-march=native -ffast-math -flto=auto" -DCMAKE_C_FLAGS="-march=native -ffast-math -flto=auto"

cmake --build build/release --config Release
```

### Build Debug

```bash
conan install . --build=missing -of=build/conan/debug --settings=build_type=Debug

cmake -B build/debug -S . -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE="build/conan/debug/conan_toolchain.cmake"

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

First time setup - vendor eigen

```bash
mkdir -p external
git clone --depth 1 https://gitlab.com/libeigen/eigen.git external/eigen
```

First time setup - add conan profile from [`conan_profile_emscripten`](conan_profile_emscripten)

Every time

```bash
source ~/emsdk/emsdk_env.sh

conan install . --profile=emscripten --build=raylib --build=missing --output-folder=build/conan/web

emcmake cmake -B build/web -DCMAKE_BUILD_TYPE=Release -DPLATFORM_WEB=ON -DCMAKE_TOOLCHAIN_FILE="build/conan/web/conan_toolchain.cmake"

emmake cmake --build build/web -j
```

#### Run

```bash
source ~/emsdk/emsdk_env.sh

emrun index.html
```
