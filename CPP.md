# C++ Related Commands

Collection of bash commands relate to building C++ projects.

## CMake

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

or use Ninja instead of Makefiles:

```bash
mkdir build && cd build
cmake -G Ninja ..
ninja
```

## Clang / GCC

### Check

```bash
update-alternatives --query c
update-alternatives --query c++
```

### Add

```bash
# Register GCC as an option for C
sudo update-alternatives --install /usr/bin/cc c /usr/bin/gcc 100

# Register Clang as an option for C
sudo update-alternatives --install /usr/bin/cc c /usr/bin/clang 90

# Register GCC as an option for C++
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100

# Register Clang as an option for C++
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 90
```

### Default

```bash
sudo update-alternatives --config c
sudo update-alternatives --config c++
```

### Validate

```bash
cc --version
c++ --version
```
