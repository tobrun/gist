# C++ Related Commands

Collection of bash commands relate to building C++ projects.

## CMake

```bash
cmake . -B build
cmake --build build -j ${nproc}
```

### Disable warnings

```bash
CXXFLAGS="-w" cmake . -B build
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

Switch between GCC and Clang

```bash
sudo update-alternatives --config c
sudo update-alternatives --config c++
```

Switch between used GCC versions

```bash
sudo update-alternatives --config g++
sudo update-alternatives --config gcc
```

### Validate

```bash
cc --version
c++ --version
```
