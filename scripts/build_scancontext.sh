#!/usr/bin/env bash

set -e
echo "Building scancontext python bindings..."
cd "$(dirname "$0")/../third_party/scancontext-pybind"
mkdir -p build && cd build
cmake ..
make -j$(nproc) && echo "Done"
cd python && uv pip install .
echo "scancontext installed into $(python -c 'import sys; print(sys.prefix)')"
