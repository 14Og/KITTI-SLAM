# LIO-SAM inspired GraphSLAM ok KITTI benchmark LiDAR sequences

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

Currently tested environment (Linux x86_64):
* Ubuntu 24.04.4 LTS
* g++ 13.3.0
* python3.12
* additional dependencies: libeigen3-dev

## Reproduce:
1. Install libeigen3-dev.
2. `git submodule update`
2. Run `uv sync`.
3. Run `scripts/build_scancontext.sh`
4. Test it with `uv run scripts/test_scancontext.py`