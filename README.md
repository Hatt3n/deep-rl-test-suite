# master-project

# Requirements
The following software must be installed before the installation can be performed:
- Python 3.7 or 3.9.10 
- ipm-python https://github.com/br4sco/ipm-python
- pyglet version 1.5.11 or 1.5.14
- All software listed in [src/deps/SLM_Lab/environment.yml](src/deps/SLM_Lab/environment.yml) except Roboschool
- Windows 11 or macOS 12 (might work on other OS:es)
- **NOTE:** This list is not complete. Also, some of the modules specified above may not necessarily be used in the current implementation.

# Installation
1. Open a terminal window in the [src]() folder.
2. Go to [src/deps/spinningup/](src/deps/spinningup/) and run `pip install -e .`
3. Go to [src/deps/baselines/](src/deps/baselines/) and run `pip install -e .`
4. Put ipm-python in [src/deps/](src/deps/) and rename it to ipm_python
5. There are functions defined within functions in src/deps/ipm_python/furuta.py, which will cause the Pickler to crash. To fix this, the code must manually be rewritten so that they become stand-alone.
6. It might be necessary to create some folders beforehand, although they should be created automatically. An example of such a directory is [out/res/]()