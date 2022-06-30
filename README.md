# master-project

# Requirements
The following software must be installed before the installation can be performed:
- Python 3.7, 3.8, or 3.9.10 
- ipm-python https://github.com/br4sco/ipm-python
- pyglet version 1.5.11 or 1.5.14
- cpprb
- cmake
- swig
- mpi
- To save videos of the Furuta Pendulum environment, ffmpeg is required
- To use some of the environments, such as Walker 2D, MuJoCo is required
- All software listed in [src/deps/SLM_Lab/environment.yml](src/deps/SLM_Lab/environment.yml) except Roboschool and Ray
- To use the Quanser QUBE-Servo 2 environments, the [quanser-openai-driver](https://github.com/BlueRiverTech/quanser-openai-driver) is required
- Windows 11, macOS 12.3, or Ubuntu 18.04 LTS (might work on other OS:es)
- **NOTE:** This list is not complete. Also, some of the modules specified above may not necessarily be used in the current implementation.

# Installation
1. Open a terminal window in the [src]() folder.
2. Go to [src/deps/spinningup/](src/deps/spinningup/) and run `pip install -e .`
3. Go to [src/deps/baselines/](src/deps/baselines/) and run `pip install -e .`
4. Put ipm-python in [src/deps/](src/deps/) and rename it to ipm_python

# Usage
1. Open a terminal window in the root directory of the repository.
2. Run `python src/testing.py -a ddpg -r 64_64_relu -n cartpole -t -i 100000 -s 0 -p`. This will train a DDPG agent with a two-layer MLP with 64 nodes in each layer and ReLU activations, on the cartpole environment of OpenAI Gym for 100000 timesteps with seed 0, and then open a webbrowser showing a graph with the performance as a function of the number of timesteps.
3. To evaluate the trained agent and see it interact with the environment, run `python src/testing.py -a ddpg -r 64_64_relu -n cartpole -s 0 -e`.

Ready-made experiments can also be run by using the `-x <experiment-name>` argument, where the details of the available experiments are available in [src/exp_config.py](src/exp_config.py). Do not forget that you must also supply the `-t`, `-e`, and/or `-p` argument when running an experiment, to specify whether you want to run training, evaluation, and/or print the performance graph(s).
