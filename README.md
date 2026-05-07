# D-NAV: End-to-End Dynamic UAV Navigation with Dual-Resolution Motion Awareness

[![Python](https://img.shields.io/badge/python-3.10-4B8BBE.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Isaac Sim](https://img.shields.io/badge/IsaacSim-2023.1.0--hotfix.1-C0392B.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Platform](https://img.shields.io/badge/platform-Ubuntu%2022.04-27AE60.svg)](https://releases.ubuntu.com/22.04/)

Simple implementation of **D-NAV**, an end-to-end reinforcement learning framework for autonomous UAV navigation in dense dynamic environments.

D-NAV directly maps raw LiDAR observations to control actions through a saliency-driven dual-resolution spatio-temporal representation. The framework is designed to jointly reason about large-scale dynamic context and fine-grained local motion cues, enabling robust navigation in cluttered environments with heterogeneous obstacle scales and complex motion patterns.

---

# Overview

Autonomous navigation in dynamic cluttered environments remains a challenging problem for UAV systems due to:

- Irregular obstacle geometries
- Fast and unpredictable motion patterns
- Real-time perception and control constraints

D-NAV addresses these challenges using:

- A spatio-temporal spherical depth representation constructed directly from sequential LiDAR observations
- A saliency-guided local refinement mechanism for dynamically critical regions
- A mission-aware waypoint-goal conditioning strategy for balancing collision avoidance and task completion

Extensive simulation and real-world experiments demonstrate strong robustness and real-time performance in challenging dynamic scenarios.

---


# Environment Setup

## System Requirements

The framework has been tested on:

- Ubuntu 22.04
- Python 3.10
- NVIDIA RTX 4090 GPU
- CUDA 12.1
- NVIDIA Isaac Sim 2023.1.0-hotfix.1

> **Important:**  
> D-NAV depends on Isaac Sim **2023.1.0-hotfix.1**. Other versions may introduce compatibility issues.

---

# Isaac Sim Installation

Please ensure that:

- NVIDIA drivers are correctly installed
- Docker is available
- Conda or Miniconda is installed

## Step 1: Install Isaac Sim via Docker

Follow the official NVIDIA container setup guide:

- https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html

Then pull the required Isaac Sim image:

```bash
docker pull nvcr.io/nvidia/isaac-sim:2023.1.0-hotfix.1
```

Launch the container:

```bash
docker run --name isaac-sim \
    --entrypoint bash \
    -it \
    --runtime=nvidia \
    --gpus all \
    --network=host \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    nvcr.io/nvidia/isaac-sim:2023.1.0-hotfix.1
```

## Step 2: Copy Isaac Sim to Local Directory

In a new terminal:

```bash
docker ps
```

Find the container ID and copy Isaac Sim to a local directory:

```bash
docker cp <container_id>:/isaac-sim/. /path/to/isaac-sim
```

---

# D-NAV Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/qinliangql/D-NAV.git
cd D-NAV
```

Set the Isaac Sim path:

```bash
echo 'export ISAACSIM_PATH="/path/to/isaac-sim"' >> ~/.bashrc
source ~/.bashrc
```

Install the training environment:

```bash
cd isaac-training
bash setup.sh
```

After installation, activate the Conda environment:

```bash
conda activate D-NAV
```

---

## Additional Dependencies

Install the following dependencies manually.


```bash
pip install torch-cluster \
    --no-index \
    --find-links https://data.pyg.org/whl/torch-2.0.0+cu118.html

pip install spconv-cu121
```

> The CUDA version used here must match the CUDA version required by the installed PyTorch build, rather than the system CUDA version.


# Verify Installation

Run a simple training example to verify the installation:

```bash
conda activate D-NAV
bash run.sh
```

If the installation is successful, the Isaac Sim window should launch correctly.

---

# Quick Evaluation Demo

We provide pretrained checkpoints and evaluation scripts for quick testing.

Run:

```bash
conda activate NavRL
bash test.sh
```

---



# Acknowledgements

The Isaac Sim training framework is built upon:

- OmniDrones: https://github.com/btx0424/OmniDrones

We thank the authors for open-sourcing their excellent work.

---
