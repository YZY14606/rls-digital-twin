# This file talk about how to create a environment for object_centric_policy.

`conda create -n real_robot_policy python==3.12.4`

`conda activate real_robot_policy`

`pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124`

`conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit=12.4.0 --strict-channel-priority`

`cd push_policy`

`pip install -r requirements.txt`

`pip install -e .`

# Vulkan install
`sudo apt-get update`

`sudo apt-get install libvulkan1`

`sudo apt-get install vulkan-utils (if ubuntu 22.04, install vulkan-tools)`

`vulkaninfo`

# SAM-2 install

`cd third_part`

`git clone https://github.com/facebookresearch/sam2.git && cd sam2`

`pip install -e .`

Download all checkpoints:
`cd checkpoints && \
./download_ckpts.sh && \
cd ..`

Or download single one in: https://github.com/facebookresearch/sam2.

# Object-planner install

`cd .. (cd my_porject)`

`git clone https://github.com/H-tr/object_planner.git`

`cd object_planner`

`conda install -c conda-forge eigen==3.4.0`

`pip install -e .`
