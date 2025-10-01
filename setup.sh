#!/bin/bash

export NV_DRV_VERSION=570.133.07
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/$NV_DRV_VERSION/NVIDIA-Linux-x86_64-$NV_DRV_VERSION.run
chmod +x NVIDIA-Linux-x86_64-$NV_DRV_VERSION.run
./NVIDIA-Linux-x86_64-$NV_DRV_VERSION.run --kernel-module-type=open


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo systemctl restart docker.service
sudo nvidia-smi -i 0 --lock-gpu-clocks=600,600
sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 -mig 1
# H100
# sudo nvidia-smi mig -cgi 9,9 -C
# A30
sudo nvidia-smi mig -cgi 5,5 -C


cd storm
sudo docker build -t ai_image .
sudo docker run --rm --gpus "device=0:0" -v $PWD:/workspace -v ~/.cache/torch/:/root/.cache/torch -v ~/.cache/huggingface:/root/.cache/huggingface --shm-size=16g -w /workspace ai_image python3 -u predownload_models.py
cd ../pf-latency
sudo docker build -t ai_image .
sudo docker run --rm --gpus "device=0:0" -v $PWD:/workspace -v ~/.cache/torch/:/root/.cache/torch -v ~/.cache/huggingface:/root/.cache/huggingface --shm-size=16g -w /workspace ai_image python3 -u predownload_models.py
sudo docker run --rm --gpus "device=0:0" -v $PWD:/workspace -v ~/.cache/huggingface:/root/.cache/huggingface --shm-size=16g -w /workspace ai_image python3 -u predownload_llm.py

