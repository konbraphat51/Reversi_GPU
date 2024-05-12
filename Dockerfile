# Nvidia HPC Contianerベースイメージを使用
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# 必要なパッケージのインストール
RUN apt-get update && \
    apt-get install -y \
    bash-completion \
    build-essential \
    wget \
    make \
    cmake \
    gcc \
    gfortran \
    tar \
    git \
    python3 \
    vim \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install NVIDA HPC SDK
WORKDIR /installs/nvidia
RUN wget  https://developer.download.nvidia.com/hpc-sdk/24.3/nvhpc_2024_243_Linux_x86_64_cuda_multi.tar.gz
RUN tar xpzf nvhpc_2024_243_Linux_x86_64_cuda_multi.tar.gz
RUN nvhpc_2024_243_Linux_x86_64_cuda_multi/install
RUN rm -rf nvhpc_2024_243_Linux_x86_64_cuda_multi.tar.gz nvhpc_2024_243_Linux_x86_64_cuda_multi
RUN echo "/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/compilers/bin/" >> /root/.bashrc

# ソースコードのコピー
COPY . /app/source

# ビルド作業ディレクトリの作成と移動
WORKDIR /app/build