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
    gdb \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#git settings
RUN git config --global credential.helper store 

# Install NVIDA HPC SDK
WORKDIR /installs/nvidia
RUN wget https://developer.download.nvidia.com/hpc-sdk/22.11/nvhpc_2022_2211_Linux_x86_64_cuda_11.8.tar.gz
RUN tar xpzf nvhpc_2022_2211_Linux_x86_64_cuda_11.8.tar.gz
RUN nvhpc_2022_2211_Linux_x86_64_cuda_11.8/install
RUN rm -rf nvhpc_2022_2211_Linux_x86_64_cuda_11.8.tar.gz nvhpc_2022_2211_Linux_x86_64_cuda_11.8
RUN echo "export PATH=\$PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin/" >> /root/.bashrc

# give path for -lcudart
RUN ln -s /usr/local/cuda/lib64/libcudart.so.11.0 /usr/local/cuda/lib64/libcudart.so
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/libcudart.so" >> /root/.bashrc

# ソースコードのコピー
COPY . /app/source

# ビルド作業ディレクトリの作成と移動
WORKDIR /app/source