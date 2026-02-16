conda create -n grec python==3.10
conda install gxx \
    av \
    pandas \
    cmake \
    make \
    ninja \
    pkg-config \
    gcc_linux-64 \
    gxx_linux-64 \
    pyarrow \
    libxcrypt \
    binutils_linux-64 -y

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

pip install packaging psutil ninja einops

pip install flash-attn==2.7.4.post1 --no-build-isolation

pip install -r requirements_backup.txt
cd trl
pip install .

cd bitsandbytes
pip install .
