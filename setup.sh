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
pip install packaging

pip install packaging psutil ninja einops \
  --index-url http://pip.sankuai.com/simple/ \
  --extra-index-url http://pypi.sankuai.com/simple/ \
  --trusted-host pip.sankuai.com \
  --trusted-host pypi.sankuai.com

pip install flash-attn==2.7.4.post1 --no-build-isolation

cd trl
pip install .

cd bitsandbytes
pip install .