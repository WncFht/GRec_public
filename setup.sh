pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
#  --index-url https://download.pytorch.org/whl/cu124
pip install packaging
# pip install flash-attn==2.8.0.post2 --no-build-isolation

conda install gxx -y
pip install flash-attn==2.7.4.post1 --no-build-isolation

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
    binutils_linux-64 -y
    
conda install av -y
pip install -r requirements.txt

conda install pandas -y
pip install seaborn hope

pip install hope

conda install pyarrow

cd trl
pip install -e .

cd MiniOneRec
pip install -r requirements.txt

cd bitsandbytes
pip install .