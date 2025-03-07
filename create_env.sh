# # Create a conda virtual environment and activate it
conda create -n patho python=3.8
conda activate patho

# Install requirements
pip install \
        yacs==0.1.8 \
        termcolor==2.2.0 \
        timm==0.6.12 \
        pykeops==2.1.1 \
        ptflops==0.6.9 \
        numpy==1.22.4 
conda install -c conda-forge opencv
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# Install the custom CUDA kernels for patho
cd clusten/src/ && python setup.py install
