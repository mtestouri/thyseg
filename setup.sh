#!/bin/bash


# install pytorch
pip install wheel numpy torch torchvision

# install SLDC
dpkg -s openslide-tools &> /dev/null
if [ $? -ne 0 ]; then
    sudo apt update && sudo apt install openslide-tools
fi
pip install openslide-python sldc
git clone https://github.com/waliens/sldc-openslide.git
(cd sldc-openslide; python setup.py install)

# clean
rm -rf sldc-openslide
