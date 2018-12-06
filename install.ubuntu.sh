#!/bin/bash
#
# install.ubuntu.sh
# ---
#
#   This sets the repository up for running the training loop.
#
#   Usage: `./install.ubuntu.sh`
#

echo "Installing Python3.6, required for AllenNlp"
sudo add-apt-repository ppa:deadsnakes/ppa; sudo apt-get update
sudo apt-get install python3.6

echo "Installing venv for Python3.6"
sudo apt install python3.6-venv

echo "Creating venv..."
# python3.6 -m venv .

echo "In the virtualenv, run the following"
echo "
pip install spacy allennlp
pip install numpy torchvision_nightly
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
python3.6 -m spacy download en
"


