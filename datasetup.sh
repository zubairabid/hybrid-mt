#!/bin/bash

echo "Welcome to the setup tool!"

echo "Downloading the WMT '14 EN-FR data"
mkdir -p data/wmt-14/json
cd data/wmt-14
wget https://www.statmt.org/europarl/v7/fr-en.tgz
tar -zxvf fr-en.tgz
cd ../../

echo "Making a JSON for the downloaded data. Requires a working python install"
echo "with all necessary packages installed. Follow the setup recommended."
python preprocess.py 'data/wmt-14'
