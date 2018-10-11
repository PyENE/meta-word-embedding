#!/usr/bin/env bash

wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip
unzip v0.1.0.zip
cd fastText-0.1.0
make
cd ..
rm v0.1.0.zip
