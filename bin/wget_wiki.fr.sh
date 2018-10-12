#!/usr/bin/env bash

mkdir -p embeddings
cd embeddings
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fr.zip
gzip -d wiki.fr.zip
rm wiki.fr.zip
cd ..