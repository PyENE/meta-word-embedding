#!/usr/bin/env bash

mkdir -p embeddings
cd embeddings

wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.fr.300.bin.gz
gzip -d cc.fr.300.bin.gz
rm cc.fr.300.bin.gz

wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.fr.300.vec.gz
gzip -d cc.fr.300.vec.gz
rm cc.fr.300.vec.gz

cd ..