# meta-word-embedding

## Installation

```bash
git clone https://github.com/brice-olivier/meta-word-embedding.git
cd meta-word-embedding
sudo python setup.py install --user
chmod +x ./bin/setup_fasttext.sh
./bin/setup_fasttext.sh
```

## If you do not already have fasttext models and embeddings (in french)

```bash
chmod +x ./bin/wget_cc.fr.300.sh
./bin/wget_cc.fr.300.sh
chmod +x ./bin/wget_wiki.fr.sh
./bin/wget_wiki.fr.sh
```

## Usage

```python
import os
import pandas
import metawordembedding
```
