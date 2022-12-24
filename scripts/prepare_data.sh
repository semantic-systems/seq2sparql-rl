#!/bin/bash

cd preprocess

python preprocess.py --dataset "lcq2" --kb_endpoint "https://skynet.coypu.org/wikidata/"
python context_paths_retrieval.py --dataset "lcq2"
python do_bert_encoding.py --dataset "lcq2"

