

python data_preparation/preprocess.py --dataset "lcq2" --kb_endpoint "https://skynet.coypu.org/wikidata/"
python data_preparation/context_paths_retrieval.py --dataset "lcq2"
python data_preparation/do_bert_encoding.py --dataset "lcq2"