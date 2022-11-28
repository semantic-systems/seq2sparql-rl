
python preprocess.py \
--dataset lcq2 \
--train_data "../data/LC-QuAD2/train.json" \
--test_data "../data/LC-QuAD2/test.json" \
--dest_dir "../processed_data/LC-QuAD2" \
--kb_endpoint "https://query.wikidata.org/sparql"