python preprocess.py \
--dataset lcq \
--train_data "../data/LC-QuAD/train_data.json" \
--test_data "../data/LC-QuAD/test-data.json" \
--dest_dir "../processed_data/LC-QuAD" \
--kb_endpoint "http://localhost:12345/dbpedia/sparql"
