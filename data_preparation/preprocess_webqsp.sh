python preprocess.py \
--dataset webqsp \
--train_data "../data/WebQSP/WebQSP.train.json" \
--test_data "../data/WebQSP/WebQSP.test.json" \
--dest_dir "../processed_data/WebQSP" \
--kb_endpoint "http://localhost:12345/freebase/sparql"