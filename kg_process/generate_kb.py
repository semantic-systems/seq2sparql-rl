import os
import sys
sys.path.append("..")

import json
from utils.knowledge_graph import WikidataKG

wikidata = WikidataKG("https://skynet.coypu.org/wikidata/")
labels_dict = json.load(open("../data/labels_dict.json"))

kb = dict()
if os.path.exists("../data/kg.json"):
    with open("../data/kb.json", "r") as kb_file:
        kb = json.load(kb_file)

for idx, entity in enumerate(labels_dict.keys(), 1):

    if entity in kb:
        print(f"{idx}/{len(labels_dict)} Entity: {entity} exists in kb.")
        continue

    one_hop_paths = wikidata.get_one_hop_paths(entity)
    all_one_hop_paths = list()
    for path in one_hop_paths:
        all_one_hop_paths.append([path[0], [path[1]], path[2]])

    if len(all_one_hop_paths) == 0:
        print(f"{idx}/{len(labels_dict)} Entity: {entity} has zero one-hop paths.")
        continue

    kb[entity] = all_one_hop_paths

    print(f"{idx}/{len(labels_dict)} Entity: {entity} has {len(all_one_hop_paths)} one-hop paths")

    if idx % 10000 == 0:
        with open("../data/kb.json", "w") as kb_file:
            json.dump(kb, kb_file)

with open("../data/kb.json", "w") as kb_file:
    json.dump(kb, kb_file)

print(f"Total entities: {len(kb)}")



