import re
import json
import KB_query
import utils

# with open("../data/labels_dict.json") as labelFile:
#    labels_dict = json.load(labelFile)


with open("../processed_data/labels_dict.json") as labelFile:
    labels_dict = json.load(labelFile)

PREFIXES_WIKIDATA = {
    " p:": "PREFIX p: <http://www.wikidata.org/prop/>",
    "wdt:": "PREFIX wdt: <http://www.wikidata.org/prop/direct/>",
    "wd:": "PREFIX wd: <http://www.wikidata.org/entity/>",
    "xsd:": "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>",
    "pq:": "PREFIX pq: <http://www.wikidata.org/prop/qualifier/>",
    "ps:": "PREFIX ps: <http://www.wikidata.org/prop/statement/>",
    "rdfs:": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"
}


# def get_ent_label(answers):
#     gold_answer_texts = []
#
#     for ans in answers:
#         if type(ans) == bool:
#             gold_answer_texts.append(str(ans))
#             continue
#
#         if re.match(utils.ENTITY_PATTERN, ans) or re.match(utils.PREDICATE_PATTERN, ans):
#             label = KB_query.query_ent_label(ans, kb_endpoint)
#         else:
#             if utils.is_timestamp(ans):
#                 label = utils.convertTimestamp(ans)
#             elif ans.startswith("+"):
#                 label = ans.split("+")[1]
#             else:
#                 label = ans
#         gold_answer_texts.append(label)
#
#     return gold_answer_texts


def getLabel(entity):
    label = ""
    if entity.startswith("Q") or entity.startswith("P"):
            #for predicates: P10-23, split away counting
        if "-" in entity:
            e = entity.split("-") [0]
        else:
            e = entity
        if e in labels_dict.keys():
            label = labels_dict[e]
    else:
        if utils.is_timestamp(entity):
            label = utils.convertTimestamp(entity)
        elif entity.startswith("+"):
            label = entity.split("+")[1]
        else:
            label = entity

    return label


def get_gold_answers(answers):

    gold_answers = []

    for ans in answers:
        if type(ans) == bool:
            gold_answers.append(ans)
            continue
        if len(ans) > 0:
            ans = list(ans.values())[0]
            if "/" in ans:
                ans = ans.rsplit("/", 1)[1]
            gold_answers.append(ans)

    return gold_answers


if __name__ == '__main__':
    #query_1 = "SELECT ?obj WHERE { wd:Q213611 p:P1411 ?s . ?s ps:P1411 ?obj . ?s pq:P805 wd:Q917076 }"
    #query_1 = "ASK WHERE { wd:Q740 wdt:P2404 ?obj filter(?obj = 0.1) } "
    query_1 = "SELECT ?answer WHERE { wd:Q4890903 wdt:P161 ?X . ?X wdt:P26 ?answer}"
    kb_endpoint = "https://query.wikidata.org/sparql"
    extracted_answers = KB_query.kb_query(query_1, kb_endpoint)
    gold_answers = get_gold_answers(extracted_answers)
    print("asd")
