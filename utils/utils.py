import re
import json

ENTITY_PATTERN = re.compile('Q[0-9]+')
PREDICATE_PATTERN = re.compile('P[0-9]+')

with open("../data/labels_dict.json") as labelFile:
    labels_dict = json.load(labelFile)

def is_timestamp(timestamp):
    pattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    if not(pattern.match(timestamp)):
        return False
    else:
        return True


def convertTimestamp(timestamp):
    yearPattern = re.compile('^[0-9][0-9][0-9][0-9]-00-00T00:00:00Z')
    monthPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-00T00:00:00Z')
    dayPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    timesplits = timestamp.split("-")
    year = timesplits[0]
    if yearPattern.match(timestamp):
        return year
    month = convertMonth(timesplits[1])
    if monthPattern.match(timestamp):
        return month + " " + year
    elif dayPattern.match(timestamp):
        day = timesplits[2].rsplit("T")[0]
        return day + " " + month + " " + year

    return timestamp

def convertMonth(month):
    return{
        "01": "january",
        "02": "february",
        "03": "march",
        "04": "april",
        "05": "may",
        "06": "june",
        "07": "july",
        "08": "august",
        "09": "september",
        "10": "october",
        "11": "november",
        "12": "december"
    }[month]


def get_label(entity):
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
        if is_timestamp(entity):
            label = convertTimestamp(entity)
        elif entity.startswith("+"):
            label = entity.split("+")[1]
        else:
            label = entity

    return label


def fill_missing_prefixes(prefixes, sparql):
    new_sparql = sparql
    for alias, uri in prefixes.items():
        if sparql.find(alias) != -1 and sparql.find(uri) == -1:
            new_sparql = uri + " " + new_sparql
    return new_sparql

# if __name__ == '__main__':
#     sparql = "SELECT ?obj WHERE { wd:Q567 p:P39 ?s . ?s ps:P39 ?obj . ?s pq:P580 ?x filter(contains(YEAR(?x),'1994')) }"
#     PREFIXES_WIKIDATA = {
#         " p:": "PREFIX p: <http://www.wikidata.org/prop/>",
#         "wdt:": "PREFIX wdt: <http://www.wikidata.org/prop/direct/>",
#         "wd:": "PREFIX wd: <http://www.wikidata.org/entity/>",
#         "xsd:": "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>",
#         "pq:": "PREFIX pq: <http://www.wikidata.org/prop/qualifier/>",
#         "ps:": "PREFIX ps: <http://www.wikidata.org/prop/statement/>",
#         "rdfs:": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"
#     }
#     print(fill_missing_prefixes(PREFIXES_WIKIDATA, sparql))