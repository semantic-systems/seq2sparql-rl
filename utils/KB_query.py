
import json
import re
from datetime import datetime

from SPARQLWrapper import SPARQLWrapper, JSON

def kb_query(_query, kb_endpoint):
    sparql = SPARQLWrapper(kb_endpoint)
    sparql.setQuery(_query)
    sparql.setReturnFormat(JSON)
    response = sparql.query().convert()
    results = parse_query_results(response)
    return results


def query_ent_label(x, kb_endpoint, lang="en"):
    query = "PREFIX wd: <http://www.wikidata.org/entity/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT * WHERE { wd:"+x+" rdfs:label ?label . filter(lang(?label) = '"+lang+"') }"
    result = kb_query(query, kb_endpoint)
    if len(result) == 0:
        print(x, "does not have {0} label !".format(lang))
        return x
    label = result[0]["label"]
    return label

def parse_query_results(response):

    if "boolean" in response:  # ASK
        results = ["Yes"] if response["boolean"] else ["No"]
    else:
        if len(response["results"]["bindings"]) > 0 and "callret-0" in response["results"]["bindings"][0]: # COUNT
            results = [int(response['results']['bindings'][0]['callret-0']['value'])]
        else:
            results = []
            for res in response['results']['bindings']:
                res = {k: v["value"] for k, v in res.items()}
                results.append(res)
    return results

def formalize(query):
    p_where = re.compile(r'[{](.*?)[}]', re.S)
    select_clause = query[:query.find("{")].strip(" ")
    select_clause = [x.strip(" ") for x in select_clause.split(" ")]
    select_clause = " ".join([x for x in select_clause if x != ""])
    select_clause = select_clause.replace("DISTINCT COUNT(?uri)", "COUNT(?uri)")

    where_clauses = re.findall(p_where, query)[0]
    where_clauses = where_clauses.strip(" ").strip(".").strip(" ")
    triples = [[y.strip(" ") for y in x.strip(" ").split(" ") if y != ""]
               for x in where_clauses.split(". ")]
    triples = [" ".join(["?x" if y[0] == "?" and y[1] == "x" else y for y in x]) for x in triples]
    where_clause = " . ".join(triples)
    query = select_clause + "{ " + where_clause + " }"
    return query

def query_answers(query, kb_endpoint):
    query = formalize(query)
    sparql = SPARQLWrapper(kb_endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(5)
    response = sparql.query().convert()

    if "ASK" in query:
        results = [str(response["boolean"])]
    elif "COUNT" in query:
        tmp = response["results"]["bindings"]
        assert len(tmp) == 1 and ".1" in tmp[0]
        results = [tmp[0][".1"]["value"]]
    else:
        tmp = response["results"]["bindings"]
        results = [x["uri"]["value"] for x in tmp]
    return results


if __name__ == '__main__':

    query_1 = "ASK WHERE { " \
               "<http://dbpedia.org/resource/James_Watt> <http://dbpedia.org/ontology/field> <http://dbpedia.org/resource/Mechanical_engineering> }"
    # query_2 = "SELECT DISTINCT COUNT(?uri) WHERE { " \
    #           "?x <http://dbpedia.org/property/partner> <http://dbpedia.org/resource/Dolores_del_R\u00edo> . " \
    #           "?uri <http://dbpedia.org/property/director> ?x  . " \
    #           "?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Film>}"
    # query_3 = "SELECT DISTINCT ?uri WHERE { " \
    #           "?x <http://dbpedia.org/property/partner> <http://dbpedia.org/resource/Dolores_del_R\u00edo> . " \
    #           "?uri <http://dbpedia.org/property/director> ?x  . " \
    #           "?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Film>}"

    #query_4 = "SELECT ?obj WHERE { wd:Q213611 p:P1411 ?s . ?s ps:P1411 ?obj . ?s pq:P805 wd:Q917076 }"

    kb_endpoint = "https://skynet.coypu.org/dbpedia/"
    #kb_endpoint = "https://query.wikidata.org/sparql"
    kb_query(query_1, kb_endpoint)
