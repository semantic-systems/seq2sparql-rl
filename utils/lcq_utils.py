

import KB_query


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
            if "_" in ans:
                ans = " ".join(ans.split("_"))
            gold_answers.append(ans)

    return gold_answers


if __name__ == '__main__':
    #query_1 = "SELECT DISTINCT ?uri WHERE { ?x <http://dbpedia.org/property/international> <http://dbpedia.org/resource/Muslim_Brotherhood> . ?x <http://dbpedia.org/ontology/religion> ?uri  . ?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/PoliticalParty>}"
    query_1 = "ASK WHERE { <http://dbpedia.org/resource/New_Way_(Israel)> <http://dbpedia.org/ontology/mergedIntoParty> <http://dbpedia.org/resource/One_Israel> }"
    kb_endpoint = "https://dbpedia.org/sparql"
    extracted_answers = KB_query.kb_query(query_1, kb_endpoint)
    gold_answers = get_gold_answers(extracted_answers)
    print("asd")
