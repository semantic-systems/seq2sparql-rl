
import KB_query

def get_ent_label(answers, kb_endpoint):
    gold_answer_texts = []

    for ans in answers:
        if type(ans) == bool:
            gold_answer_texts.append(str(ans))
            continue
        label = KB_query.query_ent_label(ans, kb_endpoint)
        gold_answer_texts.append(label)

    return gold_answer_texts


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
