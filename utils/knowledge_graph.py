from typing import List, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib
import re

class KnowledgeGraph(object):

    def __init__(self, endpoint: str) -> None:
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)

    def is_entity(self, uri: str) -> bool:
        raise NotImplementedError()

    def is_predicate(self, uri: str) -> bool:
        raise NotImplementedError()

    def get_entity_name(self, entity: str, lang: str = "en") -> str:
        raise NotImplementedError()

    def get_relations(self, entity: str, limit: int = 100) -> List[str]:
        raise NotImplementedError()

    def get_tails(self, src: str, relation: str) -> List[str]:
        raise NotImplementedError()

    def get_single_tail_relation_triple(self, src: str) -> List[str]:
        raise NotImplementedError()

    def get_all_paths(self, src_: str, tgt_: str) -> List[List]:
        raise NotImplementedError()

    def get_one_hop_paths(self, src_: str) -> List[Tuple[str, str, str]]:
        raise NotImplementedError()

    def get_shortest_path_limit(self, src_: str, tgt_: str) -> List[List]:
        raise NotImplementedError()

    def search_one_hop_relations(self, src: str, tgt: str) -> List[Tuple[str]]:
        raise NotImplementedError()

    def search_two_hop_relations(self, src: str, tgt: str) -> List[Tuple[str, str]]:
        raise NotImplementedError()


class FreebaseKG(KnowledgeGraph):

    def __init__(self, endpoint: str) -> None:
        self.name = "freebase"
        super(FreebaseKG, self).__init__(endpoint)

    def get_entity_name(self, entity: str, lang: str = "en") -> str:
        entity = ':' + entity
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>

                SELECT ?t0 WHERE {{
                    {entity} :type.object.name ?t0 .
                    FILTER LANGMATCHES(LANG(?t0), '{lang}')
                }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        return results['results']['bindings'][0]['t0']['value']

    def get_relations(self, entity: str, limit: int = 100) -> List[str]:
        entity = ':' + entity
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>

                SELECT DISTINCT ?r0 WHERE {{
                    {entity} ?r0_ ?t0 .
                    BIND(STRAFTER(STR(?r0_),STR(:)) AS ?r0)
                }} LIMIT {limit}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        return [i['r0']['value'] for i in results['results']['bindings'] if i['r0']['value'] != 'type.object.type']

    def get_tails(self, src: str, relation: str) -> List[str]:
        src = ':' + src
        relation = ':' + relation
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>

                SELECT DISTINCT ?t0 WHERE {{
                    {src} {relation} ?t0_ .
                    BIND(STRAFTER(STR(?t0_),STR(:)) AS ?t0)
                }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        return [i['t0']['value'] for i in results['results']['bindings']]

    def get_single_tail_relation_triple(self, src: str) -> List[str]:
        src = ':' + src
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>

                SELECT DISTINCT ?r ?t0 WHERE {{
                    {src} ?r_ ?t0_ .
                    BIND(STRAFTER(STR(?r_),STR(:)) AS ?r)
                    BIND(STRAFTER(STR(?t0_),STR(:)) AS ?t0)
                }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        cnt = {}
        for i in results['results']['bindings']:
            if i['r']['value'] == 'type.object.type':
                continue
            if i['r']['value'] not in cnt:
                cnt[i['r']['value']] = 0
            if cnt[i['r']['value']] > 1:
                continue
            cnt[i['r']['value']] += 1
        return [k for k, v in cnt.items() if v == 1]

    def get_one_hop_paths(self, src_: str) -> List[Tuple[str, str, str]]:
        src = src_
        src = ":"+src
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>

                SELECT DISTINCT ?r0 ?t0 WHERE {{
                    {src} ?r0_ ?t0_ .
                    BIND(STRAFTER(STR(?r0_),STR(:)) AS ?r0)
                    BIND(STRAFTER(STR(?t0_),STR(:)) AS ?t0)
                }}
                """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except Exception as e:
            print(query)
            exit(0)

        return [(src, i['r0']['value'], i['t0']['value']) for i in results['results']['bindings'] if i['r0']['value'] != 'type.object.type']


    def get_all_paths(self, src_: str, tgt_: str) -> List[List]:
        src = src_
        tgt = tgt_
        src = ':' + src
        tgt = ':' + tgt
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>

                SELECT DISTINCT ?h0 ?r0 WHERE {{
                    ?h0_ ?r0_ {tgt} .
                    BIND(STRAFTER(STR(?r0_),STR(:)) AS ?r0)
                    BIND(STRAFTER(STR(?h0_),STR(:)) AS ?h0)
                }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        last_hop = [i for i in results['results']['bindings']
                    if i['r0']['value'] != 'type.object.type']

        one_hop = [[(src_, i['r0']['value'], tgt_)]
                   for i in last_hop if i['h0']['value'] == src_]

        # two hop
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>

                SELECT DISTINCT ?r0 ?e0 ?r1 ?e1 WHERE {{
                    {src} ?r0_ ?e0_ .
                    ?e0_ ?r1_ ?e1_ .
                    BIND(STRAFTER(STR(?r0_),STR(:)) AS ?r0)
                    BIND(STRAFTER(STR(?r1_),STR(:)) AS ?r1)
                    BIND(STRAFTER(STR(?e0_),STR(:)) AS ?e0)
                    BIND(STRAFTER(STR(?e1_),STR(:)) AS ?e1)
                }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        hop_twice = [i for i in results['results']['bindings']
                     if i['r0']['value'] != 'type.object.type' and i['r1']['value'] != 'type.object.type']

        two_hop = [[(src_, i['r0']['value'], i['e0']['value']), (i['e0']['value'], i['r1']['value'], tgt_)]
                   for i in hop_twice if i['e1']['value'] == tgt_]

        # three hop
        three_hop = []
        last_hop_dict = {}
        for i in last_hop:
            if i['h0']['value'] not in last_hop_dict:
                last_hop_dict[i['h0']['value']] = []
            last_hop_dict[i['h0']['value']].append(
                [(i['h0']['value'], i['r0']['value'], tgt_)]
            )

        for i in hop_twice:
            if i['e1']['value'] not in last_hop_dict:
                continue
            tmp = [
                (src_, i['r0']['value'], i['e0']['value']),
                (i['e0']['value'], i['r1']['value'], i['e1']['value'])
            ]
            for j in last_hop_dict[i['e1']['value']]:
                three_hop.append(tmp + j)
        return one_hop + two_hop + three_hop

    def get_shortest_path_limit(self, src_: str, tgt_: str) -> List[List]:

        src = src_
        tgt = tgt_
        src = ':' + src
        tgt = ':' + tgt

        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>

                SELECT DISTINCT ?r0 WHERE {{
                    {src} ?r0_ {tgt} .
                    BIND(STRAFTER(STR(?r0_),STR(:)) AS ?r0)
                }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        one_hop = [
            [(src_, i['r0']['value'], tgt_)] for i in results['results']['bindings']
            if i['r0']['value'] != 'type.object.type'
        ]

        if len(one_hop) > 0:
            return one_hop

        single_hop_relations = self.get_single_tail_relation_triple(src)
        two_hop = []
        for r0 in list(single_hop_relations):
            query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>

                SELECT DISTINCT ?e0 ?r1 WHERE {{
                    {src} ':{r0}' ?e0_ .
                    ?e0_ ?r1_ {tgt} .
                    BIND(STRAFTER(STR(?r1_),STR(:)) AS ?r1)
                    BIND(STRAFTER(STR(?e0_),STR(:)) AS ?e0)
                }}
            """
            self.sparql.setQuery(query)
            try:
                results = self.sparql.query().convert()
            except urllib.error.URLError:
                print(query)
                exit(0)

            two_hop += [
                [(src_, r0, i['e0']['value']), (i['e0']['value'], i['r1']['value'], tgt_)]
                for i in results['results']['bindings'] if i['r1']['value'] != 'type.object.type'
            ]

        return two_hop

    def search_one_hop_relations(self, src: str, tgt: str) -> List[Tuple[str]]:

        src = ':' + src
        tgt = ':' + tgt

        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>

                SELECT DISTINCT ?r1 WHERE {{
                    {src} ?r1_ {tgt} .
                    BIND(STRAFTER(STR(?r1_),STR(:)) AS ?r1)
                }}
        """

        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        return [(i['r1']['value']) for i in results['results']['bindings']]

    def search_two_hop_relations(self, src: str, tgt: str) -> List[Tuple[str, str]]:

        src = ':' + src
        tgt = ':' + tgt

        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>

                SELECT DISTINCT ?r1 ?r2 WHERE {{
                    {src} ?r1_ ?e1 .
                    ?e1 ?r2_ {tgt} .
                    BIND(STRAFTER(STR(?r1_),STR(:)) AS ?r1)
                    BIND(STRAFTER(STR(?r2_),STR(:)) AS ?r2)
                }}
        """

        self.sparql.setQuery(query)

        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        return [(i['r1']['value'], i['r2']['value']) for i in results['results']['bindings']]

    def deduce_subgraph_by_path_one(self, src: str, rels: List[str]):

        src = ':' + src

        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>

                SELECT DISTINCT ?e1 WHERE {{
                    {src}
                }}
        """


class WikidataKG(KnowledgeGraph):

    def __init__(self, endpoint: str) -> None:
        self.name = "wikidata"
        super(WikidataKG, self).__init__(endpoint)

    def is_uri(self, uri):
        URI_PATTERN = re.compile("[A-z]*://[A-z.-/#]+.*")

        if re.match(URI_PATTERN, uri):
            return True

        return False

    def is_entity(self, uri: str) -> bool:

        if not self.is_uri(uri):
            return False

        entity = uri.rsplit("/", 1)[1]
        ENTITY_PATTERN = re.compile('Q[0-9]+')

        if re.match(ENTITY_PATTERN, entity):
            return True

        return False

    def is_predicate(self, uri: str) -> bool:

        if not self.is_uri(uri):
            return False

        predicate = uri.rsplit("/", 1)[1]
        PREDICATE_PATTERN = re.compile('P[0-9]+')

        if re.match(PREDICATE_PATTERN, predicate):
            return True

        return False

    def get_entity_name(self, entity: str, lang: str = "en") -> str:
        entity = 'wd:' + entity
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX wd: <http://www.wikidata.org/entity/>

                SELECT ?t0 WHERE {{
                    {entity} rdfs:label ?t0 .
                    FILTER LANGMATCHES(LANG(?t0), '{lang}')
                }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        return results['results']['bindings'][0]['t0']['value']

    def get_relations(self, entity: str, limit: int = 100) -> List[str]:
        entity = 'wd:' + entity
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>

                SELECT DISTINCT ?r0 WHERE {{
                    {entity} ?r0_ ?t0 .
                    BIND(STRAFTER(STR(?r0_),STR(wdt:)) AS ?r0)
                }} LIMIT {limit}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        return [i['r0']['value'] for i in results['results']['bindings']]

    def get_tails(self, src: str, relation: str) -> List[str]:
        src = 'wd:' + src
        relation = 'wdt:' + relation
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>

                SELECT DISTINCT ?t0 WHERE {{
                    {src} {relation} ?t0_ .
                    BIND(STRAFTER(STR(?t0_),STR(wd:)) AS ?t0)
                }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        return [i['t0']['value'] for i in results['results']['bindings']]

    def get_single_tail_relation_triple(self, src: str) -> List[str]:
        src = 'wd:' + src
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>

                SELECT DISTINCT ?r ?t0 WHERE {{
                    {src} ?r_ ?t0_ .
                    BIND(STRAFTER(STR(?r_),STR(wdt:)) AS ?r)
                    BIND(STRAFTER(STR(?t0_),STR(wd:)) AS ?t0)
                }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        cnt = {}
        for i in results['results']['bindings']:
            if i['r']['value'] == '':
                continue
            if i['r']['value'] not in cnt:
                cnt[i['r']['value']] = 0
            if cnt[i['r']['value']] > 1:
                continue
            cnt[i['r']['value']] += 1
        return [k for k, v in cnt.items() if v == 1]

    def get_one_hop_paths(self, src_: str) -> List[Tuple[str, str, str]]:
        src = src_
        src = "wd:"+src
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>

                SELECT DISTINCT ?r0 ?t0 WHERE {{
                    {src} ?r0 ?t0 .
                }}
                """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except Exception as e:
            print(query)
            exit(0)

        rtn = list()

        for i in results['results']['bindings']:
            try:
                if i['r0']['type'] != "uri":
                    continue
                if not self.is_predicate(i['r0']['value']):
                    continue

                predicate = i['r0']['value'].rsplit("/", 1)[1]

                if i['t0']['type'] == "uri":
                    if not self.is_entity(i['t0']['value']):
                        continue
                    tail = i['t0']['value'].rsplit('/', 1)[1]
                else:
                    if i['t0']['type'] != "literal":
                        continue
                    tail = i['t0']['value']

                rtn.append((src_, predicate, tail))
            except Exception as e:
                print(e)
                print(i)

        return rtn

    def get_all_paths(self, src_: str, tgt_: str) -> List[List]:
        src = src_
        tgt = tgt_
        src = 'wd:' + src
        tgt = 'wd:' + tgt
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>

                SELECT DISTINCT ?h0 ?r0 WHERE {{
                    ?h0_ ?r0_ {tgt} .
                    BIND(STRAFTER(STR(?r0_),STR(wdt:)) AS ?r0)
                    BIND(STRAFTER(STR(?h0_),STR(wd:)) AS ?h0)
                }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except Exception as e:
            print(query)
            exit(0)

        last_hop = [i for i in results['results']['bindings']
                    if i['r0']['value'] != '' and i['h0']['value'] != '']

        one_hop = [[(src_, i['r0']['value'], tgt_)]
                   for i in last_hop if i['h0']['value'] == src_]

        # two hop
        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>

                SELECT DISTINCT ?r0 ?e0 ?r1 ?e1 WHERE {{
                    {src} ?r0_ ?e0_ .
                    ?e0_ ?r1_ ?e1_ .
                    BIND(STRAFTER(STR(?r0_),STR(wdt:)) AS ?r0)
                    BIND(STRAFTER(STR(?r1_),STR(wdt:)) AS ?r1)
                    BIND(STRAFTER(STR(?e0_),STR(wd:)) AS ?e0)
                    BIND(STRAFTER(STR(?e1_),STR(wd:)) AS ?e1)
                }}
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        hop_twice = [i for i in results['results']['bindings']
                     if i['r0']['value'] != '' and i['r1']['value'] != '']

        two_hop = [[(src_, i['r0']['value'], i['e0']['value']), (i['e0']['value'], i['r1']['value'], tgt_)]
                   for i in hop_twice if i['e1']['value'] == tgt_]

        # three hop
        three_hop = []
        last_hop_dict = {}
        for i in last_hop:
            if i['h0']['value'] not in last_hop_dict:
                last_hop_dict[i['h0']['value']] = []
            last_hop_dict[i['h0']['value']].append(
                [(i['h0']['value'], i['r0']['value'], tgt_)]
            )

        for i in hop_twice:
            if i['e1']['value'] not in last_hop_dict:
                continue
            tmp = [
                (src_, i['r0']['value'], i['e0']['value']),
                (i['e0']['value'], i['r1']['value'], i['e1']['value'])
            ]
            for j in last_hop_dict[i['e1']['value']]:
                three_hop.append(tmp + j)
        return one_hop + two_hop + three_hop

    def search_one_hop_relations(self, src: str, tgt: str) -> List[Tuple[str]]:

        src = 'wd:' + src
        tgt = 'wd:' + tgt

        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>

                SELECT DISTINCT ?r1 WHERE {{
                    {src} ?r1_ {tgt} .
                    BIND(STRAFTER(STR(?r1_),STR(wdt:)) AS ?r1)
                }}
        """

        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        return [(i['r1']['value']) for i in results['results']['bindings']]


    def search_two_hop_relations(self, src: str, tgt: str) -> List[Tuple[str, str]]:

        src = 'wd:' + src
        tgt = 'wd:' + tgt

        query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>

                SELECT DISTINCT ?r1 ?r2 WHERE {{
                    {src} ?r1_ ?e1 .
                    ?e1 ?r2_ {tgt} .
                    BIND(STRAFTER(STR(?r1_),STR(wdt:)) AS ?r1)
                    BIND(STRAFTER(STR(?r2_),STR(wdt:)) AS ?r2)
                }}
        """

        self.sparql.setQuery(query)

        try:
            results = self.sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)

        return [(i['r1']['value'], i['r2']['value']) for i in results['results']['bindings']]

if __name__ == '__main__':
    # freebase = FreebaseKG("https://skynet.coypu.org/freebase/")
    # print(freebase.get_entity_name("m.030qb3t"))
    # print(freebase.get_relations("m.030qb3t", 100))
    # print(freebase.get_tails("m.030qb3t", "location.location.people_born_here"))
    # print(freebase.get_single_tail_relation_triple("m.030qb3t"))
    # print(freebase.get_all_paths("m.030qb3t", "m.09c7w0"))

    #wikidata = WikidataKG("https://query.wikidata.org/")
    wikidata = WikidataKG("https://skynet.coypu.org/wikidata/")
    # print(wikidata.get_entity_name("Q3624078"))
    # print(wikidata.get_relations("Q3624078"))
    # print(wikidata.get_tails("Q3624078", "P6366"))
    # print(wikidata.get_single_tail_relation_triple("Q3624078"))
    print(wikidata.get_one_hop_paths("Q188920"))
    # print(wikidata.get_all_paths("Q188920", "Q135744"))
    # print(wikidata.search_one_hop_relations("Q188920", "Q32374551"))
    # print(wikidata.search_two_hop_relations("Q188920", "Q93662"))

    # print(wikidata.is_entity("http://www.wikidata.org/entity/Q13677"))
    # print(wikidata.is_predicate("http://www.wikidata.org/prop/direct/P2031"))