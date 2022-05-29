from elasticsearch import Elasticsearch, helpers
import time
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--es_host_port', help='host and port for es', default='http://localhost:9200')
parser.add_argument('--index_name', help='name of the index', default='test_index')
parser.add_argument('--query', help='the query for the search', default='什么 时候 发货')
args = parser.parse_args()
es = Elasticsearch(args.es_host_port)

body = {"query": {
    "bool": {
      "should": [
        {"match": {"query": args.query}},
      ]
    }
  }}

resp = es.search(index=args.index_name, body=body, size=10)
print(f'query: {args.query}')
print(f'--------results----------')
count = 0
for i in resp['hits']['hits']:
    print(f'query{count}: {i["_source"]["query"]}')
    print(f'response{count}: {i["_source"]["response"]}')
    count += 1
    