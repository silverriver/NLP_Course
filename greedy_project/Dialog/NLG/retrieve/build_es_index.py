from elasticsearch import Elasticsearch, helpers
import time
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--es_host_port', help='host and port for es', default='http://localhost:9200')
parser.add_argument('--index_name', help='name of the index', default='test_index')
parser.add_argument('--data_file', help='path to the data file', default='data\E-commerce-dataset\dev.txt')
args = parser.parse_args()
es = Elasticsearch(args.es_host_port)

with open(args.data_file, encoding='utf-8') as f:
    res = [i.strip().split('\t') for i in f.readlines()]
    res = [i for i in res if i[0] == '1' and len(i) == 3 and len(i[1]) > 0 and len(i[2]) > 0]

print(len(res))

index = {
    "mappings": {
        "properties": { 
            "query": { 
                "type": "text", 
                }, 
              "response": { 
                "type": "text", 
                } 
            } 
        } 
    }

es.indices.delete(index=args.index_name, ignore=[400, 404])
es.indices.create(index=args.index_name, body=index, ignore=[400])

actions = []
for i, (_, query, response) in enumerate(res):
    action = {
        "_index": args.index_name,
        "_id": i,
        "_source": {
            "query": query,
            "response": response,
        }
    }
    actions.append(action)
    if i % 100 == 0:
        helpers.bulk(es, actions)
        actions = []
        print(i)
    
helpers.bulk(es, actions)
print(es.indices.flush())

time.sleep(0.5)
print(es.count(index=args.index_name))
print('fin.')
