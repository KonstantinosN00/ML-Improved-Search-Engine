from elasticsearch import Elasticsearch, helpers
import pandas as pd

def doc_generator(dataframe):
    df_iter = dataframe.iterrows()
    for index, document in df_iter:
        yield {
            "_index": 'books',
            "_source": document.to_dict(),
        }

if __name__ == "__main__":
    es = Elasticsearch(hosts="http://localhost:9200")
    df = pd.read_csv("BX-Books.csv")
    helpers.bulk(es, doc_generator(df))
    es.indices.put_settings(index="books", settings={'index.max_result_window': 20000})

