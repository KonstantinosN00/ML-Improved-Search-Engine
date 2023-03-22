import pandas as pd
from elasticsearch import Elasticsearch, helpers

PERCENTAGE=0.01
es = Elasticsearch(hosts="http://localhost:9200")


def elastic_query(search_term):
    total = es.count(index="books")['count']
    # Return info for 10% of total books
    resp = es.search(index="books", query={"query_string": {"query": search_term}},size=total*PERCENTAGE)
    return resp

def aggregate(book,df,maxscore):
    scaled_score=book['_score']*10/maxscore
    isbn=book['_source']['isbn']

    user_rating = df.query(f"isbn=='{isbn}'")

    if user_rating.empty: agg_rating = scaled_score / 2
    else:
        rating = user_rating.iloc[0]['rating']
        agg_rating = scaled_score / 2 + rating / 2
    book['custom_rating'] = round(agg_rating,3)
    return agg_rating

def print_default(query_res):
    print("Got %d Hits" % (query_res['hits']['total']['value']))
    print("Top 10% of books are displayed below:")
    for i, hit in enumerate(query_res['hits']['hits']):

        print(f"{i + 1}. Score: {hit['_score']}",
              " %(book_title)s, %(book_author)s: %(year_of_publication)s" % hit["_source"])

# Get elasticsearch results, user id, ratings dataframe and print custom order of results
def print_custom(query_res,df):
    maxscore = query_res['hits']['max_score']
    books=query_res['hits']['hits']
    srt = sorted(books, key=lambda book: aggregate(book, df, maxscore), reverse=True)

    print("Got %d Hits" % (query_res['hits']['total']['value']))
    print("Top 10% of books are displayed below:")
    for i, hit in enumerate(srt):
        print(f"{i+1}. Score: {hit['custom_rating']}",
              " %(book_title)s, %(book_author)s: %(year_of_publication)s" % hit["_source"])

if "__main__" == __name__:
    # 6575,0449005410,9 Horse

    uid = int(input("User id: "))
    df_ratings = pd.read_csv("BX-Book-Ratings.csv").query("uid==@uid")
    # TEST QUESTION 2
    # df_ratings = pd.read_csv("complete_ratings.csv").query("uid==@uid")
    query = input("Search for: ")
    res = elastic_query(query)
    #print_default(res)
    print_custom(res,df_ratings)

