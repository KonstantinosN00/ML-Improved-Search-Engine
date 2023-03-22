import pandas as pd
import numpy as np
import string
from os.path import exists
from q1 import elastic_query, print_custom

from gensim.models import KeyedVectors, Word2Vec

from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
WV_NAME = "word2vec.wordvectors"


def text_to_vector(wv, text):
    text = text.split()
    vector = np.zeros(100)
    for word in text:
        try:
            vector += wv[word]
        except KeyError:
            pass
    return vector / len(text)


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return text


def train_word2vec_model(df):
    model = Word2Vec(df['cleaned_summary'], vector_size=100, window=5, min_count=1, workers=4)
    model.train(df['cleaned_summary'], total_examples=len(df['cleaned_summary']), epochs=10)
    print("Word2Vec model trained...")
    model.save('book_summary_word2vec.model')
    model.wv.save('word2vec.wordvectors')
    print("Word vectors saved for future use...")


def vectorize_summaries(df, using):
    df = df.drop(
        columns=['book_title', 'book_author', 'year_of_publication', 'publisher', 'category'])
    df['vector'] = df['summary'].apply(lambda x: text_to_vector(using, x))
    print("All summaries are now translated into vectors.")
    return df.drop(columns='summary')


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


def make_neural_network(dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(dim,)))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_ratings_model(model, df_vec_summaries, df_ratings, cluster):
    training_df = pd.merge(df_ratings.query("cluster_100==@cluster"), df_vec_summaries, how="inner", on="isbn")
    X = training_df['vector'].tolist()
    y = training_df['rating'].tolist()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    history = model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32,
                        validation_data=(np.array(X_val), np.array(y_val)))
    print(f"Model trained with cluster {cluster} ratings...")
    plot_loss(history)
    model.save("model_cluster_1.h5")
    return model


def predict_rating(df_vectors, isbn):
    summary_vector = df_vectors[df_vectors['isbn'] == isbn]['vector'].iloc[0].tolist()
    prediction = model.predict([summary_vector])
    if prediction[0][0] > 10: return 10
    if prediction[0][0] < 0: return 0
    return prediction[0][0]


def get_isbn_list(query_res):
    maxscore = query_res['hits']['max_score']
    books = query_res['hits']['hits']
    L = []
    for hit in books: L.append(hit['_source']['isbn'])
    return L


def new_search(vec_sum):
    uid = int(input("User id: "))
    while uid!=0:
        df_ratings = pd.read_csv("BX-Book-Ratings.csv").query("uid==@uid")

        query = input("Search for: ")
        res = elastic_query(query)
        isbns = get_isbn_list(res)
        for isbn in isbns:
            pred = predict_rating(vec_sum, isbn)
            data = {"uid":[uid],"isbn":[isbn],"rating":pred}
            df_ratings=pd.concat([df_ratings,pd.DataFrame(data)],ignore_index=True)
        print_custom(res, df_ratings)
        uid = int(input("User id: "))


if __name__ == "__main__":
    ratings_df = pd.read_csv('complete_ratings.csv')
    clusters_df = pd.read_csv('clustered_users.csv')
    ratings_df = pd.merge(ratings_df, clusters_df.drop(columns=['age', 'country']), how='left', on='uid')
    summaries_df = pd.read_csv('BX-Books.csv')

    if not exists(WV_NAME):
        # Clean Summaries
        summaries_df['cleaned_summary'] = summaries_df['summary'].apply(clean_text)
        print("Summaries cleaned!")
        train_word2vec_model(summaries_df)
    else:
        print("Word2Vec model and vectors already exists.\nLoading it...")
    wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')


    # Vectorize
    vectored_summaries_df = vectorize_summaries(summaries_df, using=wv)  # isbn, vector
    #vectored_summaries_df.to_csv("vectored_summaries.csv",index=False)
    print("Summary vectors created and saved.")

    # For example, suppose user 54 -> cluster=1
    # We use cluster-1 ratings to train our model,
    # and we save model because it takes 10 mins to train
    if not exists("model_cluster_1.h5"):
        model = make_neural_network(dim=100)
        model = train_ratings_model(model, vectored_summaries_df, ratings_df, 1)
    else:
        model = keras.models.load_model("model_cluster_1.h5", compile=False)

    new_search(vectored_summaries_df)

