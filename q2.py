import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

from os.path import exists
K=100

def preprocessing(df):
    # We get 2 columns from right splitting and we keep the 2nd: country
    # we assign it to column 'country'
    df['country'] = df['location'].str.rsplit(pat=',', n=1, expand=True)[1].str.strip('\"\' ')
    df.dropna(inplace=True)
    # Many users were over 110 years old
    df = df[df['age'] < 100].copy()

    # One-hot encoding the countries
    encoder = OneHotEncoder(handle_unknown="ignore", categories="auto")
    country_enc = encoder.fit_transform(df[['country']])
    df[['onehot']] = country_enc
    # Combine the ages and one-hot encoded countries
    data = np.hstack((df[['age']].values, country_enc.toarray()))
    return df, data


def plot_country_age(df, k):
    # un=df['country'].unique()
    # mapping ={}
    # for i,cn in enumerate(un):
    #     mapping[cn] = i
    # df['code'] = df['country'].apply(lambda x : mapping[x])
    plt.scatter(x=df['age'], y=df.index, c=df[f'cluster_{k}'], s=10)
    plt.xlabel("Age")
    plt.ylabel("Uid")
    plt.grid(True)
    plt.show()


def optimise_k_means(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(data)
        means.append(k)
        inertias.append(kmeans.inertia_)

    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


def apply_kmeans(k, data, df):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(data)

    # Get the cluster labels for each data point
    df[f'cluster_{k}'] = kmeans.labels_
    clustered_users = df.drop(columns=['location', 'onehot'])
    clustered_users.to_csv('clustered_users.csv')


def step1():
    df_users = pd.read_csv("BX-Users.csv", index_col='uid')
    df_users, data = preprocessing(df_users)
    # optimise_k_means(data,50)
    apply_kmeans(K, data, df_users)
    plot_country_age(df_users, K)


def fill_all():
    users = pd.read_csv('clustered_users.csv').dropna()
    rat = pd.read_csv('BX-Book-Ratings.csv')
    # keep useful ratings their user clusters
    ratings = pd.merge(rat, users[['uid', f'cluster_{K}']], how="inner", on=["uid"])
    clusterbooks = ratings.groupby([f'cluster_{K}', 'isbn'])['rating'].mean() \
        .reset_index()  # cluster, isbn, rating

    for c in range(K):
        userdata = ratings[ratings[f'cluster_{K}'] == c].drop(columns=[f'cluster_{K}'])  # uid, isbn, rating
        clusterdata = clusterbooks[clusterbooks[f'cluster_{K}'] == c][['isbn', 'rating']]  # isbn, rating

        uids = pd.Series(users[users[f'cluster_{K}'] == c]['uid'].unique(), name='uid')
        isbns = pd.Series(clusterbooks[clusterbooks[f'cluster_{K}'] == c]['isbn'].unique(), name='isbn')
        full_table = pd.merge(uids, isbns, how='cross')  # full table of uid-isbn for every user and book in the cluster
        full_table = pd.merge(full_table, userdata, how='left', on=['uid', 'isbn'])  # uid, isbn, rating

        # replace NaN with values from clusters
        to_fill = full_table[full_table['rating'].isna()][['uid', 'isbn']]
        filled = pd.merge(to_fill, clusterdata, how='left', on='isbn')
        print(f"Filled cluster {c} users..")
        rat = pd.concat([rat, filled])
    rat.to_csv("complete_ratings.csv",index=False)

if __name__ == '__main__':
    if not exists("clustered_users.csv"): step1()
    fill_all()

