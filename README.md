# ML-Improved-Search-Engine
A system application that customizes the elasticsearch search engine using Machine Learning methods for a book dataset


#### Data (https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)
- BX-Books.csv\
Our dataset of books and their attributes.
- BX-Users.csv\
The list of users, their id, location and age.
- BX-Book-Ratings.csv\
The list of ratings, containing user id, isbn and rating.

#### Installation
1. Download elasticsearch-8.5.2
2. Execute ../elasticsearch-8.5.2/bin/elasticsearch.bat and wait for it to initialize.
3. Execute load_books.py to insert the book contents to the elasticsearch client.

Now we are ready to perform the searches

### q1.py
This script gets a user_id and a query and creates a custom metrics, 
combining the elasticsearch score of the query results with the user rating score, so that a more personalied search is performed.

### q2.py
This script uses the **K-Means** algorithm to perform a classification on the user data based on their age and country of origin. Then it fills, for each user, the ratings of books that they have not rated, but the cluster in which they are classified has. We use the average rating of users in their cluster to fill the values. Now the personalization is more effective, because we use data from other users with similar characteristics

### q3.py
In this script I created a **neural network** that predicts the rating that a cluster of users would give to a book.
Firstly, the words in the already rated summaries have to be translated to Word Embeddings. The Word2Vec model is used, to create an 100-dimension vector for each word that will produce the 100-dimension summary (by taking the average of all word vectors). Then, the neural network is trained with the input vectors and the output ratings.
The trained network will now estimate a rating for every available book in the dataset.

As a result the final Search Engine will return a more personalized list of books, based on their summary.
