# recommendation_system
recommendation system for podcast

Podcast is a word that combines pod of iPod and cast of broadcast, and it is a radio that you listen to on your smartphone. Episodes of desired programs can be stored, and the program is recommended by reflecting users and content records.
![image](https://user-images.githubusercontent.com/71868763/235978829-fbfca65d-87f2-4eea-b2ce-8d20deb4fdfd.png)

We make a recommendation system of Podcast with below methods.
-	Content based filtering (k-modes, W2V, TF-IDF)
-	Collaborative filtering (SVD & Matrix Factorization)
And we will compare these three methods to determine which method is suitable as a recommendation system.

To compare the performance of the models, we entered an episodes of the same name for content based methods. In content based methods, we use Tfidfvectorizer and Count vectorizer and W2V. 

1. Data Exploration
show.csv 
Podcasts Episodes (2007-2016) | Kaggle
![image](https://user-images.githubusercontent.com/71868763/235979193-069365f3-1aeb-439f-adf5-2c17175a06cb.png)

This data set contains Podcast episodes published between 2007 and 2016. In this dataset, we will find over 30,000 episodes of several different podcasts shows.

![image](https://user-images.githubusercontent.com/71868763/235979013-e9863c1c-8e04-449e-a9c8-6ed26e0c1ffd.png)
![image](https://user-images.githubusercontent.com/71868763/235979067-8e2a9bcf-d908-4412-939f-9ab512498654.png)


2. Modeling<br/>Content Based Filtering
  - K-modes
  - W2V
  - TF-IDF
<br/>Collaborative Filtering
   - SVD & Matrix Factorization


3. Model Evaluation and Analysis
