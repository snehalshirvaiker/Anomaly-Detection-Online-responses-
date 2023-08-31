#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import docx2txt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import pandas as pd


# In[12]:


# set up the NLTK stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')


# In[ ]:





# In[13]:


# define the path to your Word documents
path_to_docs = "C:/Users/hrush/Downloads/Prashant sir question 7 docs-20230417T074201Z-001/Prashant sir question 7 docs"


# In[14]:


# read the documents and preprocess the text
docs = []
doc_names = []
for file_name in os.listdir(path_to_docs):
    if file_name.endswith('.docx'):
        doc_text = docx2txt.process(os.path.join(path_to_docs, file_name))
        doc_text = doc_text.lower()
        doc_text = ' '.join([word for word in doc_text.split() if word not in stop_words])
        docs.append(doc_text)
        doc_names.append(file_name)


# In[15]:


len(docs)
print(docs)


# In[16]:


vectorizer = TfidfVectorizer(max_features=100000, max_df=0.5, smooth_idf=True)
X = vectorizer.fit_transform(docs)
print(X)
X.shape


# In[17]:


# Access the vocabulary
vocabulary = vectorizer.vocabulary_
sorted_voca=sorted(vocabulary.items(),key=lambda x:x[1])
# Print the vocabulary
print("Vocabulary:")
for term, index in sorted_voca:
    print(f"Term: '{term}', Index: {index}")


# In[18]:


num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, init='random', max_iter=100, n_init=10,random_state=12)
kmeans.fit(X)


# In[19]:


# print out the document names in each cluster
for i in range(num_clusters):
    cluster_docs = [doc_names[j] for j in range(len(doc_names)) if kmeans.labels_[j] == i]
    print(f'Cluster {i+1}: {", ".join(cluster_docs)}')


# In[20]:


# visualize the clusters using PCA and t-SNE
pca = PCA(n_components=2).fit(X.toarray())
data2D = pca.transform(X.toarray())

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=1).fit_transform(X.toarray())

plt.figure(figsize=(10, 10))
plt.scatter(data2D[:, 0], data2D[:, 1], c=kmeans.labels_)
plt.title('PCA Clustering')
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(tsne[:, 0], tsne[:, 1], c=kmeans.labels_)
plt.title('t-SNE Clustering')
plt.show()


# In[21]:


Z = linkage(X.todense(), method='ward')
fig, ax = plt.subplots(figsize=(30, 18))
dendrogram(Z, labels=doc_names)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Documents')
plt.ylabel('Distance')
plt.show()
plt.savefig('C:/Hrushi/M.Sc SEM 2/Project/Offline Results/dendrogram.pdf')


# In[12]:


cosine_similarity_matrix = cosine_similarity(X)

# create a dictionary to store the similarity scores for each document pair
similarity_scores = {}
for i in range(len(docs)):
    for j in range(i+1, len(docs)):
        similarity_scores[(doc_names[i], doc_names[j])] = cosine_similarity_matrix[i, j]

# sort the similarity scores in descending order
sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

# print the sorted similarity scores
for pair, score in sorted_scores:
    doc1, doc2 = pair
    percent_similarity = round(score * 100, 2)
    print(f"Similarity between {doc1} and {doc2}: {percent_similarity}%")


# In[13]:


df = pd.DataFrame(sorted_scores, columns=['Document Pair', 'Similarity Score'])

# add columns for individual document names and similarity percentages
df[['Document 1', 'Document 2']] = pd.DataFrame(df['Document Pair'].tolist(), index=df.index)
df['Similarity Percentage'] = round(df['Similarity Score'] * 100, 2)

# drop the 'Document Pair' column
df = df.drop(columns=['Document Pair'])

# print the dataframe
print(df)


# In[14]:


df.to_csv('C:/Hrushi/M.Sc SEM 2/Project/Offline Results/similarity_scores.csv', index=False)


# In[15]:


labels = kmeans.labels_

# compute the cluster centroids
centroids = kmeans.cluster_centers_


# In[16]:


for cluster in range(num_clusters):
    indices = [i for i, x in enumerate(labels) if x == cluster]
    cluster_docs = [docs[i] for i in indices]
    cluster_names = [doc_names[i] for i in indices]
    cluster_X = vectorizer.transform(cluster_docs)
    similarity_matrix = cosine_similarity(cluster_X)
    print(f"Cluster {cluster}:")
    for i in range(len(cluster_docs)):
        for j in range(i+1, len(cluster_docs)):
            similarity_score = similarity_matrix[i, j]
            doc1_name = cluster_names[i]
            doc2_name = cluster_names[j]
            similarity_percentage = similarity_score * 100
            print(f"Cosine similarity between '{doc1_name}' and '{doc2_name}': {similarity_percentage:.2f}%")


# In[17]:


similarity_scores = {}
for cluster in range(num_clusters):
    indices = [i for i, x in enumerate(labels) if x == cluster]
    cluster_docs = [docs[i] for i in indices]
    cluster_names = [doc_names[i] for i in indices]
    cluster_X = vectorizer.transform(cluster_docs)
    similarity_matrix = cosine_similarity(cluster_X)
    cluster_similarity_scores = []
    for i in range(len(cluster_docs)):
        for j in range(i+1, len(cluster_docs)):
            similarity_score = similarity_matrix[i, j]
            doc1_name = cluster_names[i]
            doc2_name = cluster_names[j]
            similarity_percentage = similarity_score * 100
            if similarity_percentage > 85:
                cluster_similarity_scores.append((doc1_name, doc2_name, similarity_percentage))
    similarity_scores[f"Cluster {cluster}"] = cluster_similarity_scores
for cluster, scores in similarity_scores.items():
    print(f"\n{cluster}:")
    if scores:
        for score in scores:
            print(f"Cosine similarity between '{score[0]}' and '{score[1]}': {score[2]:.2f}%")
    else:
        print("No similar documents found in this cluster.")


# In[20]:


df_clus = pd.DataFrame.from_dict({(i,j): similarity_scores[i][j] 
                             for i in similarity_scores.keys() 
                             for j in range(len(similarity_scores[i]))},
                            orient='index',
                            columns=['Doc1', 'Doc2', 'Similarity(%)'])

df_clus.index.names = ['Cluster,pair no.',]
df_clus.reset_index(inplace=True)
print(df_clus)


# In[22]:


df_clus.to_csv('C:/Hrushi/M.Sc SEM 2/Project/Offline Results/cluster_similarity_scores.csv', index=False)


# In[24]:


from scipy.spatial.distance import pdist, squareform

# compute pairwise distances between documents
distances = pdist(X.toarray(), metric='euclidean')

# convert the pairwise distances to a square distance matrix
distance_matrix = squareform(distances)

# print the distance matrix
print(distance_matrix)


# In[28]:


distance_matrix.shape


# In[ ]:





# In[ ]:




