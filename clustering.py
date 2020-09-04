# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 14:01:50 2020

@author: Shaila Sarker
"""
# Agglomerative or Hierarchical Clustering (called both)
import pandas as pd
Uni = pd.read_csv("D:/DS/4. University using Clustering/Universities.csv")

# --- The attribute "Expenses" has max influence and SFRatio has min influence, so we'll normalize the Dataframe---
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())  #[(X-X(max)) / X(max)-X(min)]
    return (x)
# run this func as a whole to avoid EOF
# Normalized data frame (considering the numerical part of data)
NormUni = norm_func(Uni.iloc[:,1:]) #[all rows, column1 to onwards(column0 contains uni_name)]

import matplotlib.pylab as plt
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
type(NormUni)
z = linkage(NormUni, method="complete",metric="euclidean") # z stores the euclidean distances to find clusters: sqrt[(a1-b1)^2 + (a2-b2)^2]

plt.figure(figsize=(15, 5));
plt.title('Hierarchical Clustering Dendrogram'); #helps to find out the number of major clusters
plt.xlabel('Index');
plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from sklearn.cluster import	AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity = "euclidean").fit(NormUni)
cluster_labels = pd.Series(h_complete.labels_) #shows which uni (according to index numbers) belongs to which cluster 
Uni['cluster'] = cluster_labels # creating a new column in the Uni Dataframe, named "cluster" and assigning it to the new column 
Uni = Uni.iloc[:,[7,0,1,2,3,4,5,6]] #rearrange the columns for convenience to find the clusters along with UniNames

# getting aggregate mean of each cluster
Uni.iloc[:,2:].groupby(Uni.cluster).mean() #to decied Tier1, Tier2 universities with the resulted avg or mean 
