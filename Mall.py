import streamlit as st
import numpy as np  
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, rand_score
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')
#---------------------------------------------------------------------------------------------------------------------------------------
add_selectbox = st.header('Project Name: Mall Customer Segment!!!!')
add_selectbox = st.sidebar.header('Name:Ajinkya Pramod Chate')
add_selectbox = st.sidebar.header('Student ID: 41770')
add_selectbox = st.sidebar.markdown("------------------------------")

#----------------------------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('Mall_Customers.csv')
st.write(df.head())
#df.shape
#df.isnull().sum()
#df.info()
#---------------------------------------------------------------------------
st.write('Displot:')
fig=plt.figure(figsize = (10,5))
sns.distplot(df['Age'])
st.pyplot(fig)
fig=plt.figure(figsize = (10,5))
sns.distplot(df['Annual Income (k$)'])
st.pyplot(fig)
fig=plt.figure(figsize = (10,5))
sns.distplot(df['Spending Score (1-100)'])
st.pyplot(fig)
#-------------------------------------------------------------------------
gender = df.Gender.value_counts()
fig1=plt.figure(figsize=(10, 5))
sns.barplot(gender.index, gender)
plt.title('Barplot of gender', fontsize=20)
plt.xlabel('gender', fontsize=15)
plt.ylabel('count', fontsize=15);
st.pyplot(fig1)
#---------------------------------------------------------------------------
fig2=plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Matrix', fontsize=20);
st.pyplot(fig2)
#----------------------------------------------------------------------------
for column in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    fig=plt.figure(figsize=(10, 5))
    sns.boxplot(df[column])
    plt.title(column, fontsize=20);
    st.pyplot(fig)
#-------------------------------------------------------------------------
df_new = df[['Annual Income (k$)', 'Spending Score (1-100)']]
st.write(df_new.head())
#-----------------------------------------------------------------------------
X_train, X_test = train_test_split(df_new, random_state=0, test_size=0.2)
st.write("X_train shape: ")
st.write(X_train.shape)
st.write("X_test shape: ")
st.write(X_test.shape)
#------------------------------------------------------------------------------
min_max_sc = MinMaxScaler()

X_train = pd.DataFrame(min_max_sc.fit_transform(X_train), columns=df_new.columns)
X_test = pd.DataFrame(min_max_sc.transform(X_test), columns=df_new.columns)
###-------------------------------------------------------------------------------
##''' K-Means Clustering '''
##inertia = []
##silh_sc = []
##
##for i in range(2, 10):
##    kmeans = KMeans(n_clusters=i, init='k-means++')
##    ''' fit on data '''
##    kmeans.fit(X_train)
##    inertia.append(kmeans.inertia_)
##    
##    ''' silhoutte score'''
##    silh_sc.append(silhouette_score(X_train, kmeans.predict(X_train)))
##
##fig, ax1 = plt.subplots(figsize=(8, 5))
##fig.text(0.1, 1, 'Spending Score (1-100) and Annual Income (k$)', fontfamily='serif', fontsize=12, fontweight='bold')
##fig.text(0.1, 0.95, 'We want to select a point where Inertia is low & Silhouette Score is high, and the number of clusters is not overwhelming for the business.',
##         fontfamily='serif',fontsize=10)
##fig.text(1.4, 1, 'Inertia', fontweight="bold", fontfamily='serif', fontsize=15, color='#244747')
##fig.text(1.51, 1, "|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
##fig.text(1.53, 1, 'Silhouette Score', fontweight="bold", fontfamily='serif', fontsize=15, color='#91b8bd')
##
##ax1.plot(range(2,10), inertia, '-', color='#244747', linewidth=5)
##ax1.plot(range(2,10), inertia, 'o', color='#91b8bd')
##ax1.set_ylabel('Inertia')
##
##ax2 = ax1.twinx()
##ax2.plot(range(2,10), silh_sc, '-', color='#91b8bd', linewidth=5)
##ax2.plot(range(2,10), silh_sc, 'o', color='#244747', alpha=0.8)
##ax2.set_ylabel('Silhouette Score')
##
##plt.xlabel('Number of clusters')
##plt.show()

#----------------------------------------------------------------------------------------------------------------------------------

kmeans_model = KMeans(n_clusters=5, init='k-means++', random_state=111, algorithm='elkan')
pred = kmeans_model.fit_predict(X_test)

fig=plt.figure(figsize=(10,5))
sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=pred, palette=sns.color_palette('hls',len(np.unique(pred))), s=100)
plt.title('Cluster of Customers'.format(X_test.columns[0], X_test.columns[1]), size=15, pad=10)
plt.xlabel(X_test.columns[0], size=12)
plt.ylabel(X_test.columns[1], size=12)
plt.legend(loc=0, bbox_to_anchor=[1,1])
plt.show()
st.pyplot(fig)
###---------------------------------------------------------------------------------------------------------------------------------
##''' using 6 clusters '''
##kmeans_model = KMeans(n_clusters=6, init='k-means++', random_state=19, algorithm='elkan')
##''' prediction '''
##pred = kmeans_model.fit_predict(X_test)
##fig = px.scatter_3d(X_test, x="Annual Income (k$)", y="Spending Score (1-100)", z="Age", color=pred, opacity=0.8, size=pred+1)
##st.pyplot(fig)
#---------------------------------------------------------------------------------------------------------------------------------------------
model = NearestNeighbors(n_neighbors=2)
neighbours = model.fit(df_new)
dist, idx = neighbours.kneighbors(df_new)

dist = np.sort(dist, axis=0)
dist = dist[:,1]

fig=plt.figure(figsize=(10,5))
plt.text(-10, 17, 'TK-distance Graph', fontfamily='serif', fontsize=15, fontweight='bold')
plt.text(-10, 16, 'The optimum value of epsilon is at the point of maximum curvature in the K-Distance Graph, which is 6 in this case.',
        fontfamily='serif', fontsize=12)
plt.plot(dist)
plt.xlabel('Data Points sorted by distance', fontsize=14)
plt.ylabel('Epsilon', fontsize=14)
st.pyplot(fig)
#----------------------------------------------------------------------------------------------------------
model_dbscan = DBSCAN(eps=6, min_samples=3)
pred = model_dbscan.fit_predict(df_new)

fig=plt.figure(figsize=(10,5))
sns.scatterplot(x=df_new.iloc[:, 0], y=df_new.iloc[:, 1], hue=pred, palette=sns.color_palette('hls', 
                                                                                              len(np.unique(pred))), s=100)
plt.title('Cluster of Customers'.format(df_new.columns[0], df_new.columns[1]), size=15, pad=10)
plt.xlabel(df_new.columns[0], size=12)
plt.ylabel(df_new.columns[1], size=12)
plt.legend(loc=0, bbox_to_anchor=[1,1])
st.pyplot(fig)

#---------------------------------------------------------------------------------------------------
fig=plt.figure(figsize = (10, 5))
plt.text(5, 465, 'Spending Score (1-100) and Annual Income (k$)', fontfamily='serif', fontsize=15, fontweight='bold')
plt.text(5, 440, 'The no. of clusters is the no. of vertical lines in the dendrogram cut by a horizontal line that can transverse the maximum distance vertically without intersecting a cluster.',
         fontfamily='serif',fontsize=12)
dendo = dendrogram(linkage(df_new, method = 'ward'))
plt.plot([115]*2000, color='r')
plt.plot([240]*2000, color='r')
plt.text(5, -50, 'Here, we can have either 5 clusters or 3 clusters',fontfamily='serif',fontsize=12)
st.pyplot(fig)
#------------------------------------------------------------------------------------------

model = AgglomerativeClustering(n_clusters = 5, affinity='euclidean', linkage='ward')
pred = model.fit_predict(df_new)

fig=plt.figure(figsize=(10,5))
sns.scatterplot(x=df_new.iloc[:, 0], y=df_new.iloc[:, 1], hue=pred, palette=sns.color_palette('hls', 5), s=100)
plt.title('Cluster of Customers'.format(df_new.iloc[:, 0], df_new.iloc[:, 1]), size=15, pad=10)
plt.xlabel(df_new.columns[0], size=12)
plt.ylabel(df_new.columns[1], size=12)
plt.legend(loc=0, bbox_to_anchor=[1,1])
st.pyplot(fig)
