# -*- coding: utf-8 -*-
"""
This script runs each test sample in four algorithms,
If we got 2-2 (teko) we return unknown classification
Created on Tue Dec 19 00:19:37 2023

@author: andre
"""
import re, seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go


import warnings
warnings.filterwarnings("ignore")

dataframe = pd.read_csv("2 person lengths.csv").iloc[:,1:]

corr = dataframe.corr()
#dataframe = dataframe[["Ankle to Heel (L)", "Hip to Knee (L)", "Heel to Foot (R)", "Knee to Ankle (R)", "label"]]

X = dataframe.iloc[:,:-1]
y = dataframe.iloc[:,-1]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

test_sample = np.array([[25.816, 28.625, 65.092, 42.153, 53.255, 5.369, 30.947, 31.661, 27.891, 57.535, 46.904, 51.274, 5.95, 25.42]])


#X_test = X_test.to_numpy()

#X = X.to_numpy()

if test_sample in X.to_numpy():
    print("yessssss")
    

accuracy = []
index = 0
unknown_class = 0

    
for test_sample in tqdm(X_test.iloc[:1,:]):
    
    #test_sample = test_sample.reshape(1,14)
    temp_results = []
    
    test_sample = [[25.816, 28.625, 65.092, 42.153, 53.255, 5.369, 30.947, 31.661, 27.891, 57.535, 46.904, 51.274, 5.95, 25.42]]
    
    
    clf = GaussianNB()
    y_pred_nb = clf.fit(X_train, y_train).predict(test_sample)
    temp_results.append(y_pred_nb.max())
    
    clf = tree.DecisionTreeClassifier()
    y_pred_DT = clf.fit(X_train, y_train).predict(test_sample)
    temp_results.append(y_pred_DT.max())
    
    clf = RandomForestClassifier()
    y_pred_RF = clf.fit(X_train, y_train).predict(test_sample)
    temp_results.append(y_pred_RF.max())
    
    clf = AdaBoostClassifier()
    y_pred_AdaBoost = clf.fit(X_train, y_train).predict(test_sample)
    temp_results.append(y_pred_AdaBoost.max())
    
    if (sum(temp_results)!=0 and sum(temp_results)!=4):
        unknown_class+=1
        
    accuracy.append(temp_results)

print("Number of Unknown classes == 0: ", unknown_class)

"""
predict_prob = clf.predict_proba([[25.816, 28.625, 65.092, 42.153, 53.255, 5.369, 30.947, 31.661, 27.891, 57.535, 46.904, 51.274, 5.95, 25.42]])

print("Predict Proba:", predict_prob)
"""





### Scatter Plot: ##################################################################

"""
plt.scatter(dataframe["Hip to Knee (L)"], dataframe["Hip to Knee (R)"], c = dataframe["label"], cmap= "rainbow")
plt.title("Hip to Knee (L) vs. Knee to Ankle (R) Scatter Plot")
plt.xlabel("Hip to Knee (L)")
plt.ylabel("Knee to Ankle (R)")
plt.show()

sns.jointplot(data=dataframe, x="Hip to Knee (L)", y="Hip to Knee (R)", hue = y)
#sns.jointplot(data=dataframe, x="Hip to Knee (L)", y="Hip to Knee (R)", kind = "hex")

plt.show()

### Line Pllot: ##################################################################



plt.plot(dataframe["Hip to Knee (L)"])
plt.show()
"""

### T-SNE: ##################################################################
from sklearn.manifold import TSNE
#import plotly.express as px

andre_sample = [[25.816, 28.625, 65.092, 42.153, 53.255, 5.369, 30.947, 31.661, 27.891, 57.535, 46.904, 51.274, 5.95, 25.42]]
X_test = np.vstack((X_test,andre_sample))

scaler = MinMaxScaler()

column_names = X.columns
X_test_scaled = scaler.fit_transform(X_test)
#X_test_scaled = pd.DataFrame(X_test_scaled, index=column_names, columns=column_names)


andre_sample_y = np.array([[2]])

y_test = y_test.to_numpy().reshape(-1,1)
y_test = np.vstack((y_test, andre_sample_y)).reshape(-1,1)

tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X_test_scaled)
tsne.kl_divergence_



scaler = MinMaxScaler()
X_tsne_scaled = scaler.fit_transform(X_tsne)



plt.scatter(x=X_tsne_scaled[:, 0], y=X_tsne_scaled[:, 1], c=y_test, cmap= "rainbow")
"""fig.update_layout(
    title="t-SNE visualization of Custom Classification dataset",
    xaxis_title="First t-SNE",
    yaxis_title="Second t-SNE",
)"""
plt.show()


col1 = X_tsne_scaled[:, 0]
col2 = X_tsne_scaled[:, 1]
col3 = X_tsne_scaled[:, 2]


print("shape", np.shape(y))

test= np.array([col1, col2, col3]).reshape(len(X_test),3)


X_tsne_pd = pd.DataFrame(test, columns=['a', 'b', 'c'])

x1 = X_tsne_pd.iloc[:,0].tolist()
y1 = X_tsne_pd.iloc[:,1].tolist()
z1 = X_tsne_pd.iloc[:,2].tolist()


print(type(X_tsne_pd["a"]))                   
print(type(X_tsne_pd["b"])) 
print(type(y))          


# axes instance
fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)


# plot
sc = ax.scatter(x1, y1, z1, s=40, c=y_test, cmap = "viridis", marker='o', alpha=1)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

# save
#plt.savefig("scatter_hue", bbox_inches='tight')


         

for i in range(len(X_test_scaled[0,:])-2,): # iterate through all columns
    

    x_column_name = column_names[i]
    y_column_name = column_names[i+1]
    z_column_name = column_names[i+2]


    x1 = X_test_scaled[:,i].tolist()
    y1 = X_test_scaled[:,i+1].tolist()
    z1 = X_test_scaled[:,i+2].tolist()

    # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    

    # plot
    sc = ax.scatter(x1, y1, z1, s=40, c=y_test, cmap = "viridis", marker='o', alpha=1)
    ax.set_xlabel(x_column_name)
    ax.set_ylabel(y_column_name)
    ax.set_zlabel(z_column_name)
    
    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    

    """
       
    fig = go.Figure(data=[go.Scatter3d(
        x=x1,
        y=y1,
        z=z1,
        mode='markers',
        marker=dict(
            size=12,
            color=z1,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )])
    
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    #fig.show(renderer = "browser")
    
    """

    """
    # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    
    
    # plot
    sc = ax.scatter(x1, y1, z1, s=40, c=y_test, cmap = "viridis", marker='o', alpha=1)
    ax.set_xlabel(x_column_name)
    ax.set_ylabel(y_column_name)
    ax.set_zlabel(z_column_name)
    
    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    
    # save
    #plt.savefig("scatter_hue", bbox_inches='tight')
    """

    plt.show()











"""

sns.jointplot(data=X_tsne_pd, x='a', y='b', hue = y) # , hue = y
#sns.jointplot(data=dataframe, x="Hip to Knee (L)", y="Hip to Knee (R)", kind = "hex")

plt.show()

"""

def person_classification_by_nearest_centroid(dataframe, new_sample, number_of_persons = 1):
    
        
    kmeans = KMeans(n_clusters=number_of_persons)
    kmeans.fit(dataframe)
    
    print(kmeans.predict(new_sample))
    



corr = dataframe.corr()





