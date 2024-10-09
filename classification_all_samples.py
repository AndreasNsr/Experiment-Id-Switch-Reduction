# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 00:19:37 2023

@author: andre
"""
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
import plotly.express as px

import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, silhouette_score
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, estimate_bandwidth
from sklearn.ensemble import AdaBoostClassifier
from mpl_toolkits.mplot3d import Axes3D
import os

from scipy.stats import gaussian_kde
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import adjusted_rand_score

import warnings


os.environ["OMP_NUM_THREADS"] = "3"
warnings.filterwarnings("ignore", module = "sklearn")
warnings.filterwarnings("ignore", module = "pandas")




dataframe = pd.read_csv("2 sides MOT16 det_SEP.csv")
dataframe.replace(0, np.nan, inplace = True)


unique_GT_tracks = np.unique(dataframe["GT_Cluster"])


all_landmarks_short_names_list = [   
    
   "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",  "LEFT_SHOULDER", "LEFT_HIP", 
  
  "LEFT_KNEE", "LEFT_ANKLE", "LEFT_HEEL", "LEFT_FOOT_INDEX", 
  
  "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_SHOULDER", "RIGHT_HIP", 
  
  "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_HEEL", "RIGHT_FOOT_INDEX"
  
  
  ]







"""
dataframe.replace(0, np.nan, inplace = True)
dataframe_dropped_na = dataframe.dropna()

left_side_dropped_na = dataframe_dropped_na.iloc[:,:7]
left_side_dropped_na.columns = ['Sho to Elb','Elb to Wrist','Sho to Hip', 'Hip to Knee', 'Knee to Ankle', 'Ankle to Heel', 'Heel to Foot']
left_side_dropped_na["Side"] = "left"



right_side_dropped_na = dataframe_dropped_na.iloc[:,7:-1]
right_side_dropped_na.columns = ['Sho to Elb','Elb to Wrist','Sho to Hip', 'Hip to Knee', 'Knee to Ankle', 'Ankle to Heel', 'Heel to Foot'] 
right_side_dropped_na["Side"] = "right"

dataframe_dropped_na = pd.concat([left_side_dropped_na, right_side_dropped_na], ignore_index=True, axis=0)

dataframe_dropped_cols = dataframe_dropped_na.columns

for i in range(7):
    
    sns.set(font_scale=1)
    
    sns.displot(dataframe_dropped_na, x=dataframe_dropped_cols[i], hue="Side", kind="kde", fill=True)




    plt.show()



dataframe.replace(0, np.nan, inplace = True)

dataframe = dataframe.dropna()


print(dataframe.isna().sum(), "\n")


dataframe2 = dataframe.dropna(thresh=dataframe.shape[0]*0.7,how='all',axis=1, inplace= False)


print(dataframe2.isna().sum())
"""

columns = ['Sho to Elb (L)','Elb to Wrist (L)','Sho to Hip (L)', 'Hip to Knee (L)', 'Knee to Ankle (L)', 'Ankle to Heel (L)', 'Heel to Foot (L)',
           'Sho to Elb (R)','Elb to Wrist (R)','Sho to Hip (R)', 'Hip to Knee (R)', 'Knee to Ankle (R)', 'Ankle to Heel (R)', 'Heel to Foot (R)',
           "GT_Cluster", "Track ID"]


GT_Cluster = dataframe["GT_Cluster"]
track_id_column = dataframe["Track ID"]


dataframe = dataframe.groupby("Track ID").transform(lambda x: x.fillna(x.median()))
dataframe = pd.concat([dataframe, track_id_column], ignore_index=True, axis=1)
dataframe.columns = columns

dataframe.loc[dataframe['Elb to Wrist (R)'].isnull()]["Elb to Wrist (R)"] = dataframe.loc[dataframe['Elb to Wrist (R)'].isnull()]["Elb to Wrist (L)"]

#dataframe = dataframe.drop('Elb to Wrist (R)', axis=1)

column_nan_counts = dataframe.isnull().sum()
print(column_nan_counts)

dataframe = dataframe.dropna() # Drop samples (rows) with NaN values
#dataframe = dataframe.fillna(dataframe.median()) # Fill NaN values with column-wise medians


column_names = dataframe.columns

#Resetting the indices after droping the rows
dataframe.reset_index(inplace = True)
dataframe.drop("index", axis = 1, inplace = True)

###############################################################################


print("Old Shape: ", dataframe.shape)

''' Detection '''
# IQR
# Calculate the upper and lower limits

for i in range(14): 
    
    Q1 = dataframe[str(column_names[i])].quantile(0.25)
    Q3 = dataframe[str(column_names[i])].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    
    
    # Create arrays of Boolean values indicating the outlier rows
    upper_array = np.where( dataframe[str(column_names[i])] >= upper)[0]
    lower_array = np.where( dataframe[str(column_names[i])] <= lower)[0]
    
    # Removing the outliers
    dataframe = dataframe.drop(index=upper_array, axis = 0)

    dataframe = dataframe.drop(index=lower_array, axis = 0)
    dataframe.reset_index(inplace = True)
    dataframe.drop("index", axis = 1, inplace = True)

    
# Print the new shape of the DataFrame
print("New Shape: ", dataframe.shape)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
last_two_cols_not_for_scaling  = dataframe.iloc[:,-2:]
dataframe_scaled = pd.DataFrame(scaler.fit_transform(dataframe.iloc[:,:-2]))

dataframe = pd.concat([dataframe_scaled, last_two_cols_not_for_scaling], axis = 1)
dataframe.columns = columns


"""
fig = px.scatter_matrix(dataframe, dimensions=columns[:7], color="GT_Cluster")
fig.show()
plt.show()
"""

###############################################################################



#dataframe = dataframe[[" Sho to Hip (L)", " Sho to Hip (R)", "Track ID"]]

X = dataframe.iloc[:,:-2]
y = dataframe.iloc[:,-1] # -2 will put GT_Cluster results (GT annotations)


#dataframe.to_csv("2 sides MOT16 det - without nan.csv")


unique_tracks = np.unique(y)
unique_GT_tracks = np.unique(dataframe["GT_Cluster"])




corr = dataframe.corr()
mask = np.triu(np.ones_like(dataframe.corr())) 
plt.title("Heatmap Correlation")
sns.heatmap(corr, cmap="Blues", annot=True, mask=mask) 
plt.show()

#sns.heatmap(corr, cmap="Blues", annot=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#X_test = dataframe.loc[dataframe["Track ID"] == 1778]


### Naive Bayes: ##################################################################

clf = GaussianNB()
scores_nb = np.mean(cross_val_score(clf, X, y, cv=5))

y_pred_nb = clf.fit(X_train, y_train).predict(X_test)

 
cm = confusion_matrix(y_test, y_pred_nb, labels=clf.classes_)

#sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
#plt.title("Naive Bayes - confusion matrix")
#plt.show()

#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb.classes_, cmap = "Blues")
#disp.plot()

accuracy = accuracy_score(y_test, y_pred_nb)

print("NB Accuracy", scores_nb, "\n")



### Decision Tree Classifier: ##################################################################



from sklearn import tree

clf = tree.DecisionTreeClassifier()
scores_dt = cross_val_score(clf, X, y, cv=5).mean()

y_pred_dt = clf.fit(X_train, y_train).predict(X_test)

tree.plot_tree(clf, feature_names = X.columns, max_depth = 2, filled = True, fontsize= 8)
plt.show()


cm = confusion_matrix(y_test, y_pred_dt, labels=clf.classes_)
#sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
#plt.title("Decision Tree  - confusion matrix")
#plt.show()

#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb.classes_)
#disp.plot()


"""predict_prob = clf.predict_proba([[25.816, 28.625, 65.092, 42.153, 53.255, 5.369, 30.947, 31.661, 27.891, 57.535, 46.904, 51.274, 5.95, 25.42]])

print("Predict Proba:", predict_prob)
"""
accuracy = accuracy_score(y_test, y_pred_dt)
print("DT Accuracy:", scores_dt, "\n")

feature_importances = clf.feature_importances_
#print("Feature Importance:", feature_importances)


import seaborn as sns

# Sort the feature importances from greatest to least using the sorted indices
sorted_indices = feature_importances.argsort()[::-1]
sorted_feature_names = X.columns[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# Create a bar plot of the feature importances
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(sorted_importances, sorted_feature_names)
plt.title("Decision Tree - Featurfe Importance")
plt.show()


### Random Forest Classifier: ##################################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(random_state=0)
scores_rf = cross_val_score(clf, X, y, cv=5).mean()

clf.fit(X_train, y_train)
y_pred_rf = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rf, labels=clf.classes_)
#sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
#plt.title("Random Forest Classifier  - confusion matrix")
#plt.show()

"""
predict_prob = clf.predict_proba([[25.816, 28.625, 65.092, 42.153, 53.255, 5.369, 30.947, 31.661, 27.891, 57.535, 46.904, 51.274, 5.95, 25.42]])

print("Predict Proba:", predict_prob)
"""
accuracy = accuracy_score(y_test, y_pred_rf)
print("RF Accuracy:", scores_rf, "\n")

feature_importances = clf.feature_importances_

# Sort the feature importances from greatest to least using the sorted indices
sorted_indices = feature_importances.argsort()[::-1]
sorted_feature_names = X.columns[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

feature_importances = clf.feature_importances_
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(sorted_importances, sorted_feature_names)
plt.title("Random Forest - Featurfe Importance")
plt.show()


### Adaboost Classifier: ##################################################################

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()
scores_adaB = cross_val_score(clf, X, y, cv=5).mean()

clf.fit(X_train, y_train)
y_pred_adaB = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred_adaB, labels=clf.classes_)

#sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
#plt.title("AdaBoostClassifier - confusion matrix")
#plt.show()

"""
predict_prob = clf.predict_proba([[25.816, 28.625, 65.092, 42.153, 53.255, 5.369, 30.947, 31.661, 27.891, 57.535, 46.904, 51.274, 5.95, 25.42]])

print("Predict Proba:", predict_prob)
"""
accuracy = accuracy_score(y_test, y_pred_adaB)
print("AdaBoostClassifier Accuracy:", scores_adaB, "\n")

feature_importances = clf.feature_importances_

# Sort the feature importances from greatest to least using the sorted indices
sorted_indices = feature_importances.argsort()[::-1]
sorted_feature_names = X.columns[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

feature_importances = clf.feature_importances_
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(sorted_importances, sorted_feature_names)
plt.title("AdaBoostClassifier - Featurfe Importance")
plt.show()






### Comparing y_test & y_test: ##################################################################

table = []
#table = np.array(table)

accuracies = np.vstack((y_test,y_pred_nb))
accuracies = np.vstack((accuracies,y_pred_dt))
accuracies = np.vstack((accuracies,y_pred_rf))
accuracies = np.vstack((accuracies,y_pred_adaB))

accuracies  = accuracies.T

rows, cols = np.shape(accuracies)
#accuracies = np.reshape(accuracies, [cols, rows])



"""
dt = {'names':['y_test', 'nb', 'DT', 'RF', 'AB'], 'formats':[int, int, int, int, int]}
accuracies = np.zeros( [rows, cols] , dtype=dt)
accuracies[:,0] = list(y_test)
accuracies[:,1] = list(y_pred_nb)
accuracies[:,2] = list(y_pred_dt)
accuracies[:,3] = list(y_pred_rf)
accuracies[:,4] = list(y_pred_adaB)
"""






### kelbow_visualizer: ##################################################################

from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer



# Instantiate the clustering model and visualizer
#kelbow_visualizer(KMeans(), X, k=(2,len(X)))
#plt.show()




### KMEANS: ##################################################################

from sklearn.cluster import MeanShift
from sklearn.feature_selection import SelectKBest, f_classif


unique_tracks_x_test = np.unique(y_test)

temp_y_test = y_test.to_numpy().reshape(-1,1)


"""
selector = SelectKBest(f_classif, k=4)
X_new = selector.fit_transform(X, y)
X = X_new
"""

def visualize_clustering_result(cluster_pred, data, algo_name):
    
        
    for i in range (7):
    
        for j in range (i+1, 7):  
            
            
            fig, ax = plt.subplots()
                
            scatter = plt.scatter(x=data.iloc[:,i], y=data.iloc[:,j], c=cluster_pred, cmap= "rainbow")
            
            legend1 = ax.legend(*scatter.legend_elements(),
                                loc="upper right", title="Classes")
            ax.add_artist(legend1)  
            plt.title(algo_name + " - Visualizatiom of Clustering Result")
            plt.xlabel(columns[i])
            plt.ylabel(columns[j])
            
            plt.show()

#visualize_clustering_result(kmeans_pred, X, "Kmeans")    



rand_index_matrix = np.empty([5,3])
silhouette_matrix = np.empty([5,3])


for i in range(5):
    
    kmeans_model = KMeans(31)
    kmeans_pred = kmeans_model.fit_predict(X) #temp_cluster_xtest_and_ytest
    #kmeans_pred = kmeans_model.predict(temp_cluster_xtest_and_ytest[:,:])
    iterations = kmeans_model.n_iter_
    features = kmeans_model.n_features_in_
    
    #print("Kmeans # iter", iterations)
    #print("Kmeans # features", features)
    
    
    
    #print("SILHOUETTE Kmeans", silhouette_score(X, kmeans_pred))
    
    
    
    """
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)    
    """
    
    
    estimated_bandwidth = estimate_bandwidth(X, quantile=0.1)
    #print("estimated_bandwidth Mean Shift", estimated_bandwidth)
    
    ms_model = MeanShift(bandwidth = estimated_bandwidth)
    ms_pred = ms_model.fit_predict(X)
    
    #print("SILHOUETTE Mean Shift", silhouette_score(X, ms_pred))
    
    
    hc_model = AgglomerativeClustering(31)
    hc_pred = hc_model.fit_predict(X)
    
    #print("SILHOUETTE Hierarchical Clustering", silhouette_score(X, hc_pred))
    

    
    DBSCAN_pred = DBSCAN(min_samples=2).fit_predict(X)
    #print("SILHOUETTE DBSCAN", silhouette_score(X, DBSCAN_pred))
    
    kde = gaussian_kde(X.to_numpy().T)
    density = kde(X.to_numpy().T)
    #print("Density:", np.min(density))
    
    
    #temp30_cluster_xtest_and_ytest["actual cluster"] = GT_Cluster
    
    dataframe["kmeans"] = kmeans_pred
    dataframe["meanshift"] = ms_pred
    dataframe["hier_clustering"] = hc_pred
    dataframe["DBSCAN"] = DBSCAN_pred
    
    
    
    temp30_cluster_xtest_and_ytest = dataframe.to_numpy()
    
    
    
    #numbers = (temp30_cluster_xtest_and_ytest[ ((temp30_cluster_xtest_and_ytest[:,-3] == 0) & (temp30_cluster_xtest_and_ytest[:,-1] == 0) ) |
    #                             ((temp30_cluster_xtest_and_ytest[:,-3] == 1) & (temp30_cluster_xtest_and_ytest[:,-1] == 1 )) ])
    
    purple_women_ids = [1058, 1268, 1641, 1699, 1778, 2898]
    
    
    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.width', None)
    #pd.set_option('display.max_colwidth', -1)
    
    """
    for i in purple_women_ids:
        
        #print("ID: ", i, "\n")
        temp = dataframe[ (dataframe["Track ID"] == i)  ]
        
        print(temp.iloc[:,-4:].to_string())
    
        print()
        #print("Count: ", len(temp.iloc[:,-4:]))
    """
    
    #pd.reset_option('all')
    
    
    cluster_number_zero = dataframe[ (dataframe["Track ID"] == 0)  ]
    
    rand_index_matrix[i, 0] = rand_score(dataframe["kmeans"], dataframe["GT_Cluster"])
    rand_index_matrix[i, 1] = rand_score(dataframe["meanshift"], dataframe["GT_Cluster"])
    rand_index_matrix[i, 2] = rand_score(dataframe["hier_clustering"], dataframe["GT_Cluster"])
    
                       
    silhouette_matrix[i, 0] = silhouette_score(X, kmeans_pred)
    silhouette_matrix[i, 1] = silhouette_score(X, ms_pred)
    silhouette_matrix[i, 2] = silhouette_score(X, hc_pred)



print("Rand Index (KMeans and GT):", np.mean(rand_index_matrix, axis = 0)[0])
#print("Adj. Rand Index (KMeans and GT):", adjusted_rand_score(dataframe["kmeans"], dataframe["GT_Cluster"]))


print("Rand Index (Mean Shift and GT):", np.mean(rand_index_matrix, axis = 0)[1])
#print("Adj. Rand Index (Mean Shift and GT):", adjusted_rand_score(dataframe["meanshift"], dataframe["GT_Cluster"]))

print("Rand Index (Hierarchical Clustering and GT):", np.mean(rand_index_matrix, axis = 0)[2])
#print("Adj. Rand Index (KMeans and Mean Shift):", adjusted_rand_score(dataframe["kmeans"], dataframe["meanshift"]))


"""
print("Rand Index (DBSCAN and GT):", rand_score(dataframe["DBSCAN"], dataframe["GT_Cluster"]))
#print("Adj. Rand Index (KMeans and Mean Shift):", adjusted_rand_score(dataframe["DBSCAN"], dataframe["GT_Cluster"]))


print("Rand Index (KMeans and Mean Shift):", rand_score(dataframe["kmeans"], dataframe["meanshift"]))
#print("Adj. Rand Index (KMeans and Mean Shift):", adjusted_rand_score(dataframe["kmeans"], dataframe["meanshift"]))


print("Rand Index (KMeans and Hierarchical Clustering):", rand_score(dataframe["kmeans"], dataframe["hier_clustering"]))
#print("Adj. Rand Index (KMeans and Mean Shift):", adjusted_rand_score(dataframe["kmeans"], dataframe["meanshift"]))


print("Rand Index (hier_clustering and Mean Shift):", rand_score(dataframe["hier_clustering"], dataframe["meanshift"]))
#print("Adj. Rand Index (KMeans and Mean Shift):", adjusted_rand_score(dataframe["kmeans"], dataframe["meanshift"]))
"""

print("\n\n\n ----------- T-SNE------------\n\n\n")

### T-SNE: ##################################################################


from sklearn.manifold import TSNE
#import plotly.express as px


tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X)
#tsne.kl_divergence_


fig, ax = plt.subplots()

scatter = plt.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], c=y, cmap= "rainbow")
"""fig.update_layout(
    title="t-SNE visualization of Custom Classification dataset",
    xaxis_title="First t-SNE",
    yaxis_title="Second t-SNE",
)"""
    
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)    
    
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y, cmap= "rainbow")

ax.set_xlabel('First Component')
ax.set_ylabel('Second Component')
ax.set_zlabel('Third Component')
ax.set_title('TSNE - 3D Scatter Plot')

plt.show()





#inline
y_pred_cluster = []




# Instantiate the clustering model and visualizer
#kelbow_visualizer(KMeans(), X_tsne, k=(2,len(X_tsne)))
#plt.show()

"""
selector = SelectKBest(f_classif, k=4)
X_new = selector.fit_transform(X, y)
X = X_new
"""



rand_index_matrix_tsne = np.empty([5,3])
silhouette_matrix_tsne = np.empty([5,3])

for i in range(5):

    kmeans_model = KMeans(31)
    kmeans_pred = kmeans_model.fit_predict(X_tsne) #temp_cluster_xtest_and_ytest
    #kmeans_pred = kmeans_model.predict(temp_cluster_xtest_and_ytest[:,:])
    iterations = kmeans_model.n_iter_
    features = kmeans_model.n_features_in_
    
    #print("Kmeans X_tsne # iter", iterations)
    #print("Kmeans X_tsne # features", features)
    
    
    
    #print("X_tsne SILHOUETTE Kmeans", silhouette_score(X_tsne, kmeans_pred))
    
    
    
    """
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)    
    """
    
    estimated_bandwidth = estimate_bandwidth(X_tsne, quantile=0.1)
    
    
    ms_model = MeanShift(bandwidth = estimated_bandwidth)
    ms_pred = ms_model.fit_predict(X_tsne)
    
    #print("X_tsne SILHOUETTE Mean Shift", silhouette_score(X_tsne, ms_pred))
    
    
    
    
    hc_model = AgglomerativeClustering(31)
    hc_pred = hc_model.fit_predict(X_tsne)
    
    #print("X_tsne SILHOUETTE Agg. Clustering", silhouette_score(X_tsne, hc_pred))
    
    
            
    #DBSCAN_pred = DBSCAN(min_samples=2).fit_predict(X_tsne)
    #print("X_tsne SILHOUETTE DBSCAN", silhouette_score(X_tsne, DBSCAN_pred))
    
    
    
    
    #temp30_cluster_xtest_and_ytest["actual cluster"] = GT_Cluster
    
    dataframe["kmeansX_tsne"] = kmeans_pred
    dataframe["meanshiftX_tsne"] = ms_pred
    dataframe["hier_clusteringX_tsne"] = hc_pred
    #dataframe["DBSCANX_tsne"] = DBSCAN_pred
    
    
    
    
    
    #numbers = (temp30_cluster_xtest_and_ytest[ ((temp30_cluster_xtest_and_ytest[:,-3] == 0) & (temp30_cluster_xtest_and_ytest[:,-1] == 0) ) |
    #                             ((temp30_cluster_xtest_and_ytest[:,-3] == 1) & (temp30_cluster_xtest_and_ytest[:,-1] == 1 )) ])
    
    purple_women_ids = [1058, 1268, 1641, 1699, 1778, 2898]
    
    
    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.width', None)
    #pd.set_option('display.max_colwidth', -1)
    
    """
    for i in purple_women_ids:
        
        print("ID: ", i, "\n")
        temp = dataframe[ (dataframe["Track ID"] == i)  ]
        
        print(temp.iloc[:,-4:].to_string())
    
        print()
        print("Count: ", len(temp.iloc[:,-4:]))
        
    """
    #pd.reset_option('all')
    
    
    cluster_number_zero = dataframe[ (dataframe["Track ID"] == 0)  ]
    
    
    rand_index_matrix_tsne[i, 0] = rand_score(dataframe["kmeansX_tsne"], dataframe["GT_Cluster"])
    rand_index_matrix_tsne[i, 1] = rand_score(dataframe["meanshiftX_tsne"], dataframe["GT_Cluster"])
    rand_index_matrix_tsne[i, 2] = rand_score(dataframe["hier_clusteringX_tsne"], dataframe["GT_Cluster"])
    
                       
    silhouette_matrix_tsne[i, 0] = silhouette_score(X_tsne, kmeans_pred)
    silhouette_matrix_tsne[i, 1] = silhouette_score(X_tsne, ms_pred)
    silhouette_matrix_tsne[i, 2] = silhouette_score(X_tsne, hc_pred)
               
    
    




print("T-SNE Rand Index (KMeans and GT):", np.mean(rand_index_matrix_tsne, axis = 0)[0])
#print("Adj. Rand Index (KMeans and GT):", adjusted_rand_score(dataframe["kmeans"], dataframe["GT_Cluster"]))


print("T-SNE Rand Index (Mean Shift and GT):", np.mean(rand_index_matrix_tsne, axis = 0)[1])
#print("Adj. Rand Index (Mean Shift and GT):", adjusted_rand_score(dataframe["meanshift"], dataframe["GT_Cluster"]))

print("T-SNE Rand Index (Hierarchical Clustering and GT):", np.mean(rand_index_matrix_tsne, axis = 0)[2])
#print("Adj. Rand Index (KMeans and Mean Shift):", adjusted_rand_score(dataframe["kmeans"], dataframe["meanshift"]))

#print("T-SNE Rand Index (DBSCAN and GT):", rand_score(dataframe["DBSCANX_tsne"], dataframe["GT_Cluster"]))
#print("Adj. Rand Index (KMeans and Mean Shift):", adjusted_rand_score(dataframe["kmeans"], dataframe["meanshift"]))


print("T-SNE Rand Index (KMeans and Mean Shift):", rand_score(dataframe["kmeansX_tsne"], dataframe["meanshiftX_tsne"]))
#print("Adj. Rand Index (KMeans and Mean Shift):", adjusted_rand_score(dataframe["kmeans"], dataframe["meanshift"]))


print("T-SNE Rand Index (KMeans and Hierarchical Clustering):", rand_score(dataframe["kmeansX_tsne"], dataframe["hier_clusteringX_tsne"]))
#print("Adj. Rand Index (KMeans and Mean Shift):", adjusted_rand_score(dataframe["kmeans"], dataframe["meanshift"]))


print("T-SNE Rand Index (hier_clustering and Mean Shift):", rand_score(dataframe["hier_clusteringX_tsne"], dataframe["meanshiftX_tsne"]))
#print("Adj. Rand Index (KMeans and Mean Shift):", adjusted_rand_score(dataframe["kmeans"], dataframe["meanshift"]))




#sns.boxplot(dataframe.query("kmeans == 0").all())
#plt.show()







def clustering_id_results (dataframe, algo):
    
    
    
    number_of_clusters = np.max(dataframe[algo]) + 1 # since numbering starts from 0
    
    print("number_of_clusters", number_of_clusters)
    all_clustering_results = []
    
    for i in range(number_of_clusters):
        
        
        temp = dataframe.query(algo + "==" + str(i) )
            
        all_clustering_results.append(temp) #temp.iloc[:,-8]
        
    return all_clustering_results


clus_results_by_name = clustering_id_results(dataframe, "kmeans")






clus_results_by_name_new = []

''' Cluster Outliers Removal in each cluster '''
# IQR
# Calculate the upper and lower limits
indexx = 0
for clus in clus_results_by_name:
    
    #print("INDEX:", indexx)
    #print("Old Shape: ", clus.shape)
    
    indices_to_remove = []
    
    clus.reset_index(inplace = True)
    clus.drop("index", axis = 1, inplace = True)

    for i in range (14):
       
        Q1 = clus.iloc[:,i].quantile(0.25)
        Q3 = clus.iloc[:,i].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        
        # Create arrays of Boolean values indicating the outlier rows
        
        temp_indices_to_remove = list(np.where( clus.iloc[:,i] >= upper)[0])
        
        for index in temp_indices_to_remove:
            
            if index not in indices_to_remove:
                
                indices_to_remove.append(index)
                
        
        temp_indices_to_remove = list(np.where( clus.iloc[:,i] <= lower)[0])
        
        for index in temp_indices_to_remove:
            
            if index not in indices_to_remove:
                
                indices_to_remove.append(index)

    # Removing the outliers
    clus = clus.drop(index=indices_to_remove, axis = 0)  

    if(len(clus) > 2):      
        clus_results_by_name_new.append(clus.iloc[:,-10:]) #temp.iloc[:,-8]

      
    #print("New Shape: ", clus.shape)
    indexx += 1
        
 
dataframe_filtered = pd.DataFrame()

for clus in clus_results_by_name_new:
    
    dataframe_filtered = pd.concat([dataframe_filtered, clus])

    


"""
clus.reset_index(inplace = True)
clus.drop("index", axis = 1, inplace = True)
"""




print("Rand Index (KMeans and GT):", rand_score(dataframe_filtered["kmeans"], dataframe_filtered["GT_Cluster"]))
#print("Adj. Rand Index (KMeans and GT):", adjusted_rand_score(dataframe["kmeans"], dataframe["GT_Cluster"]))


print("Rand Index (Mean Shift and GT):", rand_score(dataframe_filtered["meanshift"], dataframe_filtered["GT_Cluster"]))
#print("Adj. Rand Index (Mean Shift and GT):", adjusted_rand_score(dataframe["meanshift"], dataframe["GT_Cluster"]))

print("Rand Index (Hierarchical Clustering and GT):", rand_score(dataframe_filtered["hier_clustering"], dataframe_filtered["GT_Cluster"]))
#print("Adj. Rand Index (KMeans and Mean Shift):", adjusted_rand_score(dataframe["kmeans"], dataframe["meanshift"]))







### Mahalanobis Distance Classification: ##################################################################



import numpy as np
from scipy.spatial.distance import mahalanobis

"""
rslt_df_1 = dataframe[dataframe.iloc[:-1,14] == 1810] 
rslt_df_2 = dataframe[dataframe.iloc[:-1,14] == 1689] 
rslt_df_3 = dataframe[dataframe.iloc[:-1,14] == 1513] 
rslt_df_4 = dataframe[dataframe.iloc[:-1,14] == 2023] 

#dataframes = [rslt_df_1, rslt_df_2, rslt_df_3, rslt_df_4]

"""

def mahalanobis_distance_from_4_known_ids_women(data_sample, dataframes):
    
        
    for i in range(4):
        
        # Calculate the mean vector and covariance matrix of the dataset
        mu = np.mean(X_test.iloc[:-5,9], axis=0)
        sigma = np.cov(X_test.iloc[:-5,:].T)
        
        # Calculate the Mahalanobis Distance between two points
        
            
        data_sample = X_test.iloc[i, :]
    
        dist_x1 = mahalanobis(data_sample, mu, np.linalg.inv(sigma))
        print(f"Distance 1: {dist_x1}")

    





