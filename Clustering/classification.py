# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 00:19:37 2023

@author: andre
"""
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, silhouette_score
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from mpl_toolkits.mplot3d import Axes3D


dataframe = pd.read_csv("2 sides MOT16 det_SEP.csv")
dataframe.replace(0, np.nan, inplace = True)




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


dataframe = dataframe.dropna()

    

#dataframe = dataframe[[" Sho to Hip (L)", " Sho to Hip (R)", "Track ID"]]

X = dataframe.iloc[:,:-2]
y = dataframe.iloc[:,-1] # -2 will put GT_Cluster results (GT annotations)


#dataframe.to_csv("2 sides MOT16 det - without nan.csv")

column_names = dataframe.columns

unique_tracks = np.unique(y)



"""
#relevant_indices = dataframe.index[dataframe['Track ID'] == 1778  | dataframe['Track ID'] == 2989 ].tolist()
relevant_indices = list(np.where(dataframe["Track ID"] == 1778)[0])
relevant_samples =  dataframe.iloc[relevant_indices]
"""

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



table = accuracies
#relevant_indices = np.where((table == 2023) | (table == 1689))


relevant_indices2 = (table[ (table[:,0] == 1689)|
                             (table[:,0] == 2023) |
                             (table[:,0] == 1513) |
                             (table[:,0] == 1810) |
                             (table[:,0] == 1060) |
                             (table[:,0] == 1282) ])



relevant_indices_nb = (relevant_indices2[((relevant_indices2[:,0] == 1689) & (relevant_indices2[:,1] == 1689)) |
                             ((relevant_indices2[:,0] == 2023) & (relevant_indices2[:,1] == 2023)) |
                             ((relevant_indices2[:,0] == 1513) & (relevant_indices2[:,1] == 1513)) |
                             ((relevant_indices2[:,0] == 1810) & (relevant_indices2[:,1] == 1810)) |
                             ((relevant_indices2[:,0] == 1060) & (relevant_indices2[:,1] == 1060)) |
                             ((relevant_indices2[:,0] == 1282) & (relevant_indices2[:,1] == 1282))])




relevant_indices_dt = (relevant_indices2[((relevant_indices2[:,0] == 1689) & (relevant_indices2[:,2] == 1689)) |
                             ((relevant_indices2[:,0] == 2023) & (relevant_indices2[:,2] == 2023)) |
                             ((relevant_indices2[:,0] == 1513) & (relevant_indices2[:,2] == 1513)) |
                             ((relevant_indices2[:,0] == 1810) & (relevant_indices2[:,2] == 1810)) |
                             ((relevant_indices2[:,0] == 1060) & (relevant_indices2[:,2] == 1060)) |
                             ((relevant_indices2[:,0] == 1282) & (relevant_indices2[:,2] == 1282))])

relevant_indices_rf = (relevant_indices2[((relevant_indices2[:,0] == 1689) & (relevant_indices2[:,3] == 1689)) |
                             ((relevant_indices2[:,0] == 2023) & (relevant_indices2[:,3] == 2023)) |
                             ((relevant_indices2[:,0] == 1513) & (relevant_indices2[:,3] == 1513)) |
                             ((relevant_indices2[:,0] == 1810) & (relevant_indices2[:,3] == 1810)) |
                             ((relevant_indices2[:,0] == 1060) & (relevant_indices2[:,3] == 1060)) |
                             ((relevant_indices2[:,0] == 1282) & (relevant_indices2[:,3] == 1282))])

relevant_indices_adaB = (relevant_indices2[((relevant_indices2[:,0] == 1689) & (relevant_indices2[:,4] == 1689)) |
                             ((relevant_indices2[:,0] == 2023) &   (relevant_indices2[:,4] == 2023)) |
                             ((relevant_indices2[:,0] == 1513) &   (relevant_indices2[:,4] == 1513)) |
                             ((relevant_indices2[:,0] == 1810) &   (relevant_indices2[:,4] == 1810)) |
                             ((relevant_indices2[:,0] == 1060) &   (relevant_indices2[:,4] == 1060)) |
                             ((relevant_indices2[:,0] == 1282) &   (relevant_indices2[:,4] == 1282))])



relevant_indices2 = relevant_indices2[relevant_indices2[:, 0].argsort()]









"""

relevant_indices2 = table[(table[:,0] == 1689) & (table[:,1] == 1689) |
                          (table[:,0] == 1689) & (table[:,1] == 2023) |
                          (table[:,0] == 1689) & (table[:,1] == 1513) |
                          (table[:,0] == 1689) & (table[:,1] == 1810) |
                          (table[:,0] == 1689) & (table[:,1] == 1060) |
                          (table[:,0] == 1689) & (table[:,1] == 1282)]




Considering only 2 features:
"1058": 0,
"1268": 0,
"1641": 0,
"1699" : 0,
"1778" : 0,
"2989" : 0


Total for this specific women: 178
Considering total frames: improvment: 84.13%
1060     0/3
1513:    1/16
1689: ID SWITCH  12 + 103 (same)  / 145
1810     4/9 common
2023 ID SWITCH: 2 /5


Total for this specific women: 178
Considering previous numbers:
1060     0/3
1513:    1/16
1689: ID SWITCH  5  / 145
1810     0/9 common
2023 ID SWITCH: 0 /5
"""

### SECOND: ##################################################################
"""
Total for this specific women: 140
Considering previous numbers:
1060     0/3
1513:    1/16
1689: ID SWITCH  5  / 145
1810     0/9 common
2023 ID SWITCH: 0 /5
"""

### Third: ##################################################################
"""
Total for this specific women: 140
Considering previous numbers:
1060     0/3 -> 2/3 same person
1513:    1/16 -> 10/same person: clus: 7
1689: ID SWITCH  5  / 107 ->   62/107:6
1810     0/9 common  -> 3/9
2023 ID SWITCH: 0 /5  -> 2/9
"""






### 3D Plots: ##################################################################


"""

X_test = X_test.to_numpy()

for i in range(len(X_test[0,:])-2): # iterate through all columns
    

    x_column_name = columns[i]
    y_column_name = columns[i+1]
    z_column_name = columns[i+2]


    x1 = X_test[:,i].tolist()
    y1 = X_test[:,i+1].tolist()
    z1 = X_test[:,i+2].tolist()

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
column_names = X.columns


### kelbow_visualizer: ##################################################################

from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer

unique_tracks_x_test = np.unique(y_test)

"""
# Instantiate the clustering model and visualizer
kelbow_visualizer(KMeans(), X, k=(2,25))
plt.show()
"""


temp_y_test = y_test.to_numpy().reshape(-1,1)

temp_cluster_xtest_and_ytest = np.hstack((X_test, temp_y_test))

"""
temp_cluster_xtest_and_ytest2 = (temp_cluster_xtest_and_ytest[ (temp_cluster_xtest_and_ytest[:,-1] != 1689)&
                             (temp_cluster_xtest_and_ytest[:,-1] != 2023) &
                             (temp_cluster_xtest_and_ytest[:,-1] != 1513) &
                             (temp_cluster_xtest_and_ytest[:,-1] != 1810) &
                             (temp_cluster_xtest_and_ytest[:,-1] != 1060) &
                             (temp_cluster_xtest_and_ytest[:,-1] != 1282) ])
"""


### KMEANS: ##################################################################

from sklearn.cluster import MeanShift


temp_cluster_relevant = (temp_cluster_xtest_and_ytest[ (temp_cluster_xtest_and_ytest[:,-1] == 1689)|
                             (temp_cluster_xtest_and_ytest[:,-1] == 2023) |
                             (temp_cluster_xtest_and_ytest[:,-1] == 1513) |
                             (temp_cluster_xtest_and_ytest[:,-1] == 1810) |
                             (temp_cluster_xtest_and_ytest[:,-1] == 1060) |
                             (temp_cluster_xtest_and_ytest[:,-1] == 1282) ])

temp_cluster_xtest_and_ytest = pd.DataFrame(temp_cluster_xtest_and_ytest) 

temp30_cluster_xtest_and_ytest = temp_cluster_xtest_and_ytest.iloc[-30:,:]



kmeans_model = KMeans()
kmeans_pred = kmeans_model.fit_predict(temp30_cluster_xtest_and_ytest.iloc[:,:-1]) #temp_cluster_xtest_and_ytest
#kmeans_pred = kmeans_model.predict(temp_cluster_xtest_and_ytest[:,:])

print("SILHOUTTE", silhouette_score(temp30_cluster_xtest_and_ytest.iloc[:,:-1], kmeans_pred))


for i in range (7):

    for j in range (i, 7):   
        
        if (i != j):
            
            fig, ax = plt.subplots()
                
            scatter = plt.scatter(x=temp30_cluster_xtest_and_ytest.iloc[:,i], y=temp30_cluster_xtest_and_ytest.iloc[:,j], c=kmeans_pred, cmap= "rainbow")
            
            legend1 = ax.legend(*scatter.legend_elements(),
                                loc="upper right", title="Classes")
            ax.add_artist(legend1)  
            plt.title("KMeans - Visualizatiom of Clustering Result")
            plt.xlabel(columns[i])
            plt.ylabel(columns[j])
            
            plt.show()






"""
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)    
"""




ms_model = MeanShift()
ms_pred = ms_model.fit_predict(temp30_cluster_xtest_and_ytest.iloc[:,:-1])

print("SILHOUTT MEAN SHIFT", silhouette_score(temp30_cluster_xtest_and_ytest.iloc[:,:-1], ms_pred))


for i in range (7):

    for j in range (i+1, 7):  
        
        
        fig, ax = plt.subplots()
            
        scatter = plt.scatter(x=temp30_cluster_xtest_and_ytest.iloc[:,i], y=temp30_cluster_xtest_and_ytest.iloc[:,j], c=ms_pred, cmap= "rainbow")
        
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="upper right", title="Classes")
        ax.add_artist(legend1)  
        plt.title("MeanShift - Visualizatiom of Clustering Result")
        plt.xlabel(columns[i])
        plt.ylabel(columns[j])
        
        plt.show()
        
        
        
        




# Instantiate the clustering model and visualizer
kelbow_visualizer(KMeans(), temp30_cluster_xtest_and_ytest.iloc[:,:-1], k=(2,25))
plt.show()



temp30_cluster_xtest_and_ytest["actual cluster"] = [0,0,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1]

temp30_cluster_xtest_and_ytest["kmeans"] = kmeans_pred[-30:,]
temp30_cluster_xtest_and_ytest["meanshift"] = ms_pred[-30:,]

temp30_cluster_xtest_and_ytest = temp30_cluster_xtest_and_ytest.to_numpy()



numbers = (temp30_cluster_xtest_and_ytest[ ((temp30_cluster_xtest_and_ytest[:,-3] == 0) & (temp30_cluster_xtest_and_ytest[:,-1] == 0) ) |
                             ((temp30_cluster_xtest_and_ytest[:,-3] == 1) & (temp30_cluster_xtest_and_ytest[:,-1] == 1 )) ])

purple_women_ids = [1058, 1268, 1641, 1699, 1778, 2898]



for i in purple_women_ids:
    
    print("ID: ", i, "\n")
    temp = temp30_cluster_xtest_and_ytest[ (temp30_cluster_xtest_and_ytest[:,14] == i)  ]

    print(temp[:,-4:])




from sklearn.metrics.cluster import rand_score

print("Rand Index (KMeans and GT):", rand_score(temp30_cluster_xtest_and_ytest[:,16],
                                                temp30_cluster_xtest_and_ytest[:,15]))


print("Rand Index (Mean Shift and GT):", rand_score(temp30_cluster_xtest_and_ytest[:,17],
                                                temp30_cluster_xtest_and_ytest[:,15]))


print("Rand Index (KMeans and Mean Shift):", rand_score(temp30_cluster_xtest_and_ytest[:, 16],
                                                temp30_cluster_xtest_and_ytest[:,17]))


### T-SNE: ##################################################################


from sklearn.manifold import TSNE
#import plotly.express as px


tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
tsne.kl_divergence_


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


y_pred_cluster = []




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

    























def person_classification_by_nearest_centroid(X_test, new_sample = 0, number_of_persons = 3):
    
    global y_pred_cluster
 
    kmeans = KMeans(n_clusters=number_of_persons)
    y_pred_cluster = kmeans.fit_predict(X_test)


person_classification_by_nearest_centroid(X_test)

y_pred_cluster = np.reshape(y_pred_cluster, [-1,1])

cluster_table = np.hstack((table, y_pred_cluster))

cluster_table = cluster_table[(cluster_table[:,0] == 1689) |
                          (cluster_table[:,0] == 2023) |
                          (cluster_table[:,0] == 1513) |
                          (cluster_table[:,0] == 1810) |
                          (cluster_table[:,0] == 1060) |
                          (cluster_table[:,0] == 1282) ]

cluster_table = cluster_table[cluster_table[:, 0].argsort()]



corr = dataframe.corr()





