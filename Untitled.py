#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdatasheet-1.xlsx")
df


# In[5]:


class_x_data = df[df['Label'] == 0]  
class_y_data = df[df['Label'] == 1]  
intra_class_var_x = np.var(class_x_data[['embed_1', 'embed_2']], ddof=1)  
intra_class_var_y = np.var(class_y_data[['embed_1', 'embed_2']], ddof=1)  
mean_class_x = np.mean(class_x_data[['embed_1', 'embed_2']], axis=0)  
mean_class_y = np.mean(class_y_data[['embed_1', 'embed_2']], axis=0)  
inter_class_distance = np.linalg.norm(mean_class_x - mean_class_y)
print(f'Intraclass spread (variance) for Class X: {intra_class_var_x}')
print(f'Intraclass spread (variance) for Class Y: {intra_class_var_y}')
print(f'Interclass distance between Class X and Class Y: {inter_class_distance}')


# In[6]:


unique_classes = df['Label'].unique()
class_centroids = {}

for class_label in unique_classes:
    class_data = df[df['Label'] == class_label]
    class_mean = np.mean(class_data[['embed_1', 'embed_2']], axis=0)
    class_centroids[class_label] = class_mean

for class_label, centroid in class_centroids.items():
    print(f'Class {class_label} Centroid: {centroid}')


# In[7]:


grouped = df.groupby('Label')
class_standard_deviations = {}
for class_label, group_data in grouped:
    class_std = group_data[['embed_1', 'embed_2']].std(axis=0)
    class_standard_deviations[class_label] = class_std
for class_label, std_deviation in class_standard_deviations.items():
    print(f'Standard Deviation for Class {class_label}:')
    for col, std in zip(std_deviation.index, std_deviation.values):
        print(f'  {col}: {std}')


# In[8]:


grouped = df.groupby('Label')


class_centroids = {}
for class_label, group_data in grouped:
    class_mean = group_data[['embed_1', 'embed_2']].mean(axis=0)
    class_centroids[class_label] = class_mean


class_labels = list(class_centroids.keys())
num_classes = len(class_labels)
class_distances = {}

for i in range(num_classes):
    for j in range(i + 1, num_classes):
        class_label1 = class_labels[i]
        class_label2 = class_labels[j]
        distance = np.linalg.norm(class_centroids[class_label1] - class_centroids[class_label2])
        class_distances[(class_label1, class_label2)] = distance


for (class_label1, class_label2), distance in class_distances.items():
    print(f'Distance between Class {class_label1} and Class {class_label2}: {distance}')


# In[10]:


import numpy as np
import matplotlib.pyplot as plt


feature1_data = df['embed_1']


num_bins = 5

hist_counts, bin_edges = np.histogram(feature1_data, bins=num_bins)

mean_feature1 = np.mean(feature1_data)
variance_feature1 = np.var(feature1_data, ddof=1)
plt.hist(feature1_data, bins=num_bins, edgecolor='black', alpha=0.7)
plt.xlabel('Feature1')
plt.ylabel('Frequency')
plt.title('Histogram of Feature1')
plt.grid(True)
plt.show()
print(f'Mean of Feature1: {mean_feature1}')
print(f'Variance of Feature1: {variance_feature1}')


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


vector1 = np.array([df['embed_1'][0], df['embed_2'][0]])
vector2 = np.array([df['embed_1'][3], df['embed_2'][3]])


r_values = list(range(1, 11))


distances = []
for r in r_values:
    minkowski_distance = distance.minkowski(vector1, vector2, p=r)
    distances.append(minkowski_distance)


plt.plot(r_values, distances, marker='o', linestyle='-', color='b')
plt.xlabel('r (Minkowski Parameter)')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance vs. r')
plt.grid(True)
plt.show()


# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split



binary_df = df[df['Label'].isin([0, 1])]


X = binary_df[['embed_1', 'embed_2']]
y = binary_df['Label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[10]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier

 

# Assuming you have already split your data into X_train and y_train
# If not, please refer to the previous code for splitting the data.

 

# Create a k-NN classifier with k=3
neigh = KNeighborsClassifier(n_neighbors=3)

 

# Fit the classifier to your training data
neigh.fit(X_train, y_train)


# In[9]:


accuracy = neigh.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[18]:


# Choose a specific test vector (for example, the first vector in the test set)
test_vector = X_test.iloc[0]  # You can select any index you prefer

# Use neigh.predict() to classify the test vector
predicted_class = neigh.predict([test_vector])

print("Predicted Class:", predicted_class[0])


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


accuracies_kNN = []
accuracies_NN = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
k_values = range(1, 12)

for k in k_values:
    
    kNN_classifier = KNeighborsClassifier(n_neighbors=k)
    kNN_classifier.fit(X_train, y_train)
    
    
    y_pred_kNN = kNN_classifier.predict(X_test)
    
   
    accuracy_kNN = accuracy_score(y_test, y_pred_kNN)
    accuracies_kNN.append(accuracy_kNN)

    
        
    NN_classifier = KNeighborsClassifier(n_neighbors=1)
    NN_classifier.fit(X_train, y_train)
        
       
    y_pred_NN = NN_classifier.predict(X_test)
        
        
    accuracy_NN = accuracy_score(y_test, y_pred_NN)
    accuracies_NN.append(accuracy_NN)


plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies_kNN, marker='o', label='kNN (k=3)')
plt.plot(k_values, accuracies_NN, marker='o', label='NN (k=1)')

plt.title('Accuracy vs. k Value')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()


# In[21]:


from sklearn.metrics import confusion_matrix, classification_report


y_train_pred = neigh.predict(X_train)
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)


y_test_pred = neigh.predict(X_test)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)


print("Confusion Matrix (Training Data):\n", confusion_matrix_train)
print("\nConfusion Matrix (Test Data):\n", confusion_matrix_test)


classification_report_train = classification_report(y_train, y_train_pred)
print("\nClassification Report (Training Data):\n", classification_report_train)


classification_report_test = classification_report(y_test, y_test_pred)
print("\nClassification Report (Test Data):\n", classification_report_test)


# In[ ]:




