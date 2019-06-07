#!/usr/bin/env python
# coding: utf-8

# In[9]:


########## MACHINE LEARNING ASSIGNMENT -3 ############
### DECISION TREE, RANDOM FOREST, SUPPORT VECTOR CLASSIFICATION
## IMPLEMENTATION USING SCI-KIT LEARN
# Note : NOT FROM SCRATCH


# In[10]:


# IMPORT LIBRARIES
import numpy as np
import seaborn as sn
from sklearn import *
from sklearn.ensemble import *
from sklearn.datasets import *
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.neural_network import *
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import graphviz


# In[11]:


n = []
for i in range(0,60):
    n.append(i)
print(n)


# In[12]:


n.append("o")
print(n)


# In[13]:


df = pd.read_csv("sonar.all-data" , names =n)


# In[14]:


df.head()
df["o"].value_counts()
repl = {"o": {"M": 1, "R": 0 }}
df_new = df.replace(repl)
df_new.head()


# In[15]:


df1 = df_new.drop(df.columns[len(df.columns)-1], axis=1)


# In[16]:


df1.head()


# In[17]:


X = df1
print(X)
Y = df_new["o"].tolist()
print(Y)
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=63)


# In[36]:


# Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
tree.plot_tree(clf)


# In[38]:


# Decision Tree Visualization Graph
dot_data = tree.export_graphviz(clf,out_file=None)
graph = graphviz.Source(dot_data)
graph.render("tree")
print(clf.feature_importances_)
predicted = clf.predict(X_test)
print(accuracy_score(y_test,predicted))
print(predicted)


# In[20]:


# Random Forest classifier
clf1 = RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
clf1.fit(X_train,y_train)


# In[21]:


print(clf1.feature_importances_)
predicted = clf1.predict(X_test)
print(accuracy_score(y_test,predicted))


# In[22]:


# Support Vector Machine Classification
clf2 = svm.SVC(gamma='scale')
clf2.fit(X_train,y_train)
predicted = clf2.predict(X_test)
print(accuracy_score(y_test,predicted))


# In[23]:


clf_e= tree.DecisionTreeClassifier(criterion="entropy",random_state=60,max_depth=3,min_samples_leaf=5)
clf_e = clf_e.fit(X_train,y_train)
Ypred = clf_e.predict(X_test)
print(accuracy_score(y_test,Ypred))


# In[32]:


##### K --- FOLD CROSS VALIDATION ON DECISION TREES #####
kf = KFold(n_splits=10)
kf.get_n_splits(X)
accuracy = []

clf = tree.DecisionTreeClassifier(criterion='gini', random_state = 42)
clf_entropy = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 63)
scores = cross_val_score(clf, X, Y, cv=kf)
avg_score = np.mean(scores)
print(avg_score)
scores_en = cross_val_score(clf_entropy, X, Y, cv=kf)
avg_score_en = np.mean(scores_en)
print(scores)
print(avg_score_en) 


# In[33]:


##### K --- FOLD CROSS VALIDATION ON RANDOM FOREST #####
kf = KFold(n_splits=10)
kf.get_n_splits(X)
accuracy = []
clf1 = RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
scores = cross_val_score(clf1, X, Y, cv=kf)
avg_score = np.mean(scores)
print(scores)
print(avg_score)


# In[34]:


##### K --- FOLD CROSS VALIDATION ON SUPPORT VECTOR MACHINES #####
kf = KFold(n_splits=10)
kf.get_n_splits(X)
accuracy = []
clf2 = svm.SVC(gamma='scale')
scores = cross_val_score(clf2, X, Y, cv=kf)
avg_score = np.mean(scores)
print(scores)
print(avg_score)


# In[25]:


# Neural Network --- Multi Layer Perceptron
nn_mlp = MLPClassifier(hidden_layer_sizes=(24,), activation='logistic', solver='adam', alpha=0.0001,
                    batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
                    power_t=0.5, max_iter=200, shuffle=True, random_state=40, tol=0.0001, 
                    verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
                    epsilon=1e-08, n_iter_no_change=10)
nn_mlp.fit(X_train,y_train)
predictions = nn_mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[26]:


cm = confusion_matrix(y_test, predictions)

print(cm)

# Confusion matrix on test samples
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[27]:


# Plotting heat map of confusion matrix -- dependency of different variables
# Note to view completely please click on the image
# if want to know the test dataset kindly convert it into a array and put the index out there in 
# pd.DataFrame command.
# Execution takes time so if you want to view just reduce the range.
# Mentioned at last to avoid the time delay in running the entire code.
df_cm = pd.DataFrame(df_new, range(209), range(62))
plt.figure(figsize = (100,50))
sn.set(font_scale=2)
sn.heatmap(df_cm, annot=True)
plt.show()


# In[28]:


####################################... NOTES ... #########################################
''' 

'''


# In[ ]:





# In[ ]:




