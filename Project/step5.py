import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.evaluate.bootstrap_point632 import accuracy
from networkx.algorithms.bipartite.cluster import clustering
from utils import remove_outliers, impute, categorize,print_nan
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV


pd.set_option('future.no_silent_downcasting', True)

orig_df = pd.read_csv("Airline_Satisfaction.csv")
df = pd.get_dummies(orig_df,
                            columns=['Class',],
                            prefix=['Class'])

### Step 1: Data pre-processing phase

# after checking the data, we found that there are no outliers in the columns;
# there are some missing values in Arrival Delay in Minutes
# but there are Categorical columns that need to be converted to numerical values
df = categorize(df)
df = impute(df)


cols_to_norm = ['Age','Flight Distance','Departure Delay in Minutes', 'Arrival Delay in Minutes']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.mean()) /x.std())

# plot the correlation between Age and Glucose
'''
plt.scatter(df_imputed["Age"],df_imputed["Glucose"])
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.title("Correlation between Age and Glucose")
plt.show()
'''

### Step 3: Feature Extraction
# Split data into test and training sets (consider 20% for test).
from sklearn.model_selection import train_test_split

y = df["satisfied"]
X = df.drop(columns=["satisfied"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Use PCA on the training data to create 3 new components from existing features (all columns except outcome).
# Transfer training and test data to the new dimensions (PCs).
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

X_train_pca_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
X_test_pca_df = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

y_train_pca = np.ravel(y_train.astype(int))
y_test_pca = np.ravel(y_test.astype(int))

y_train_pca_df = pd.DataFrame(y_train_pca, columns=["Outcome"])
y_test_pca_df = pd.DataFrame(y_test_pca, columns=["Outcome"])

#combine X_train_pca and X_test_pca(only for check)
'''
X_train_pca_df["Outcome"] = y_train_pca_df
X_test_pca_df["Outcome"] = y_test_pca_df
combined_pca_df = pd.concat([X_train_pca_df, X_test_pca_df])
combined_pca_df.to_csv("pcv.csv",index=False)
covariance_matrix = pca.get_covariance()
pc_feature_relationship = pd.DataFrame(pca.components_, columns=X.columns, index=['PC1', 'PC2', 'PC3'])
'''

# Plot the proportion of explained variance by Cumulative sum of principal components
'''x_axis = range(pca.n_components_)
plt.plot(x_axis, np.cumsum(pca.explained_variance_ratio_), marker="o")
plt.xlabel('Principal component')
plt.ylabel('Cumulative sum of explained variance')
plt.xticks(x_axis)
plt.show()
'''
# Show eigenvalues.Kaiser’s rule:  principal components are retained, whose eigenvalue exceed 1.
print("Eigenvalues:")
print(pca.explained_variance_)
# Step 4: Classification using a super learner (Work with the pca features ?)
'''
Define three classification models as base classifiers consisting of Naïve Bayes, Neural Network, and KNN.
Define a decision tree as the meta learner.
Train decision tree (meta learner) on outputs of three base classifiers using 5-fold cross validation.
Find hyperparameters for all these models which provide the best accuracy rate.
Report accuracy of the model on the test data.

'''
# Train Naive Bayes model
from nb import train_nb
model_nb = train_nb(X_train_pca_df, y_train_pca, X_test_pca_df, y_test_pca)

# Train KNN model
from knn import train_knn
model_knn = train_knn(X_train_pca_df, y_train_pca, X_test_pca_df, y_test_pca)

# Neural Network Model
from nn import train_nn
best_hp = {'hidden_layers':1,'neurons':5, 'activation':'relu','optimizer': 'adam'}
model_nn = train_nn(X_train_pca_df, y_train_pca, X_test_pca_df, y_test_pca,best_hp)

from dtree import train_dtree
model_dtree = train_dtree(X_train_pca_df, y_train_pca, X_test_pca_df, y_test_pca)

exit()
