from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import matplotlib.pyplot as plt
import math

param_grid = {
    'n_neighbors': list(range(10, 30)),  # Values of k
    'weights': [ 'distance'],  # Weighting scheme
    'metric': ['euclidean', 'manhattan']  # Distance metrics
}


def train_knn(X_train,y_train,X_test,y_test, best_hp=None):
    print('*** Training KNN')

    # KNN Model
    '''
    #Elbow Method to find the best value of k
    from sklearn.model_selection import cross_val_score
    accuracies = []
    k_values = range(5, int(math.sqrt(X_train.shape[0]))+1)
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        accuracies.append(scores.mean())
    # Plot the results
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validated Accuracy')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(k_values)
    plt.show()  # we can see 5 is a good option
    exit()
    '''
    ### Grid Search with Cross-Validation
    knn = KNeighborsClassifier()
    # Define the grid of parameters to search

    if best_hp:
        print(f"Best Hyperparameters: {best_hp}")
        best_model = KNeighborsClassifier(n_neighbors=best_hp['n_neighbors'], weights=best_hp['weights'], metric=best_hp['metric'])
    else:
        # Perform grid search
        grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        # Print the best parameters and best score
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation Score: {grid_search.best_score_}")

    # Fit the best model with the whole training data
    best_model.fit(X_train, y_train)
    # Accuracy on train data
    train_pred_label = best_model.predict(X_train)
    train_accuracy = accuracy_score(train_pred_label, y_train)
    print(f"Train Data Accuracy: {train_accuracy}")


    # Accuracy on test data
    test_pred_label = best_model.predict(X_test)
    test_accuracy = accuracy_score(test_pred_label, y_test)
    print(f"Test Data Accuracy: {test_accuracy}")


    return best_model
