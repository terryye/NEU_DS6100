from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *

param_grid = {
    'n_neighbors': list(range(5, 21)),  # Values of k
    'weights': ['uniform', 'distance'],  # Weighting scheme
    'metric': ['euclidean', 'manhattan']  # Distance metrics
}


def train_knn(X_train,y_train,X_test,y_test):
    print('*** Training KNN')

    # KNN Model
    '''
    #Elbow Method to find the best value of k
    from sklearn.model_selection import cross_val_score
    accuracies = []
    k_values = range(1, 21)
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_pca_df, y_train_pca, cv=5, scoring='accuracy')
        accuracies.append(scores.mean())
    # Plot the results
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validated Accuracy')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(k_values)
    plt.show()  # we can see 5 is a good option
    '''

    ### Grid Search with Cross-Validation
    knn = KNeighborsClassifier()
    # Define the grid of parameters to search

    # Perform grid search
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    # Print the best parameters and best score
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_}")

    # Accuracy on test data
    test_pred_label = best_model.predict(X_test)
    test_accuracy = accuracy_score(test_pred_label, y_test)
    print(f"Test Data Accuracy: {test_accuracy}")


    return best_model
