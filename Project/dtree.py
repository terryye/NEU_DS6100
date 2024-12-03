from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *

param_grid = {
    'max_depth': [3, 5],
    'min_samples_split': [3, 5 ],
    'min_samples_leaf': [3, 5, 10, 20],
    'min_impurity_decrease': [0.01, 0.001],
    'ccp_alpha': [0.01, 0.001, 0.0001]
}

def train_dtree(X_train,y_train,X_test,y_test):
    print('*** Training Decision Tree')
    # Define hyperparameter grid
    dtree = tree.DecisionTreeClassifier(criterion="gini")
    grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print('best_params =', best_params)

    # Accuracy on test data
    test_pred_label = best_model.predict(X_test)
    test_accuracy = accuracy_score(test_pred_label, y_test)
    print('test data accuracy =', test_accuracy)
    return best_model
