from keras.models import Sequential
from keras.layers import Dense, Input
from keras.metrics import BinaryAccuracy
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV


def train_nn(X_train,y_train,X_test,y_test,best_hp):
    # Define hyperparameter grid
    print('*** Training Neural Network')

    param_grid = {
        'batch_size': [16],
        'epochs': [100],
        'model__hidden_layers': [1],
        'model__neurons': [15],
        'model__optimizer': ['adam'],
        'model__activation': ['relu']
    }
    '''
    param_grid = {
        'batch_size': [8, 16],
        'epochs': [50,100],
        'model__hidden_layers': [1,2],
        'model__neurons': [3,9],
        'model__optimizer': ['adam'],
        'model__activation': ['relu']
    }
    '''
    def create_model(hidden_layers=1, neurons=5, activation='relu', optimizer='adam'):
        print('hidden_layers=', hidden_layers, 'neurons=', neurons, 'activation=', activation, 'optimizer=', optimizer)
        model_nn = Sequential()
        model_nn.add(
            Input(shape=(X_train.shape[1],)))  # Input layer

        for _ in range(hidden_layers):
            # use same neurons and activation function for all hidden layers
            model_nn.add(Dense(units=neurons, activation=activation))

        model_nn.add(Dense(units=1, activation='sigmoid'))  # output layer, use sigmoid for binary classification

        metric = BinaryAccuracy(
            name="binary_accuracy", dtype=None, threshold=0.5
        )
        model_nn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[metric])
        return model_nn

    if best_hp :
        best_model = create_model(best_hp['hidden_layers'],
                                  best_hp['neurons'],
                                  best_hp['activation'],
                                  best_hp['optimizer'])
        best_model.fit(X_train, y_train, batch_size=16, epochs=100, verbose=0)
    else :

        # Create a KerasClassifier
        nn = KerasClassifier(model=create_model,verbose=0)

        grid_search = GridSearchCV(estimator=nn, param_grid=param_grid, cv=2)
        grid_result = grid_search.fit(X_train, y_train)

        print("Best Parameters:", grid_result.best_params_)
        print("Best Cross-Validation Score:", grid_result.best_score_)

        # Access the best KerasClassifier and extract the model
        best_keras_classifier = grid_search.best_estimator_
        best_model = best_keras_classifier.model_

    loss, accuracy = best_model.evaluate(X_test, y_test)
    print('test data accuracy=', accuracy, ' , loss=', loss)
    return best_model