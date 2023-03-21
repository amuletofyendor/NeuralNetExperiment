import os
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def create_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a CNN model for MNIST classification and memoize the model in a file.

    Returns:
        model: a Keras model instance
    """
    # Generate a file name based on the provided arguments
    file_name = f'model_{input_shape}_{num_classes}.pkl'

    # Check if the model file exists and load the model
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            model = pickle.load(f)
    else:
        # Create the model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        # Save the model to a file
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)

    return model
