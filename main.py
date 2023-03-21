import os
import hashlib
import numpy as np
from data_preprocessing import load_preprocess_data
from model import create_model
from visualization import plot_history, display_weights, display_dense_weights
from keras.models import load_model
from keras.utils import plot_model
from keras.layers import Conv2D, Dense


def data_hash(x_train, y_train):
    data = np.hstack((x_train.ravel(), y_train.ravel()))
    return hashlib.md5(data.tobytes()).hexdigest()


if __name__ == '__main__':
    num_classes = 10
    x_train, y_train, x_test, y_test = load_preprocess_data(num_classes)

    batch_size = 128
    epochs = 5

    model_file_prefix = f'model_{data_hash(x_train, y_train)}_{batch_size}_{epochs}'
    model_file_name = f'{model_file_prefix}.h5'

    if os.path.exists(model_file_name):
        # Load the memoized model
        print('Loading memoized model...')
        model = load_model(model_file_name)
    else:
        # Train a new model
        model = create_model()
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                x_test,
                y_test))

        # Save the trained model
        model.save(model_file_name)

        # Plot and save the training history
        plot_history(history)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plot_model(model, to_file=f'{model_file_prefix}.png',
               show_shapes=True, show_layer_names=True)

    # Iterate through the layers of the model
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D):
            # Retrieve the weights of the convolutional layer
            conv_weights = layer.get_weights()[0][:, :, 0, :]

            # Define an output file name for the weight plot
            output_file = f"conv_weights_layer_{i}.png"

            # Display the weights as images and save them as files
            display_weights(
                conv_weights, f"Convolutional Layer {i}", output_file)

    # Iterate through the layers of the model
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            # Retrieve the weights of the dense layer
            dense_weights = layer.get_weights()[0]

            # Define an output file name for the weight plot
            output_file = f"dense_weights_layer_{i}.png"

            # Display the weights as heatmaps and save them as files
            display_dense_weights(
                dense_weights, f"Dense Layer {i}", output_file)
