import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    """
    Plot training history for accuracy and loss, and save the plots as images.

    Parameters:
        history: a Keras History object
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy_plot.png')  # Save the accuracy plot as an image
    plt.show()
    plt.close()  # Close the current figure before creating a new one

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss_plot.png')  # Save the loss plot as an image
    plt.show()
    plt.close()  # Close the current figure


def display_weights(layer_weights, layer_name, output_file):
    min_val = np.min(layer_weights)
    max_val = np.max(layer_weights)
    layer_weights = (layer_weights - min_val) / (max_val - min_val)

    num_filters = layer_weights.shape[-1]
    grid_size = int(np.ceil(np.sqrt(num_filters)))

    weight_image = np.empty(
        (grid_size * layer_weights.shape[0], grid_size * layer_weights.shape[1]))

    for i in range(num_filters):
        row, col = divmod(i, grid_size)
        weight_image[row * layer_weights.shape[0]:(row + 1) * layer_weights.shape[0], col * layer_weights.shape[1]:(
            col + 1) * layer_weights.shape[1]] = layer_weights[:, :, i]

    plt.imshow(weight_image, cmap='gray')
    plt.title(f"Weights of {layer_name}")
    plt.axis('off')
    plt.savefig(output_file, dpi=300)
    plt.clf()


def display_dense_weights(layer_weights, layer_name, output_file):
    plt.imshow(layer_weights, cmap='viridis', aspect='auto')
    plt.title(f"Weights of {layer_name}")
    plt.xlabel('Output Neurons')
    plt.ylabel('Input Neurons')
    plt.colorbar(label='Weight')
    plt.savefig(output_file, dpi=300)
    plt.clf()
