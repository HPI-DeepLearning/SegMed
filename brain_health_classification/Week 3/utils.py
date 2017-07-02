import keras
from keras.utils.vis_utils import model_to_dot
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def plot_history(model_results):
    ax = plt.figure().gca()
    ax.plot(model_results.history['loss'], label='train_loss')
    ax.plot(model_results.history['acc'], label='train_acc')
    ax.plot(model_results.history['val_loss'], label='val_loss')
    ax.plot(model_results.history['val_acc'], label='val_acc')
    ax.legend()
    ax.set_title('Model history')
    ax.set_xlabel('epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def plot_sample(ax, sample, title, shape):
    img = sample.reshape(*shape)
    ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.axis('off')
    ax.set_title(title)

def has_tumor(one_hot_vector):
    return one_hot_vector.argmax()
    
def plot_predicted_samples(count, samples, labels, predicted, main_title, shape):
    # Shuffle datapoints
    idx = np.random.choice(np.arange(samples.shape[0]), count, replace=False)
    samples, labels, predicted = (samples[idx], labels[idx], predicted[idx])
    cols = 4
    rows = count // cols
    assert rows * cols == count, 'Number of samples must be a multiple of 4'
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    for i, ax in enumerate(axes.flat):
        plot_sample(ax, samples[i], '#{}, Is HGG: {}, Predicted: {}'.format(
            idx[i], has_tumor(labels[i]), has_tumor(predicted[i])), shape)
    fig.suptitle(main_title)

def plot_samples(count, samples, labels, main_title):
    # Always the same results
    plot_predicted_samples(4, samples, labels, np.array(['-1'] * len(labels)), 'Testing set')
    
def plot_model(model, filename=None, show=True):
    if filename:
        keras.utils.plot_model(model, show_shapes=True, to_file=filename)
    if show:
	from IPython.display import SVG
        return SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
