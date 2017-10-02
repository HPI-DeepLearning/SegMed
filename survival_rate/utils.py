from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.utils.vis_utils import model_to_dot
from matplotlib.ticker import MaxNLocator

def plot_history(model_results, filename=None):
    ax = plt.figure().gca()
    ax.plot(model_results.history['aux_output_mse'], label='aux_output_mse')
    ax.plot(model_results.history['main_output_mse'], label='main_output_mse')
    ax.plot(model_results.history['val_aux_output_mse'], label='val_aux_output_mse')
    ax.plot(model_results.history['val_main_output_mse'], label='val_main_output_mse')
    ax.legend()
    ax.set_title('Model history')
    ax.set_xlabel('epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if filename:
        ax.get_figure().savefig(filename)
        
    
def plot_model(model, filename=None):
    if filename:
        keras.utils.plot_model(model, show_shapes=True, to_file=filename)
    else:
        from IPython.display import SVG
        return SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

def plot_samples(X1, X2, Y, predicted=None, samples=10, seed=0):
    fig, axes = plt.subplots(samples, 7, figsize=(16, samples*2))
    np.random.seed(seed)
    choices = np.random.choice(np.arange(len(X1)), samples)
    for i, patient in enumerate(choices):
        age = int(X2[patient])
        result = '({})'.format(int(predicted[patient])) if predicted is not None else ''
        survival_rate = int(Y[patient])
        for j in range(7):
            axes[i][j].imshow(X1[patient,j])
        axes[i][0].set_title('#{} & {} -> {}{}'.format(patient, age, survival_rate, result))
    fig.suptitle('Samples: #ID & Age -> SR{}'.format('(Predicted SR)' if predicted is not None else ''))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.subplots_adjust(top=0.85)


def as_classes(days, classes):
    percents = np.linspace(0, 1, classes+1)[1:-1]
    cuts = [0] + list(days.quantile(percents)) + [days.max()+1]
    results = np.zeros((len(days), classes))
    print('Class seperators: {}'.format(cuts))
    groups = [[]] * classes
    for i in range(classes):
        groups[i] = (cuts[i] <= days) & (days < cuts[i+1])
        print('Group {} has {} elements'.format(i, sum(groups[i])))
        results[groups[i], i] = 1
    return results

def plot_corr(df, size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    x = ax.matshow(corr)
    plt.colorbar(x)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    ax.set_title('Pearson correlation matrix')
    
def plot_pixels_vs_survival(df, X1_train, X1_test, X1_val, Y_train, Y_test, Y_val, MAX_SURVIVAL, **kwargs):
    # _, ax = plt.subplots(1, 1)
    # x = ax.scatter(df['TSize'], df['Survival'], label='Whole dataset', c=df['Age'])
    # plt.colorbar(x)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].scatter(X1_train[:, :3].sum(axis=(1, 2, 3, 4)), Y_train * MAX_SURVIVAL, label='train')
    axes[0].set_xlabel('Pixels')
    axes[0].set_ylabel('Days')
    axes[0].set_title('Size of tumor vs survival rate - training set')
    axes[1].scatter(X1_test[:, :3].sum(axis=(1, 2, 3, 4)), Y_test * MAX_SURVIVAL, label='test')
    axes[1].set_xlabel('Pixels')
    axes[1].set_ylabel('Days')
    axes[1].set_title('Size of tumor vs survival rate - test set')
    axes[2].scatter(X1_val[:, :3].sum(axis=(1, 2, 3, 4)), Y_val * MAX_SURVIVAL, label='val')
    axes[2].set_xlabel('Pixels')
    axes[2].set_ylabel('Days')
    axes[2].set_title('Size of tumor vs survival rate - validation set')
    fig.tight_layout()
    fig.savefig('analysis/pixels_vs_survival_rate.jpg')

def plot_age_vs_survival(df, X2_train, X2_test, X2_val, Y_train, Y_test, Y_val, MAX_SURVIVAL, **kwargs):
    # _, ax = plt.subplots(1, 1)
    # ax.scatter(df['Age'], df['Survival'], label='Whole dataset')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].scatter(X2_train.argmax(axis=1), Y_train * MAX_SURVIVAL, label='train')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Days')
    axes[0].set_title('Age vs survival rate - training set')
    axes[1].scatter(X2_test.argmax(axis=1), Y_test * MAX_SURVIVAL, label='test')
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Days')
    axes[1].set_title('Age vs survival rate - test set')
    axes[2].scatter(X2_val.argmax(axis=1), Y_val * MAX_SURVIVAL, label='val')
    axes[2].set_xlabel('Age')
    axes[2].set_ylabel('Days')
    axes[2].set_title('Age vs survival rate - validation set')
    fig.tight_layout()
    fig.savefig('analysis/age_vs_survival_rate.jpg')
    
def plot_tumor_position(X):
    choices = np.random.choice(range(X.shape[0]), 10)
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    for layer, ax in enumerate(axes.flat[:7]):
        for patient in choices:
            region = X[patient, layer, :, :, :]
            ax.plot(region.sum(axis=(1,2)))
        ax.set_xlabel('Slices')
        ax.set_ylabel('Amount')
    axes.flat[0].set_title('Pixels of tumor region #1')
    axes.flat[1].set_title('Pixels of tumor region #2')
    axes.flat[2].set_title('Pixels of tumor region #3')
    axes.flat[3].set_title('Pixels of brain modality #1')
    axes.flat[4].set_title('Pixels of brain modality #2')
    axes.flat[5].set_title('Pixels of brain modality #3')
    axes.flat[6].set_title('Pixels of brain modality #4')
    fig.tight_layout()
    fig.savefig('analysis/slices_amount_of_pixels.jpg')