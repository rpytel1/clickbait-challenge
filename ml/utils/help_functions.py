import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Citation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['non-clickbaits', 'clickbaits']


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    save_title = '_'.join([word.lower() for word in title.split()])

    plt.savefig('../../figures/' + save_title + '.png', bbox_inches='tight')


def compute_diffs(truthClass_test, truthClass_pred, truthMean_test, truthMean_pred):
    miss_diffs = []
    corr_diffs = []
    for ct, cp, mt, mp in zip(truthClass_test, truthClass_pred, truthMean_test, truthMean_pred):
        if ct != cp:
            miss_diffs += [abs(mt - mp)]
        else:
            corr_diffs += [abs(mt - mp)]

    avg_miss_diff = np.mean(miss_diffs)
    print("Number of missclassified: {}".format(len(miss_diffs)))
    print("Average difference of missclassified: {}\n".format(avg_miss_diff))

    avg_corr_diff = np.mean(corr_diffs)
    print("Number of correctly classified: {}".format(len(corr_diffs)))
    print("Average difference of correctly classified: {}\n".format(avg_corr_diff))