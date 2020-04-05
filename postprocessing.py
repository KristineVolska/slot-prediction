import csv

import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
from openpyxl import load_workbook

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20, 16))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    x_label = 'Accuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass)
    plt.xlabel('Predicted label. {0}'.format(x_label))
    print(x_label)
    plt.savefig('confusion_matrix.png', dpi=300)


def read_results(file):
    with open(file, 'r', encoding="utf-8") as f:
        tsv = csv.reader(f, delimiter='\n')
        sentences = list()
        sentence = list()
        for row in tsv:
            try:
                sentence.append(row[0])
            except IndexError:
                sentences.append(sentence)
                context = len(sentence)
                sentence = list()
        return sentences, context


def draw_confusion(class_file, input, output):
    text_before, context = read_results(input)
    text_after, context = read_results(output)

    with open(class_file, 'r', encoding="utf-8") as file:
        tsv = csv.reader(file, delimiter='\n')
        classes = {}
        target_names = list()
        for i, row in enumerate(tsv):
            tag = row[0].split("\tNOUN;", maxsplit=1)[1]
            feats = tag.split(";")
            feat = ""
            for f in feats:
                abr = f.split("=")[1][:2]
                feat += abr
            classes[tag] = feat
            target_names.append(feat)

    columns = [-x for x in reversed(range(1, context // 2 + 1))] + ["target"] + [x for x in range(1, context // 2 + 1)]

    input_data = pd.DataFrame(text_before, columns=columns).fillna(0)
    output_data = pd.DataFrame(text_after, columns=columns).fillna(0)

    conf_table = pd.DataFrame(columns=['y_Actual', 'y_Predicted'])
    search_table = pd.DataFrame(index=None, columns=["tag_pair_key", "i", "index", "Actual", "Predicted"])

    for i in range(0, input_data.shape[0]):
        c_val = input_data.iloc[i]['target']
        p_val = output_data.iloc[i]['target']

        c_val = c_val.split("|NOUN;", maxsplit=1)[1]
        c_val = classes.get(c_val)
        try:
            p_val = p_val.split("|NOUN;", maxsplit=1)[1]
            p_val = classes.get(p_val)
            if p_val is None:
                p_val = "other"
        except IndexError:
            p_val = "other"
        conf_table = conf_table.append({'y_Actual': c_val, 'y_Predicted': p_val}, ignore_index=True)

        input_row = pd.DataFrame.from_dict(input_data.iloc[i].to_dict(), orient='index', columns=['Actual'])
        output_row = pd.DataFrame.from_dict(output_data.iloc[i].to_dict(), orient='index', columns=['Predicted'])
        actual_predicted = pd.concat([input_row, output_row], axis=1, sort=False)
        actual_predicted.reset_index(level=0, inplace=True)
        actual_predicted['tag_pair_key'] = "{0}-{1}".format(c_val, p_val)

        order_values = list()
        for x in range(1, context + 1):
            order_values.append("{0}.{1}".format(i, x))
        actual_predicted['i'] = order_values
        search_table = search_table.append(actual_predicted, ignore_index=True)
    target_names.append("other")
    target_names.sort()

    search_table = search_table.sort_values(by=['tag_pair_key', 'i'])
    conf_matrix = confusion_matrix(conf_table['y_Actual'].to_numpy(), conf_table['y_Predicted'].to_numpy())
    file = "Confusion_matrix.xlsx"
    writer = pd.ExcelWriter(file, engine='openpyxl')
    writer.book = load_workbook(file)
    pd.DataFrame(conf_matrix, index=target_names, columns=target_names).to_excel(writer, 'input')
    writer.save()

    search_table.to_excel(writer, 'search', index=False)
    writer.save()

    plot_confusion_matrix(cm=conf_matrix,
                          normalize=False,
                          target_names=target_names,
                          title="Confusion Matrix")
