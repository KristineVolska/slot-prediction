import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from openpyxl import load_workbook


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
    contains_other = False

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
                contains_other = True
        except IndexError:
            p_val = "other"
            contains_other = True
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
    if contains_other:
        target_names.append("other")
    target_names.sort()

    search_table = search_table.sort_values(by=['tag_pair_key', 'i'])
    conf_matrix = confusion_matrix(conf_table['y_Actual'].to_numpy(), conf_table['y_Predicted'].to_numpy())
    file = "Confusion_matrix.xlsx"
    writer = pd.ExcelWriter(file, engine='openpyxl')
    try:
        writer.book = load_workbook(file)
    except KeyError:
        print("ERROR: The existing file", file, "is corrupted. Creating a new file.")
        writer = pd.ExcelWriter("Confusion_matrix_2.xlsx", engine='openpyxl')

    pd.DataFrame(conf_matrix, index=target_names, columns=target_names).to_excel(writer, 'input')
    search_table.to_excel(writer, 'search', index=False)
    writer.save()
