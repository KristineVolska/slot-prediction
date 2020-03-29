import argparse
import time
from textblob import TextBlob
from textblob_aptagger.taggers import PerceptronTagger
from io import open
import csv
from preprocessing import preprocessing
from preprocessing import create_fn
from datetime import timedelta
import pandas as pd
import seaborn as sn

def _read_tagged(text, sep='|'):
    sentences = []
    for sent in text.split('\n'):
        tokens = []
        tags = []
        for token in sent.split():
            word, pos = token.split(sep)
            tokens.append(word)
            tags.append(pos)
        sentences.append((tokens, tags))
    return sentences


def read_file(file):
    with open(file, 'r', encoding="utf-8") as f:
        tsv = csv.reader(f, delimiter='\n')
        text = ""
        for row in tsv:
            try:
                text += " {0}".format(row[0])
            except IndexError:
                text += "\n"
        return text


def prepare_test_data(sentences):
    test_sentences = []
    for j, sentence in enumerate(sentences):
        test_string = ""
        for i, word in enumerate(sentence[0]):
            if i == len(sentence[0])//2:
                test_string += " {0}|None".format(word)
            else:
                test_string += " {0}|{1}".format(word, sentence[1][i])
        test_string += "\n"
        test_sentences.append(test_string)
    return test_sentences


def train_tagger(file_path):
    pos_tagger = PerceptronTagger(load=False)
    text = read_file(file_path)
    sentences = _read_tagged(text)
    pos_tagger.train(sentences, save_loc="textblob_aptagger/model.pickle")


def test_data(input_file):
    pos_tagger = PerceptronTagger()
    text = read_file(input_file)
    sentences = _read_tagged(text)
    test_sentences = prepare_test_data(sentences)
    result_file = create_fn(input_file, "RESULTS", ".tsv")
    with open(result_file, "w+", newline='', encoding="utf-8") as out:
        for sentence in test_sentences:
            blob = TextBlob(sentence, pos_tagger=pos_tagger)
            row = str(blob.tags).replace("[('", "").replace("', '", "|").replace("'), ('", "\r\n").replace("')]", "\r\n\r\n")
            out.write(row)
    return result_file


def read_results(file):
    with open(file, 'r', encoding="utf-8") as f:
        tsv = csv.reader(f, delimiter='\n')
        sentences = list()
        text = ""
        for row in tsv:
                try:
                    if "-START-" not in row[0] and "-END-" not in row[0]:
                        text += " {0}".format(row[0])
                except IndexError:
                    sentences.append(text)
                    text = ""
        return sentences


def compare(input, output):

    text_before = read_results(input)
    text_after = read_results(output)
    wrong = 0

    with open("differences.tsv", "w+", newline='', encoding="utf-8") as f:
        output = csv.writer(f, delimiter='\n')
        for bef, aft in zip(text_before, text_after):
            if bef != aft:
                output.writerow([bef])
                output.writerow([aft])
                output.writerow([])
                wrong += 1
    total = len(text_before)
    print("Accuracy:")
    print(round((total - wrong)/total * 100, 2), "%")


def draw_confusion(class_file, diffs_file):
    with open(class_file, 'r', encoding="utf-8") as file:
        tsv = csv.reader(file, delimiter='\n')
        classes = {}
        for row in tsv:
            tag = row[0].split("\tNOUN;", maxsplit=1)[1]
            feats = tag.split(";")
            feat = ""
            for f in feats:
                abr = f.split("=")[1][:2]
                feat += abr
            classes[tag] = feat

    diffs = pd.read_csv(diffs_file, sep=' ', header=None).fillna(0)
    conf_table = pd.DataFrame(columns=['y_Actual', 'y_Predicted'])

    for i in range(0, diffs.shape[0], 2):
        correct_row = diffs.iloc[i]
        wrong_row = diffs.iloc[i + 1]
        for c_val, w_val in zip(correct_row, wrong_row):
            if c_val != w_val:
                c_val = c_val.split("|NOUN;", maxsplit=1)[1]
                c_val = classes.get(c_val)
                try:
                    w_val = w_val.split("|NOUN;", maxsplit=1)[1]
                    w_val = classes.get(w_val)
                except IndexError:
                    w_val = "other"
                conf_table = conf_table.append({'y_Actual': c_val, 'y_Predicted': w_val}, ignore_index=True)

    confusion_matrix = pd.crosstab(conf_table['y_Actual'], conf_table['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    fig = sn.heatmap(confusion_matrix, annot=True, xticklabels=True, yticklabels=True, linewidths=.01).get_figure()
    fig.set_size_inches(20, 15)
    fig.savefig('confusion_matrix.png', dpi=300)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", help="Word count before and after target")
    args = parser.parse_args()

    start = time.time()

    dev = "https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-dev.conllu"
    train = "https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-train.conllu"

    test = preprocessing(dev, int(args.context))
    train = preprocessing(train, int(args.context))

    train_tagger(train)
    results = test_data(test)

    compare(test, results)
    elapsed = (time.time() - start)

    # Draw confusion matrix
    draw_confusion(create_fn(test, "NOUNS", ".tsv"), "differences.tsv")

    print("Execution time: ")
    print(str(timedelta(seconds=elapsed)))

if __name__ == "__main__":
    main()
