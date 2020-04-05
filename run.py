import argparse
import time
from textblob import TextBlob
from textblob_aptagger.taggers import PerceptronTagger
from io import open
import csv
from preprocessing import preprocessing
from preprocessing import create_fn
from datetime import timedelta
from postprocessing import draw_confusion

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
            if i == len(sentence[0]) // 2:
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
            row = str(blob.tags).replace("[('", "").replace("', '", "|").replace("'), ('", "\r\n").replace("')]",
                                                                                                           "\r\n\r\n")
            out.write(row)
    return result_file


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

    # Draw confusion matrix
    draw_confusion(create_fn(test, "NOUNS", ".tsv"), test, results)

    print("Execution time: ")
    print(str(timedelta(seconds=(time.time() - start))))


if __name__ == "__main__":
    main()
