import argparse
import time
from textblob import TextBlob
from textblob_aptagger.taggers import PerceptronTagger
from io import open
import csv
from preprocessing import preprocessing
from preprocessing import create_fn

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", help="Word count before and after target")
    args = parser.parse_args()

    dev = "https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-dev.conllu"
    train = "https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-train.conllu"

    test = preprocessing(dev, int(args.context))
    train = preprocessing(train, int(args.context))

    start = time.time()
    start = start/60

    train_tagger(train)
    results = test_data(test)

    compare(test, results)
    end = time.time()
    end = end/60

    print("Execution time: ")
    print(round(end - start, 2), "sec")

if __name__ == "__main__":
    main()
