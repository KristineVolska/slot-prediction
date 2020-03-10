import argparse

from textblob import TextBlob
from textblob_aptagger.taggers import PerceptronTagger
from io import open
import csv


def _read_tagged(text, sep='|'):
    sentences = []
    for sent in text.split('\n'):
        tokens = []
        tags = []
        for token in sent.split():
            try:
                word, pos = token.split(sep)
                tokens.append(word)
                tags.append(pos)
            except ValueError:
                if token in ["-START-", "-END-"]:
                    tokens.append(token)
                    tags.append(token)  # add a dummy tag (placeholder)
                else:
                    pass
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
            if i == 1:
                test_string += " {0}".format(word)
            else:
                if sentence[0][i] in ["-START-", "-END-"]:
                    pass
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
    with open("textblob_aptagger/test_results.tsv", "w+", newline='', encoding="utf-8") as out:
        for sentence in test_sentences:
            blob = TextBlob(sentence, pos_tagger=pos_tagger)
            row = str(blob.tags).replace("[('", "").replace("', '", "|").replace("'), ('", "\r\n").replace("')]", "\r\n\r\n")
            out.write(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Input file")
    parser.add_argument("--train", help="Boolean. 'true' if model training is necessary")
    args = parser.parse_args()
    if args.train == "true":
        train_tagger(args.data)
    else:
        test_data(args.data)

if __name__ == "__main__":
    main()
