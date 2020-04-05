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


def train_tagger(file_path, nr_iter, suffix, part_tag):
    pos_tagger = PerceptronTagger(load=False, use_suffix=suffix, part_tag=part_tag)
    text = read_file(file_path)
    sentences = _read_tagged(text)
    pos_tagger.train(sentences, save_loc="textblob_aptagger/model.pickle", nr_iter=nr_iter)


def test_data(input_file, suffix, part_tag):
    pos_tagger = PerceptronTagger(use_suffix=suffix, part_tag=part_tag)
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
    parser.add_argument("--context", help="Word count before and after target", default=3)
    parser.add_argument("--iter", help="Number of training iterations", default=5)
    parser.add_argument('--suffix', help="Add this argument to use suffix analysis in training", action='store_true')
    parser.add_argument('--part_tag', help="Add this argument to separate tags into morphological features", action='store_true')
    args = parser.parse_args()
    start = time.time()

    dev = "https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-dev.conllu"
    train = "https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-train.conllu"

    test = preprocessing(dev, int(args.context))
    train = preprocessing(train, int(args.context))
    train_tagger(train, int(args.iter), bool(args.suffix), bool(args.part_tag))
    results = test_data(test, bool(args.suffix), bool(args.part_tag))

    # Draw confusion matrix
    draw_confusion(create_fn(test, "NOUNS", ".tsv"), test, results)

    print("Execution time: ")
    print(str(timedelta(seconds=(time.time() - start))))


if __name__ == "__main__":
    main()
