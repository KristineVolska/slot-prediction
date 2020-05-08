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
from postprocessing import read_results

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


def train_tagger(load, file_path, nr_iter, suffix, part_tag):
    pos_tagger = PerceptronTagger(load=load, use_suffix=suffix, part_tag=part_tag)
    text = read_file(file_path)
    sentences = _read_tagged(text)
    return pos_tagger.train(sentences, save_loc="textblob_aptagger/model.pickle", nr_iter=nr_iter)


def test_data(input_file, pos_tagger):
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


def compare(input, output):
    text_before, context = read_results(input)
    text_after, context = read_results(output)
    wrong = 0
    for bef, aft in zip(text_before, text_after):
        if bef != aft:
            wrong += 1
    total = len(text_before)
    accuracy = round(total - wrong) / total
    print('Accuracy: {0}/{1}={2:0.4f}'.format(total-wrong, total, accuracy))
    return round(accuracy, 6)


def run(load, train_set, test_set, iter, suffix, part_tag):
    train_tagger(load, train_set, iter, suffix, part_tag)
    pos_tagger = PerceptronTagger(use_suffix=suffix, part_tag=part_tag)
    results = test_data(test_set, pos_tagger)
    compare(test_set, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", help="Word count before and after target", default=3)
    parser.add_argument("--iter", help="Number of training iterations", default=5)
    parser.add_argument('--random', help="Add this argument to generate a random english word for suffix", action='store_true')
    parser.add_argument('--suffix', help="Add this argument to use target word suffix analysis in training", action='store_true')
    parser.add_argument('--part_tag', help="Add this argument to separate tags into morphological features", action='store_true')
    parser.add_argument('--tag_iter', help="Add this argument to tag the test data set after each training iteration", action='store_true')
    parser.add_argument('--conf_m', help="Add this argument to create a confusion matrix", action='store_true')
    args = parser.parse_args()
    start = time.time()

    train = "https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-train.conllu"
    dev = "https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-dev.conllu"
    test = "https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-test.conllu"

    train = preprocessing(train, bool(args.random), int(args.context))
    validation = preprocessing(dev, bool(args.random), int(args.context))
    test = preprocessing(test, bool(args.random), int(args.context))

    # Tag validation data set after each iteration
    if bool(args.tag_iter):
        stop_criterion = False
        iter_n = 1
        print("Iteration", iter_n)
        train_tagger(False, train, 1, bool(args.suffix), bool(args.part_tag))
        while not stop_criterion:
            iter_n += 1
            print("Iteration", iter_n)
            stop_criterion = train_tagger(True, train, 1, bool(args.suffix), bool(args.part_tag))
        print("Starting the TEST data set tagging")
        results = test_data(test, PerceptronTagger(model_file="model_max.pickle", use_suffix=bool(args.suffix), part_tag=bool(args.part_tag)))
        print("Final results in TEST set:")
        compare(test, results)
    # Tag validation data set only after all training iterations
    else:
        run(False, train, validation, int(args.iter), bool(args.suffix), bool(args.part_tag))
        test = validation

    if bool(args.conf_m):
        print("Creating confusion matrix for", create_fn(test, "RESULTS", ".tsv"))
        draw_confusion(create_fn(test, "NOUNS", ".tsv"), test, create_fn(test, "RESULTS", ".tsv"))
    print("Execution time: ")
    print(str(timedelta(seconds=(time.time() - start))))


if __name__ == "__main__":
    main()
