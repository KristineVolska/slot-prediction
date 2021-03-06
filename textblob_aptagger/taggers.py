# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import random
from collections import defaultdict
import pickle
import logging
from textblob.base import BaseTagger
from textblob.exceptions import MissingCorpusError
from textblob_aptagger._perceptron import AveragedPerceptron
from array import *

PICKLE = "model.pickle"


class PerceptronTagger(BaseTagger):
    '''
    Code modified from:
    Greedy Averaged Perceptron tagger, as implemented by Matthew Honnibal.
    See more implementation details here:
        http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
    :param load: Load the pickled model upon instantiation.
    '''

    START = "START"
    END = "END"

    def __init__(self, load=True, model_file=PICKLE, use_suffix=False, part_tag=False, use_dropout=False, use_comb=0):
        load_path = os.path.join(os.path.dirname(__file__), model_file)
        logging.basicConfig(filename="logging_file.txt", level=logging.INFO)
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        self.use_suffix = use_suffix
        self.part_tag = part_tag
        self.use_dropout = use_dropout
        self.use_comb = use_comb
        self.dev_accuracy = array('f', [0])
        if load:
            self.load(load_path)

    def tag(self, corpus, tokenize=False):
        '''Tags a string `corpus`.'''
        # Assume untokenized corpus has \n between sentences and ' ' between words
        s_split = lambda t: t.split('\n')
        w_split = lambda s: s.split()

        def split_sents(corpus):
            for s in s_split(corpus):
                yield w_split(s)

        tokens = []
        for words in split_sents(corpus):
            context_len = len(words) // 2
            for i, word in enumerate(words):
                word_n_tag = tuple(map(str, word.split('|')))
                tag = word_n_tag[1]
                word = word_n_tag[0]
                if tag == "None":
                    w_t = tuple(tuple(map(str, item.split('|'))) for item in words)
                    features = self._get_features(i, context_len, word, w_t)
                    tag = self.model.predict(features)
                tokens.append((word, tag))
        return tokens

    def train(self, sentences, save_loc=None, nr_iter=5):
        '''Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.
        :param sentences: A list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        '''
        self._count_classes(sentences)
        self.model.classes = self.classes
        for iter_ in range(nr_iter):
            c = 0
            n = 0
            random.shuffle(sentences)
            for words, tags in sentences:
                context_len = len(words) // 2
                for i, word in enumerate(words):
                    if i == context_len:
                        feats = self._get_features(i, context_len, word, tuple(zip(words, tags)))
                        if self.use_dropout:
                            th = random.random()
                            feats = defaultdict(int, {k: v for k, v in feats.items() if random.random() > th})
                        guess = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                        c += guess == tags[i]
                        n += 1
            logging.info("Iter {0}: {1}/{2}={3}".format(iter_ + 1, c, n, _pc(c, n)))
            print("Iter {0}: {1}/{2}={3}".format(iter_ + 1, c, n, _pc(c, n)))
        self.model.average_weights()
        if nr_iter == 1:
            cr_iter = 3
            dev_input = "lv_lvtb-ud-dev_input.tsv"
            import run
            dev_results = run.test_data(dev_input, self)
            curr_accuracy = run.compare(dev_input, dev_results)
            self.dev_accuracy.append(curr_accuracy)
            max_acc = max(self.dev_accuracy)
            if round(max_acc, 6) <= curr_accuracy:
                pickle.dump((self.model.weights, self.tagdict, self.classes, self.dev_accuracy),
                            open("textblob_aptagger/model_max.pickle", 'wb'), -1)
            max_index = self.dev_accuracy.index(max_acc)
            size = len(self.dev_accuracy)
            if size - 1 - cr_iter == max_index:
                print("Stopping criterion is met, stopping the training")
                return True
            else:
                # Pickle as a binary file
                if save_loc is not None:
                    pickle.dump((self.model.weights, self.tagdict, self.classes, self.dev_accuracy),
                                open(save_loc, 'wb'), -1)
        else:
            # Pickle as a binary file
            if save_loc is not None:
                pickle.dump((self.model.weights, self.tagdict, self.classes, self.dev_accuracy),
                            open(save_loc, 'wb'), -1)
        return None

    def load(self, loc):
        '''Load a pickled model.'''
        try:
            w_td_c = pickle.load(open(loc, 'rb'))
        except IOError:
            msg = ("Missing model.pickle file.")
            raise MissingCorpusError(msg)
        self.model.weights, self.tagdict, self.classes, self.dev_accuracy = w_td_c
        self.model.classes = self.classes
        return None

    def _get_features(self, i, context_len, word, context):
        '''Map tokens into a feature representation, implemented as a
        {hashable: float} dict. If the features change, a new model must be
        trained.
        '''

        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add('bias')
        for j in range(1, context_len + 1):
            try:
                prev_tag = context[i - j][1]
            except IndexError:
                if i - j < 0:
                    prev_tag = self.START
                else:
                    prev_tag = context[i - j]
            if self.part_tag:
                prev_tag = prev_tag.split(';')
                for p, prev_feat in enumerate(prev_tag):
                    curr_feat = prev_feat.split('=')
                    if p == 0:
                        add('Pos_i-{0}'.format(j), curr_feat[0])
                    else:
                        add('{0}_i-{1}'.format(curr_feat[0], j), curr_feat[1])
            else:
                if self.use_comb == 1:  # Use one combined feature representing context_len words before target
                    add('i-{0} tag'.format(j), prev_tag)
                    tag_list = list()
                    if j != 1 and j == context_len:
                        for t in range(1, j + 1):
                            tag_list.append(context[i - t][1])
                        add('{0} prev tags'.format(j), *tag_list)
                elif self.use_comb == 2:  # Use all combined features
                    tag_list = list()
                    for t in range(1, j + 1):
                        tag_list.append(context[i - t][1])
                    add('i-{0} tag'.format(j), prev_tag)
                    if j != 1:
                        add('{0} prev tags'.format(j), *tag_list)
                else:
                    add('i-{0} tag'.format(j), prev_tag)

        if self.use_suffix:
            for k in range(1, 5):
                add('ch-{0}'.format(k), word[-k:])

        for j in range(1, context_len + 1):
            try:
                next_tag = context[i + j][1]
            except IndexError:
                if i + j >= len(context):
                    next_tag = self.END
                else:
                    next_tag = context[i + j]
            if self.part_tag:
                next_tag = next_tag.split(';')
                for n, next_feat in enumerate(next_tag):
                    curr_feat = next_feat.split('=')
                    if n == 0:
                        add('Pos_i+{0}'.format(j), curr_feat[0])
                    else:
                        add('{0}_i+{1}'.format(curr_feat[0], j), curr_feat[1])
            else:
                if self.use_comb == 1:  # Use one combined feature representing context_len words after target
                    add('i+{0} tag'.format(j), next_tag)
                    tag_list = list()
                    if j != 1 and j == context_len:
                        for t in range(1, j + 1):
                            tag_list.append(context[i + t][1])
                        add('{0} next tags'.format(j), *tag_list)
                elif self.use_comb == 2:  # Use all combined features
                    tag_list = list()
                    for t in range(1, j + 1):
                        tag_list.append(context[i + t][1])
                    add('i+{0} tag'.format(j), next_tag)
                    if j != 1:
                        add('{0} next tags'.format(j), *tag_list)
                else:
                    add('i+{0} tag'.format(j), next_tag)

        return features

    def _count_classes(self, sentences):
        for words, tags in sentences:
            context_len = len(words)//2
            for i, tag in enumerate(tags):
                if i == context_len:
                    self.classes.add(tag)


def _pc(n, d):
    return (float(n) / d) * 100
