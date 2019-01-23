import os
from typing import List

import numpy as np
from gensim.models import KeyedVectors

from amazon.data import DATA_DIR
from amazon.documents import Document


class Vectorizer:
    """ Transform a string into a vector representation"""

    def __init__(self, word_embedding_path: str):
        """
        :param word_embedding_path: path to gensim embedding file
        """
        # Load word embeddings from file
        self.word_embeddings = KeyedVectors.load_word2vec_format(os.path.join(DATA_DIR, 'glove.txt'), binary=False)
        # Create POS to index dictionary
        self.pos2index = {'PAD': 0, 'TO': 1, 'VBN': 2, "''": 3, 'WP': 4, 'UH': 5, 'VBG': 6, 'JJ': 7, 'VBZ': 8, '--': 9,
                          'VBP': 10, 'NN': 11, 'DT': 12, 'PRP': 13, ':': 14, 'WP$': 15, 'NNPS': 16, 'PRP$': 17,
                          'WDT': 18, '(': 19, ')': 20, '.': 21, ',': 22, '``': 23, '$': 24, 'RB': 25, 'RBR': 26,
                          'RBS': 27, 'VBD': 28, 'IN': 29, 'FW': 30, 'RP': 31, 'JJR': 32, 'JJS': 33, 'PDT': 34, 'MD': 35,
                          'VB': 36, 'WRB': 37, 'NNP': 38, 'EX': 39, 'NNS': 40, 'SYM': 41, 'CC': 42, 'CD': 43, 'POS': 44,
                          'LS': 45}
        # Create shape to index dictionary
        self.shapes = {'NL': 0, 'NUMBER': 1, 'SPECIAL': 2, 'ALL-CAPS': 3, '1ST-CAP': 4, 'LOWER': 5, 'MISC': 6}
        # Create labels to index dictionary
        self.indexes = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}

    def encode_features(self, documents: List[Document]):
        """
        Creates a feature matrix for all documents in the sample list
        :param documents: list of all samples as document objects
        :return: lists of numpy arrays for word, pos and shape features.
                 Each item in the list is a sentence, i.e. a list of indices (one per token)
        """

        max_length = max([len(document.tokens) for document in documents])

        words, pos, shapes = np.zeros((len(documents), max_length)), np.zeros((len(documents), max_length)), np.zeros(
        (len(documents), max_length))
        for i, document in enumerate(documents):
            for j, token in enumerate(document.tokens):
                if(token.pos != "#"):
                    pos[i][j] = self.pos2index[token.pos]
                    shapes[i][j] = self.shapes[token.shape]
                    if token.text.lower() in self.word_embeddings.index2word:
                        words[i][j] = self.word_embeddings.index2word.index(token.text.lower())
        return words, pos, shapes

    def encode_annotations(self, documents: List[Document]):
        """
        Creates the Y matrix representing the annotations (or true positives) of a list of documents
        :param documents: list of documents to be converted in annotations vector
        :return: numpy array. Each item in the list is a sentence, i.e. a list of labels (one per token)
        """

        labels = []
        for document in documents:
            labels.append(self.indexes[document.overall])
        return labels
