import os
from unittest import TestCase

from amazon.data import DATA_DIR
from amazon.documents.recurrentNeuralNetwork import RecurrentNeuralNetwork
from amazon.documents.vectorizer import Vectorizer


class TestRecurrentNeuralNetwork(TestCase):
    def test_initClass(self):
        self.object = RecurrentNeuralNetwork()

    def test_typeOfClass(self):
        bc = RecurrentNeuralNetwork.build_classification(Vectorizer(os.path.join(DATA_DIR, 'glove.txt')).word_embeddings,
                                                                          {'pos':(25, 10),'shape':(10,5)}, 5)
        bs = RecurrentNeuralNetwork.build_sequence(Vectorizer(os.path.join(DATA_DIR, 'glove.txt')).word_embeddings,
                                                   {'pos':(25, 10),'shape':(10,5)}, 5,)
        self.assertIsInstance(bc, RecurrentNeuralNetwork,"Not an instance")
        self.assertIsInstance(bs, RecurrentNeuralNetwork,"Not an instance")
