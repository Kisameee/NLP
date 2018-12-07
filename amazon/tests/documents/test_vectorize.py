import os
from unittest import TestCase

from amazon.data import DATA_DIR
from amazon.documents import Vectorizer
from amazon.documents.amazonReviewParser import AmazonReviewParser


class TestVectorizer(TestCase):

    def test_vectorize(self):
        document = AmazonReviewParser().read_file(os.path.join(DATA_DIR, 'test.json'))
        v = Vectorizer(os.path.join(DATA_DIR, 'glove.txt'))
        words, pos, shapes = v.encode_features(document)
        self.assertEqual(len(words.tolist()), 13, "mauvaise liste mots")
        self.assertEqual(len(pos.tolist()), 13, "mauvaise liste pos")
        self.assertEqual(len(shapes.tolist()), 13, "mauvaise shape list")
