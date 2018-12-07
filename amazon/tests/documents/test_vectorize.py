import os
from unittest import TestCase

from amazon.data import DATA_DIR
from amazon.documents import Vectorizer
from amazon.documents.amazonReviewParser import AmazonReviewParser


class TestVectorizer(TestCase):

    def test_vectorise(self):
        document = (AmazonReviewParser().read_file(filename=os.path.join(DATA_DIR, 'digital_music_reviews.json')))
        v = Vectorizer(os.path.join(DATA_DIR, 'glove.txt'))
        words, pos, shapes = v.encode_features(document)
        self.assertEqual(words.tolist() == [[13075, 85, 805]])
        self.assertEqual(pos.tolist() == [[38, 11, 21]])
        self.assertEqual(shapes.tolist() == [[4, 5, 2]])
