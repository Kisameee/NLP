import json
import os
from unittest import TestCase

from amazon.data import DATA_DIR
from amazon.documents.amazonReviewParser import AmazonReviewParser


class TestParser(TestCase):

    def test_read_file(self, file = os.join(DATA_DIR, 'digital_music_reviews.json')):
        self.assertEqual(self.read_file(file).length(), 64706, "Not enough documents founded")