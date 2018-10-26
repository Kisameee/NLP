import json
import os
from unittest import TestCase

from amazon.data import DATA_DIR


class TestParser(TestCase):
    def test_parser_json_is_true(self):
        file = json.load(os.join(DATA_DIR, 'digital_music_reviews.json'))

