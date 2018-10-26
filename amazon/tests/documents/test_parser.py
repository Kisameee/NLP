import json
from unittest import TestCase


class TestParser(TestCase):
    def test_parser_json_is_true(self):
        file = json.load("/Users/HERNANDEZPierre/PycharmProjects/NLP/amazon/data/digital_music_reviews.json")

        self.assertEqual(file.