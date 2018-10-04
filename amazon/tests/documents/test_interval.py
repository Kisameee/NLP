from unittest import TestCase

from amazon.documents import Interval


class TestInterval(TestCase):
    def setUp(self):
        self.object = Interval(0, 10)

    def test_intersection(self, other = Interval(1, 5)):
        self.assertEqual(self.object.intersection(other), Interval(1, 5), "wrong intersection")

    def test_overlaps(self, other = Interval(1, 5)):
        self.assertEqual(self.object.overlaps(other), True, "wrong overlaps")