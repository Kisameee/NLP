from unittest import TestCase

from amazon.documents import Token


class TestToken(TestCase):
    def setUp(self):
        self.object = Token(None, 0, 10, "pos", shape="shape", text="text")

    def test_text(self):
        self.assertEqual(self.object.text, "text", "wrong text")

    def test_pos(self):
        self.assertEqual(self.object.pos, "pos", "wrong pos")

    def test_shape(self):
        self.assertEqual(self.object.shape, "shape", "wrong shape")
