from unittest import TestCase

from amazon.documents import Document


class TestDocument(TestCase):
    def test_create_from_text(self):
        text = "As an example, consider the process of boarding a train." \
               " In which the reward is measured by the shape."

        doc = Document.create_from_text(text)
        self.assertEqual(len(doc.tokens), 22)
        self.assertEqual(len(doc.sentences), 2)
        self.assertEqual(doc.tokens[0].text, "As")
        self.assertEqual(doc.tokens[-1].text, ".")