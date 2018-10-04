from amazon.documents.interval import Interval


class Token(Interval):
    """ A Interval representing word like units of text with a dictionary of features """

    def __init__(self, document, start: int, end: int, pos: str, shape: str, text: str):
        """
        Note that a token has 2 text representations.
        1) How the text appears in the original document e.g. doc.text[token.start:token.end]
        2) How the tokeniser represents the token e.g. nltk.word_tokenize('"') == ['``']
        :param document: the document object containing the token
        :param start: start of token in document text
        :param end: end of token in document text
        :param pos: part of speach of the token
        :param shape: integer label describing the shape of the token
        :param text: this is the text representation of token
        """

        Interval.__init__(self, start, end)
        self._doc = document
        self._pos = pos
        self._shape = shape
        self._text = text

    @property
    def text(self):
        return self._text

    @property
    def pos(self):
        return self._pos

    @property
    def shape(self):
        return self._shape



    def __repr__(self):
        return 'Token({}, {}, {})'.format(self._text, self._start, self._end)
