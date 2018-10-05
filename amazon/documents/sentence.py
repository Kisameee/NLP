from amazon.documents.interval import Interval


class Sentence(Interval):
    """ Interval corresponding to a Sentence"""

    def __init__(self, document, start: int, end: int):
        Interval.__init__(self, start, end)
        self._doc = document

    def __repr__(self):
        return 'Sentence({}, {})'.format(self.start, self.end)

    @property
    def tokens(self):
        """Returns the list of tokens contained in a sentence"""
        list = []
        for token in self._doc.tokens:
            if Interval.overlaps(self, token):
                list.append(token)
        return list

        #return [token for token in self._doc.tokens if Interval.overlaps(self, token)]