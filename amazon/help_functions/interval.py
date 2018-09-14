class Interval:
    """A class for representing a contiguous range of integers"""

    def __init__(self, start: int, end: int):
        """
        :param start: start of the range
        :param end: first integer not included the range
        """
        self.start = int(start)
        self.end = int(end)
        if self.start > self.end:
            raise ValueError('Start "{}" must not be greater than end "{}"'.format(self.start, self.end))
        if self.start < 0:
            raise ValueError('Start "{}" must not be negative'.format(self.start))

    def __len__(self):
        """ Return end - start """
        return self.end - self.start

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __ne__(self, other):
        return self.start != other.start or self.end != other.end

    def __lt__(self, other):
        return (self.start, -len(self)) < (other.start, -len(other))

    def __le__(self, other):
        return (self.start, -len(self)) <= (other.start, -len(other))

    def __gt__(self, other):
        return (self.start, -len(self)) > (other.start, -len(other))

    def __ge__(self, other):
        return (self.start, -len(self)) >= (other.start, -len(other))

    def __hash__(self):
        return hash(tuple(v for k, v in sorted(self.__dict__.items())))

    def __contains__(self, item: int):
        """ Return self.start <= item < self.end """
        return self.start <= item < self.end

    def __repr__(self):
        return 'Interval[{}, {}]'.format(self.start, self.end)

    def __str__(self):
        return repr(self)

    def intersection(self, other) -> 'Interval':
        """ Return the interval common to self and other """
        a, b = sorted((self, other))
        if a.end <= b.start:
            return Interval(self.start, self.start)
        return Interval(b.start, min(a.end, b.end))

    def overlaps(self, other) -> bool:
        """ Return True if there exists an interval common to self and other """
        a, b = sorted((self, other))
        return a.end > b.start

    def shift(self, i: int):
        self.start += i
        self.end += i
