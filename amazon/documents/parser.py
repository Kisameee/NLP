from amazon.documents import Document


class Parser(object):
    """Classe parente pour tous les parsers"""
    def create(self):
        return self

    def read_file(self, filename: str) -> Document:
        with open(filename, 'r', encoding='utf-8') as fp:
            content = fp.read()
        return self.read(content)