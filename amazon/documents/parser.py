class Parser(object):
    """Classe parente pour tous les parsers"""
    def create(self):
        return self

    def read_file(self, filename: str) -> Document:
        with open(filename, 'r', encoding='utf-8') as fp:
            content = fp.read()
        return self.read(content)

class SimpleTextParser(Parser):
    def read(self, content: str) -> Document:
        return Document().create_from_text(content)