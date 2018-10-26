from amazon.documents import Document
from amazon.documents.parser import Parser


class SimpleTextParser(Parser):
    def read(self, content: str) -> Document:
        return Document().create_from_text(content)