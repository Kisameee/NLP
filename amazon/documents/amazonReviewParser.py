from amazon.documents import Document
from amazon.documents.parser import Parser


class AmazonReviewParser(Parser):
    @staticmethod
    def read(content: str):
        """Reads the content of an amazon data file and returns one document instance per document it finds."""
        import json
        documents = []
        for line in content.splitlines():
            comment = json.loads(line)
            doc = Document.create_from_text(comment["reviewText"])
            doc.overall = comment["overall"]
            documents.append(doc)
            # documents.append(create_from_text())
            # Split lines and loop over them
            # Read json with: data = json.loads(line)
            # Instantiate Document object from "reviewText" and label from "overall"

        return documents
