from amazon.documents import Document
from amazon.documents.parser import Parser


class AmazonReviewParser(Parser):
    def read(self, content: str) -> Document:
        """Reads the content of an amazon data file and returns one document instance per document it finds."""
        import json
        documents = []
        all_json = json.load(content)
        for comment in all_json:
            doc = Document.create_from_text(comment['review_text'])
            doc.overall = comment['overall']
            documents.append(doc)
        #documents.append(create_from_text())
        # Split lines and loop over them
        # Read json with: data = json.loads(line)
        # Instantiate Document object from "reviewText" and label from "overall"

        return documents