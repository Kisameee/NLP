import re
from typing import List
import nltk
from amazon.documents import *
from .token import Token
from .sentence import Sentence


class Document:
    """
    A document is a combination of text and the positions of the tags and elements in that text.
    """

    def __init__(self) -> None:
        """
        Constructor of the Document class
        """
        self.text = None
        self.tokens = None
        self.sentences = None
        self.overall = None

    @classmethod
    def create_from_text(cls, text: str = None) -> 'Document':
        """
        Initialize a Document object from a text
        :param text: document text as a string
        :return: The document text as Document object
        """
        doc = Document()
        doc.text = text
        # 1. Tokenize text (tokens & sentences)
        words, pos_tags = zip(*nltk.pos_tag(nltk.word_tokenize(text)))
        sentences = nltk.sent_tokenize(text.replace('\n', ' '))
        # 2. Find tokens intervals
        doc.tokens = Document._find_tokens(doc, words, pos_tags, text)
        # 3. Find sentences intervals
        doc.sentences = Document._find_sentences(doc, sentences, text)
        return doc

    @staticmethod
    def _find_tokens(doc: 'Document', word_tokens: List[str], pos_tags: List[str], text: str) -> 'List[Token]':
        """
        Calculate the span of each token, find which element it belongs to and create a new Token instance
        :param doc: Reference to documents instance
        :param word_tokens: list of strings(tokens) coming out of nltk.word_tokenize
        :param pos_tags: list of strings(pos tag) coming out of nltk.pos_tag
        :param text: Document text as a string
        :return: list of tokens as Token class
        """
        offset = 0
        tokens = []
        missing = None
        for token, pos_tag in zip(word_tokens, pos_tags):
            # TODO: Handle linebreak '\n' with 'NL'
            pos = text.find(token, offset, offset + max(50, len(token)))
            if pos > -1:
                if missing:
                    # TODO: Handle linebreak '\n' with 'NL'
                    t = Token(doc, offset + pos, offset + pos + len(missing['token']), missing['pos_tag'],
                              get_shape_category(missing['token']), missing['token'])
                    tokens.append(t)
                    offset += len(missing['token'])
                    missing = None
                t = Token(doc, offset + pos, offset + pos + len(token), pos_tag, get_shape_category(token), token)
                tokens.append(t)
                offset += len(token)
            else:
                missing = {
                    'token': token,
                    'pos_tag': pos_tag
                }
        return tokens

    @staticmethod
    def _find_sentences(doc: 'Document', sentences_tokens: List[str], text: str) -> 'List[Sentence]':
        """
        List Sentence objects each time a sentence is found in the text
        :param doc: reference to documents instance
        :param sentences_tokens: list of strings(sentences) coming out of nltk.sent_tokenize
        :param text: Document text as a string
        """
        offset = 0
        sentences = []
        missing = None
        for sentence in sentences_tokens:
            pos = text.find(sentence, offset, offset + max(500, len(sentence)))
            if pos > -1:
                if missing:
                    s = Sentence(doc, offset + pos, offset + pos + len(missing))
                    sentences.append(s)
                    offset += len(missing)
                    missing = None
                s = Sentence(doc, offset + pos, offset + pos + len(sentence))
                sentences.append(s)
                offset += len(sentence)
            else:
                missing = sentence
        return sentences


def get_shape_category(token):
    if re.match('^[\n]+$', token):  # IS LINE BREAK
        return 'NL'
    if any(char.isdigit() for char in token) and re.match('^[0-9.,]+$', token):  # IS NUMBER (E.G., 2, 2.000)
        return 'NUMBER'
    if re.fullmatch('[^A-Za-z0-9\t\n ]+', token):  # IS SPECIAL CHARS (E.G., $, #, ., *)
        return 'SPECIAL'
    if re.fullmatch('^[A-Z\-.]+$', token):  # IS UPPERCASE (E.G., AGREEMENT, INC.)
        return 'ALL-CAPS'
    if re.fullmatch('^[A-Z][a-z\-.]+$', token):  # FIRST LETTER UPPERCASE (E.G. This, Agreement)
        return '1ST-CAP'
    if re.fullmatch('^[a-z\-.]+$', token):  # IS LOWERCASE (E.G., may, third-party)
        return 'LOWER'
    if not token.isupper() and not token.islower():  # WEIRD CASE (E.G., 3RD, E2, iPhone)
        return 'MISC'
    return 'MISC'
