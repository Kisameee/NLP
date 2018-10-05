from typing import List

import nltk


class Document:
    """
    A document is a combination of text and the positions of the tags and elements in that text.
    """

    def __init__(self):
        self.text = None
        self.tokens = None
        self.sentences = None

    @classmethod
    def create_from_text(cls, text: str = None):
        """
        :param text: document text as a string
        """
        doc = Document()
        doc.text = text
        # TODO: To be implemented
        # 1. Tokenize texte (tokens & phrases)
        words, pos_tags = zip(*nltk.pos_tag(nltk.word_tokenize(text)))
        sentences = nltk.sent_tokenize(text.replace('\n', ' '))
        # 2. Corriger la tokenisation (retokenize)
        words, pos_tags = Document._retokenize(words, pos_tags)
        # 3. Trouver les intervalles de Tokens
        doc.tokens = Document._find_tokens(doc, words, pos_tags, text)
        # 4. Trouver les intervalles de phrases
        doc.sentences = Document._find_sentences(sentences, text)

        return doc

    @staticmethod
    def _retokenize(word_tokens: List[str], pos_tags: List[str]):
        """
        Correct NLTK tokenization. We separate symbols from words, such as quotes, -, *, etc
        :param word_tokens: list of strings(tokens) coming out of nltk.word_tokenize
        :param pos_tags:  list of strings(pos tag) coming out of nltk.pos_tag
        :return: new list of tokens
        """
        split_ends = re.escape('-*·')
        split_always = re.escape('’`"\'“”/\\')
        # declare new lists
        for token, pos in zip(word_tokens, pos_tags):
            split_tokens = re.split('([' + split_always + ']+)|(\n)|(^[' + split_ends + '])|([' + split_ends + ']$)',
                                    token)
            split_tokens = [t for t in split_tokens if t is not None and t != '']
            # extend new token list
            # find new pos tags, extend pos tag list
        # return new lists

    @staticmethod
    def _find_tokens(doc, word_tokens, pos_tags, text):
        """ Calculate the span of each token, find which element it belongs to and create a new Token instance
            :param doc: Reference to documents instance
            :param word_Tokens:  list of strings(tokens) coming out of nltk.word_tokenize
            :param pos_tags:  list of strings(pos tag) coming out of nltk.pos_tag
            :return: list of tokens as Token class
         """
        offset = 0
        tokens = []
        missing = None
        for token, pos_tag in zip(word_tokens, pos_tags):
            # Traiter le changement de ligne '\n' avec pos tag 'NL'

            pos = text.find(token, offset, offset + max(50, len(token)))
            if pos > -1:
                if missing:
            # Find missing
            # Make Token instance + add it to list
            # Traiter le changement de ligne '\n' avec pos tag 'NL'

            # Make Token instance: Token(doc, start, end, pos, shape, text) shape = get_shape_category(token)
            # Add to list of tokens
            else:
        # Record missing

    @staticmethod
    def _find_sentences(doc, sentences: List[str], doc_text: str):
        """ yield Sentence objects each time a sentence is found in the text """
        # TODO: To be implemented
