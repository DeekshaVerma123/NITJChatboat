from .models import Course
import numpy as np
import spacy
from spacy.matcher import Matcher
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    Example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # Lowercase sentence words to match training
    sentence_words = [word.lower() for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

btech_pattern = [
    [{"LOWER": "b."}, {"LOWER": "tech"}],
    [{"LOWER": "b"}, {"LOWER": "tech"}],
    [{"LOWER": "btech"}],
]

matcher.add(
    "mtech", [[{"LOWER": "mtech"}], [{"LOWER": "m", "OP": "+"}, {"LOWER": "tech"}]]
)
matcher.add("btech", btech_pattern)


def course_matcher(sentence):
    doc = nlp(sentence)
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # 'mtech' or 'btech'
        return string_id
    return None
