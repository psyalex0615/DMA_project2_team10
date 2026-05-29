from whoosh.analysis import Filter, RegexTokenizer, LowercaseFilter, StopFilter
from nltk.stem import PorterStemmer

class NLTKPorterFilter(Filter):
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, tokens):
        for t in tokens:
            t.text = self.stemmer.stem(t.text)
            yield t

def get_porter_analyzer():
    return RegexTokenizer() | LowercaseFilter() | StopFilter() | NLTKPorterFilter()
