from whoosh.analysis import Filter, RegexTokenizer, LowercaseFilter, StopFilter
from nltk.stem import PorterStemmer, SnowballStemmer

class NLTKPorterFilter(Filter):
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, tokens):
        for t in tokens:
            t.text = self.stemmer.stem(t.text)
            yield t

def get_porter_analyzer():
    return RegexTokenizer() | LowercaseFilter() | StopFilter() | NLTKPorterFilter()

class NLTKSnowballFilter(Filter):
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
    def __call__(self, tokens):
        for t in tokens:
            t.text = self.stemmer.stem(t.text)
            yield t

def get_snowball_analyzer():
    return RegexTokenizer() | LowercaseFilter() | StopFilter() | NLTKSnowballFilter()
