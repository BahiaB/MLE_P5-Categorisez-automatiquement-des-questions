from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
#Preprocessing 
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


class TagsCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.clean_tags(text) for text in X]

    def clean_tags(self, text):
        tags_to_transform = BeautifulSoup(text, "html.parser")
        return [tag.name for tag in tags_to_transform.find_all()]
    
    '''
    def transform(self, X, y=None):
        return [BeautifulSoup(text, "html.parser").get_text() for text in X]'''
    
class HtmlCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.clean_text(text) for text in X]

    def clean_text(self, text):
    # Delete HTML tags
        clean_text = re.sub(r'<.*?>', '', text)
    # Delete special characters and punctuation
        clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)
        return clean_text
    
    
    '''def transform(self, X, y=None):
        return [BeautifulSoup(text, "html.parser").get_text() for text in X]'''


class TextTokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [nltk.word_tokenize(text) for text in X]

class TextLower(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [[word.lower() for word in text] for text in X]
    
class TextStopWordRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        return [[word for word in text if word not in stop_words] for text in X]
    
class TextLemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [[self.lemmatize(word) for word in text] for text in X]

    def lemmatize(self, word):
        pos_tag = self.get_wordnet_pos(word)
        if pos_tag:
            return self.lemmatizer.lemmatize(word, pos=pos_tag)
        else:
            return word

    def get_wordnet_pos(self, word):
        """Map POS tag to first character used by WordNetLemmatizer"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)
    

def create_pipeline():
    pipeline = Pipeline([
        ('html_cleaner', HtmlCleaner()),
        ('tokenizer', TextTokenizer()),
        ('stop_words', TextStopWordRemover()),
        ('text_lower', TextLower()),
        #('tags_cleaner', TagsCleaner()),
        ('lemmatizer',TextLemmatizer())
    ])
    return pipeline



# Pipeline des diffrentes vectorisation
# CountVectorizer (Bag of Words)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


class CustomCountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = CountVectorizer(min_df=0.01)

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X, y=None):
        return self.vectorizer.transform(X)

pipeline_vec = Pipeline([
    ("count_vectorizer", CustomCountVectorizer())
])


def create_vectors_pipeline():
    pipeline_vec = Pipeline([
        ("count_vectorizer", CustomCountVectorizer())
    ])
    return pipeline_vec