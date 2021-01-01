from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download(['punkt','stopwords'])
def tokenize(text):
    """
    Normalize by converting to lower case
    Tokenize by converting sentence to tokens
    Remove stop words
    Convert words to root form by Stemming
    """
    
    text=text.lower()
    token=word_tokenize(text)
    final_token=[]
    stemmer=PorterStemmer()
    for tok in token:
        if tok not in stopwords.words('english'):
            stem=stemmer.stem(tok)
            final_token.append(stem)
    return final_token