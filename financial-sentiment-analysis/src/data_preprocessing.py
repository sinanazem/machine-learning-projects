from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re

#import stopwords and text processing libraries

from nltk.corpus import stopwords

def text_preprocessing(text):

    #convert all to lowercase
    text = text.lower()

    #removing puntuations
    text = re.sub(r'[^\w\s]', ' ', text)

    #remove stopwords
    word_token = [word for word in word_tokenize(text) if word.isalpha() and word not in stopwords.words('english')]

    #stemizing
    stemmer = PorterStemmer()
    stem_word = [stemmer.stem(word) for word in word_token]



    #lemmitizing
    wnl = WordNetLemmatizer()
    lema_word = [wnl.lemmatize(word) for word in stem_word]

    return ' '.join(lema_word)



if __name__ == '__main__':

    text = preprocessing('Today the weather is sunny!')
    print(text)