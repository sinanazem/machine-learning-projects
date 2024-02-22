import spacy
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def preprocessing(text):

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)   
    lemma_words = [token.lemma_ for token in doc]
    words = [word for word in lemma_words if word.isalpha() and word not in stopwords.words('english')]
    final_text = ' '.join(words)
    return final_text

 

if __name__ == '__main__':
    sample_text = """It is a long established fact that a reader will be distracted by the readable content of a 
    page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters,
     as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and 
     web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover
      many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident,
       sometimes on purpose (injected humour and the like)."""
   
    print(word_tokenize(sample_text))
    