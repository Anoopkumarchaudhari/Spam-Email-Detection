import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
ps = PorterStemmer()

def preprocessing_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    temp = []
    for i in text:
        if(i.isalnum()):
            temp.append(i)
    text = temp[:]
    temp.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            temp.append(i)
    text = temp[:]
    temp.clear()
    for i in text:
        temp.append(ps.stem(i))
    return " ".join(temp);

tfidf = pickle.load(open('Vectorizer.pkl','rb'))
MNNB = pickle.load(open('MultinomialNB.pkl','rb'))
BNB = pickle.load(open('bernoulliNB.pkl','rb'))
st.title('Email/Message spam Classifier ')

input_sms = st.text_area("Enter Email/Message")
if st.button('Predict'):
    #data preprocess
    transformed_sms = preprocessing_text(input_sms)
    # vectorization
    vector_input = tfidf.transform([transformed_sms])
    #predict
    result = BNB.predict(vector_input)

    #display
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')

