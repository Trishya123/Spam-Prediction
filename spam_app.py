import streamlit as  slt
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
slt.title("Email/ SMS Classifier")
input_sms = slt.text_area("Enter the message")
def transform_text(text):
    text = text.lower()
    text= nltk.word_tokenize(text)
    y =[]
    for i in text :
        if i.isalnum():
         y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation :
            y.append(ps.stem(i))
    return " ".join(y)
if slt.button("Predict"):
            
# 1. Preprocess
     transformed_sms = transform_text(input_sms)
#2. Vectorize
     vector_input = tfidf.transform([transformed_sms])
#3.predict
     result = model.predict(vector_input)[0]
#4. Result
     if result == 1:
          slt.header("Spam")
     else:
           slt.header("Not Spam")

    