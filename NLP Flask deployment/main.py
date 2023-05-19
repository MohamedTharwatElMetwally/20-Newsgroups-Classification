from flask import Flask,request,jsonify,redirect,url_for
from flask import render_template
import numpy as np
import pickle
from datetime import  date

#########################
import pandas as pd
import string

import re
import nltk
import spacy
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA, LEMMA
from spacy.tokens import Doc
nlp = spacy.load('en_core_web_lg')


app=Flask(__name__,template_folder='templates')

def pre_model(data):
        vectorizer = pickle.load(open('TfidfVectorizer.pkl', 'rb'))
    
        data=vectorizer.transform([data])
        loaded_model=pickle.load(open('XGBClassifierModel.pkl', 'rb'))

        pred=loaded_model.predict(data)
        pred=pred[0]

        return  get_name(pred)

def get_name(i):
    names = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
       'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
       'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
       'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
       'sci.electronics', 'sci.med', 'sci.space',
       'soc.religion.christian', 'talk.politics.guns',
       'talk.politics.mideast', 'talk.politics.misc',
       'talk.religion.misc']
    return names[i]

def remove_stopwords(doc):
    indexes = []
    for index, token in enumerate(doc):
        if token.is_stop:
            indexes.append(index)
    np_array = doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA, LEMMA])
    np_array = np.delete(np_array, indexes, axis = 0)
    doc2 = Doc(doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in indexes])
    doc2.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA, LEMMA], np_array)
    return doc2
def clean_header(text):
    text = re.sub(r'(From:\s+[^\n]+\n)', '', text)
    text = re.sub(r'(Subject:[^\n]+\n)', '', text)
    text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', '', text)
    text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
    text = re.sub(r'(Version:[^\n]+\n)', '', text)

    return text
def clean_text(text): 
    
    re_url = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
    re_email = re.compile('(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')                 
    text = text.lower()
    text = text.strip()
    text = re.sub(re_url, '', text)
    text = re.sub(re_email, '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'(\d+)', ' ', text)
    text = re.sub(r'(\s+)', ' ', text)
    
    return text

def preprocessing(data):


    data = clean_text(clean_header(data))
    data=remove_stopwords(nlp(data))
    data= ' '.join(token.lemma_ for token in data)

    return data 


@app.route("/")
def home():
    message="hello world"
    return render_template('index.html',message=message)


@app.route('/predict')
def predict():
    return render_template('predict.html')
import json
# this is an endpoint
@app.route('/result',methods=['POST'])
def result():
    if request.method=='POST':
        to_file=request.files.get('file2')
        content = to_file.read().decode('latin-1')
        print('file:\n',content)
        prep_data=preprocessing(content)
        
        output=pre_model(prep_data)
        # output=output.response
        print(output)

        # out=out.values()
        return render_template('twoforms.html',prediction=output,date=date.today(),result='result')

@app.route('/result2',methods=['POST'])
def result2():
    if request.method=='POST':
        to_predict=request.form['file1']
        print("to_predict",to_predict)

        content = to_predict
        print('file:\n',content)
        prep_data=preprocessing(content)
        
        output=pre_model(prep_data)
        # output=output.response
        print(output)


        return render_template('twoforms.html',prediction=output,date=date.today(),result2='result2')


@app.route('/twoforms')
def twoforms():
    return render_template('twoforms.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__=='__main__':
    app.run(debug=True)
    