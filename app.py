## install requirements
## python -m pip install nltk==3.5 joblib==1.0.1 scikit-learn numpy fastapi uvicorn[standard]

## run with
## uvicorn main:app --reload

import joblib
from asyncio import Lock
import numpy as np
from fastapi import FastAPI

import nltk

nltk_resources = [
    'tokenizers/punkt',
    'taggers/averaged_perceptron_tagger',
    'corpora/wordnet'
]

for res in nltk_resources:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res.split('/')[1])

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
wn.ensure_loaded()

app = FastAPI()

# load trained classifier
clf_path = 'models/classifier.joblib'
vec_path = 'models/TfidfVectorizer.joblib'
wnl = WordNetLemmatizer()

simplify = {'N':wn.NOUN, 'V':wn.VERB, 'J':wn.ADJ, 'R':wn.ADV}

lock = Lock()

async def lemmatize(wnl, text, tagfilter = list('NVJR')):
    newtext = []
    async with lock:
        for token,tag in nltk.pos_tag(nltk.word_tokenize(text)):
            if tag[0] in tagfilter:
                try:
                    lemma = wnl.lemmatize(token, simplify[tag[0]])
                    newtext.append(lemma.lower())
                except KeyError:
                    pass
    return ' '.join(newtext)

@app.get("/predict/")
async def predict(query : str = ''):
    model = joblib.load(clf_path)
    vec = joblib.load(vec_path)
    # vectorize the user's query and make a prediction
    tags = list('NVJ')
    text = await lemmatize(wnl, query, tags)
    X = vec.transform(np.array([text]))
    yp = model.predict_proba(X)[0]
    output = {
        cls:round(prob,3)
            for cls,prob in sorted(
                zip(model.classes_, yp),
                key=lambda x: x[1],
                reverse=True
                )
    }
    return output
