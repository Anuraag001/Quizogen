from django.shortcuts import render
import spacy
from spacy import displacy
import pandas as pd
import numpy as np
import nltk
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelWithLMHead, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nlp=spacy.load('en_core_web_lg')

# Create your views here.
def preprocess(sent,type=0):
    doc=nlp(sent)
    new_sent=[]
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        else:
            new_sent.append(token.lemma_)
    if type==1:
        return new_sent
    else:
        return ' '.join(new_sent)
    
def preprocess_with_ner(sent, type=0):
    doc = nlp(sent)
    new_sent = set()
    
    for ent in doc.ents:
        new_sent.add(ent.text)
    
    return new_sent

def get_question(answer, context, tokenizer,model,max_length=64):
  input_text = "answer: %s  context: %s </s>" % (answer, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0])

def get_questions(text):
    doc=nlp(text)
    
    b=preprocess_with_ner(text)
    b=list(b)
    b=np.array(b)

    preprocessed_sent=[preprocess(sent.text) for sent in doc.sents]
    preprocessed_sent=np.array(preprocessed_sent)

    v=TfidfVectorizer()
    tdidf_vector=v.fit_transform(preprocessed_sent)
    tdidf_vector.toarray()

    sent_score=tdidf_vector.sum(axis=1)
    sent_score=np.array(sent_score)

    sent_score_map=np.hstack((sent_score,preprocessed_sent.reshape(-1,1)))
    sent_score_map=np.sort(sent_score_map,axis=0)[::-1]

    v=TfidfVectorizer()
    tdidf_matrix=v.fit_transform(b)
    tdidf_matrix.toarray()
    score=tdidf_matrix.sum(axis=1)

    word_score=np.hstack((score,b.reshape(-1,1)))
    word_score=np.sort(word_score,axis=0)[::-1]
    word_score=np.array(word_score)
    word_score_map=word_score[:,1]
    
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

    x=20
    questions=[]
    answers=[]
    while x>0:
        flag=True
        for sent in sent_score_map[:,1]:
            for word in word_score_map:
                if word in sent:
                    d=nlp(str(word))
                    explanation = spacy.explain(d.ents[0].label_)
                    question=get_question(word,sent,tokenizer,model)
                    #print(f"{question} '|' {word} '|' Type: {d.ents[0].label_} '|' Explanation: {explanation}")
                    word_score_map = np.delete(word_score_map, np.where(word_score_map == word))
                    x-=1
                    flag=False
                    questions.append(str(question[16:-4]))
                    answers.append(str(word))
                    break
            if x==0 or word_score_map.size==0:
                break
        if flag:
            break
    return questions,answers



def front(request):
    if request.method=='POST':
        text=request.POST['textinput']
        questions,answers=get_questions(text)
        count=len(questions)
        count=np.arange(count)
        for question in questions:
            print(question)
        return render(request,'front.html',{"questions":questions,"answers":answers,"count":count})
    return render(request, "front.html",{"questions":[""],"answers":[""],"count":[0]})
