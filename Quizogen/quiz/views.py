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
final_list=[]

Tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
Model = AutoModel.from_pretrained("bert-base-uncased")

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
    data_list=[]
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
                    data_list.append({
                        'question': str(question[16:-4]),
                        'answer': str(word),
                        'type': str(d.ents[0].label_),
                        'explanation': str(explanation)
                    })
                    break
            if x==0 or word_score_map.size==0:
                break
        if flag:
            break
    return data_list



def front(request):
    data_list=[]
    data_list.append({
        'question': 'a',
        'answer': 'b',
        'type': 'c',
        'explanation': 'd'
    })
    return render(request, "front.html",{"count":False,"dict_":data_list})

def que_gen_form(request):
    global final_list
    if request.method=='POST':
        text=request.POST['textinput']
        data_list=get_questions(text)
    final_list=data_list
    return render(request,'front.html',{"count":True,"dict_":data_list})

def cosine_similarity_(sentence1, sentence2):

    inputs1 = Tokenizer(sentence1, return_tensors="pt", truncation=True)
    inputs2 = Tokenizer(sentence2, return_tensors="pt", truncation=True)
    with torch.no_grad():
        embeddings1 = Model(**inputs1).last_hidden_state.mean(dim=1).numpy()
        embeddings2 = Model(**inputs2).last_hidden_state.mean(dim=1).numpy()

    similarity_score = cosine_similarity(embeddings1, embeddings2)[0, 0]
    if similarity_score>0.75 :
        return "..."
    else:
        return ""

def check_ans_form(request):
    final_view=[]
    if request.method=='POST':
        for i in range(len(request.POST)-1):
            user_answer=str(request.POST[f"qn_{i+1}"])
            final_view.append({
                'question':final_list[i]['question'],
                'answer':final_list[i]['answer'],
                'user_answer':user_answer,
                'similarity':cosine_similarity_(str(final_list[i]['answer']),user_answer)
            })
    return render(request,"final.html",{"final_view":final_view})

