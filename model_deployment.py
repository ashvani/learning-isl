# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 21:51:26 2025

@author: dubey
"""

from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
import joblib

class Dataset(BaseModel):
  title: str
  description: str
  
  
class Response(BaseModel):
    prediction: int

app = FastAPI()
joblib_in = open("rf_model","rb")
model=joblib.load(joblib_in)

@app.get('/')
def index():
    return {'message': 'ML API to predict label for AG dataset'}

@app.post('/predict')
def predict_news_label(data:Dataset):
    data = data.model_dump()
    title=data['title']
    description=data['description']
    prediction = model.predict([title + ' ' + description])
    
    return Response(prediction=prediction[0])

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
    