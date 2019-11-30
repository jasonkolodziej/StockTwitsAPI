from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
from pydantic import BaseModel
import json
from custom_ml_util import run_prediction_on_json

def check_params(q):
    x = True
    try:
        assert 'SentimentText' in json.loads(q.json()), "key `SentimentText` should appear in hash [dict] argument"
        assert 'Sentiment' in json.loads(q.json()), "key `Sentiment` should appear in hash [dict] argument"
        assert q.dict() == dict(q) != {'SentimentText': [None], 'Sentiment': [None]}, "Check `body` keys are valid and values are not null"
        assert q.dict() == dict(q) != {'SentimentText': None, 'Sentiment': None}, "Check `body` keys are valid and values are not null"
        assert q.dict() == dict(q) != {'SentimentText': [None], 'Sentiment': None}, "Check `body` keys are valid and values are not null"
        assert q.dict() == dict(q) != {'SentimentText': None, 'Sentiment': [None]}, "Check `body` keys are valid and values are not null"
    except AssertionError as a:
        print(a)
        x = a
    return x


app = FastAPI()

class Prediction(BaseModel):
    SentimentText: list = None
    Sentiment: list = None


@app.get("/")
def read_root():
    return {"Message": "Hello! Where would you like to invest?"}


@app.post("/prediction", response_model=Prediction)
def send_prediction(predict: Prediction):
    check = check_params(predict)
    if check is not True:
        raise HTTPException(status_code=404, detail=str(check))
    else:
        predicted_response = run_prediction_on_json(predict.json())
        predict.Sentiment = predicted_response
    return JSONResponse(content=jsonable_encoder(predict))

