from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
from pydantic import BaseModel
import json
from custom_ml_util import run_prediction_on_json

def check_params(q):
    try:
        assert 'SentimentText' in json.loads(q), "key `SentimentText` should appear in hash [dict] argument"
        assert 'Sentiment' in json.loads(q), "key `Sentiment` should appear in hash [dict] argument"
    except AssertionError as a:
        return a
    return True


app = FastAPI()

class Prediction(BaseModel):
    SentimentText: list = None
    Sentiment: list = None


@app.get("/")
def read_root():
    return {"Message": "Hello! Where would you like to invest?"}


@app.post("/prediction", response_model=Prediction)
def send_prediction(predict: Prediction):
    check = check_params(predict.json())
    if len(predict.Sentiment) == 1 and predict.Sentiment[0] == None:
        check = False
    elif len(predict.SentimentText) == 1 and predict.SentimentText[0] == None:
        check = False
    if check is not True:
        check = 'Check body keys body null' if check is False else check
        raise HTTPException(status_code=404, detail=check)
    else:
        predicted_response = run_prediction_on_json(predict.json())
        predict.Sentiment = predicted_response
    return JSONResponse(content=jsonable_encoder(predict))

