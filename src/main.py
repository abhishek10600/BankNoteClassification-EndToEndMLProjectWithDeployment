from fastapi import FastAPI
import pickle
from schemas import BankNote



app = FastAPI()

# classification model
pickle_in = open("../classifier.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.get("/test")
async def test():
    return {"success": True, "message": "Api is working"}


@app.post("/predict")
async def predict(inputdata: BankNote):
    data = inputdata.model_dump()
    variance = data["variance"]
    skewness = data["skewness"]
    curtosis = data["curtosis"]
    entropy = data["entropy"]
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if prediction[0] > 0.5:
        prediction_response = "Fake note."
    else:
        prediction_response = "It is a bank note."
    return {"success": True, "message": prediction_response}
