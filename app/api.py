from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import pickle as pk

app = FastAPI()

try:
    datas = pd.read_csv("./datas/application_test.csv")
    datas.replace([np.inf, -np.inf], np.nan, inplace=True)
    datas.fillna(-1, inplace=True)
except:
    raise HTTPException(status_code=404, detail="Données non trouvées")

try:
    model = pk.load(open("./model/model.pkl", "rb"))
except:
    raise HTTPException(status_code=404, detail="Modèle non trouvé")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(num_client: int):
    predict = model.predict_proba(datas.iloc[num_client])
    if predict >= 0.4669:
        return {"prediction": "Potentiel mauvais payeur"}

    return {"prediction": "Potentiel bon payeur"}
