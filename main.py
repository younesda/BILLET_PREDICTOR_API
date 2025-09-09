from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import io

from utils import load_model, preprocess_data  # ton module utilitaire

# Charger le modèle une seule fois
model = load_model("rf_model_060825.sav")

# Créer l'app FastAPI
app = FastAPI(title="Billet Prediction API")

@app.post("/predict-file/")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Lecture CSV avec détection automatique du séparateur ("," ou ";")
        df = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")

        # Prétraitement
        df_processed = preprocess_data(df.copy(), model)

        # Prédictions
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]

        # Résultats
        df["prediction"] = predictions
        df["proba"] = probabilities.round(4)

        # Conversion en CSV
        output = io.StringIO()
        df.to_csv(output, sep=";", index=False)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict-json/")
async def predict_json(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")

        # Prétraitement
        df_processed = preprocess_data(df.copy(), model)

        # Prédictions
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]

        # Résultats
        df["prediction_label"] = predictions
        df["probability"] = probabilities.round(4)

        # Stats globales
        total = len(df)
        vrais = int((df["prediction_label"] == 1).sum())
        faux = int((df["prediction_label"] == 0).sum())
        pourcentage = round(vrais / total * 100, 2) if total > 0 else 0

        return JSONResponse(
            content={
                "results": df.to_dict(orient="records"),
                "stats": {
                    "total": total,
                    "vrais": vrais,
                    "faux": faux,
                    "pourcentage_vrais": pourcentage,
                }
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

