import pandas as pd
import pickle
from flask import Flask, request, jsonify


with open('../transformers/encoder.pkl', 'rb') as file:
    target = pickle.load(file)
with open('../transformers/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('../transformers/model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    
    if not input_data:
        return jsonify({"Error": "No hay datos"}), 400

    df_encoded = target.transform(input_data)

    df_scaled = pd.DataFrame(scaler.transform(df_encoded), columns = scaler.get_feature_names_out())
    prediccion = model.predict(df_scaled)
    probabilidad = model.predict_proba(df_scaled)[0, 1]

    respuesta = {
        'attrition_risk': 'Yes' if prediccion == 1 else 'No',
        'probability': round(float(probabilidad), 2)
    }

    return jsonify(respuesta)

if __name__ == '__main__':
    app.run(debug=True)
