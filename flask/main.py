import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS  # Importamos para manejar CORS y evitar problemas de conexión

# Cargar los archivos pickle (encoder, scaler y modelo)
with open('../transformers/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
    print("Encoder cargado correctamente.")

with open('../transformers/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    print("Scaler cargado correctamente.")

with open('../transformers/model.pkl', 'rb') as file:
    model = pickle.load(file)
    print("Modelo cargado correctamente.")

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados al servidor
        datos = request.get_json(silent=True)
        print("Datos recibidos en Flask:", datos)

        if not datos or not isinstance(datos, dict):
            return jsonify({"Error": "Formato inválido", "Detalles": "Se esperaba un objeto JSON"}), 400

        # Convertir los datos en un DataFrame
        df = pd.DataFrame([datos])  # Asegurar que los datos sean bidimensionales
        print("Dimensiones iniciales:", df.shape)

        # Encoding
        df['Attrition'] = 0  # valor temporal para el encoding
        df_encoded = encoder.transform(df)
        df_encoded.drop(columns='Attrition', inplace= True)

        print("Dimensiones después del encoding:", df_encoded.shape)

        # Escalar los datos
        col_num=['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
            'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany',
            'YearsSinceLastPromotion', 'YearsWithCurrManager']
        df_encoded[col_num] = scaler.transform(df_encoded[col_num])

        print("Dimensiones después de la estandarización:", df_encoded.shape)

        # Realizar predicción
        prediccion = model.predict(df_encoded)
        probabilidad = model.predict_proba(df_encoded)[0, 1]
        print("Predicción:", prediccion, "Probabilidad:", probabilidad)

        # Construir la respuesta
        respuesta = {
            'attrition_risk': 'Yes' if prediccion[0] == 1 else 'No',
            'probability': round(float(probabilidad), 2)
        }

        return jsonify(respuesta)

    except Exception as e:
        print("Error ocurrido:", str(e))
        return jsonify({"Error": "Error interno en el servidor", "Detalles": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)