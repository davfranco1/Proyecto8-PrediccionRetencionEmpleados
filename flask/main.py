import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS  # Importamos para manejar CORS y evitar problemas de conexión

# Cargar los archivos pickle (encoder, scaler y modelo)
with open('../transformers/encoder.pkl', 'rb') as file:
    target = pickle.load(file)
    print("Encoder cargado correctamente.")

with open('../transformers/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    print("Scaler cargado correctamente.")

with open('../transformers/model.pkl', 'rb') as file:
    model = pickle.load(file)
    print("Modelo cargado correctamente.")

# Inicializar Flask y habilitar CORS
app = Flask(__name__)
CORS(app)  # Permitir solicitudes desde cualquier origen

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos en formato JSON
        datos = request.get_json()
        print("Datos recibidos:", datos)  # Registrar los datos recibidos para depuración

        if not datos:
            return jsonify({"Error": "No hay datos en la solicitud"}), 400

        # Convertir los datos en un DataFrame
        df = pd.DataFrame([datos])
        print("Datos convertidos a DataFrame:", df)

        # Aplicar transformación con el encoder
        df_encoded = target.transform(df)
        print("Datos codificados (encoded):", df_encoded)

        # Escalar los datos
        df_scaled = pd.DataFrame(scaler.transform(df_encoded), columns=scaler.get_feature_names_out())
        print("Datos escalados (scaled):", df_scaled)

        # Realizar predicción
        prediccion = model.predict(df_scaled)
        probabilidad = model.predict_proba(df_scaled)[0, 1]
        print("Predicción:", prediccion, "Probabilidad:", probabilidad)

        # Construir la respuesta
        respuesta = {
            'attrition_risk': 'Yes' if prediccion[0] == 1 else 'No',
            'probability': round(float(probabilidad), 2)
        }

        return jsonify(respuesta)

    except Exception as e:
        # Registrar cualquier error ocurrido
        print("Error ocurrido:", str(e))
        return jsonify({"Error": "Error interno en el servidor", "Detalles": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Asegúrate de que el puerto sea accesible