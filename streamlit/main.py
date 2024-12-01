import pandas as pd
import streamlit as st
import requests
import json
import time

st.set_page_config(
    page_title="Predictor de la Renuncia de un Empleado",
    page_icon="🏢",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6f3ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💼 Predictor de la Renuncia de un Empleado")
st.write("Este modelo es capaz de predecir la probabilidad de que un empleado deje la empresa, según su perfil.")

st.image(
    "../images/header.jpg",
    use_container_width=True,
)

df = pd.read_pickle("../datos/pickles_transformados/modelo1/df_sinnulos.pkl")

# Formularios de entrada
st.header("👨🏻‍💼👩🏻‍💼 Perfil del empleado")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider('Edad', min_value=df["Age"].astype(int).min(), max_value=df["Age"].astype(int).max(), step=1)
        distance_from_home = st.slider('Distancia desde casa', min_value=df["DistanceFromHome"].astype(int).min(), max_value=df["DistanceFromHome"].astype(int).max(), step=1)
        monthly_income = st.number_input('Ingreso mensual', min_value=df["MonthlyIncome"].astype(int).min(), max_value=df["MonthlyIncome"].astype(int).max(), step=100)
        num_companies_worked = st.slider('Numero de empresas trabajadas', min_value=df["NumCompaniesWorked"].astype(int).min(), max_value=df["NumCompaniesWorked"].astype(int).max(), step=1)
        percent_salary_hike = st.slider('Aumento porcentual de salario', min_value=df["PercentSalaryHike"].astype(int).min(), max_value=df["PercentSalaryHike"].astype(int).max(), step=1)
        total_working_years = st.slider('Años totales trabajados', min_value=df["TotalWorkingYears"].astype(int).min(), max_value=df["TotalWorkingYears"].astype(int).max(), step=1)
        training_times_last_year = st.slider('Cantidad de formaciones durante el último año', min_value=df["TrainingTimesLastYear"].astype(int).min(), max_value=df["TrainingTimesLastYear"].astype(int).max(), step=1)
        years_at_company = st.slider('Años en la empresa', min_value=df["YearsAtCompany"].astype(int).min(), max_value=df["YearsAtCompany"].astype(int).max(), step=1)
        years_since_last_promotion = st.slider('Años desde la última promoción', min_value=df["YearsSinceLastPromotion"].astype(int).min(), max_value=df["YearsSinceLastPromotion"].astype(int).max(), step=1)
        years_with_curr_manager = st.slider('Años con el manager actual', min_value=df["YearsWithCurrManager"].astype(int).min(), max_value=df["YearsWithCurrManager"].astype(int).max(), step=1)
    
    with col2:
        environment_satisfaction = st.selectbox('Satisfacción con el ambiente laboral (1-4)', [1, 2, 3, 4])
        job_satisfaction = st.selectbox('Satisfacción con el trabajo (1-4)', [1, 2, 3, 4])
        work_life_balance = st.selectbox('Equilibrio vida-trabajo (1-4)', [1, 2, 3, 4])
        job_involvement = st.selectbox('Involucramiento en el trabajo (1-4)', [1, 2, 3, 4])
        business_travel = st.selectbox('Viajes de empresa', df["BusinessTravel"].unique())
        department = st.selectbox('Departamento', df["Department"].unique())
        education = st.selectbox('Nivel formativo', ['Below College', 'College', 'Bachelor', 'Master', 'Doctor'])
        education_field = st.selectbox('Sector formativo', df["EducationField"].unique())
        job_role = st.selectbox('Puesto de trabajo', df["JobRole"].unique())
        marital_status = st.selectbox('Estado civil', df["MaritalStatus"].unique())
        gender = st.selectbox('Género', df["Gender"].unique())
        job_level = st.selectbox('Nivel del puesto', [1, 2, 3, 4, 5])
        stock_option_level = st.selectbox('Nivel de opciones de acciones', [0, 1, 2, 3])
        performance_rating = st.selectbox('Calificación de desempeño', [3, 4])
    
        boton_predecir = st.form_submit_button("Predecir")


if boton_predecir:
    datos = {
        'Age': float(age),
        'BusinessTravel': str(business_travel),
        'Department': str(department),
        'DistanceFromHome': float(distance_from_home),
        'Education': str(education),
        'EducationField': str(education_field),
        'Gender': str(gender),
        'JobLevel': str(job_level),
        'JobRole': str(job_role),
        'MaritalStatus': str(marital_status),
        'MonthlyIncome': float(monthly_income),
        'NumCompaniesWorked': float(num_companies_worked),
        'PercentSalaryHike': float(percent_salary_hike),
        'StockOptionLevel': str(stock_option_level),
        'TotalWorkingYears': float(total_working_years),
        'TrainingTimesLastYear': float(training_times_last_year),
        'YearsAtCompany': float(years_at_company),
        'YearsSinceLastPromotion': float(years_since_last_promotion),
        'YearsWithCurrManager': float(years_with_curr_manager),
        'EnvironmentSatisfaction': str(environment_satisfaction),
        'JobSatisfaction': str(job_satisfaction),
        'WorkLifeBalance': str(work_life_balance),
        'JobInvolvement': str(job_involvement),
        'PerformanceRating': str(performance_rating),
    }

    # Llamada a la API
    url_flask = 'http://127.0.0.1:5000/predict'
    response = requests.post(url_flask, json=datos)  # Pass 'datos' directly

    # Respuesta de la API
    if response.status_code == 200:
        result = response.json()
        with st.spinner('Estamos calculando el valor del alquiler...'):
            time.sleep(3)
        if result['attrition_risk'] == 'Yes':
            st.success(f"Un empleado con las características dadas tiene una probabilidad ALTA de dejar la empresa. Probabilidad: {result['probability']:.2f}")
        else:
            st.success(f"Un empleado con las características dadas tiene una probabilidad BAJA de dejar la empresa. Probabilidad: {result['probability']:.2f}")
    else:
        st.error(f"Error llamando a la API. Código de estado: {response.status_code}")
        try:
            error_details = response.json()
            st.write("Detalles del error:", error_details)
        except json.JSONDecodeError:
            st.write("No se pudo decodificar la respuesta del error:", response.text)