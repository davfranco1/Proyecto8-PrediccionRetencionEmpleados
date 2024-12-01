import pandas as pd
import streamlit as st
import requests
import json

# Load DataFrame
df = pd.read_pickle("../datos/df_str.pkl")

# Streamlit App
st.title('Predictor de la Renuncia de un Empleado')
st.write("Este modelo es capaz de predecir la probabilidad de que un empleado deje la empresa, según su perfil.")

with st.form("predict_form"):
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

    # Submit button
    boton_predecir = st.form_submit_button("Predecir")

if boton_predecir:
    datos = {
        'Age': age,
        'BusinessTravel': business_travel,
        'Department': department,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EducationField': education_field,
        'Gender': gender,
        'JobLevel': job_level,
        'JobRole': job_role,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'NumCompaniesWorked': num_companies_worked,
        'PercentSalaryHike': percent_salary_hike,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'YearsAtCompany': years_at_company,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager,
        'EnvironmentSatisfaction': environment_satisfaction,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'JobInvolvement': job_involvement,
        'PerformanceRating': performance_rating,
    }

    # Convert data to DataFrame
    df_consulta = pd.DataFrame(datos, index=[0])
    json_consulta = json.dumps(df_consulta.iloc[0].to_dict())

    # Call API
    url_flask = 'http://127.0.0.1:5000/predict'
    response = requests.post(url_flask, json=json_consulta)

    # Handle API response
    if response.status_code == 200:
        result = response.json()
        if result['attrition_risk'] == 'Yes':
            st.success(f"Un empleado con las características dadas tiene una probabilidad ALTA de dejar la empresa. Probabilidad: {result['probability']:.2f}")
        else:
            st.success(f"Un empleado con las características dadas tiene una probabilidad BAJA de dejar la empresa. Probabilidad: {result['probability']:.2f}")
    else:
        st.error(f"Error llamando a la API. Código de estado: {response.status_code}")
        st.write(response.text)