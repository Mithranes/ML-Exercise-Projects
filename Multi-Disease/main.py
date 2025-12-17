import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load models
diabetes_model = pickle.load(open('/home/mithranes/Documents/old/Machine Learning/Multi-Disease/savedModels/diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('/home/mithranes/Documents/old/Machine Learning/Multi-Disease/savedModels/heart_model.sav', 'rb'))
parkinson_model = pickle.load(open('/home/mithranes/Documents/old/Machine Learning/Multi-Disease/savedModels/parkinson_model.sav', 'rb'))

# Sidebar menu
with st.sidebar:
    selection = option_menu('Multiple Disease System',
                            ['Diabetes Prediction', 'Heart Prediction', 'Parkinson Prediction'],
                            icons=['activity', 'heart', 'person'],
                            default_index=0)

# ==================== Diabetes Prediction ====================
if selection == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input('Number of Pregnancies', 0, 20, 0, key='diab_preg')
        skin_thickness = st.number_input('Skin Thickness', 0, 100, 20, key='diab_skin')
        dpf = st.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.5, key='diab_dpf')
    with col2:
        glucose = st.number_input('Glucose Level', 0, 300, 120, key='diab_glucose')
        insulin = st.number_input('Insulin Level', 0, 900, 79, key='diab_insulin')
        age = st.number_input('Age', 1, 120, 25, key='diab_age')
    with col3:
        bp = st.number_input('Blood Pressure', 0, 200, 70, key='diab_bp')
        bmi = st.number_input('BMI', 0.0, 70.0, 20.0, key='diab_bmi')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result', key='diab_btn'):
        diab_prediction = diabetes_model.predict([[pregnancies, glucose, bp, skin_thickness,
                                                  insulin, bmi, dpf, age]])
        diab_diagnosis = 'Diabetic' if diab_prediction[0] == 1 else 'Not Diabetic'
    st.success(diab_diagnosis)

# ==================== Heart Prediction ====================
elif selection == 'Heart Prediction':
    st.title('Heart Disease Prediction')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input('Age', 1, 120, 50, key='heart_age')
        sex = st.number_input('Sex (0=Female,1=Male)', 0, 1, 1, key='heart_sex')
        cp = st.number_input('Chest Pain Type (0-3)', 0, 3, 0, key='heart_cp')
    with col2:
        trestbps = st.number_input('Resting BP', 80, 200, 120, key='heart_trestbps')
        chol = st.number_input('Serum Cholesterol', 100, 600, 200, key='heart_chol')
        fbs = st.number_input('Fasting Blood Sugar >120 mg/dl (0=No,1=Yes)', 0, 1, 0, key='heart_fbs')
    with col3:
        restecg = st.number_input('Resting ECG (0-2)', 0, 2, 1, key='heart_restecg')
        thalach = st.number_input('Max Heart Rate', 60, 220, 150, key='heart_thalach')
        exang = st.number_input('Exercise Induced Angina (0=No,1=Yes)', 0, 1, 0, key='heart_exang')
    with col4:
        oldpeak = st.number_input('ST Depression', 0.0, 10.0, 1.0, key='heart_oldpeak')
        slope = st.number_input('Slope of ST Segment (0-2)', 0, 2, 1, key='heart_slope')
        ca = st.number_input('Major Vessels (0-3)', 0, 3, 0, key='heart_ca')
        thal = st.number_input('Thalassemia (1=Normal,2=Fixed,3=Reversible)', 1, 3, 2, key='heart_thal')

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result', key='heart_btn'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,
                                                thalach, exang, oldpeak, slope, ca, thal]])
        heart_diagnosis = 'Heart Disease Detected' if heart_prediction[0] == 1 else 'No Heart Disease'
    st.success(heart_diagnosis)

# ==================== Parkinson Prediction ====================
elif selection == 'Parkinson Prediction':
    st.title('Parkinson Prediction')
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', 0.0, 500.0, 120.0, key='park_fo')
        fhi = st.number_input('MDVP:Fhi(Hz)', 0.0, 500.0, 130.0, key='park_fhi')
        flo = st.number_input('MDVP:Flo(Hz)', 0.0, 500.0, 110.0, key='park_flo')
        jitter_percent = st.number_input('MDVP:Jitter(%)', 0.0, 1.0, 0.005, key='park_jitterp')
    with col2:
        jitter_abs = st.number_input('MDVP:Jitter(Abs)', 0.0, 1.0, 0.00005, key='park_jittera')
        rap = st.number_input('MDVP:RAP', 0.0, 1.0, 0.003, key='park_rap')
        ppq = st.number_input('MDVP:PPQ', 0.0, 1.0, 0.004, key='park_ppq')
        ddp = st.number_input('Jitter:DDP', 0.0, 1.0, 0.006, key='park_ddp')
    with col3:
        shimmer = st.number_input('MDVP:Shimmer', 0.0, 1.0, 0.03, key='park_shimmer')
        shimmer_db = st.number_input('MDVP:Shimmer(dB)', 0.0, 1.0, 0.3, key='park_shimmerdb')
        apq3 = st.number_input('Shimmer:APQ3', 0.0, 1.0, 0.02, key='park_apq3')
        apq5 = st.number_input('Shimmer:APQ5', 0.0, 1.0, 0.025, key='park_apq5')
    with col4:
        apq = st.number_input('MDVP:APQ', 0.0, 1.0, 0.027, key='park_apq')
        dda = st.number_input('Shimmer:DDA', 0.0, 1.0, 0.03, key='park_dda')
        nhr = st.number_input('NHR', 0.0, 1.0, 0.02, key='park_nhr')
        hnr = st.number_input('HNR', 0.0, 50.0, 20.0, key='park_hnr')
    with col5:
        rpde = st.number_input('RPDE', 0.0, 1.0, 0.5, key='park_rpde')
        dfa = st.number_input('DFA', 0.0, 1.0, 0.6, key='park_dfa')
        spread1 = st.number_input('spread1', -10.0, 10.0, -5.0, key='park_spread1')
        spread2 = st.number_input('spread2', -5.0, 5.0, 0.3, key='park_spread2')
        d2 = st.number_input('D2', 0.0, 10.0, 2.0, key='park_d2')
        ppe = st.number_input('PPE', 0.0, 1.0, 0.1, key='park_ppe')

    parkinsons_diagnosis = ''
    if st.button('Parkinson Test Result', key='park_btn'):
        parkinson_prediction = parkinson_model.predict([[fo, fhi, flo, jitter_percent, jitter_abs,
                                                         rap, ppq, ddp, shimmer, shimmer_db,
                                                         apq3, apq5, apq, dda, nhr, hnr, rpde,
                                                         dfa, spread1, spread2, d2, ppe]])
        parkinsons_diagnosis = 'Parkinson Detected' if parkinson_prediction[0] == 1 else 'No Parkinson'
    st.success(parkinsons_diagnosis)
