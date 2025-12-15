import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data = pd.read_csv("data/data.csv")

    data = data.drop(['Unnamed: 32','id'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data = get_clean_data()

    slider_labels = [
        ('Radius mean','radius_mean'), 
        ('Texture mean','texture_mean'),
        ('perimeter mean','perimeter_mean'), 
        ('area mean','area_mean'), 
        ('smoothness mean','smoothness_mean'),
        ('compactness mean','compactness_mean'), 
        ('concavity mean','concavity_mean'), 
        ('concave points mean','concave points_mean'), 
        ('symmetry mean','symmetry_mean'), 
        ('fractal dimension mean','fractal_dimension_mean'),
        ('radius se','radius_se'), 
        ('texture se','texture_se'), 
        ('perimeter se','perimeter_se'), 
        ('area se','area_se'), 
        ('smoothness se','smoothness_se'),
        ('compactness se','compactness_se'), 
        ('concavity se','concavity_se'), 
        ('concave points se','concave points_se'), 
        ('symmetry se','symmetry_se'),
        ('fractal dimension se','fractal_dimension_se'),
        ('radius worst','radius_worst'), 
        ('texture worst','texture_worst'), 
        ('perimeter worst','perimeter_worst'), 
        ('area worst','area_worst'), 
        ('smoothness worst','smoothness_worst'),
        ('compactness worst','compactness_worst'), 
        ('concavity worst','concavity_worst'),
        ('concave points worst','concave points_worst'), 
        ('symmetry worst','symmetry_worst'),
        ('fractal dimension worst','fractal_dimension_worst') 
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
        label,
        min_value=float(0 ),
        max_value=float(data[key].max()),
        value = float(data[key].mean()), 
         )

    return input_dict

def get_scalled_data(input_dict):
    data = get_clean_data()

    X = data.drop('diagnosis', axis=1)

    scale_dict = {}

    for key, value in input_dict.items():
        min_val = X[key].min()
        max_val = X[key].max()
        scaled_value = (value - min_val) / (max_val - min_val)
        scale_dict[key] = scaled_value

    return scale_dict

def get_radar_chart(input_data):
    input_data = get_scalled_data(input_data)

    categories = [
        'Radius', 'Texture', 'Perimeter',
          'Area', 'Smoothness', 'Compactness', 
          'Concavity', 'Concave Points', 'Symmetry',
            'Fractal Dimension',''
            ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'],
            input_data['texture_mean'],
            input_data['perimeter_mean'],
            input_data['area_mean'],
            input_data['smoothness_mean'],
            input_data['compactness_mean'],
            input_data['concavity_mean'],
            input_data['concave points_mean'],
            input_data['symmetry_mean'],
            input_data['fractal_dimension_mean'],
        ],
        theta=categories,
        fill='toself',
        name='Mean Values'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'],
            input_data['texture_se'],
            input_data['perimeter_se'],
            input_data['area_se'],
            input_data['smoothness_se'],
            input_data['compactness_se'],
            input_data['concavity_se'],
            input_data['concave points_se'],
            input_data['symmetry_se'],
            input_data['fractal_dimension_se'],
        ],
        theta=categories,
        fill='toself',
        name='Standard Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],
            input_data['texture_worst'],
            input_data['perimeter_worst'],
            input_data['area_worst'],
            input_data['smoothness_worst'],
            input_data['compactness_worst'],
            input_data['concavity_worst'],
            input_data['concave points_worst'],
            input_data['symmetry_worst'],
            input_data['fractal_dimension_worst'],
        ],
        theta=categories,
        fill='toself',
        name='Worst  Error',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )

    return fig

def add_prediction(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scalar = pickle.load(open("model/scalar.pkl", "rb"))

    input_array = np.array([list(input_data.values())]).reshape(1, -1)
    
    input_scaled_array = scalar.transform(input_array)

    prediction = model.predict(input_scaled_array)

    st.header("Cell Clustor Prediction")

    if prediction[0] == 1:
        prediction = "The Abnormality of Cell is Detected"

    else:
        prediction = "The Cell is Normal"

    st.write(prediction)

    st.write("Probability of being benign: ", model.predict_proba(input_scaled_array)[0][0]*100 , "%")
    st.write("Probability of being malignant: ", model.predict_proba(input_scaled_array)[0][1]*100, "%")

    st.write("This app can assist the professionals in diagnosing breast cancer based on cytology data. However, it should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")
    
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="PredictaForge", 
        page_icon="female-doctor", 
        layout="wide",
        initial_sidebar_state="expanded"      
    )

    load_css()

    input_data = add_sidebar()

    with st.container():
        st.title("PredictaForge")
        st.write("Please collect this app your cytology to help diagnose breast cancer form your tissue sample.")

    col1 , col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)

    with col2:
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        add_prediction(input_data)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()