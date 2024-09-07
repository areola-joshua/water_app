import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msn
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

def load_data(file_path):
    if file_path:
        return pd.read_csv(file_path)
    else:
        # Provide default example dataset
        example_data = {
            'ph': np.random.uniform(0.0, 14.0, 1000),
            'hardness': np.random.uniform(0.0, 250.0, 1000),
            'solids': np.random.uniform(20.0, 500.0, 1000),
            'chloramines': np.random.uniform(1.0, 4.0, 1000),
            'sulfate': np.random.uniform(0.0, 250.0, 1000),
            'conductivity': np.random.uniform(0.0, 800.0, 1000),
            'organic_carbon': np.random.uniform(0.0, 2.0, 1000),
            'trihalomethanes': np.random.uniform(0.0, 0.08, 1000),
            'turbidity': np.random.uniform(0.0, 7.0, 1000),
            'Potability': np.random.randint(0, 2, 1000)
        }
        return pd.DataFrame(example_data)

# Load your dataset
upload = st.sidebar.file_uploader('Upload your file here:', type=['csv', 'xlsx'])

@st.cache_data
def load_data():
    if upload:
        return pd.read_csv(upload) if upload.type == 'csv' else pd.read_excel(upload)
    else:
        return load_data(None)

data = load_data()

st.image("1.jpg", use_column_width=True)
st.markdown("---")
st.title(":wave: `Water Potability Prediction App`")
st.markdown("---")
st.subheader(":blush:`About:`")
st.markdown("""
    :smile: Welcome to the `Water Potability Prediction App`. This Streamlit Web 
    application enables users to `predict water potability` based on various 
    water :clap: quality parameters. :open_mouth: Through interactive `visualization` tools 
    and a `user-friendly interface`, users can explore water quality data, input 
    parameter values, and obtain predictions regarding water 
    potability.:stuck_out_tongue_closed_eyes: The app 
    also provides `evaluation metrics` to assess the performance of the `machine 
    learning model`, making it a valuable tool for researchers and environmentalists
    interested in `water quality assessment` :wave:
    """)


# Data Exploration
st.sidebar.subheader("Features Selection")

# Checkbox for showing raw data
show_data = st.sidebar.checkbox("Show Raw Data", help="Check this box to display the raw data.")
if show_data:
    st.subheader("Raw Data")
    st.write(data)

# Display missing values
st.markdown("---")
st.write(':persevere:**`shape of the data`:**')
a = data.shape
b = a[0]
c = a[1]
st.write('The data has: ', b, 'rows and ', c, 'columns')
st.write(":astonished:**`Missing Values`:**")
sum_isna = data.isna().sum()
st.write(sum_isna)
st.markdown("---")
st.write(':yum:**`Plot of missing values in data`:**')
msn.matrix(data)
st.pyplot()
st.markdown("---")

data.dropna(inplace=True)
a = data.shape
b = a[0]
c = a[1]
st.write('After Cleaning, the data has: ', b, 'rows and ', c, 'columns')
st.markdown("---")

# Iterate over each parameter and create slider
parameters = {
    'ph': {'potable_range': (6.5, 8.5), 'slider_range': (0.0, 14.0)},
    'hardness': {'potable_range': (0.0, 250.0), 'slider_range': (0.0, 1000.0)},
    'solids': {'potable_range': (20.0, 500.0), 'slider_range': (0.0, 1000.0)},
    'chloramines': {'potable_range': (0.0, 4.0), 'slider_range': (0.0, 10.0)},
    'sulfate': {'potable_range': (0.0, 250.0), 'slider_range': (0.0, 1000.0)},
    'conductivity': {'potable_range': (0.0, 800.0), 'slider_range': (0.0, 1500.0)},
    'organic_carbon': {'potable_range': (0.0, 2.0), 'slider_range': (0.0, 10.0)},
    'trihalomethanes': {'potable_range': (0.0, 0.08), 'slider_range': (0.0, 0.20)},
    'turbidity': {'potable_range': (0.0, 7.0), 'slider_range': (0.0, 15.0)}
}
input_values = sidebar_inputs(parameters)

# Model training and evaluation
X = data.drop("Potability", axis=1)
y = data["Potability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Train SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Train Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Predictions Random Forest
y_pred = model.predict(X_test)

# Make predictions using SVM model
svm_prediction = svm_model.predict(X_test)

# Make predictions using Logistic Regression model
lr_prediction = lr_model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display Accuracy
st.write(":sparkles: **Model Accuracy**")
st.subheader("RandomForestClassifier")
st.write("The accuracy of the model is: {:.2f}%".format(accuracy * 100))
st.subheader("Support Vector Machine")
svm_accuracy = accuracy_score(y_test, svm_prediction)
st.write("The accuracy of the model is: {:.2f}%".format(svm_accuracy * 100))
st.subheader("Logistic Regression")
lr_accuracy = accuracy_score(y_test, lr_prediction)
st.write("The accuracy of the model is: {:.2f}%".format(lr_accuracy * 100))

# Display evaluation metrics
st.subheader("Model Evaluation Metrics")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1-score: {f1:.2f}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
st.pyplot()

# Prediction function
def predict_potability(input_values):
    for param, value in input_values.items():
        if not parameters[param]['potable_range'][0] <= value <= parameters[param]['potable_range'][1]:
            return "Not Potable"
    return "Potable"

# Predict water potability
prediction = predict_potability(input_values)

# Display Prediction
st.subheader(":sparkles: **Prediction**")
if prediction == "Potable":
    st.write("Based on the input parameters, the water is predicted to be: :star2:**Potable**:grey_exclamation:")
else:
    st.write("Based on the input parameters, the water is predicted to be: 💔 :broken_heart:**Not Potable**:grey_exclamation:")

# Footer
st.markdown("---")
st.write("`Built with ❤️ by KinplusTech`")
