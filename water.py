import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import missingno as msno

#st.markdown('<style>header{visibility: hidden;}</style>', unsafe_allow_html=True)
st.markdown('<style>footer{visibility: hidden;}</style>', unsafe_allow_html=True)

# Function to load data
def load_data(file_path):
    if file_path:
        return pd.read_csv(file_path)
    else:
        return None

# Function to process data
def process_data(df):
    if df is not None:
        # Add your processing code here
        pass

# Function to generate plots based on selected columns
def generate_plots(data, selected_columns, plot_type):
    if plot_type == 'Line Plot':
        st.write("### Line Plot")
        sns.lineplot(data=data[selected_columns])
        st.pyplot()

    elif plot_type == 'Bar Plot':
        st.write("### Bar Plot")
        sns.barplot(data=data[selected_columns])
        st.pyplot()

    elif plot_type == 'Scatter Plot':
        st.write("### Scatter Plot")
        sns.scatterplot(data=data[selected_columns])
        st.pyplot()

    elif plot_type == 'Histogram':
        st.write("### Histogram")
        sns.histplot(data=data[selected_columns], kde=True)
        st.pyplot()

    elif plot_type == 'Box Plot':
        st.write("### Box Plot")
        sns.boxplot(data=data[selected_columns])
        st.pyplot()

# Main function
def main():
    # Page title

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    # Load data
    data = load_data(uploaded_file)

    if data is not None:
        st.title("Exploratory Data Analysis ")

        # Display the dataset
        st.write("### Dataset")
        st.write(data)
        st.markdown("---")

        # Process data
        process_data(data)
        st.write(':persevere:**`shape of the data`:**')
        a = data.shape
        b = a[0]
        c = a[1]
        st.write('The data has: ', b, 'rows and ', c, 'columns')
        st.markdown("---")

        st.write(':yum:**`checking missing values in data`:**')
        msno.matrix(data)
        st.pyplot()
        st.markdown("---")

        d = data.isna().sum()
        st.write("missing values per column: ",d)
        st.markdown("---")

        # Select columns
        selected_columns = st.multiselect("Select columns for plotting", data.columns)
        st.markdown("---")

        # Select plot type
        plot_type = st.selectbox("Select plot type", ['Line Plot', 'Bar Plot', 'Scatter Plot', 'Histogram', 'Box Plot'])
        st.markdown("---")

        if len(selected_columns) > 0:
            # Generate plots
            generate_plots(data, selected_columns, plot_type)
        else:
            st.warning("Please select at least one column for plotting.")
    


    else:
        data = pd.read_csv("water_potability.csv")
        st.image("1.jpg", use_column_width=True)
        st.markdown("---")
        st.title("Water Potability Prediction App")
        st.markdown("---")
        st.subheader(""" 
                                     :blush:`About:` 

            :smile: Welcome to the `Water Potability Prediction App`. This Streamlit web 
            application enables users to `predict water potability` based on various 
            water :clap: quality parameters. :open_mouth: Through interactive `visualization` tools 
            and a `user-friendly interface`, users can explore water quality data, input 
            parameter values, and obtain predictions regarding water potability.:stuck_out_tongue_closed_eyes: The app 
            also provides `evaluation metrics` to assess the performance of the `machine 
            learning model`, making it a valuable tool for researchers and environmentalists
            interested in `water quality assessment` :wave:
            """)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown("---")
        st.write(':persevere:**`shape of the data`:**')
        a = data.shape
        b = a[0]
        c = a[1]
        st.write('The data has: ', b, 'rows and ', c, 'columns')
        st.markdown("---")
        st.write(':yum:**`checking missing values in data`:**')
        msno.matrix(data)
        st.pyplot()
        st.write(":sweat_smile:**`missing values in percentage`:**")
        st.markdown("---")
        def per_missing(df):
            '''
            Finds the percentage of missing values in each column
            '''
            tot_len = len(df)
            per_missing = df.isna().sum().to_frame()
            per_missing['% missing'] = (per_missing[0]/tot_len) *100
            return per_missing.drop(columns = [0]).sort_values(by = ['% missing'], ascending= False)
        perc = per_missing(data)
        st.write(perc)
        data.dropna(inplace=True)
        st.sidebar.header("Features Selection")
        show_data = st.sidebar.checkbox("Show Raw Data", help="Check this box to display the raw data.")
        if show_data:
            st.subheader("Raw Data")
            st.write(data)
        st.sidebar.subheader("Data Statistics")
        if st.sidebar.checkbox("Show Data Information", help="Check this box to display data information."):
            st.write("Data Information")
            st.write(data.info())
            st.markdown("---")
        if st.sidebar.checkbox("Show Data Summary", help="Check this box to display summary statistics."):
            st.write("Data Summary")
            st.write(data.describe())
        st.sidebar.text("Potability; 1:potable 0:not_potable")
        st.sidebar.subheader("Exploratory Data Analysis")
        if st.sidebar.checkbox("Potability vs PH"):
            potability_ph_fig = px.violin(data, x='Potability', y='ph', box=True, points="all", color="Potability", color_discrete_map={0: "blue", 1: "orange"})
            st.plotly_chart(potability_ph_fig)
            st.markdown("---")
        if st.sidebar.checkbox("Potability vs Hardness"):
            st.sidebar.subheader("Potability vs Hardness")
            potability_hardness_fig = px.violin(data, x='Potability', y='Hardness', box=True, points="all", color="Potability", color_discrete_map={0: "blue", 1: "orange"})
            st.plotly_chart(potability_hardness_fig)
            st.markdown("---")
        if st.sidebar.checkbox("Potability vs Solids"):
            st.sidebar.subheader("Potability vs Solids")
            potability_solids_fig = px.violin(data, x='Potability', y='Solids', box=True, points="all", color="Potability", color_discrete_map={0: "blue", 1: "orange"})
            st.plotly_chart(potability_solids_fig)
            st.markdown("---")
        if st.sidebar.checkbox("Pairplot"):
            st.write("### Pairplot")
            pairplot_fig = sns.pairplot(data, hue='Potability')
            st.pyplot(pairplot_fig)
            st.markdown("---")
        if st.sidebar.checkbox("Correlation Heatmap"):
            st.write("### Correlation Heatmap:")
            corr = data.corr()
            plt.figure(figsize=(10, 8))
            heatmap_fig = sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
            st.pyplot(heatmap_fig.get_figure())
            st.markdown("---")
        st.sidebar.subheader("Data Visualization")
        if st.sidebar.checkbox("Histogram of Water Potability"):
            st.sidebar.write("### Histogram of Water Potability:")
            fig, ax = plt.subplots()
            data["Potability"].value_counts().plot(kind="bar", ax=ax)
            st.sidebar.pyplot(fig)
            st.markdown("---")
        if st.sidebar.checkbox("Scatter Plot of pH vs. Hardness"):
            st.sidebar.write("### Scatter Plot of pH vs. Hardness")
            scatter_fig = px.scatter(data, x="ph", y="Hardness", color="Potability")
            st.plotly_chart(scatter_fig)
            st.markdown("---")
        st.sidebar.subheader("Model Prediction")
        ranges = {
            'ph': (6.5, 8.5),
            'hardness': (0.0, 250.0),
            'solids': (0.0, 5000.0),
            'chloramines': (1.0, 4.0),
            'sulfate': (0.0, 250.0),
            'conductivity': (0.0, 800.0),
            'organic_carbon': (0.0, 2.0),
            'trihalomethanes': (0.0, 0.080),
            'turbidity': (0.0, 1.0)
        }
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
        st.sidebar.subheader("Enter Water Quality Parameters")
        input_values = {}
        for param, param_info in parameters.items():
            potable_min, potable_max = param_info['potable_range']
            slider_min, slider_max = param_info['slider_range']
            input_values[param] = st.sidebar.slider(f"{param.capitalize()}", min_value=slider_min, max_value=slider_max, value=(potable_min + potable_max) / 2, step=0.1)
        X = data.drop("Potability", axis=1)
        y = data["Potability"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        svm_model = SVC()
        svm_model.fit(X_train, y_train)
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        svm_prediction = svm_model.predict(X_test)
        lr_prediction = lr_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        st.markdown("---")
        f1 = f1_score(y_test, y_pred)
        st.write(" :sparkles: `Model Accuracy`")
        st.markdown("---")
        st.subheader("RandomForestClassifier")
        st.write("The accuracy of the model is: {:.2f}%".format(accuracy * 100))
        st.markdown("---")
        st.subheader("Support Vector Machine")
        svm_accuracy = accuracy_score(y_test, svm_prediction)
        st.write("The accuracy of the model is: {:.2f}%".format(svm_accuracy * 100))
        st.markdown("---")
        st.subheader("LogisticRegression")
        lr_accuracy = accuracy_score(y_test, lr_prediction)
        st.write("The accuracy of the model is: {:.2f}%".format(lr_accuracy * 100))
        st.markdown("---")
        st.subheader("Model Evaluation Metrics")
        st.write(f"Precision: {precision:.2f}")
        st.markdown("---")
        st.write(f"Recall: {recall:.2f}")
        st.markdown("---")
        st.write(f"F1-score: {f1:.2f}")
        st.markdown("---")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        st.pyplot()
        st.markdown("---")
        def predict_potability(input_values):
            for param, value in input_values.items():
                if not parameters[param]['potable_range'][0] <= value <= parameters[param]['potable_range'][1]:
                    return "Not Potable"
            return "Potable"
        prediction = predict_potability(input_values)
        st.subheader(":sparkles:`Prediction`")
        if prediction == "Potable":
            st.write("Based on the input parameters, the water is predicted to be: :star2:**Potable**:grey_exclamation:")
        else:
            st.write("Based on the input parameters, the water is predicted to be: 💔 :broken_heart:**Not Potable**:grey_exclamation:")
        st.markdown("---")
        st.write("`Built with ❤️ by KinplusTech`")

# Run the app
if __name__ == "__main__":
    main()