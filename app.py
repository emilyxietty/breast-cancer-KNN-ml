import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset
breast_cancer_data = load_breast_cancer()
feature_names = breast_cancer_data.feature_names

X = pd.DataFrame(breast_cancer_data.data, columns=feature_names)
y = pd.Series(breast_cancer_data.target)

# Load the pre-trained model
with open('knn_bc_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def print_title():
    st.title('Breast Cancer Classifier')
    st.write("This model using a KNN algorithm (k=23). Accuracy of a KNN model where k=23 is 96.49%, which is the highest accuracy of all k values between 1 and 100. Features and data are drawn from the Breast cancer wisconsin (diagnostic) dataset.")
    st.write("Please note: This model is not considered a reliable source of medical information and cannot replace consultation from a medical professional.")
    st.markdown("""---""")

def print_credits():
    st.write("Dataset from: July-August 1995. - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 163-171.")

# Define the Streamlit app
def main():
    print_title()
    
    st.write("Enter the features below to make a prediction.")
    input_features = {}
    for feature_name in feature_names:
        default_value = 0.0  # You can set a default value for each feature
        input_features[feature_name] = st.number_input(feature_name, value=default_value)

    # Convert the input features into a DataFrame for plotting
    input_data = pd.DataFrame([input_features])

    # Make a prediction when a button is clicked
    if st.button('Make Prediction'):
        # Prepare the input features as a list
        input_values = [input_features[feature_name] for feature_name in feature_names]
        input_values = [input_values]  # Convert to 2D array for prediction

        # Use the loaded model to make a prediction
        prediction = loaded_model.predict(input_values)

        # Display the prediction
        target_names = breast_cancer_data.target_names
        st.markdown("""---""")
        st.title("Results")
        st.write("Please note: This model is not considered a reliable source of medical information and cannot replace consultation from a medical professional.")
        st.write(f"If a tumour is present, your results indicate a {target_names[prediction[0]]} tumour.")
        if target_names[prediction[0]] == 'malignant':
            st.write("A consultation with a doctor may be beneficial.")
        st.markdown("""---""")
        # Create a scatter plot to visualize the new input in relation to existing data points
        st.write("Below is a scatter plot demonstrating the relationship of 2 variables in your value vs. our testing data values.")
        fig, ax = plt.subplots()
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='viridis', label='Existing Data Points')
        ax.scatter(input_data.iloc[:, 0], input_data.iloc[:, 1], c='red', marker='x', label='New Input')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.legend()
        st.pyplot(fig)

        st.markdown("""---""")

        print_credits()



if __name__ == '__main__':
    main()
