import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Autism_Adult_Data.csv')
    return data

data = load_data()

# Print column names to debug
#st.write("Columns in the dataset:", data.columns.tolist())

# Display the title and description
st.title("Autism Screening Application")
st.write("This application helps in identifying adults with potential autism based on their behavioral features.")

st.image("image1.jpg")

# Display the dataset
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

# Define input features
st.sidebar.header('Please answer the following questions:')
gender = st.sidebar.selectbox('Gender', ('m', 'f'))
age = st.sidebar.slider('Age', 1, 100, 25)
ethnicity = st.sidebar.selectbox('Ethnicity', data['ethnicity'].unique())
jundice = st.sidebar.selectbox('Jundice', ('yes', 'no'))
austim = st.sidebar.selectbox('Autism Family History', ('yes', 'no'))

# Define the 10 autism-related questions
questions = [
    'Do you often notice small sounds when others do not?',
    'Do you usually concentrate more on the whole picture, rather than the small details?',
    'In a social group, can you easily keep track of several different people’s conversations?',
    'Do you find it easy to go back and forth between different activities?',
    'Do you find it difficult to imagine what it would be like to be someone else?',
    'Do you find yourself drawn more strongly to people than to things?',
    'Do you prefer to do things the same way over and over again?',
    'Do you find it hard to make new friends?',
    'Do you notice patterns in things all the time?',
    'Do you find it easy to work out what someone is thinking or feeling just by looking at their face?'
]

answers = []
for i, question in enumerate(questions):
    answer = st.sidebar.slider(question, 0, 1, 0)
    answers.append(answer)

# Prepare the input features DataFrame
input_features = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'ethnicity': [ethnicity],
    'jundice': [jundice],
    'austim': [austim],
    'A1_Score': [answers[0]],
    'A2_Score': [answers[1]],
    'A3_Score': [answers[2]],
    'A4_Score': [answers[3]],
    'A5_Score': [answers[4]],
    'A6_Score': [answers[5]],
    'A7_Score': [answers[6]],
    'A8_Score': [answers[7]],
    'A9_Score': [answers[8]],
    'A10_Score': [answers[9]]
})

# Convert categorical variables to numeric for both training and prediction data
data = pd.get_dummies(data, drop_first=True)
input_features = pd.get_dummies(input_features)

# Ensure input_features has all columns used during training
missing_cols = set(data.columns) - set(input_features.columns)
for col in missing_cols:
    input_features[col] = 0

# Ensure input_features does not have extra columns
input_features = input_features[data.columns[:-1]]

# Handle missing values by filling them with the median value
data.fillna(data.median(), inplace=True)
input_features.fillna(input_features.median(), inplace=True)

# Automatically detect the target column
target_column = 'Class/ASD' if 'Class/ASD' in data.columns else data.columns[-1]
st.write(f"Target column for prediction: {target_column}")

# Prepare training data
X = data.drop(columns=[target_column])
y = data[target_column]

# Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predicting and displaying the result
if st.button('Predict'):
    prediction = model.predict(input_features)
    st.write('Prediction: ', 'You have autism' if prediction[0] == 1 else 'You do not have autism')

    # Show progress bar
    with st.spinner('Loading...'):
        import time
        time.sleep(2)
    st.success('Done!')

    # Display model accuracy
    st.write('Model Accuracy: ', accuracy_score(y_test, model.predict(X_test)))

    # Display confusion matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    st.write('Confusion Matrix:')
    st.write(cm)

    # Displaying the graph
    st.subheader('Feature Importance')
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    feature_importance.nlargest(10).plot(kind='barh')
    st.pyplot(plt)

# Upload and display media files
st.subheader("Upload and Display Media")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "mp4", "mp3", "wav"])
if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        st.image(uploaded_file)
    elif uploaded_file.type.startswith('video'):
        st.video(uploaded_file)
    elif uploaded_file.type.startswith('audio'):
        st.audio(uploaded_file)
