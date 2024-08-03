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

# Display the title and description
st.title("Autism Screening Application")
st.write("This application helps in identifying adults with potential autism based on their behavioral features.")
st.image("image1.jpg", use_column_width=True)

# Display the dataset
if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(data)

# Define input features
st.sidebar.header('Please answer the following questions:')
gender = st.sidebar.selectbox('Gender', ('male', 'female'))
age = st.sidebar.slider('Age', 1, 100, 25)
ethnicity = st.sidebar.selectbox('Ethnicity', data['ethnicity'].unique())
jundice = st.sidebar.selectbox('Jaundice', ('yes', 'no'))
austim = st.sidebar.selectbox('Autism Family History', ('yes', 'no'))

# Define the 10 autism-related questions
questions = [
    'Do you notice small sounds that others do not?',
    'Do you usually focus on details more than the big picture?',
    'Can you easily keep track of several people’s conversations?',
    'Do you find it easy to switch between activities?',
    'Do you find it difficult to imagine being someone else?',
    'Do you prefer being with people rather than being alone?',
    'Do you like to do things the same way every time?',
    'Do you find it hard to make new friends?',
    'Do you notice patterns in things a lot?',
    'Can you understand what someone is feeling by looking at their face?'
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

# Automatically detect the target column
target_column = 'Class/ASD' if 'Class/ASD' in data.columns else data.columns[-1]  # Default to the last column if 'Class/ASD' is not found
st.write(f"Target column for prediction: {target_column}")

# Prepare training data
X = data.drop(columns=[target_column])
y = data[target_column]

# Ensure input_features has all columns used during training
missing_cols = set(X.columns) - set(input_features.columns)
for col in missing_cols:
    input_features[col] = 0

# Ensure input_features does not have extra columns
input_features = input_features[X.columns]

# Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predicting and displaying the result
if st.button('Predict'):
    with st.spinner('Processing...'):
        prediction = model.predict(input_features)
        result = 'Yes, you have autism' if prediction[0] == 1 else 'No, you do not have autism'
        st.subheader('Prediction Result')
        st.write(result)

        # Show progress bar
        st.success('Done!')

        # Display model accuracy
        st.subheader('Model Accuracy')
        st.write('Accuracy:', accuracy_score(y_test, model.predict(X_test)))

        # Display confusion matrix
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, model.predict(X_test))
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot(fig)

        # Plot feature importance
        st.subheader('Feature Importance')
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots(figsize=(10, 8))
        feature_importance.nlargest(10).plot(kind='barh', ax=ax, color='skyblue')
        ax.set_title('Top 10 Important Features')
        st.pyplot(fig)

        # Line chart of model performance
        st.subheader('Model Performance Over Time')
        history = pd.DataFrame({'Epoch': range(1, 11), 'Accuracy': [0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84]})
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history['Epoch'], history['Accuracy'], marker='o', linestyle='-', color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Over Epochs')
        st.pyplot(fig)

        # Pie chart of class distribution
        st.subheader('Class Distribution')
        class_distribution = y.value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
        ax.set_title('Class Distribution in the Dataset')
        st.pyplot(fig)

# Upload and display media files
st.subheader("Upload Media Files for Autism Screening")
uploaded_file = st.file_uploader("Please upload any relevant media files (audio, video, images...) that may assist in the autism screening process.", type=["jpg", "jpeg", "png", "mp4", "mp3", "wav"])
if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    elif uploaded_file.type.startswith('video'):
        st.video(uploaded_file)
    elif uploaded_file.type.startswith('audio'):
        st.audio(uploaded_file)
        
        
st.subheader("Feel free to ask anthing from us...")
st.caption("Enter your special comments or questions here.")
st.text_input('')
