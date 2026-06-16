import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
import re
from collections import Counter

# Page configuration
st.set_page_config(page_title="Spam/Ham Email Analysis", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv('spam_ham_dataset.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_data()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_data
def prepare_features(_df):
    _df = _df.copy()
    _df['clean_text'] = _df['text'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_sparse = vectorizer.fit_transform(_df['clean_text'])
    X_dense = X_sparse.toarray()       # convert to dense for hashing
    return X_dense, vectorizer, _df

X, vectorizer, df_clean = prepare_features(df)
y = df_clean['label_num'].values


@st.cache_resource
def train_model(_X, _y):
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)


st.sidebar.title("Navigation")
sections = [
    "EDA & Visualizations",
    "Case Study",
    "Machine Learning",
    "Prediction Playground",
    "Raw Data & Export"
]
choice = st.sidebar.radio("Go to", sections)

st.title("Spam/Ham Email Analysis Dashboard")


if choice == "EDA & Visualizations":
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots()
        df_clean['label'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red'])
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Text Length Distribution")
        df_clean['text_length'] = df_clean['text'].apply(len)
        fig, ax = plt.subplots()
        sns.histplot(data=df_clean, x='text_length', hue='label', bins=30, kde=True, ax=ax)
        ax.set_xlabel('Text Length (characters)')
        st.pyplot(fig)
    
    st.subheader("Word Clouds")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Spam")
        spam_text = ' '.join(df_clean[df_clean['label']=='spam']['clean_text'])
        if spam_text:
            wc = WordCloud(width=400, height=300, background_color='white').generate(spam_text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    with col2:
        st.write("Ham")
        ham_text = ' '.join(df_clean[df_clean['label']=='ham']['clean_text'])
        if ham_text:
            wc = WordCloud(width=400, height=300, background_color='white').generate(ham_text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    st.subheader("Most Common Words")
    spam_words = ' '.join(df_clean[df_clean['label']=='spam']['clean_text']).split()
    ham_words = ' '.join(df_clean[df_clean['label']=='ham']['clean_text']).split()
    spam_counter = Counter(spam_words).most_common(10)
    ham_counter = Counter(ham_words).most_common(10)
    col1, col2 = st.columns(2)
    with col1:
        st.write("Top 10 spam words")
        st.dataframe(pd.DataFrame(spam_counter, columns=['Word', 'Frequency']))
    with col2:
        st.write("Top 10 ham words")
        st.dataframe(pd.DataFrame(ham_counter, columns=['Word', 'Frequency']))

elif choice == "Case Study":
    st.header("Case Study: Misclassified Emails")
    st.write(
        "We examine emails that the model predicted incorrectly. "
        "This helps identify patterns where the model gets confused."
    )
    
    # Re-create the same train/test split to obtain original text indices
    n = len(df_clean)
    indices = np.arange(n)
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    test_df = df_clean.iloc[test_idx].reset_index(drop=True)
    
    y_pred = model.predict(X_test)
    misclass_mask = y_test != y_pred
    misclass_indices = np.where(misclass_mask)[0]
    
    if len(misclass_indices) == 0:
        st.success("No misclassifications in the test set! The model performed perfectly.")
    else:
        st.write(f"**{len(misclass_indices)} misclassified emails** out of {len(y_test)} test samples.")
        misclass_data = []
        for idx in misclass_indices:
            true_label = 'spam' if y_test[idx] == 1 else 'ham'
            pred_label = 'spam' if y_pred[idx] == 1 else 'ham'
            text_preview = test_df.loc[idx, 'text'][:200] + "..." if len(test_df.loc[idx, 'text']) > 200 else test_df.loc[idx, 'text']
            misclass_data.append({
                "True Label": true_label,
                "Predicted": pred_label,
                "Email Preview": text_preview
            })
        st.dataframe(pd.DataFrame(misclass_data), use_container_width=True)
    
    st.subheader("What can we learn?")
    st.markdown("""
    - **False positives (ham predicted as spam):** Often contain words commonly found in spam (e.g., 'free', 'offer') but in a legitimate context.
    - **False negatives (spam predicted as ham):** May use subtle language or mimic normal conversation to evade filters.
    - Reviewing these examples can guide improvements such as adding new features, adjusting threshold, or collecting more diverse training data.
    """)

elif choice == "Machine Learning":
    st.header("Machine Learning Model")
    st.write("Training a Logistic Regression classifier on TF-IDF features.")
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Test Accuracy: {acc:.2%}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=['ham', 'spam'], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

elif choice == "Prediction Playground":
    st.header("Prediction Playground")
    st.write("Enter an email text to classify as spam or ham.")
    
    user_input = st.text_area("Email text", height=200)
    if st.button("Predict"):
        if user_input:
            cleaned = preprocess_text(user_input)
            input_vec = vectorizer.transform([cleaned]).toarray()
            pred = model.predict(input_vec)[0]
            prob = model.predict_proba(input_vec)[0]
            label = 'spam' if pred == 1 else 'ham'
            confidence = prob[pred]
            st.write(f"**Prediction:** {label.upper()} (confidence: {confidence:.2%})")
            if label == 'spam':
                st.error("This email is classified as SPAM.")
            else:
                st.success("This email is classified as HAM.")
        else:
            st.warning("Please enter some text.")

elif choice == "Raw Data & Export":
    st.header("Raw Data")
    st.write("View and export the dataset.")
    
    st.subheader("Data Preview")
    st.dataframe(df_clean[['label', 'text']].head(100))
    
    st.subheader("Download Data")
    csv = df_clean[['label', 'text']].to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="spam_ham_clean.csv", mime="text/csv")