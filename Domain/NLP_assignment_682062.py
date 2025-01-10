import unicodedata
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from contractions import CONTRACTION_MAP
import re
from autocorrect import Speller
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd


def preprocess_text(text, use_lemmatization=True):
    def character_normalization(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) \
                if contraction_mapping.get(match) \
                else contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    def remove_extra_new_lines(text):
        return re.sub(r'[\r|\n|\r\n]+', ' ', text)

    def case_conversion(text):
        return text.lower()

    def autocorrect(text):
        spell = Speller(fast=True)
        return spell(text)

    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')
    # Keeping not to preserve negative sentiment
    stopword_list.remove('not')

    def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopwords]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    def special_char_removal(text, remove_digits=False):
        def remove_special_characters(text, remove_digits):
            pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
            text = re.sub(pattern, '', text)
            return text

        special_char_pattern = re.compile(r'([{.(-)!}])')
        text = special_char_pattern.sub(" \\1 ", text)
        return remove_special_characters(text, remove_digits)

    def extra_white_space_removal(text):
        return re.sub(' +', ' ', text)

    # Apply the preprocessing steps
    text = character_normalization(text)
    text = expand_contractions(text)
    text = case_conversion(text)
    text = remove_extra_new_lines(text)
    if use_lemmatization:
        text = lemmatize_text(text)
    text = special_char_removal(text, remove_digits=True)
    text = extra_white_space_removal(text)
    text = remove_stopwords(text, is_lower_case=True, stopwords=stopword_list)
    text = autocorrect(text)

    return text


def text_to_df(file_path):
    # reading the data
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split each line on the tab character
            parts = line.strip().split('\t')
            if len(parts) == 2:  # Ensure the line has exactly two parts
                data.append(parts)
            # Convert the processed data into a DataFrame
    return pd.DataFrame(data)


def train_random_forest(df, vectorizer=CountVectorizer(), use_embedding=False, X_embeddings=None):
    # Step 1: Prepare the data
    X = df['Review']  # Feature: Sentences
    y = df['Score']  # Target: Sentiment scores

    # Step 2: Create a Bag-of-Words representation or other passed representation
    if use_embedding:
        X = X_embeddings
    else:
        X = vectorizer.fit_transform(X)

    # Define the model
    rf = RandomForestClassifier(random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

    # Train-test split

    # Separating training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best parameters and accuracy
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")

    # Evaluate on test set
    best_rf = grid_search.best_estimator_
    accuracy = best_rf.score(X_test, y_test)
    print(f"Test Set Accuracy: {accuracy}")

    # Confusion Matrix
    y_pred = best_rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_rf.classes_, yticklabels=best_rf.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap")
    plt.show()

    return accuracy


# Load GloVe embeddings
def load_glove_embeddings(glove_file):
    glove_embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove_embeddings[word] = vector
    return glove_embeddings


def get_average_glove_embedding(review, glove_embeddings):
    words = review.split()
    word_vectors = [glove_embeddings[word] for word in words if word in glove_embeddings]
    if not word_vectors:
        return np.zeros(300)  # Adjust the size based on the GloVe dimension (300 in this case)
    return np.mean(word_vectors, axis=0)


# Create a new feature matrix using average GloVe embeddings
def create_glove_embeddings(df, glove_embeddings):
    return np.array([get_average_glove_embedding(review, glove_embeddings) for review in df['Review']])


# Function to extract keyphrases using KeyBERT
def extract_keyphrases(text, model):
    keyphrases = model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
    return " ".join([phrase[0] for phrase in keyphrases])


# Importing data
df_amazon = text_to_df('../Data/sentiment labelled sentences/amazon_cells_labelled.txt')
df_imdb = text_to_df('../Data/sentiment labelled sentences/imdb_labelled.txt')
df_yelp = text_to_df('../Data/sentiment labelled sentences/yelp_labelled.txt')
df = pd.concat([df_yelp, df_imdb, df_amazon], ignore_index=True)
df.columns = ['Review', 'Score']
print(df)

# Check if preprocessed pickle file exists
pickle_file = 'preprocessed_dataframe.pkl'

if os.path.exists(pickle_file):
    # Load the DataFrame from the pickle file
    df = pd.read_pickle(pickle_file)
else:
    df['Review'] = df['Review'].apply(preprocess_text)
    # Save the preprocessed DataFrame as a pickle file
    df.to_pickle(pickle_file)

# Training original model
original_model_accuracy = train_random_forest(df)

# Part 3
# Lemma
df_lemma = pd.concat([df_yelp, df_imdb, df_amazon], ignore_index=True)
df_lemma.columns = ['Review', 'Score']

# Check if preprocessed pickle file exists
pickle_file = 'preprocessed_dataframe_no_lemma.pkl'

if os.path.exists(pickle_file):
    # Load the DataFrame from the pickle file
    df_lemma = pd.read_pickle(pickle_file)
else:
    df_lemma['Review'] = df_lemma['Review'].apply(preprocess_text, args=(False,))
    # Save the preprocessed DataFrame as a pickle file
    df_lemma.to_pickle(pickle_file)

no_lemma_model_accuracy = train_random_forest(df_lemma)
print(f"No lemmatization accuracy: {no_lemma_model_accuracy}")

# tf.idf

tf_idf_model_accuracy = train_random_forest(df, vectorizer=TfidfVectorizer())
print(f"TF.IDF model accuracy: {tf_idf_model_accuracy}")

# N-grams
n_grams_model_accuracy = train_random_forest(df, vectorizer=CountVectorizer(ngram_range=(1, 3)))
print(f"1-3-Gram model accuracy: {n_grams_model_accuracy}")

# Embedding



# Load the GloVe embeddings (adjust the path as needed)
glove_embeddings = load_glove_embeddings('glove.6B.300d.txt')

# Preprocess the text
df['Review'] = df['Review'].apply(preprocess_text)

# Create GloVe embeddings
X_embeddings = create_glove_embeddings(df, glove_embeddings)

# Train the Random Forest model using the embeddings
glove_accuracy = train_random_forest(df, use_embedding=True, X_embeddings=X_embeddings)
print(f"GloVe model accuracy: {glove_accuracy}")

# Keyphrase extraction
df_keybert = pd.concat([df_yelp, df_imdb, df_amazon], ignore_index=True)
df_keybert.columns = ['Review', 'Score']

# Check if preprocessed pickle file exists
pickle_file = 'preprocessed_keyphrases_dataframe.pkl'

if os.path.exists(pickle_file):
    # Load the DataFrame from the pickle file
    df_keybert = pd.read_pickle(pickle_file)
else:
    # Preprocess text (your existing function)
    df_keybert['Review'] = df_keybert['Review'].apply(preprocess_text)

    # Keyphrase extraction
    kw_model = KeyBERT()
    df_keybert['Review'] = df_keybert['Review'].apply(lambda x: extract_keyphrases(x, kw_model))

    # Save the preprocessed DataFrame as a pickle file
    df_keybert.to_pickle(pickle_file)

# Train the random forest model using the enhanced reviews
keyphrase_model_accuracy = train_random_forest(df_keybert[['Review', 'Score']])
print(f"Keyphrase model accuracy: {keyphrase_model_accuracy}")
