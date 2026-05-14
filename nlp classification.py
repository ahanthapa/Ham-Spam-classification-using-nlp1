#Ham/Spam classification using nlp

import pandas as pd
# Download stopwords
nltk.download('stopwords')

# Load dataset
# Make sure spam.csv is in the same folder

df = pd.read_csv('spam.csv', encoding='latin-1')

# Select useful columns

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Initialize stemmer

ps = PorterStemmer()

# Preprocessing function

def preprocess_text(text):
    text = text.lower()
    words = text.split()

    cleaned_words = []

    for word in words:
        word = word.translate(str.maketrans('', '', string.punctuation))

        if word not in stopwords.words('english'):
            cleaned_words.append(ps.stem(word))

    return " ".join(cleaned_words)

# Apply preprocessing

df['processed_message'] = df['message'].apply(preprocess_text)

# Feature extraction

vectorizer = TfidfVectorizer(max_features=3000)

X = vectorizer.fit_transform(df['processed_message']).toarray()
y = df['label']

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model

model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Accuracy

accuracy = accuracy_