from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def compute_tfidf(documents):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(documents)
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
