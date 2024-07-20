# Load the model and vectorizer
clf = joblib.load('text_classifier_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example prediction
def predict_category(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = clf.predict(text_tfidf)
    return label_encoder.inverse_transform(prediction)[0]

# Test prediction
sample_text = "The stock market surged today due to positive economic reports."
print("Predicted Category:", predict_category(sample_text))
