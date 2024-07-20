import joblib

# Save the model and vectorizer
joblib.dump(clf, 'text_classifier_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
