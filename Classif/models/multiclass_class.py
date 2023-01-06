


def lmodel_cv(liste, X, Y, vectorizer=tvec,average_method='binary'):
  for model in liste:
    pipe = Pipeline([
                  ('tfidf', vectorizer),
                  ('classify', model)
                ])
    cv_results = cross_validate(pipe, X, Y,cv=5)
    res = pd.DataFrame(cv_results)
    print("Accuracy du modèle "f"{model} :"
        f"{res['test_score'].mean():.3f} +/- {res['test_score'].std():.3f}\n"
        f"{res['fit_time'].mean()}"
      )