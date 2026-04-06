import joblib
model = joblib.load("models/best_model.joblib")
print(type(model))

X = [[
    2,
    2000,
    8000,
    10000,
    2,
    572
]]
prediction = model.predict(X)[0]
prediction_proba = model.predict_proba(X)[0, 1]

print("prediction :", prediction)
print("prediction proba : ", prediction_proba)