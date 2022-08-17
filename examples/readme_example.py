from pyhacrf import StringPairFeatureExtractor, Hacrf

training_X = [('helloooo', 'hello'), # Matching examples
              ('h0me', 'home'),
              ('krazii', 'crazy'),
              ('non matching string example', 'no really'), # Non-matching examples
              ('and another one', 'yep')]
training_y = ['match',
              'match',
              'match',
              'non-match',
              'non-match']

# Extract features
feature_extractor = StringPairFeatureExtractor(match=True, numeric=True)
training_X_extracted = feature_extractor.fit_transform(training_X)

# Train model
model = Hacrf(l2_regularization=1.0)
model.fit(training_X_extracted, training_y)

# Evaluate
from sklearn.metrics import confusion_matrix
predictions = model.predict(training_X_extracted)

print("Confusion matrix")
print(confusion_matrix(training_y, predictions))
print()
print("Predicted probabilities")
print(model.predict_proba(training_X_extracted))
