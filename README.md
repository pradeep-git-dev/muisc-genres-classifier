**README — Music Genre Classification using Machine Learning**

**1. Problem**
• Manual classification of music genres is slow and inconsistent.
• Large music datasets require automated categorization.
• Audio features contain patterns that humans cannot easily identify manually.
• Need an accurate system to classify music automatically based on song characteristics.

**2. Goal**
• Build a machine learning model to predict music genre from audio features.
• Train the model using labeled music data.
• Evaluate model performance using accuracy and classification metrics.
• Allow users to input song features and get predicted genres.

**3. Dataset**
• Input file: music_genre_data.csv
• Contains numerical audio features and music genre labels.
• Missing values represented by "?" are removed during preprocessing.

**4. Features Used (Input Variables)**
• acousticness
• danceability
• duration_ms
• energy
• instrumentalness
• liveness
• loudness
• speechiness
• tempo
• valence
• popularity

• Target variable:

* music_genre (genre label predicted by the model)

**5. How the Problem is Solved using Machine Learning**
• Load dataset and handle missing values.
• Select relevant audio features and target label.
• Split data into training set (75%) and testing set (25%).
• Train a Random Forest Classifier to learn patterns between features and genres.
• Predict genres for unseen test data.
• Evaluate performance using accuracy score and classification report.
• Provide interactive input for real-time genre prediction.

**6. Machine Learning Model**
• Algorithm: Random Forest Classifier
• Ensemble learning method using multiple decision trees.
• Reduces overfitting and improves prediction accuracy.
• Handles complex relationships between audio features.

**7. Model Evaluation**
• Accuracy score measures overall prediction performance.
• Classification report shows precision, recall, and F1-score.
• Sample predictions compare actual vs predicted genres.

**8. Output**
• Predicts music genre based on audio feature values.
• Allows interactive user input for custom predictions.
