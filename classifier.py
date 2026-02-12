import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# read csv file
csv_path = "music_genre_data.csv"
df = pd.read_csv(csv_path, na_values=["?"])

print("Columns:", df.columns.tolist())
print("Total rows:", len(df))

# select features and target
feature_cols = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "popularity"
]

target_col = "music_genre"

# remove rows with missing values
df_model = df[feature_cols + [target_col]].dropna()

X = df_model[feature_cols]
y = df_model[target_col]

print("Usable rows after dropping NAs:", len(df_model))
print("Example class counts:")
print(y.value_counts().head(20))

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# train random forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100

print("Accuracy on test set: {:.2f}%".format(acc))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# show sample predictions
print("Sample test rows with predictions:")
sample_n = 5

for i in range(sample_n):
    row_features = X_test.iloc[i]
    true_label = y_test.iloc[i]
    pred_label = y_pred[i]

    print("Features:", row_features.to_dict())
    print("True Genre:", true_label)
    print("Predicted Genre:", pred_label)
    print("-" * 50)

# interactive prediction
print("\nEnter feature values to predict genre")
print("Press Enter to use default mean value")

col_means = X.mean()

def ask_float(prompt, default):
    value = input(f"{prompt} [{default:.4f}]: ").strip()

    if value == "":
        return float(default)

    try:
        return float(value)
    except ValueError:
        print("Invalid number, using default")
        return float(default)

while True:
    user_input = input("\nPress Enter to predict or type 'exit' to stop: ").strip().lower()

    if user_input == "exit":
        print("Exiting")
        break

    user_values = {}

    for col in feature_cols:
        user_values[col] = ask_float(f"Enter {col}", col_means[col])

    user_df = pd.DataFrame([user_values])
    prediction = model.predict(user_df)[0]

    print("Predicted Genre:", prediction)
