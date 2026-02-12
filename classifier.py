import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# 1. READ CSV
# =====================================================
csv_path = "music_genre_data.csv"   # adjust if needed
df = pd.read_csv(csv_path, na_values=['?']) # Treat '?' as a missing value

print("Columns:", df.columns.tolist())
print("Total rows:", len(df))

# =====================================================
# 2. SELECT FEATURES AND TARGET
# =====================================================
# From the header snippet, useful numeric features:
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

# Keep only rows with no missing values in these columns
df_model = df[feature_cols + [target_col]].dropna()

X = df_model[feature_cols]
y = df_model[target_col]

print("Usable rows after dropping NAs:", len(df_model))
print("Example class counts:")
print(y.value_counts().head(20))
print()

# =====================================================
# 3. TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y  # keep class balance
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])
print()

# =====================================================
# 4. TRAIN MODEL
# =====================================================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================================================
# 5. EVALUATE
# =====================================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy on test set: {:.2f}%".format(acc))
print()

print("Classification Report (first few classes):")
print(classification_report(y_test, y_pred, zero_division=0))
print()

print("=== SAMPLE TEST ROWS WITH PREDICTIONS ===")
sample_n = 5
for i in range(sample_n):
    row_features = X_test.iloc[i]
    true_label = y_test.iloc[i]
    pred_label = y_pred[i]
    print("Features:", row_features.to_dict())
    print("TRUE GENRE:", true_label)
    print("PREDICTED GENRE:", pred_label)
    print("-" * 60)

# =====================================================
# 6. INTERACTIVE INPUT LOOP
# =====================================================
print("\nNow you can enter feature values to predict genre.")
print("Press Enter with no value to use the column mean for that feature.\n")

# Precompute means to use as defaults
col_means = X.mean()

def ask_float(prompt, default):
    s = input(f"{prompt} [{default:.4f}]: ").strip()
    if s == "":
        return float(default)
    try:
        return float(s)
    except ValueError:
        print("Invalid number, using default.")
        return float(default)

while True:
    cont = input("\nType 'exit' to stop, or press Enter to predict: ").strip().lower()
    if cont == "exit":
        print("Exiting. Thank you!")
        break

    user_values = {}
    for col in feature_cols:
        val = ask_float(f"Enter {col}", col_means[col])
        user_values[col] = val

    user_df = pd.DataFrame([user_values])  # 1-row DataFrame
    pred_genre = model.predict(user_df)[0]
    print("Predicted Genre:", pred_genre)
0.40
