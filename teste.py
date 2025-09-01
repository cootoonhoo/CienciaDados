# blood_donors_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# === 1. Carregar dataset ===
df = pd.read_csv("./dataset/blood_donor_dataset.csv")  # Ajuste para o nome do seu arquivo

# === 2. Selecionar features e target ===
features = ["months_since_first_donation", "number_of_donation", "pints_donated", "blood_group"]
target = "availability"

X = df[features]
y = df[target]

# === 3. Codificação da variável categórica ===
encoder = OneHotEncoder(drop='first', sparse_output=False)
blood_group_encoded = encoder.fit_transform(X[['blood_group']])
blood_group_df = pd.DataFrame(blood_group_encoded, columns=encoder.get_feature_names_out(['blood_group']))
X = X.drop(columns=['blood_group']).reset_index(drop=True)
X = pd.concat([X, blood_group_df], axis=1)

# === 4. Codificação do alvo (Yes=1, No=0) ===
label_enc = LabelEncoder()
y = label_enc.fit_transform(y)

# === 5. Dividir dados ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === 6. Definir modelos ===
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# === 7. Treinar e avaliar ===
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    cv = cross_val_score(model, X, y, cv=5).mean()

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print(f"Cross-Validation (5-fold): {cv:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_enc.classes_))
