import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('./dataset/Coffe_sales.csv')

features = ['hour_of_day', 'cash_type', 'Time_of_Day', 'Weekday']
target = 'coffee_name'

X = df[features]
y = df[target]


X_encoded = pd.get_dummies(X, columns=['cash_type', 'Time_of_Day', 'Weekday'])

print("--- Amostra dos Dados Pré-Processados (Features) ---")
print(X_encoded.head())
print("\n")


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
print(f"Tamanho do conjunto de teste:  {X_test.shape[0]} amostras\n")


decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(X_train, y_train)

print("--- Modelo de Árvore de Decisão treinado com sucesso! ---\n")


y_pred = decision_tree.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"--- Avaliação do Modelo ---")
print(f"Acurácia no conjunto de teste: {accuracy:.2f}")


print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))


print("--- Gerando a visualização da Árvore de Decisão... ---")
plt.figure(figsize=(20, 10))
plot_tree(decision_tree,
          feature_names=X_encoded.columns,
          class_names=decision_tree.classes_,
          filled=True,
          rounded=True,
          max_depth=3, 
          fontsize=10)

plt.title("Visualização da Árvore de Decisão (Profundidade = 3)")
plt.savefig("arvore_de_decisao.png")
print("Visualização salva no arquivo 'arvore_de_decisao.png'")