import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Carregamento da Base de Dados
# Carrega o arquivo CSV para um DataFrame do pandas.
df = pd.read_csv('./dataset/Coffe_sales.csv')

# 2. Definição do Problema e Seleção de Features
# O objetivo é prever a coluna 'coffee_name'.
# Para isso, selecionamos as colunas que serão usadas como variáveis preditoras (features).
features = ['hour_of_day', 'cash_type', 'Time_of_Day', 'Weekday']
target = 'coffee_name'

X = df[features]
y = df[target]

# 3. Pré-processamento dos Dados
# O algoritmo de Árvore de Decisão do scikit-learn não aceita texto como entrada.
# Precisamos converter as colunas categóricas (como 'cash_type', 'Time_of_Day') em números.
# A técnica "One-Hot Encoding" (usando pd.get_dummies) é ideal para isso, pois cria
# novas colunas binárias (0 ou 1) para cada categoria, sem criar uma ordem falsa entre elas.
X_encoded = pd.get_dummies(X, columns=['cash_type', 'Time_of_Day', 'Weekday'])

# Exibe as primeiras linhas dos dados pré-processados para verificação
print("--- Amostra dos Dados Pré-Processados (Features) ---")
print(X_encoded.head())
print("\n")


# 4. Divisão da Base em Treino e Teste
# Conforme solicitado no PDF do trabalho, dividimos os dados para treinar e depois
# avaliar o modelo de forma imparcial. Usaremos 80% para treino e 20% para teste.
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
print(f"Tamanho do conjunto de teste:  {X_test.shape[0]} amostras\n")


# 5. Criação e Treinamento do Modelo de Árvore de Decisão
# Instanciamos o classificador. O parâmetro `random_state` garante que o resultado
# seja o mesmo sempre que o código for executado.
# O PDF sugere testar diferentes parâmetros. Você pode, por exemplo, limitar a profundidade
# da árvore para evitar overfitting, adicionando `max_depth=5`.
decision_tree = DecisionTreeClassifier(random_state=42)

# Treinamos o modelo usando os dados de treino.
decision_tree.fit(X_train, y_train)

print("--- Modelo de Árvore de Decisão treinado com sucesso! ---\n")


# 6. Avaliação do Desempenho do Modelo
# O PDF exige pelo menos duas métricas de avaliação.
# Usaremos Acurácia e o Relatório de Classificação (com Precisão, Recall e F1-Score).
y_pred = decision_tree.predict(X_test)

# Métrica 1: Acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"--- Avaliação do Modelo ---")
print(f"Acurácia no conjunto de teste: {accuracy:.2f}")

# Métrica 2: Relatório de Classificação
# Este relatório mostra as métricas para cada tipo de café.
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))


# 7. Visualização da Árvore de Decisão (Opcional, mas recomendado para o relatório)
# Esta visualização ajuda a interpretar como o modelo toma suas decisões.
# A árvore pode ser grande, então vamos limitar a profundidade da visualização para melhor leitura.
print("--- Gerando a visualização da Árvore de Decisão... ---")
plt.figure(figsize=(20, 10))
plot_tree(decision_tree,
          feature_names=X_encoded.columns,
          class_names=decision_tree.classes_,
          filled=True,
          rounded=True,
          max_depth=3, # Limita a profundidade da árvore na imagem
          fontsize=10)

# Salva a imagem em um arquivo para usar no seu relatório
plt.title("Visualização da Árvore de Decisão (Profundidade = 3)")
plt.savefig("arvore_de_decisao.png")
print("Visualização salva no arquivo 'arvore_de_decisao.png'")