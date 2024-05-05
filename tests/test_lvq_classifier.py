from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from LVQClassifier import LVQClassifier  # Importe sua classe LVQClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# Carregar o conjunto de dados do Hugging Face
dataset = load_dataset("maharshipandya/spotify-tracks-dataset")
df = dataset['train'].to_pandas()

# Remover colunas irrelevantes
data = df.drop(columns=['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name'])

# Separe os atributos (X) e os rótulos (y)
X = data.drop('track_genre', axis=1)
y = data['track_genre']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciar o classificador
lvq_classifier = LVQClassifier(n_codebooks=15, learn_rate=0.001, epochs=50)

# Convertendo os rótulos de saída em valores discretos usando LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Treinar o modelo usando o método fit com os dados de treinamento
lvq_classifier.fit(X_train.values, y_train_encoded)

# Faça previsões nos dados de teste usando o método predict
predictions = lvq_classifier.predict(X_test.values)

# Calcular a precisão do modelo nos dados de teste usando o método score
accuracy = accuracy_score(y_test_encoded, predictions)
print("Accuracy:", accuracy)

precision = precision_score(y_test_encoded, predictions, average='weighted', zero_division=1)
print("Precision:", precision)

# Calcular a métrica de recall
recall = recall_score(y_test_encoded, predictions, average='weighted', zero_division=1)
print("Recall:", recall)

# Calcular a métrica F1 Score
f1 = f1_score(y_test_encoded, predictions, average='weighted')
print("F1 Score:", f1)

# Calcular a métrica AUC
auc = lvq_classifier.auc(X_test.values, y_test_encoded)
print("AUC:", auc)
