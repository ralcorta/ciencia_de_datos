import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix, mean_absolute_error

df = pd.read_csv('kaggle_diabetes.csv')
df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})
df_copy = df.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[[
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# -------------------- GRÁFICOS --------------------
# Distribución de Outcome (0: No diabetes, 1: Diabetes)
sns.countplot(x='Outcome', data=df_copy)
plt.title("Distribución de pacientes con y sin diabetes")
plt.xlabel("Outcome (0: No diabetes, 1: Diabetes)")
plt.ylabel("Cantidad")
plt.savefig("static/distribucion_outcome.png")
# plt.show()

# Matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(df_copy.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación entre variables")
plt.savefig("static/matriz_correlacion.png")
# plt.show()

#  Distribución de glucosa según clase
sns.histplot(data=df_copy, x='Glucose', hue='Outcome', kde=True, element='step')
plt.title("Distribución de Glucosa por Outcome")
plt.xlabel("Glucose")
plt.ylabel("Frecuencia")
plt.savefig("static/histograma_glucosa.png")
# plt.show()

# Boxplots para comparar variables por clase
features_to_plot = ['Glucose', 'BloodPressure', 'BMI', 'Age']
for feature in features_to_plot:
    sns.boxplot(x='Outcome', y=feature, data=df_copy)
    plt.title(f"Distribución de {feature} según diagnóstico")
    plt.xlabel("Outcome")
    plt.ylabel(feature)
    plt.savefig(f"static/boxplot_{feature.lower()}.png")
    # plt.show()

# -------------------- ENTRENAMIENTO --------------------
df = df.drop(columns='Pregnancies')

X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# -------------------- MÉTRICAS --------------------
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(cr)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.savefig("static/matriz_confusion.png")
# plt.show()

filename = 'model/diabetes-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))