import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label_binarize
import skimpy as sk

# Charger les données avec st.cache_data
@st.cache_data
def load_data():
    df = pd.read_csv("maternal+health+risk/Maternal Health Risk Data Set.csv")
    return df

df = load_data()

# Affichage des premières lignes des données
st.title("Prédiction du Niveau de Risque de Santé Maternelle")
st.write("Voici un aperçu des premières lignes des données :")
st.write(df.head())

# Distribution des variables : Boxplots
st.subheader("Distribution des Variables Indépendantes")
df_numeric = df.select_dtypes(include=['number'])

# Utiliser des colonnes pour la mise en page
cols = st.columns(len(df_numeric.columns))
# Affichage des boxplots avec Matplotlib
for column in df_numeric.columns:
    st.write(f"**Boxplot de {column}**")
    fig, ax = plt.subplots(figsize=(8, 6))  # Créez une figure et un axe
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f'Boxplot de {column}')
    st.pyplot(fig)  # Passez la figure explicite à st.pyplot()


# Exclure les colonnes non numériques pour la corrélation
df_numeric = df.select_dtypes(include=[np.number])

# Calculer la matrice de corrélation sur les données numériques uniquement
st.subheader("Matrice de Corrélation")
# Créer une figure et un axe
fig, ax = plt.subplots(figsize=(10, 6))  # Taille de la figure ajustable
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', ax=ax)  # Tracer sur l'axe spécifié
st.pyplot(fig)  # Passer la figure explicite à st.pyplot()


# Heatmap de corrélation avec mise en forme
# Matrice de Corrélation avec Mise en Forme
st.subheader("Matrice de Corrélation avec Mise en Forme")
fig, ax = plt.subplots(figsize=[20, 15])  # Créer une figure et un axe
sns.heatmap(df_numeric.corr(), annot=True, fmt='.2f', ax=ax, cmap='magma')
ax.set_title("Correlation Matrix", fontsize=20)
st.pyplot(fig)  # Passer la figure à st.pyplot()


# Encoder la variable cible
le = LabelEncoder()
df['RiskLevel_encoded'] = le.fit_transform(df['RiskLevel'])

# Sélectionner les variables indépendantes et la cible
df_numeric = df.select_dtypes(include=['number'])
X = df_numeric.drop(columns=['RiskLevel_encoded'])  # Retirer 'RiskLevel_encoded' de X
y = df['RiskLevel_encoded']  # 'RiskLevel_encoded' est la variable cible

# Imputation des valeurs manquantes
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Diviser le jeu de données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Titre de l'application
st.sidebar.title("Modèles et Prédictions")
st.sidebar.write("Choisissez un ou plusieurs modèles pour entraîner et évaluer.")

# Liste des modèles à utiliser
models = [
    {'label': 'Régression Logistique', 'model': LogisticRegression(max_iter=10000, multi_class='ovr', solver='lbfgs')},
    {'label': 'Arbre de Décision', 'model': DecisionTreeClassifier()},
    {'label': 'SVM', 'model': SVC(probability=True)},
    {'label': 'KNN', 'model': KNeighborsClassifier()},
    {'label': 'XGBoost', 'model': XGBClassifier()},
    {'label': 'Forêt Aléatoire', 'model': RandomForestClassifier()},
    {'label': 'Gradient Boosting', 'model': GradientBoostingClassifier()}
]

# Sélectionner les modèles via des cases à cocher
selected_models = []
for model in models:
    if st.sidebar.checkbox(model['label']):
        selected_models.append(model['model'])
        
# Hyperparamètres pour GridSearchCV
param_grid = {}

for selected_model in selected_models:
    if isinstance(selected_model, LogisticRegression):
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga'],
            'max_iter': [500, 1000, 1500]
        }
    elif isinstance(selected_model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

    # GridSearchCV pour optimiser le modèle sélectionné
    if param_grid:
        grid_search = GridSearchCV(selected_model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        st.write(f"Meilleurs paramètres pour {selected_model}: {grid_search.best_params_}")
    else:
        best_model = selected_model.fit(X_train, y_train)

    # Prédiction sur les données de test
    y_pred = best_model.predict(X_test)

    # Évaluation du modèle
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Convertir la matrice de confusion en DataFrame pour un affichage correct
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=[f'Classe {i}' for i in range(conf_matrix.shape[1])],
                                  index=[f'Classe {i}' for i in range(conf_matrix.shape[0])])

    # Affichage des résultats dans un tableau
    st.subheader(f"Performance du modèle {selected_model}")
    metrics = pd.DataFrame({
        "Accuracy": [accuracy],
        "Precision (Classe 0)": [classification_rep.split()[5]],
        "Recall (Classe 0)": [classification_rep.split()[6]],
        "F1-score (Classe 0)": [classification_rep.split()[7]],
        "Matrice de Confusion": [conf_matrix_df]
    })

    st.write(metrics)

    # Calcul des courbes ROC pour chaque classe (multiclasse)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Ajoutez toutes les classes possibles
    y_pred_proba = best_model.predict_proba(X_test)

    # Tracer les courbes ROC pour chaque classe
    fig, ax = plt.subplots(figsize=(10, 6))  # Créer un objet figure et axes
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
        ax.plot(fpr, tpr, label=f'Classe {i} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel('1 - Specificité (Taux de Faux Positifs)', fontsize=12)
    ax.set_ylabel('Sensibilité (Taux de Vrais Positifs)', fontsize=12)
    ax.set_title(f'Courbes ROC pour {selected_model}', fontsize=14)
    ax.legend(loc="lower right")
    st.pyplot(fig)  # Passez explicitement la figure à st.pyplot()
