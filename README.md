# Job Salary Prediction

Prédiction du salaire à partir des caractéristiques professionnelles  
(Expérience, formation, compétences, secteur, taille d'entreprise, remote, etc.)

![Salar

## 🎯 Objectif du projet

Construire un modèle de machine learning capable de **prédire le salaire annuel** d'une personne à partir de ses informations professionnelles.

- **Type de problème** : Régression (prédiction de log(salaire))  
- **Dataset** : ~250 000 lignes synthétiques  
- **Métriques principales** : MAE & RMSE (en $)  
- **Meilleur modèle** : Gradient Boosting / Random Forest (selon runs)

## Données

| Colonne             | Type          | Description                                      |
|---------------------|---------------|--------------------------------------------------|
| `job_title`         | catégorielle  | Intitulé du poste                                |
| `experience_years`  | numérique     | Années d’expérience                              |
| `education_level`   | ordinale      | Niveau d’études                                  |
| `skills_count`      | numérique     | Nombre de compétences                            |
| `industry`          | catégorielle  | Secteur d’activité                               |
| `company_size`      | ordinale      | Taille de l’entreprise                           |
| `location`          | catégorielle  | Pays / région                                    |
| `remote_work`       | ordinale      | Remote / Hybrid / Présentiel                    |
| `certifications`    | numérique     | Nombre de certifications                         |
| `salary`            | cible         | Salaire annuel (USD) – variable à prédire       |

## 🚀 Fonctionnalités principales

- Encodage ordinal intelligent (niveau d’études, taille entreprise, remote)
- Features d’interaction métier (exp × éducation, exp × skills, etc.)
- Score composite pondéré (`total_score`)
- Transformation logarithmique de la cible → `log_salary`
- Comparaison de plusieurs modèles : Ridge, DecisionTree, RandomForest, GradientBoosting
- Sauvegarde du meilleur modèle + mappings dans un fichier `.pkl` prêt pour production

## Résultats (exemple – à mettre à jour)

| Modèle              | MAE (USD)    | RMSE (USD)   | Score combiné |
|---------------------|--------------|--------------|---------------|
| Ridge               | 18 420       | 24 150       | 0.42          |
| Decision Tree       | 16 780       | 23 890       | 0.38          |
| Random Forest       | 12 310       | 17 640       | 0.21          |
| **Gradient Boosting** | **11 890** | **16 980** | **0.18**      |

→ **Meilleur modèle** : Gradient Boosting (souvent)


## Comment utiliser le modèle (inférence)

```python
import joblib
import pandas as pd

# Charger le modèle
data = joblib.load("model/best_model.pkl")
model = data["model"]
feature_cols = data["feature_cols"]
mappings = {
    "edu": data["edu_map"],
    "size": data["size_map"],
    "remote": data["remote_map"]
}

# Exemple de données d’entrée
new_data = pd.DataFrame([{
    "job_title": "Data Scientist",
    "experience_years": 7,
    "education_level": "Master",
    "skills_count": 12,
    "industry": "Tech",
    "company_size": "Large",
    "location": "United States",
    "remote_work": "Hybrid",
    "certifications": 3
}])

# Préparer les features (même logique que dans script.py)
# ... (encoder, créer interactions, etc.)

# Prédiction
log_pred = model.predict(X_new)
salary_pred = np.expm1(log_pred)[0]

print(f"Salaire estimé : {salary_pred:,.0f} $")
