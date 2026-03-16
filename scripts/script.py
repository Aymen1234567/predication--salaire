import os
import joblib
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42


df = pd.read_csv("../data/job_salary_prediction_dataset.csv")

df_fe = df.copy()

# 1. Ordinal encoding
edu_map = {'High School':1,'Diploma':2,'Bachelor':3,'Master':4,'PhD':5}
size_map = {'Startup':1,'Small':2,'Medium':3,'Large':4,'Enterprise':5}
remote_map = {'No':0,'Hybrid':1,'Yes':2}
df_fe['edu_ord']    = df_fe['education_level'].map(edu_map)
df_fe['size_ord']   = df_fe['company_size'].map(size_map)
df_fe['remote_ord'] = df_fe['remote_work'].map(remote_map)

# 2. Label encode nominals
le = LabelEncoder()
for col in ['job_title','industry','location']:
    df_fe[col+'_enc'] = le.fit_transform(df_fe[col])

# 3. Interaction features
df_fe['exp_x_edu']      = df_fe['experience_years'] * df_fe['edu_ord']
df_fe['exp_x_skills']   = df_fe['experience_years'] * df_fe['skills_count']
df_fe['edu_x_certs']    = df_fe['edu_ord'] * df_fe['certifications']
df_fe['size_x_remote']  = df_fe['size_ord'] * df_fe['remote_ord']
df_fe['total_score']    = (df_fe['experience_years'] + df_fe['edu_ord']*2 +
                           df_fe['skills_count']*0.5 + df_fe['certifications']*2)

# 4. Squared & log features
df_fe['exp_sq']         = df_fe['experience_years'] ** 2
df_fe['log_exp_p1']     = np.log1p(df_fe['experience_years'])
df_fe['skills_sq']      = df_fe['skills_count'] ** 2

# 5. Target: log_salary for regression
df_fe['log_salary'] = np.log1p(df_fe['salary'])

# 6. Salary band for classification (Low / Mid / High)
q33 = df_fe['salary'].quantile(0.33)
q66 = df_fe['salary'].quantile(0.66)
df_fe['salary_band'] = pd.cut(df_fe['salary'], bins=[0, q33, q66, np.inf],
                               labels=['Low','Mid','High'])
df_fe['salary_band_enc'] = LabelEncoder().fit_transform(df_fe['salary_band'])

print(f"Features created. Total columns: {df_fe.shape[1]}")
print(f"Salary bands:\n{df_fe['salary_band'].value_counts()}")
df_fe.head(3)


FEATURE_COLS = [
    'experience_years','skills_count','certifications',
    'edu_ord','size_ord','remote_ord',
    'job_title_enc','industry_enc','location_enc',
    'exp_x_edu','exp_x_skills','edu_x_certs','size_x_remote',
    'total_score','exp_sq','log_exp_p1','skills_sq'
]



X = df_fe[FEATURE_COLS]
y_reg = df_fe['log_salary']       # regression target
y_cls = df_fe['salary_band_enc']  # classification target

X_train, X_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.2, random_state=RANDOM_STATE)
_, _, yc_train, yc_test           = train_test_split(X, y_cls, test_size=0.2, random_state=RANDOM_STATE,
                                                     stratify=y_cls)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Salary band split — Train: {pd.Series(yc_train).value_counts().to_dict()}")



reg_models = {
    'Ridge':              Ridge(alpha=1.0),
    'Decision Tree':      DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
    'Random Forest':      RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting':  GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                    max_depth=5, random_state=RANDOM_STATE),
}

reg_results = []
for name, model in reg_models.items():
    use_sc = name in ('Ridge','Lasso')
    Xtr = X_train_sc if use_sc else X_train
    Xte = X_test_sc  if use_sc else X_test
    model.fit(Xtr, yr_train)
    preds_log = model.predict(Xte)
    preds = np.expm1(preds_log)
    actual = np.expm1(yr_test)
    mae = mean_absolute_error(actual, preds)
    rmse= np.sqrt(mean_squared_error(actual, preds))
    reg_results.append({'Model':name,'MAE':mae,'RMSE':rmse})
    print(f"{name:22s}    MAE={mae:,.0f}  RMSE={rmse:,.0f}")

df_results = pd.DataFrame(reg_results)
print(df_results)

# Normaliser MAE et RMSE entre 0 et 1
df_results['MAE_norm'] = df_results['MAE'] / df_results['MAE'].max()
df_results['RMSE_norm'] = df_results['RMSE'] / df_results['RMSE'].max()


df_results['Score'] = (df_results['MAE_norm'] + df_results['RMSE_norm']) / 2


meilleur_model_info = df_results.loc[df_results['Score'].idxmin()]
meilleur_model_name =meilleur_model_info['Model']
print("Meilleur modèle selon score combiné (MAE + RMSE):")
print(meilleur_model_info)


meilleur_model = reg_models[meilleur_model_name]

# Pour Ridge, inclure le scaler dans un pipeline
if meilleur_model_name == 'Ridge':
   meilleur_model = Pipeline([('scaler', scaler), ('model', meilleur_model)])


os.makedirs('../model', exist_ok=True)


joblib.dump({
    'model': meilleur_model,
    'edu_map': edu_map,
    'size_map': size_map,
    'remote_map': remote_map,
    'feature_cols': FEATURE_COLS
}, '../model/meilleur_model.pkl')

print("✅ Meilleur modèle sauvegardé dans ../model/meilleur_model.pkl")