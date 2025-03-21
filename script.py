import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump

df = pd.read_csv(r"C:\Users\vasanth\Downloads\Telco_Customer_Churn.csv")


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

label_encoder = LabelEncoder()
df['Churn']= label_encoder.fit_transform(df['Churn'])

df['InternetService'] = label_encoder.fit_transform(df['InternetService'])
df['Contract'] = label_encoder.fit_transform(df['Contract'])

selected_features = ['tenure','InternetService','Contract','MonthlyCharges','TotalCharges']
X = df[selected_features]
y = df['Churn']

model = RandomForestClassifier(n_estimators=100, random_state=101)
model.fit(X, y)

dump(model, 'random_forest_model.joblib')