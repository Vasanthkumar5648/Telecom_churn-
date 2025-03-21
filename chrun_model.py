{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a4f86f2b",
   "metadata": {},
   "source": [
    "# libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c93a33bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from joblib import dump"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "18d0b869",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "46789639",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(r\"C:\\Users\\vasanth\\Downloads\\Telco_Customer_Churn.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2b6b3605",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>customerID</th>\n",
       "      <th>gender</th>\n",
       "      <th>SeniorCitizen</th>\n",
       "      <th>Partner</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>tenure</th>\n",
       "      <th>PhoneService</th>\n",
       "      <th>MultipleLines</th>\n",
       "      <th>InternetService</th>\n",
       "      <th>OnlineSecurity</th>\n",
       "      <th>...</th>\n",
       "      <th>DeviceProtection</th>\n",
       "      <th>TechSupport</th>\n",
       "      <th>StreamingTV</th>\n",
       "      <th>StreamingMovies</th>\n",
       "      <th>Contract</th>\n",
       "      <th>PaperlessBilling</th>\n",
       "      <th>PaymentMethod</th>\n",
       "      <th>MonthlyCharges</th>\n",
       "      <th>TotalCharges</th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7590-VHVEG</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>1</td>\n",
       "      <td>No</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>No</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>29.85</td>\n",
       "      <td>29.85</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5575-GNVDE</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>34</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>One year</td>\n",
       "      <td>No</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>56.95</td>\n",
       "      <td>1889.5</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3668-QPYBK</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>2</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>53.85</td>\n",
       "      <td>108.15</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7795-CFOCW</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>45</td>\n",
       "      <td>No</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>One year</td>\n",
       "      <td>No</td>\n",
       "      <td>Bank transfer (automatic)</td>\n",
       "      <td>42.30</td>\n",
       "      <td>1840.75</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9237-HQITU</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>2</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Fiber optic</td>\n",
       "      <td>No</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>70.70</td>\n",
       "      <td>151.65</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService  \\\n",
       "0  7590-VHVEG  Female              0     Yes         No       1           No   \n",
       "1  5575-GNVDE    Male              0      No         No      34          Yes   \n",
       "2  3668-QPYBK    Male              0      No         No       2          Yes   \n",
       "3  7795-CFOCW    Male              0      No         No      45           No   \n",
       "4  9237-HQITU  Female              0      No         No       2          Yes   \n",
       "\n",
       "      MultipleLines InternetService OnlineSecurity  ... DeviceProtection  \\\n",
       "0  No phone service             DSL             No  ...               No   \n",
       "1                No             DSL            Yes  ...              Yes   \n",
       "2                No             DSL            Yes  ...               No   \n",
       "3  No phone service             DSL            Yes  ...              Yes   \n",
       "4                No     Fiber optic             No  ...               No   \n",
       "\n",
       "  TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling  \\\n",
       "0          No          No              No  Month-to-month              Yes   \n",
       "1          No          No              No        One year               No   \n",
       "2          No          No              No  Month-to-month              Yes   \n",
       "3         Yes          No              No        One year               No   \n",
       "4          No          No              No  Month-to-month              Yes   \n",
       "\n",
       "               PaymentMethod MonthlyCharges  TotalCharges Churn  \n",
       "0           Electronic check          29.85         29.85    No  \n",
       "1               Mailed check          56.95        1889.5    No  \n",
       "2               Mailed check          53.85        108.15   Yes  \n",
       "3  Bank transfer (automatic)          42.30       1840.75    No  \n",
       "4           Electronic check          70.70        151.65   Yes  \n",
       "\n",
       "[5 rows x 21 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "7d030d8d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "dfcc96ec",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "3598ef8a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n",
    "df['TotalCharges'].fillna(0, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "741bf088",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "8b60c174",
   "metadata": {},
   "outputs": [],
   "source": [
    "label_encoder = LabelEncoder()\n",
    "df['Churn'] = label_encoder.fit_transform(df['Churn'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "d94a1621",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "241f7f50",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['InternetService'] = label_encoder.fit_transform(df['InternetService'])\n",
    "df['Contract'] = label_encoder.fit_transform(df['Contract'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "3d2d5426",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "009e8e56",
   "metadata": {},
   "outputs": [],
   "source": [
    "selected_features = ['tenure','InternetService','Contract','MonthlyCharges','TotalCharges']\n",
    "X= df[selected_features]\n",
    "y=df['Churn']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b8aba2bb",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d7d918c5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier(random_state=101)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(random_state=101)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier(random_state=101)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = RandomForestClassifier(n_estimators=100, random_state=101)\n",
    "model.fit(X,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "72de8417",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8e3cf209",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['random_forest_model.joblib']"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dump(model,'random_forest_model.joblib')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "9c6714f9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
