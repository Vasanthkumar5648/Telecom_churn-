{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "bf5de19a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn import metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8d606b66",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(r\"C:\\Users\\vasanth\\Downloads\\Telco_Customer_Churn.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "485d2216",
   "metadata": {},
   "outputs": [],
   "source": [
    "y = df['Churn'].values\n",
    "X = df.drop(columns = ['Churn'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "43967526",
   "metadata": {},
   "outputs": [],
   "source": [
    "non_numeric_columns = X.select_dtypes(include=['object']).columns\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "for col in non_numeric_columns:\n",
    "    encoder = LabelEncoder()\n",
    "    X[col] = encoder.fit_transform(X[col])\n",
    "features = X.columns.values\n",
    "scaler = MinMaxScaler(feature_range= (0,1))\n",
    "scaler.fit(X)\n",
    "X = pd.DataFrame(scaler.transform(X))\n",
    "X.columns = features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "68ff4cb0",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "3f2bd1e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from joblib import dump"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "7f6fc450",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "3f8ac9ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVC\n",
    "model.svm = SVC(kernel='linear')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "0793747d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7913413768630234"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.svm.fit(X_train,y_train)\n",
    "preds = model.svm.predict(X_test)\n",
    "metrics.accuracy_score(y_test, preds)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "6981a53c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[914, 109],\n",
       "       [185, 201]], dtype=int64)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "confusion_matrix(y_test, preds)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "249b7ea0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['SVM.joblib']"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dump(model,'SVM.joblib')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c7c29cee",
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
