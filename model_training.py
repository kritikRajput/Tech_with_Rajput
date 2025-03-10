import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import joblib
data = pd.read_csv('dataset/creditcard.csv')
print ('###############')
X = data[['Time', 'Amount']]  
y = data['Class']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
joblib.dump(pca, 'models/pca.pkl')
model = LogisticRegression()
model.fit(X_train_pca, y_train)
joblib.dump(model, 'models/fraud_model.pkl')