import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
diabetes_dataset = pd.read_csv('diabetess.csv')

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
y = diabetes_dataset['Outcome']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify=y, random_state=2)
X.shape, X_train.shape, X_test.shape

# Mô hình : SVM
assert len(X) == len(y), "Số lượng mẫu không khớp giữa X và y"
# Phân chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Khởi tạo mô hình RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Huấn luyện mô hình trên tập huấn luyện
rf_classifier.fit(X_train, y_train)

# Dự đoán trên tập kiểm thử
y_pred = rf_classifier.predict(X_test)

# Đánh giá hiệu suất của mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# In báo cáo chi tiết về hiệu suất của mô hình
print('Classification Report:\n', classification_report(y_test, y_pred))
import pickle
pickle.dump(rf_classifier, open('model.pkl', 'wb'))

