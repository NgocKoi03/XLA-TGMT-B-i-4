import os
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import cifar10

# Tắt tất cả thông báo từ TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Tải và chuẩn bị dữ liệu
def load_and_preprocess_data(selected_labels, train_size=600, test_size=300, random_seed=42):
    (x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()
    y_train_full, y_test_full = y_train_full.flatten(), y_test_full.flatten()
    
    # Lọc dữ liệu theo nhãn đã chọn
    train_mask = np.isin(y_train_full, selected_labels)
    test_mask = np.isin(y_test_full, selected_labels)
    
    x_train_selected, y_train_selected = x_train_full[train_mask], y_train_full[train_mask]
    x_test_selected, y_test_selected = x_test_full[test_mask], y_test_full[test_mask]

    # Lấy mẫu ngẫu nhiên từ tập huấn luyện và kiểm tra
    np.random.seed(random_seed)
    train_indices = np.random.choice(len(x_train_selected), train_size, replace=False)
    test_indices = np.random.choice(len(x_test_selected), test_size, replace=False)
    
    x_train, y_train = x_train_selected[train_indices], y_train_selected[train_indices]
    x_test, y_test = x_test_selected[test_indices], y_test_selected[test_indices]
    
    # Chuẩn hóa và làm phẳng dữ liệu
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
    
    return x_train, y_train, x_test, y_test

# Trích xuất đặc trưng bằng PCA
def apply_pca(x_train, x_test, n_components=50):
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_test_pca

# Huấn luyện và đánh giá mô hình
def train_and_evaluate_model(model, model_name, x_train, y_train, x_test, y_test):
    print(f"Training {model_name}...")
    start_time = time.time()
    model.fit(x_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(x_test)
    prediction_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"{model_name} Results:")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Prediction Time: {prediction_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("-" * 30)

# Thực thi quy trình
selected_labels = [2, 3, 5]
x_train, y_train, x_test, y_test = load_and_preprocess_data(selected_labels)
x_train_pca, x_test_pca = apply_pca(x_train, x_test)

# Khởi tạo và đánh giá các mô hình
models = {
    "SVM": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10)
}

for model_name, model in models.items():
    train_and_evaluate_model(model, model_name, x_train_pca, y_train, x_test_pca, y_test)
