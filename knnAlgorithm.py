import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class K_Nearest_Neighbors:
    def __init__(self, k_neighbors = 5):
        self.k = k_neighbors
        self.X_train = None
        self.y_train = None

    def features(self, X, y):
        self.X_train = X
        self.y_train = y

    def train_test_split(self, X, y, test_size = 0.8, random_state = 42):
    
        # Shuffle indexes randomly
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(len(X))

        # % training and % test split
        train_size = int(test_size * len(X))
        self.train_idx , self.test_idx = shuffled_indices[:train_size], shuffled_indices[train_size:] 

        # Create training and test sets
        X_train, X_test = X[self.train_idx], X[self.test_idx]
        y_train, y_test = y[self.train_idx], y[self.test_idx]

        return X_train, X_test, y_train, y_test
    
    #Ecludean distance function
    def euclidean_distance(self,x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def knn_accuracy(self, y_pred, y_test):
        return np.sum(y_pred == y_test) / len(y_test)

    def KNN(self, X_train, y_train, X_test, k=5):
        prediction = []

        for test_point in X_test:
            distances = [self.euclidean_distance(test_point, train_point) for train_point in X_train]
            k_indices = np.argsort(distances)[:k]
            k_nearest_labels = [y_train[i] for i in k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            prediction.append(most_common)

        return np.array(prediction)
   
    def get_test_idx(self):
        idx = self.test_idx
        return idx
    
    def optimal_k(self, X_train, y_train, X_test, y_test, k_neighbors=21, smoothing_window=3):
      
        errors = []
        k_values = list(range(1, k_neighbors)) 

        print("--- Calculando errores de clasificación para cada k ---")
        for k in k_values:
            y_prediction = self.KNN(X_train=X_train, y_train=y_train, X_test=X_test, k = k)
            accuracy = np.sum(y_prediction == y_test) / len(y_test)
            error = 1 - accuracy
            errors.append(error)
            print(f"k = {k:2}, Precisión = {accuracy*100:.2f}%, Error = {error:.3f}")

        # --- Generate the plot ---
        plt.figure(figsize=(10, 6))

        # Original error curve
        plt.plot(k_values, errors, marker='o', linestyle='-', color='b', label='Error Original')
        # We convert to Pandas Series to use the moving average
        errors_series = pd.Series(errors)
        # We calculate the moving average. Adjust 'smoothing_window' if you want more or less smoothing.
        smoothed_errors = errors_series.rolling(window=smoothing_window, min_periods=1, center=True).mean()
            
        plt.plot(k_values, smoothed_errors, marker='x', linestyle='--', color='red', 
                    label=f'Error Suavizado (Prom. Móvil {smoothing_window})', linewidth=2)

        plt.title("Método del Codo para determinar el mejor k (con suavizado visual)", fontsize=14)
        plt.xlabel("Número de vecinos (k)", fontsize=12)
        plt.ylabel("Error de clasificación", fontsize=12)
        plt.xticks(k_values)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend() 
        plt.show()

        best_k_idx = np.argmin(smoothed_errors)
        best_k = k_values[best_k_idx]

        return best_k

            

