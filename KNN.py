
'''
    Marco Yahir Solorio Diaz
    KNN Implementation on Titanic Dataset
'''

#=======================================================================================================================================================
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Rectangle
import warnings as wrn  
import knnAlgorithm as knn
import Imputation as imp
import Graphs as grh
import os

#=======================================================================================================================================================
# Load the Titanic dataset from a CSV file
try:
    titanic_data = pd.read_csv('E:\Documents\9no Semestre\Algoritmos de Inteligencia Artificial\Ex1_KNN_Titanic\Titanic-Dataset.csv', sep=',')
    print("\n\nDataset loaded successfully.\n\n")

except FileNotFoundError:
    print("Error: The file E:\Documents\9no Semestre\Algoritmos de Inteligencia Artificial\Ex1_KNN_Titanic\Titanic-Dataset.csv'\n\n")

# Display basic information about the dataset
print(titanic_data.shape)
print("\n\n")
print(titanic_data.head())
print("\n\n")
print(titanic_data.info())
print("\n\n")
print(titanic_data.isnull().sum())

#=======================================================================================================================================================
# Cleaning and Preprocessing the Data
print
# Delete columns that are irrelevant or have too many null values
titanic_data.drop(columns = ['Cabin', 'Name', 'Ticket'], inplace = True, errors = 'ignore')

# Convert categorical varibles  to numerical
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Identify a numerical and categorical columns
numerical_cols = ['Age', 'Fare','SibSp', 'Parch']
categorical_cols = ['Embarked', 'Sex', 'Pclass']

print("\nNumerical Columns:", numerical_cols)
print("\nCategorical Columns:", categorical_cols)
 
#=======================================================================================================================================================
# Imputation with mean and mode
print("\n--- Data imputation Applied ---")

# Numerical value imputation with mean
df_imputed = imp.mean_imputation(titanic_data.copy(), numerical_cols = numerical_cols)

# Numerical value imputation with mode
df_imputed = imp.mode_imputation(df_imputed, categorical_cols = categorical_cols)

print("\n\nImputed successfully.\n\n")

# MinMaxScaler normalization
df_normalized = imp.normalize_dataframe(df_imputed, numerical_cols)

#=======================================================================================================================================================
# Save processed dataset to a new CSV file
try:
    output_filename = 'titanic_imputed_normalized.csv'
    df_normalized.to_csv(output_filename, index=False)
    print(f"File Saved as: {output_filename}")
except Exception as e:
    print(f"Error to save file: {e}")

#=======================================================================================================================================================

# Graphical comparison with boxplots
print("\nDistribution comparations before and after to processing\n")
for col in numerical_cols:
    compare_df = pd.DataFrame({
        'Value': pd.concat([titanic_data[col], df_imputed[col]], ignore_index=True),
        'Type': ['Original'] * len(titanic_data[col]) + ['Imputado'] * len(df_normalized[col])
    })

    plt.figure(figsize=(8,5))
    sns.boxplot(x='Type', y='Value', data=compare_df, hue='Type', palette=['#6BAED6', '#E07A5F'], legend=False)
    plt.title(f'BoxPlot Comparation - {col}', fontsize=12)
    plt.ylabel(col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

#=======================================================================================================================================================
# KNN Implementation and Evaluation
# Separate objective and predictor variables
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Sex']
X = df_normalized[features].values
y = df_normalized['Survived'].values

# Initialize KNN algorithmc
n_neighbors = 11
Knn_algorithm = knn.K_Nearest_Neighbors(k_neighbors = n_neighbors)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = Knn_algorithm.train_test_split(X = X, y = y, test_size = 0.7, random_state = 42)
Knn_algorithm.features(X_train, y_train)

#=======================================================================================================================================================
# Optimal k determination using Elbow Method
print("\nOptimal k determination using Elbow Method...\n")
Knn_algorithm.optimal_k(X_train, y_train, X_test, y_test, k_neighbors = 21)
#=======================================================================================================================================================

predictions = Knn_algorithm.KNN(X_train = X_train, y_train = y_train, X_test = X_test, k = n_neighbors)
accuracy = Knn_algorithm.knn_accuracy(predictions, y_test)

print(f"\nModel Precision KNN: {accuracy * 100:.2f}%")
#=======================================================================================================================================================

'''
# Graph of Decision Boundaries
print("\n Generate Graph of Decision Boundaries...\n")
try:
    x_col, y_col = 'Pclass', 'Age'
    X_vis = df_normalized[[x_col, y_col]].values
    y_vis = df_normalized['Survived'].values

    # create mesh grid
    x_min, x_max = X_vis[:, 0].min() - 0.1, X_vis[:, 0].max() + 0.1
    y_min, y_max = X_vis[:, 1].min() - 0.1, X_vis[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Predict on mesh grid points
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            Z[i, j] = Knn_algorithm.KNN(X_vis, y_vis, np.array([[xx[i, j], yy[i, j]]]), k = n_neighbors)[0]

    # Plotting the decision boundaries
    plt.figure(figsize=(8,6))
    
    # Plotting the decision regions(contourf)
    plt.contourf(xx, yy, Z, alpha = 0.4, cmap = plt.cm.coolwarm) 
    # Blue Color (cool) survived (1)
    handle_survived = Rectangle((0, 0), 1, 1, fc=plt.cm.coolwarm(255)) 
    # Red Color (warm) No Survived (0)
    handle_not_survived = Rectangle((0, 0), 1, 1, fc=plt.cm.coolwarm(0)) 
    survived = (y_vis == 1)
    not_survived = (y_vis == 0)

    # Points No survived (0) - Red Color
    plt.scatter(X_vis[not_survived, 0], X_vis[not_survived, 1], 
                c=plt.cm.coolwarm(0), cmap = plt.cm.coolwarm, 
                s = 30, edgecolor = 'k', label='Predicción: No Sobrevivió (0)')
    
    # Point Survived (1) - Blue Color
    plt.scatter(X_vis[survived, 0], X_vis[survived, 1], 
                c=plt.cm.coolwarm(255), cmap = plt.cm.coolwarm, 
                s = 30, edgecolor = 'k', label='Predicción: Sobrevivió (1)')
    
    plt.title(f"Fronteras de Decisión KNN (k={n_neighbors}) - {x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    
    scatter_handles, scatter_labels = plt.gca().get_legend_handles_labels()
    full_handles = [handle_not_survived, handle_survived] + scatter_handles
    full_labels = ['Región: No Sobrevivió (0)', 'Región: Sobrevivió (1)'] + scatter_labels
    
    plt.legend(full_handles, full_labels, title='Clase Predicha', loc='upper right', fontsize=8)

    plt.show()

except Exception as e:
    print(f"Error to generate Decision Boundries: {e}")

'''
#=======================================================================================================================================================
# Manual Confusion Matrix Calculation
print("\nManual Confusion Matrix Calcualting...\n")

cm = grh.confusion_matrix_manual(y_test, predictions)

print("\nMatriz de Confusión:")
print(pd.DataFrame(cm, index = ['Real: No Survived (0)', 'Real: Survived (1)'], columns = ['Pred: No Survived (0)', 'Pred: Survived (1)']))

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Survived', 'Survived'], yticklabels=['No Survived', 'Survived'])
plt.title('Matriz de Confusión - KNN (k = {})'.format(n_neighbors))
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

#===================================================================================================================================================
# Additional Metrics Calculation
print("\nCalculating Additional Metrics...\n")
TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"\nAdditional Metrics:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-Score: {f1_score*100:.2f}%")


# Analysis of Survival by Sex
print("\nSurvivor analysis by sex\n")

tabla_sexo = pd.crosstab(df_imputed['Sex'], df_imputed['Survived'], rownames=['Sex (0 = Male, 1 = Female)'], colnames=['Survived (0 = No, 1 = Yes)'])
print(tabla_sexo)

tabla_sexo['Total'] = tabla_sexo.sum(axis = 1)
tabla_sexo.loc['Total'] = tabla_sexo.sum()

print("\nTotal Table:")
print(tabla_sexo)

#=======================================================================================================================================================
# Graphical Analysis of Survival by Sex
df_results = df_imputed.iloc[Knn_algorithm.test_idx].copy()
df_results['Predicted'] = predictions

plt.figure(figsize=(7,5))
sns.countplot(x='Sex', hue='Survived', data=df_results, palette={0: 'red', 1: 'green'})
plt.title('Real Survived by Sex ')
plt.xlabel('Sex (0 = Male, 1 = Female)')
plt.ylabel('Number of Passengers')
plt.legend(title='Real Survivor', labels=['No', 'Yes'])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x='Sex', hue='Predicted', data=df_results, palette={0: 'red', 1: 'green'})
plt.title('Survived Predicted by Sex')
plt.xlabel('Sexo (0 = Male, 1 = Female)')
plt.ylabel('Number of Passengers')
plt.legend(title='Predicted Survived', labels=['No', 'Yes'])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
#=======================================================================================================================================================

