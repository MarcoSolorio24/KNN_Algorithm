from sklearn.preprocessing import MinMaxScaler

def mean_imputation(df_imputed, numerical_cols):
    # Numerical value imputation with mean
    for col in numerical_cols:
        if df_imputed[col].isnull().sum() > 0:
            mean_value = df_imputed[col].mean()
            df_imputed[col] = df_imputed[col].fillna(mean_value)
            print(f"➡ Columna '{col}' imputada con la media: {mean_value:.2f}")
    return df_imputed

def mode_imputation(df_imputed, categorical_cols):
    # Numerical value imputation with mode
    for col in categorical_cols:
        if df_imputed[col].isnull().sum() > 0:
            mode_value = df_imputed[col].mode()[0]
            df_imputed[col].fillna(mode_value, inplace=True)
            print(f"➡ Columna '{col}' imputada con la moda: {mode_value}")
    return df_imputed

def normalize_dataframe(df, cols):
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[cols] = scaler.fit_transform(df_norm[cols])
    return df_norm