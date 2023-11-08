import pickle

with open('model.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    mms, pca, lrc, LRpca_Params = pickle.load(f_in)

# Predict function
def transform_data(cell):
    df_X = pd.DataFrame(cell, index=[0])
    iX = mms.transform(df_X)
    X = pca.transform(iX)
    return X

def predict_from_model(X):
    y_pred = lrc.predict_proba(X)[0,1]
    if y_pred >= 0.5:
        Diagnosis = "Malignant"
    else:
        Diagnosis = "Benign"
    return Diagnosis, y_pred
            
X = transform_data(cell)
Diagnosis, y_pred = predict_from_model(X)
text = f"Cell Diagnosis: {Diagnosis}\np Malignant: {y_pred}"
print(text)