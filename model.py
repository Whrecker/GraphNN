import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def extract_features(file_path):
    df = pd.read_excel(file_path)
    features = {}
    for col in df.columns[1:]:
        data = df[col]
            numeric_values = []
            for d in data.dropna():
                if isinstance(d, (int, float, complex)) and not isinstance(d, bool):
                    numeric_values.append(d)
                # handle numeric strings
                elif isinstance(d, str):
                    try:
                        numeric_values.append(float(d))
                    except ValueError:
                        pass
            if len(numeric_values) == len(data.dropna()):
                features[f'{col}_mean'] = data.mean() 
                features[f'{col}_sum'] = data.sum()   
                features[f'{col}_transitions'] = (data.diff() != 0).sum()  
    return features

def load_data(folder, label):
    data = []
    for file in os.listdir(folder):
        if file.endswith('.xlsx'):
            if label!=None:
                features = extract_features(os.path.join(folder, file))
                features['target'] = label
                data.append(features)
            else:
                features = extract_features(os.path.join(folder, file))
                data.append(features)
    return pd.DataFrame(data)

healthy_folder = 'C:/Users/jag7b/project ankit sir/healty'
sick_folder = 'C:/Users/jag7b/project ankit sir/different'

healthy_data = load_data(healthy_folder, label=0)[:-10]
print("done")
sick_data = load_data(sick_folder, label=1)[:-10]
print("done2")
testinghealthy=load_data(healthy_folder, label=0)[-10:]
testingsick=load_data(sick_folder, label=1)[-10:]
df = pd.concat([healthy_data, sick_data])
test=load_data("C:/Users/jag7b/project ankit sir",label=None)
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=420)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)

res_df = pd.DataFrame({"File": ["model.py"], "Technique": ["Random Forest Classifier"], "Accuracy": [acc], "Precision": [prec], "Recall": [rec]})
import os
try:
    if os.path.exists("metrics_results.xlsx"):
        with pd.ExcelWriter("metrics_results.xlsx", mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            res_df.to_excel(writer, startrow=writer.sheets["Sheet1"].max_row, index=False, header=False)
    else:
        res_df.to_excel("metrics_results.xlsx", index=False)
except Exception as e:
    print(e)
print("Metrics saved.")
        
