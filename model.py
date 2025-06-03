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
        if col!='bed sensor':
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

healthy_folder = 'C:/Users/jag7b/project ankit sir/healthy'
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

y_pred = model.predict(X_test)
print(y_pred)
print(classification_report(y_test, y_pred))
y_pred=model.predict(test)
print(y_pred)

X = testinghealthy.drop(columns=['target'])
print(model.predict(X))
print("--------------------------")
X=testingsick.drop(columns=['target'])
print(model.predict(X))
        
