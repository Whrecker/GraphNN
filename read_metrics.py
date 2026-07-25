import pandas as pd
try:
    df = pd.read_excel(r"c:\Users\jag7b\project ankit sir\metrics_results.xlsx")
    print(df.to_string())
except Exception as e:
    print(f"Error reading excel: {e}")
