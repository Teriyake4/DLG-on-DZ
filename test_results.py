import pandas as pd
import os

resultsPath = os.path.join('.', "results/baseline/")
foResults = os.path.join(resultsPath, "fo/results.csv")
zoResults = os.path.join(resultsPath, "zo/results.csv")

thresholds = [0.01, 0.005, 0.001, 0.0005, 0.0001]

def calculateStats(df):
    # for threshold in thresholds:
    #     print(f"MSE < {threshold}:", len(df[df.iloc[:, 2] < threshold]) / len(df)) # MSE threshold

    print("DLG:")
    print(f"Mean Loss: {df.iloc[:, 1].mean():.4e}, Mean MSE: {df.iloc[:, 2].mean():.4e}")
    print(f"Median Loss: {df.iloc[:, 1].median():.4e}, Median MSE: {df.iloc[:, 2].median():.4e}")
    print(f"Minimum MSE: {df.iloc[:, 2].min()}, Index: {df.iloc[:, 2].idxmin()}")

    print("iDLG:")
    print(f"Mean Loss: {df.iloc[:, 3].mean():.4e}, Mean MSE: {df.iloc[:, 4].mean():.4e}")
    print(f"Median Loss: {df.iloc[:, 3].median():.4e}, Median MSE: {df.iloc[:, 4].median():.4e}")
    print(f"Minimum MSE: {df.iloc[:, 3].min()}, Index: {df.iloc[:, 3].idxmin()}")

fodf = pd.read_csv(foResults)
zodf = pd.read_csv(zoResults)

print("FO Results:")
calculateStats(fodf)

print('\n')

print("ZO Results:")
calculateStats(zodf)