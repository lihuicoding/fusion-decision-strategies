import pandas as pd

f = pd.read_csv("dataset\winequality_red.csv")
f.to_csv("dataset\winequality_red.csv", sep=",")