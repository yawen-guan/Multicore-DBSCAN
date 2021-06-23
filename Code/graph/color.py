# Usage: python color.py <dataFileName> <outputImageName>

import matplotlib.pyplot as plt
import pandas as pd
import sys

dataFileName = sys.argv[1]
df = pd.read_csv(dataFileName)
# print(df)

colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
dfc = (df["clusterID"]%len(colors)).map({i: colors[i] for i in range(0, len(colors))})
# print(df["clusterID"])

plt.scatter(df["x"], df["y"], c=dfc, alpha=0.6, s=10)

plt.savefig(sys.argv[2])
