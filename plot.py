import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import ROOT_DIR

data = pd.read_csv(ROOT_DIR + "/data/TTF_FM_new.csv",sep=";")
df = pd.DataFrame(data, columns=["Date","Price"])
plt.plot(df["Date"],df["Price"])
plt.show()