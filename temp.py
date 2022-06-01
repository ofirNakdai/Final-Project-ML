from cmath import nan
import pandas as pd

test = pd.Series(["high","low","mid","c", nan])
unique_vals = pd.Series(test.unique())
name = "ofir"
test.fillna(name,inplace=True)
print(test)
