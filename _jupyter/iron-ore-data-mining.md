```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv



```python
# Create New Conda Environment and Use Conda Channel 
#!conda create -n newCondaEnvironment python=3.8 pandas=1.5 -y
#!source /opt/conda/bin/activate newCondaEnvironment
```


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

import optuna
from optuna.integration import LightGBMPruningCallback
from optuna.samplers import TPESampler
from lightgbm import LGBMRegressor
import lightgbm
from lightgbm import plot_importance

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import warnings

pd.options.display.max_columns = None

%matplotlib inline
```

## Dataset Loadin


```python
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
```


```python
df = pd.read_csv('/kaggle/input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv',decimal=",",parse_dates=["date"],infer_datetime_format=True)
```


```python
df = reduce_mem_usage(df)
```

    Mem. usage decreased to 70.33 Mb (47.9% reduction)



```python
df['date'] = pd.to_datetime(df['date'])
```


```python
df = df.rename(columns=lambda x: '_'.join(x.split()))
```


```python
df.duplicated().sum()
```




    1171



**Notice, we don't drop these seemly duplicated data points. Since there might be essential in the follow up continuity check.**


```python
df.head()
```




<div>
<style scoped>
    .dataframe {
      display: block;
      width: 100%;
      overflow-x: auto;
    }
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
    
    

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>%_Iron_Feed</th>
      <th>%_Silica_Feed</th>
      <th>Starch_Flow</th>
      <th>Amina_Flow</th>
      <th>Ore_Pulp_Flow</th>
      <th>Ore_Pulp_pH</th>
      <th>Ore_Pulp_Density</th>
      <th>Flotation_Column_01_Air_Flow</th>
      <th>Flotation_Column_02_Air_Flow</th>
      <th>Flotation_Column_03_Air_Flow</th>
      <th>Flotation_Column_04_Air_Flow</th>
      <th>Flotation_Column_05_Air_Flow</th>
      <th>Flotation_Column_06_Air_Flow</th>
      <th>Flotation_Column_07_Air_Flow</th>
      <th>Flotation_Column_01_Level</th>
      <th>Flotation_Column_02_Level</th>
      <th>Flotation_Column_03_Level</th>
      <th>Flotation_Column_04_Level</th>
      <th>Flotation_Column_05_Level</th>
      <th>Flotation_Column_06_Level</th>
      <th>Flotation_Column_07_Level</th>
      <th>%_Iron_Concentrate</th>
      <th>%_Silica_Concentrate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-03-10 01:00:00</td>
      <td>55.200001</td>
      <td>16.98</td>
      <td>3019.530029</td>
      <td>557.434021</td>
      <td>395.713013</td>
      <td>10.0664</td>
      <td>1.74</td>
      <td>249.214005</td>
      <td>253.235001</td>
      <td>250.576004</td>
      <td>295.096008</td>
      <td>306.399994</td>
      <td>250.225006</td>
      <td>250.884003</td>
      <td>457.395996</td>
      <td>432.962006</td>
      <td>424.954010</td>
      <td>443.558014</td>
      <td>502.255005</td>
      <td>446.369995</td>
      <td>523.343994</td>
      <td>66.910004</td>
      <td>1.31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-03-10 01:00:00</td>
      <td>55.200001</td>
      <td>16.98</td>
      <td>3024.409912</td>
      <td>563.965027</td>
      <td>397.382996</td>
      <td>10.0672</td>
      <td>1.74</td>
      <td>249.718994</td>
      <td>250.531998</td>
      <td>250.862000</td>
      <td>295.096008</td>
      <td>306.399994</td>
      <td>250.136993</td>
      <td>248.994003</td>
      <td>451.890991</td>
      <td>429.559998</td>
      <td>432.938995</td>
      <td>448.085999</td>
      <td>496.363007</td>
      <td>445.921997</td>
      <td>498.075012</td>
      <td>66.910004</td>
      <td>1.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-03-10 01:00:00</td>
      <td>55.200001</td>
      <td>16.98</td>
      <td>3043.459961</td>
      <td>568.054016</td>
      <td>399.667999</td>
      <td>10.0680</td>
      <td>1.74</td>
      <td>249.740997</td>
      <td>247.873993</td>
      <td>250.313004</td>
      <td>295.096008</td>
      <td>306.399994</td>
      <td>251.345001</td>
      <td>248.070999</td>
      <td>451.239990</td>
      <td>468.927002</td>
      <td>434.609985</td>
      <td>449.687988</td>
      <td>484.411011</td>
      <td>447.825989</td>
      <td>458.566986</td>
      <td>66.910004</td>
      <td>1.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-03-10 01:00:00</td>
      <td>55.200001</td>
      <td>16.98</td>
      <td>3047.360107</td>
      <td>568.664978</td>
      <td>397.938995</td>
      <td>10.0689</td>
      <td>1.74</td>
      <td>249.917007</td>
      <td>254.487000</td>
      <td>250.048996</td>
      <td>295.096008</td>
      <td>306.399994</td>
      <td>250.421997</td>
      <td>251.147003</td>
      <td>452.441010</td>
      <td>458.165009</td>
      <td>442.864990</td>
      <td>446.209991</td>
      <td>471.411011</td>
      <td>437.690002</td>
      <td>427.669006</td>
      <td>66.910004</td>
      <td>1.31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-03-10 01:00:00</td>
      <td>55.200001</td>
      <td>16.98</td>
      <td>3033.689941</td>
      <td>558.166992</td>
      <td>400.253998</td>
      <td>10.0697</td>
      <td>1.74</td>
      <td>250.203003</td>
      <td>252.136002</td>
      <td>249.895004</td>
      <td>295.096008</td>
      <td>306.399994</td>
      <td>249.983002</td>
      <td>248.927994</td>
      <td>452.441010</td>
      <td>452.899994</td>
      <td>450.523010</td>
      <td>453.670013</td>
      <td>462.597992</td>
      <td>443.682007</td>
      <td>425.678986</td>
      <td>66.910004</td>
      <td>1.31</td>
    </tr>
  </tbody>
</table>
</div>



* Goal is to predict **% Silica Concentrate**
* Silica Concentrate is the impurity in the iron ore which needs to be removed
* The value of modeling on this target value is that it takes at least an hour to measure the impurity


```python
total = df.isnull().sum()
percent = (df.isnull().sum()/df.isnull().count())
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data
```




<div>
<style scoped>
    .dataframe {
      display: block;
      width: 100%;
      overflow-x: auto;
    }
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>date</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>%_Iron_Feed</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>%_Silica_Feed</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Starch_Flow</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Amina_Flow</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ore_Pulp_Flow</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ore_Pulp_pH</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ore_Pulp_Density</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_01_Air_Flow</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_02_Air_Flow</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_03_Air_Flow</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_04_Air_Flow</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_05_Air_Flow</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_06_Air_Flow</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_07_Air_Flow</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_01_Level</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_02_Level</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_03_Level</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_04_Level</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_05_Level</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_06_Level</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Flotation_Column_07_Level</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>%_Iron_Concentrate</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>%_Silica_Concentrate</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe {
      display: block;
      width: 100%;
      overflow-x: auto;
    }
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>%_Iron_Feed</th>
      <th>%_Silica_Feed</th>
      <th>Starch_Flow</th>
      <th>Amina_Flow</th>
      <th>Ore_Pulp_Flow</th>
      <th>Ore_Pulp_pH</th>
      <th>Ore_Pulp_Density</th>
      <th>Flotation_Column_01_Air_Flow</th>
      <th>Flotation_Column_02_Air_Flow</th>
      <th>Flotation_Column_03_Air_Flow</th>
      <th>Flotation_Column_04_Air_Flow</th>
      <th>Flotation_Column_05_Air_Flow</th>
      <th>Flotation_Column_06_Air_Flow</th>
      <th>Flotation_Column_07_Air_Flow</th>
      <th>Flotation_Column_01_Level</th>
      <th>Flotation_Column_02_Level</th>
      <th>Flotation_Column_03_Level</th>
      <th>Flotation_Column_04_Level</th>
      <th>Flotation_Column_05_Level</th>
      <th>Flotation_Column_06_Level</th>
      <th>Flotation_Column_07_Level</th>
      <th>%_Iron_Concentrate</th>
      <th>%_Silica_Concentrate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
      <td>737453.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>56.294743</td>
      <td>14.651719</td>
      <td>2869.140381</td>
      <td>488.144653</td>
      <td>397.578400</td>
      <td>9.767639</td>
      <td>1.680380</td>
      <td>280.151886</td>
      <td>277.159973</td>
      <td>281.082489</td>
      <td>299.447845</td>
      <td>299.917816</td>
      <td>292.071503</td>
      <td>290.754883</td>
      <td>520.244812</td>
      <td>522.649597</td>
      <td>531.352661</td>
      <td>420.320984</td>
      <td>425.251648</td>
      <td>429.941040</td>
      <td>421.021271</td>
      <td>65.050049</td>
      <td>2.326763</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.157743</td>
      <td>6.807439</td>
      <td>1215.203735</td>
      <td>91.230537</td>
      <td>9.699785</td>
      <td>0.387007</td>
      <td>0.069249</td>
      <td>29.621286</td>
      <td>30.149357</td>
      <td>28.558270</td>
      <td>2.572536</td>
      <td>3.636578</td>
      <td>30.217804</td>
      <td>28.670105</td>
      <td>131.014923</td>
      <td>128.165054</td>
      <td>150.842163</td>
      <td>91.794434</td>
      <td>84.535820</td>
      <td>89.862228</td>
      <td>84.891495</td>
      <td>1.118645</td>
      <td>1.125554</td>
    </tr>
    <tr>
      <th>min</th>
      <td>42.740002</td>
      <td>1.310000</td>
      <td>0.002026</td>
      <td>241.669006</td>
      <td>376.248993</td>
      <td>8.753340</td>
      <td>1.519820</td>
      <td>175.509995</td>
      <td>175.156006</td>
      <td>176.468994</td>
      <td>292.195007</td>
      <td>286.295013</td>
      <td>189.927994</td>
      <td>185.962006</td>
      <td>149.218002</td>
      <td>210.751999</td>
      <td>126.254997</td>
      <td>162.201004</td>
      <td>166.990997</td>
      <td>155.841003</td>
      <td>175.348999</td>
      <td>62.049999</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>52.669998</td>
      <td>8.940000</td>
      <td>2076.320068</td>
      <td>431.795990</td>
      <td>394.264008</td>
      <td>9.527360</td>
      <td>1.647310</td>
      <td>250.281006</td>
      <td>250.457001</td>
      <td>250.854996</td>
      <td>298.262573</td>
      <td>298.067993</td>
      <td>262.540985</td>
      <td>256.302002</td>
      <td>416.977997</td>
      <td>441.882996</td>
      <td>411.325012</td>
      <td>356.678986</td>
      <td>357.653015</td>
      <td>358.497009</td>
      <td>356.772003</td>
      <td>64.370003</td>
      <td>1.440000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>56.080002</td>
      <td>13.850000</td>
      <td>3018.429932</td>
      <td>504.393005</td>
      <td>399.248993</td>
      <td>9.798100</td>
      <td>1.697600</td>
      <td>299.343994</td>
      <td>296.222992</td>
      <td>298.696014</td>
      <td>299.804993</td>
      <td>299.887115</td>
      <td>299.476990</td>
      <td>299.010986</td>
      <td>491.877991</td>
      <td>495.955994</td>
      <td>494.317993</td>
      <td>411.973999</td>
      <td>408.773010</td>
      <td>424.664581</td>
      <td>411.065002</td>
      <td>65.209999</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>59.720001</td>
      <td>19.600000</td>
      <td>3727.729980</td>
      <td>553.257019</td>
      <td>402.967987</td>
      <td>10.038000</td>
      <td>1.728330</td>
      <td>300.148987</td>
      <td>300.690002</td>
      <td>300.381989</td>
      <td>300.638000</td>
      <td>301.791138</td>
      <td>303.061005</td>
      <td>301.903992</td>
      <td>594.114014</td>
      <td>595.463989</td>
      <td>601.249023</td>
      <td>485.549011</td>
      <td>484.329010</td>
      <td>492.683990</td>
      <td>476.464996</td>
      <td>65.860001</td>
      <td>3.010000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>65.779999</td>
      <td>33.400002</td>
      <td>6300.229980</td>
      <td>739.538025</td>
      <td>418.640991</td>
      <td>10.808100</td>
      <td>1.853250</td>
      <td>373.871002</td>
      <td>375.992004</td>
      <td>364.346008</td>
      <td>305.871002</td>
      <td>310.269989</td>
      <td>370.910004</td>
      <td>371.592987</td>
      <td>862.273987</td>
      <td>828.919006</td>
      <td>886.822021</td>
      <td>680.359009</td>
      <td>675.643982</td>
      <td>698.861023</td>
      <td>659.901978</td>
      <td>68.010002</td>
      <td>5.530000</td>
    </tr>
  </tbody>
</table>
</div>




```python
dt_df = df.set_index('date')
```


```python
all_hours = pd.Series(data=pd.date_range(start=dt_df.index.min(), end=dt_df.index.max(), freq='H'))
mask = all_hours.isin(dt_df.index.values)
all_hours[~mask]
```




    149   2017-03-16 06:00:00
    150   2017-03-16 07:00:00
    151   2017-03-16 08:00:00
    152   2017-03-16 09:00:00
    153   2017-03-16 10:00:00
                  ...        
    462   2017-03-29 07:00:00
    463   2017-03-29 08:00:00
    464   2017-03-29 09:00:00
    465   2017-03-29 10:00:00
    466   2017-03-29 11:00:00
    Length: 318, dtype: datetime64[ns]



**Therefore, we choose only to reserve the data after 2017-03-29 12\:00\:00**


```python
dt_df = dt_df.loc["2017-03-29 12:00:00":]
```

**The measurement frequency of most rows in the dataset is every 20 seconds according to what the publisher/author of the dataset mentions. This would imply that there should be 180 measurements in each hour if this information is correct.**


```python
dt_df.groupby(dt_df.index).count()["%_Silica_Concentrate"].value_counts()
```




    180    3947
    179       1
    Name: %_Silica_Concentrate, dtype: int64




```python
dt_df.groupby(dt_df.index).count()["%_Silica_Concentrate"][dt_df.groupby(dt_df.index).count()["%_Silica_Concentrate"] < 180]
```




    date
    2017-04-10    179
    Name: %_Silica_Concentrate, dtype: int64




```python
dt_df.loc['2017-04-10'].groupby(dt_df.loc['2017-04-10'].index).count()['%_Silica_Concentrate']
```




    date
    2017-04-10 00:00:00    179
    2017-04-10 01:00:00    180
    2017-04-10 02:00:00    180
    2017-04-10 03:00:00    180
    2017-04-10 04:00:00    180
    2017-04-10 05:00:00    180
    2017-04-10 06:00:00    180
    2017-04-10 07:00:00    180
    2017-04-10 08:00:00    180
    2017-04-10 09:00:00    180
    2017-04-10 10:00:00    180
    2017-04-10 11:00:00    180
    2017-04-10 12:00:00    180
    2017-04-10 13:00:00    180
    2017-04-10 14:00:00    180
    2017-04-10 15:00:00    180
    2017-04-10 16:00:00    180
    2017-04-10 17:00:00    180
    2017-04-10 18:00:00    180
    2017-04-10 19:00:00    180
    2017-04-10 20:00:00    180
    2017-04-10 21:00:00    180
    2017-04-10 22:00:00    180
    2017-04-10 23:00:00    180
    Name: %_Silica_Concentrate, dtype: int64



Inspired by https://www.kaggle.com/code/matiasob/dataset-is-corrupt-and-should-not-be-used, it says it's an option to replace that missing timestamp with the latest datapoint behind it, which is **2017-04-09 23\:00\:00**


```python
df_before = dt_df.copy().loc[:'2017-04-10 00:00:00']
df_after = dt_df.copy().loc['2017-04-10 01:00:00':]
new_date = pd.to_datetime('2017-04-10 00:00:00')
new_data = pd.DataFrame(df_before[-1:].values, index=[new_date], columns=df_before.columns)
df_before = pd.concat([df_before,new_data],axis=0)

dt_df = pd.concat([df_before, df_after])
dt_df.reset_index(drop=True)

dt_df["duration"] = 20
dt_df.loc[0,"duration"] = 0
dt_df.duration = dt_df.duration.cumsum()

dt_df['Date_with_seconds'] = pd.Timestamp("2017-03-29 12:00:00") + pd.to_timedelta(dt_df['duration'], unit='s')

dt_df = dt_df.set_index("Date_with_seconds")
```


```python
fig = plt.figure(figsize=(18,18))

sns.heatmap(dt_df.corr(), annot=True, cmap='viridis')
```




    <AxesSubplot:>




    
![png](iron-ore-data-mining_files/iron-ore-data-mining_25_1.png)
    


* First you can tell the **% of silica** is negatively corelated to the **% of iron concentrate**, which makes sense since they are separately desired and undesired substance of the process, purity and impurity
* Secondly, the **airflow in column 1** is in general positively corelated to the **airflow in column 2-7** except for column 5, which arise a potential doubt deserve diving deeper. Also we can tell the larger the airflow is, the lower the flotation level is in the responding column.
* Thirdly, we can see from the correlation heat map that the **ore pulp flow** and **starch flow** barely have influence on the final product -- % iron and % silica. on top of the fact that they both have positive corelation with the airflow measured in the columns, we can later verify the underlying collinearity by eigenvalue.


```python
plt.figure(figsize=(20,20),dpi=200)
for i , n in enumerate(dt_df.columns.to_list()):
    plt.subplot(6,4,i+1)
    ax = sns.histplot(data=dt_df, x=n, kde=False, bins=20)#, multiple="stack")
    plt.title(f"Histograma {n}", fontdict={"fontsize":14})
    plt.xlabel("")
    plt.ylabel(ax.get_ylabel(), fontdict={"fontsize":12})
    if i not in [0,4,8,12,16,20,24]:
        plt.ylabel("")
    

plt.tight_layout();
```


    
![png](iron-ore-data-mining_files/iron-ore-data-mining_27_0.png)
    



```python
scaler = StandardScaler()
#scaler = MinMaxScaler()

z = pd.DataFrame(scaler.fit_transform(dt_df), columns=dt_df.columns, index=dt_df.index)
z = z.melt()

plt.figure(figsize=(14,6),dpi=100)
sns.boxplot(x=z["variable"], y=z["value"]);
plt.xticks(rotation=90);
plt.xlabel("");
plt.title("Boxplots Univariados");
```


    
![png](iron-ore-data-mining_files/iron-ore-data-mining_28_0.png)
    


### TimeSeries Trend


```python
df_h = dt_df.resample('H').first()
df_h.index.names = ['Date']
```


```python
result = seasonal_decompose(df_h["%_Silica_Concentrate"][:300])

def decompose_plot(result):

    fig,axes = plt.subplots(ncols=1,nrows=4,figsize=(12,7))

    axes[0].plot(result.observed.index, result.observed.values, color='tab:cyan')
    axes[0].set_ylabel("Observed",fontdict={"size":10,},labelpad=10)
    axes[0].set_xticks(axes[0].get_xticks(), rotation=45)

    axes[1].plot(result.trend.index, result.trend.values, color='tab:cyan')
    axes[1].set_ylabel("Trend",fontdict={"size":10,},labelpad=10)

    axes[2].plot(result.seasonal.index, result.seasonal.values, color='tab:cyan')
    axes[2].set_ylabel("Seasonality",fontdict={"size":10,},labelpad=10)

    axes[3].plot(result.resid.index, result.resid.values, color='tab:cyan')
    axes[3].set_ylabel("Residuals",fontdict={"size":10,},labelpad=10)
    
    for n in range(0,4):
        axes[n].autoscale(axis="both",tight=True)

    fig.suptitle("Seasonal Decompose",fontsize=15)
    plt.tight_layout()

decompose_plot(result)
```


    
![png](iron-ore-data-mining_files/iron-ore-data-mining_31_0.png)
    


### Supervised TimeDiff Learning


```python
dt_df.columns
```




    Index(['%_Iron_Feed', '%_Silica_Feed', 'Starch_Flow', 'Amina_Flow',
           'Ore_Pulp_Flow', 'Ore_Pulp_pH', 'Ore_Pulp_Density',
           'Flotation_Column_01_Air_Flow', 'Flotation_Column_02_Air_Flow',
           'Flotation_Column_03_Air_Flow', 'Flotation_Column_04_Air_Flow',
           'Flotation_Column_05_Air_Flow', 'Flotation_Column_06_Air_Flow',
           'Flotation_Column_07_Air_Flow', 'Flotation_Column_01_Level',
           'Flotation_Column_02_Level', 'Flotation_Column_03_Level',
           'Flotation_Column_04_Level', 'Flotation_Column_05_Level',
           'Flotation_Column_06_Level', 'Flotation_Column_07_Level',
           '%_Iron_Concentrate', '%_Silica_Concentrate', 'duration'],
          dtype='object')




```python
list_cols = [col for col in dt_df.columns.to_list()]  # Use this if I want to include all explanatory variables.

# list_cols = [col for col in df.columns.to_list() if "Flotation" not in col]  # Use this to exclude flotation air/level variables.
# flotation_cols = [col for col in df.columns.to_list() if "Flotation" in col] # Use this to exclude flotation air/level variables.

list_cols.remove("%_Silica_Concentrate")
list_cols.remove("%_Iron_Concentrate")

# Resample the original df every 15 minutes.
df_15 = dt_df.resample("15min").first()
df_15 = df_15.drop("%_Iron_Concentrate", axis=1)

# df_15 = df_15.drop(flotation_cols, axis=1)  # Use this only if I do not want to include flotation air/level variables.

window_size = 3

# Taking advantage of the time factor, I add lagged explanatory variables at 15min, 30min, and 45min, respectively
# (these are the ones that were updated every 20s)

for col_name in list_cols:
    for i in range(window_size):
        df_15[f"{col_name} ({-15*(i+1)}mins)"] = df_15[f"{col_name}"].shift(periods=i+1)

# Resample from 15min to 1hour
df_h = df_15.resample('H').first()

# Here I change the name of my index.
df_h.index.names = ['Date']
```


```python
# Add lagged values of the target variable as explanatory variables, lagged by 1, 2, and 3 hours respectively.
col_name = "%_Silica_Concentrate"
window_size = 3

for i in range(window_size):
    df_h[f"{col_name} ({-i-1}h)"] = df_h[f"{col_name}"].shift(periods=i+1)

# Make the target variable the first column.
x = df_h.columns.to_list()
x.remove("%_Silica_Concentrate")
x.insert(0, "%_Silica_Concentrate")
df_h = df_h[x]

df_h = df_h.dropna()  # Remove rows with missing values due to the shift().
df_h = df_h.astype("float32")  # Convert data to float32 for faster computation.
print(df_h.shape)
```

    (3946, 92)


I create performance reporting functions for my model, a tracker to keep track of metrics, and also add a function to perform a time series split (where temporal order is respected).


```python
def plot_time_series(timesteps, values, start=0, end=None, label=None):#, #color="blue"):
    """
    Plots timesteps (a series of points in time) against values (a series of values across timesteps).

    Parameters
    ----------
    timesteps : array of timestep values
    values : array of values across time
    format : style of plot, default "."
    start : where to start the plot (setting a value will index from start of timesteps & values)
    end : where to end the plot (similar to start but for the end)
    label : label to show on plot about values, default None 
    """
  # Plot the series
    sns.lineplot(x=timesteps[start:end], y=values[start:end], label=label)
    plt.xlabel("Date")
    plt.ylabel("% Silica Concentrate")
    if label:
        plt.legend(fontsize=14) # make label bigger
    plt.grid(True)
    
def timeseries_models_tracker_df():
    reg_models_scores_df = pd.DataFrame(
        columns=["model_name", "MAE", "RMSE", "MASE", "R2", "MAPE"])

    return reg_models_scores_df

def timeseries_report_model(y_test, model_preds, tracker_df="none", model_name="model_unknown", seasonality=1, naive=False):
    mae = round(mean_absolute_error(y_test, model_preds), 4)
    rmse = round(mean_squared_error(y_test, model_preds) ** 0.5, 4)
    mase = round(mean_absolute_scaled_error(y_test, model_preds, seasonality, naive),4)
    r2 = round(r2_score(y_test, model_preds), 4)
    mape = round(mean_absolute_percentage_error(y_test, model_preds), 4)

    print("MAE: ", mae)
    print("RMSE :", rmse)
    print("MASE :", mase)
    print("R2 :", r2)
    print("MAPE :", mape)

    if isinstance(tracker_df, pd.core.frame.DataFrame):
        tracker_df.loc[tracker_df.shape[0]] = [
            model_name, mae, rmse, mase, r2, mape]
    else:
        pass
    
def mean_absolute_scaled_error(y_true, y_pred, seasonality=1, naive=False):
    """
    Implement MASE (assuming no seasonality of data).
    """
    y_true = np.array(y_true)
    #y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true-y_pred))

    # Find MAE of naive forecast (no seasonality)
    if naive:
        mae_naive_no_season = np.mean(np.abs(y_true - y_pred))
        
    else :
        mae_naive_no_season = np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality])) # our seasonality is 1 day (hence the shift of 1)

    return mae / mae_naive_no_season


# Make the train/test splits
def make_train_test_splits(windows, labels, test_split=0.2):
    """
    Splits matching pairs of winodws and labels into train and test splits.
    """
    split_size = int(len(windows) * (1-test_split)) # this will default to 80% train/20% test
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels
```

### LSTM


```python
# Make the train/test splits
def make_train_test_splits(windows, labels, test_split=0.2):
    """
    Splits matching pairs of winodws and labels into train and test splits.
    """
    split_size = int(len(windows) * (1-test_split)) # this will default to 80% train/20% test
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels
```


```python
df_forecast = df_h.copy()
df_forecast.shape
```




    (3946, 92)




```python
tracker = timeseries_models_tracker_df()
```


```python
X = df_forecast.drop("%_Silica_Concentrate", axis=1)
y = df_forecast["%_Silica_Concentrate"]
```


```python
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(X,y,test_split=0.1)
len(train_windows), len(test_windows), len(train_labels), len(test_labels)
```




    (3551, 395, 3551, 395)




```python
warnings.filterwarnings("ignore")
def objective(trial, X,y):


    
    param_grid = {
        "random_state": 123,
        "verbosity": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ['gbdt']),#,"goss"]),
        "device_type": trial.suggest_categorical("device_type", ['cpu']),
        "n_estimators": trial.suggest_int("n_estimators", 25, 10000,step=100), #for large datasets this should be very high
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 10, 190), # Also change this for large datasets, should be small to avoid overfitting
        "max_depth": trial.suggest_int("max_depth", 2, 80),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 800, step=20), #modify this for large datasets, causes overfittin if too low
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0001, 1000, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0001, 1000, log=True),
        #"min_split_gain": trial.suggest_float("min_split_gain", 0, 3),
        "subsample": trial.suggest_float("subsample", 0.05, 1, step=0.05),
        "subsample_freq": trial.suggest_categorical("subsample_freq", [0,1]), #[0,1] bagging
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1, step=0.1),
    }
    
    #min_data_in_leaf # min_child_samples
    
    # Aqui defino funcion a minimizar, usare el RMSE (Root Mean Squared Error)
    def rmse(y_val,y_pred):
        is_higher_better = False
        name = "rmse"
        value = mean_squared_error(y_val,y_pred, squared=False)
        return name, value, is_higher_better
    
    cv_scores = np.empty(5)
    
    tscv = TimeSeriesSplit(n_splits=5, max_train_size=None)
    
    for idx , (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        #print(y_train.shape, y_val.shape)

        model = LGBMRegressor(objective="regression", silent=True,**param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=rmse,
            early_stopping_rounds=100,
            categorical_feature="auto",   # cat_idx, #specifiy categorial features.
            callbacks=[LightGBMPruningCallback(trial, "l2", report_interval=20)],  # Add a pruning callback
            verbose=0)

        preds =  model.predict(X_val)
        rmse = mean_squared_error(y_val,preds, squared=False)

        cv_scores[idx] = rmse

    
    return np.mean(cv_scores)
```


```python
study = optuna.create_study(direction="minimize", study_name="LGBM Regressor",sampler=TPESampler(),
                            pruner=optuna.pruners.PercentilePruner(50, n_startup_trials=5, n_warmup_steps=50))

#study.enqueue_trial(try_this_first)
func = lambda trial: objective(trial, train_windows, train_labels)
study.optimize(func, n_trials=100)
```

    [32m[I 2023-04-14 20:06:45,615][0m A new study created in memory with name: LGBM Regressor[0m
    [32m[I 2023-04-14 20:06:46,696][0m Trial 0 finished with value: 0.8109004958352836 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 8325, 'learning_rate': 0.2136801933659405, 'num_leaves': 122, 'max_depth': 24, 'min_child_samples': 290, 'reg_alpha': 0.4089594656291566, 'reg_lambda': 233.41962530839834, 'subsample': 0.55, 'subsample_freq': 1, 'colsample_bytree': 0.5}. Best is trial 0 with value: 0.8109004958352836.[0m
    [32m[I 2023-04-14 20:06:47,790][0m Trial 1 finished with value: 0.8149075483133782 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 2625, 'learning_rate': 0.20374443271553472, 'num_leaves': 24, 'max_depth': 65, 'min_child_samples': 550, 'reg_alpha': 0.00011716269934726968, 'reg_lambda': 0.005714841791728027, 'subsample': 0.45, 'subsample_freq': 0, 'colsample_bytree': 0.8}. Best is trial 0 with value: 0.8109004958352836.[0m
    [32m[I 2023-04-14 20:06:48,981][0m Trial 2 finished with value: 0.7299322021858232 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 525, 'learning_rate': 0.27680866331252735, 'num_leaves': 38, 'max_depth': 30, 'min_child_samples': 270, 'reg_alpha': 0.0035782538305419846, 'reg_lambda': 17.845327449332153, 'subsample': 0.6000000000000001, 'subsample_freq': 0, 'colsample_bytree': 0.7}. Best is trial 2 with value: 0.7299322021858232.[0m
    [32m[I 2023-04-14 20:06:53,661][0m Trial 3 finished with value: 0.7167180406329721 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 1625, 'learning_rate': 0.059660208844652896, 'num_leaves': 119, 'max_depth': 5, 'min_child_samples': 90, 'reg_alpha': 0.0005815773434101997, 'reg_lambda': 217.92130270023952, 'subsample': 0.7000000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.4}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:06:54,228][0m Trial 4 finished with value: 1.0204431478471587 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 5825, 'learning_rate': 0.24944132414726475, 'num_leaves': 110, 'max_depth': 6, 'min_child_samples': 670, 'reg_alpha': 0.3440221518083674, 'reg_lambda': 0.022656206463668692, 'subsample': 0.6000000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.9000000000000001}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:06:54,689][0m Trial 5 finished with value: 1.136575957699749 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 3825, 'learning_rate': 0.28705810714585883, 'num_leaves': 74, 'max_depth': 23, 'min_child_samples': 330, 'reg_alpha': 0.00117464219603884, 'reg_lambda': 930.423463062104, 'subsample': 0.15000000000000002, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:06:54,846][0m Trial 6 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:06:55,234][0m Trial 7 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:06:56,536][0m Trial 8 finished with value: 0.8090499737420279 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 2325, 'learning_rate': 0.11283395677878345, 'num_leaves': 119, 'max_depth': 69, 'min_child_samples': 450, 'reg_alpha': 8.173710274957399, 'reg_lambda': 0.0016016396936841674, 'subsample': 0.9000000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.8}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:06:56,717][0m Trial 9 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:06:56,966][0m Trial 10 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:06:57,176][0m Trial 11 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:06:57,615][0m Trial 12 finished with value: 0.9041904425668372 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 25, 'learning_rate': 0.03016855396951179, 'num_leaves': 59, 'max_depth': 40, 'min_child_samples': 170, 'reg_alpha': 0.019585259507660126, 'reg_lambda': 28.15569407509035, 'subsample': 0.7500000000000001, 'subsample_freq': 0, 'colsample_bytree': 0.5}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:06:59,302][0m Trial 13 finished with value: 0.8057430227791377 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 3525, 'learning_rate': 0.2839273135884679, 'num_leaves': 49, 'max_depth': 14, 'min_child_samples': 430, 'reg_alpha': 0.00011383319189951518, 'reg_lambda': 0.6079263943020092, 'subsample': 0.35000000000000003, 'subsample_freq': 0, 'colsample_bytree': 1.0}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:06:59,666][0m Trial 14 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:00,013][0m Trial 15 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:00,252][0m Trial 16 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:00,587][0m Trial 17 finished with value: 0.8149493781609001 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 25, 'learning_rate': 0.18580292763389805, 'num_leaves': 35, 'max_depth': 30, 'min_child_samples': 370, 'reg_alpha': 0.0026861115432249383, 'reg_lambda': 2.871548016458815, 'subsample': 0.8, 'subsample_freq': 0, 'colsample_bytree': 0.4}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:07:00,856][0m Trial 18 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:01,099][0m Trial 19 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:01,496][0m Trial 20 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:02,301][0m Trial 21 finished with value: 0.9311458598267224 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 3625, 'learning_rate': 0.2897041297710464, 'num_leaves': 48, 'max_depth': 16, 'min_child_samples': 610, 'reg_alpha': 0.00011179972251138886, 'reg_lambda': 0.38152607547229983, 'subsample': 0.35000000000000003, 'subsample_freq': 0, 'colsample_bytree': 1.0}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:07:03,237][0m Trial 22 finished with value: 0.8041108367396304 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 3625, 'learning_rate': 0.26733948592440315, 'num_leaves': 29, 'max_depth': 2, 'min_child_samples': 410, 'reg_alpha': 0.00029447281288036225, 'reg_lambda': 3.303977711581764, 'subsample': 0.3, 'subsample_freq': 0, 'colsample_bytree': 1.0}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:07:04,345][0m Trial 23 finished with value: 0.7965476309079488 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 4825, 'learning_rate': 0.2532355942602236, 'num_leaves': 18, 'max_depth': 2, 'min_child_samples': 350, 'reg_alpha': 0.0006355878139971462, 'reg_lambda': 49.62783909762161, 'subsample': 0.25, 'subsample_freq': 0, 'colsample_bytree': 0.9000000000000001}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:07:05,493][0m Trial 24 finished with value: 0.7951368361969313 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 6825, 'learning_rate': 0.26040870592797677, 'num_leaves': 19, 'max_depth': 8, 'min_child_samples': 350, 'reg_alpha': 0.002182715488938396, 'reg_lambda': 54.34387548459513, 'subsample': 0.05, 'subsample_freq': 0, 'colsample_bytree': 0.9000000000000001}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:07:05,706][0m Trial 25 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:06,193][0m Trial 26 finished with value: 1.136575957699749 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 7125, 'learning_rate': 0.22272302205289798, 'num_leaves': 10, 'max_depth': 24, 'min_child_samples': 270, 'reg_alpha': 0.006611179580240883, 'reg_lambda': 84.4892531883775, 'subsample': 0.05, 'subsample_freq': 1, 'colsample_bytree': 0.8}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:07:06,452][0m Trial 27 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:07,129][0m Trial 28 finished with value: 1.01392775561528 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 1325, 'learning_rate': 0.2643214843578808, 'num_leaves': 77, 'max_depth': 47, 'min_child_samples': 510, 'reg_alpha': 0.00037227074686419393, 'reg_lambda': 14.638891286775614, 'subsample': 0.5, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:07:07,397][0m Trial 29 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:07,611][0m Trial 30 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:08,898][0m Trial 31 finished with value: 0.7933250454777272 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 4925, 'learning_rate': 0.2531069435391942, 'num_leaves': 23, 'max_depth': 4, 'min_child_samples': 350, 'reg_alpha': 0.0006064327965463573, 'reg_lambda': 70.68233732837147, 'subsample': 0.05, 'subsample_freq': 0, 'colsample_bytree': 0.9000000000000001}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:07:10,098][0m Trial 32 finished with value: 0.7964545102673657 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 6125, 'learning_rate': 0.2678466575477564, 'num_leaves': 27, 'max_depth': 8, 'min_child_samples': 370, 'reg_alpha': 0.00027533885040641204, 'reg_lambda': 141.4087349096171, 'subsample': 0.05, 'subsample_freq': 0, 'colsample_bytree': 0.9000000000000001}. Best is trial 3 with value: 0.7167180406329721.[0m
    [32m[I 2023-04-14 20:07:11,722][0m Trial 33 finished with value: 0.7113949471652715 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 2925, 'learning_rate': 0.24557286207708312, 'num_leaves': 129, 'max_depth': 20, 'min_child_samples': 190, 'reg_alpha': 0.0012961636790348608, 'reg_lambda': 50.85143189171674, 'subsample': 0.15000000000000002, 'subsample_freq': 0, 'colsample_bytree': 0.8}. Best is trial 33 with value: 0.7113949471652715.[0m
    [32m[I 2023-04-14 20:07:11,970][0m Trial 34 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:12,259][0m Trial 35 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:12,484][0m Trial 36 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:13,040][0m Trial 37 finished with value: 1.136575957699749 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 4225, 'learning_rate': 0.24705260582822197, 'num_leaves': 128, 'max_depth': 21, 'min_child_samples': 310, 'reg_alpha': 0.0011167068598794217, 'reg_lambda': 19.695157765405053, 'subsample': 0.1, 'subsample_freq': 1, 'colsample_bytree': 0.8}. Best is trial 33 with value: 0.7113949471652715.[0m
    [32m[I 2023-04-14 20:07:13,307][0m Trial 38 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:15,038][0m Trial 39 finished with value: 0.7095940226374052 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 725, 'learning_rate': 0.2303245797335545, 'num_leaves': 112, 'max_depth': 5, 'min_child_samples': 70, 'reg_alpha': 0.011219092904157112, 'reg_lambda': 5.849334604493899, 'subsample': 0.55, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 39 with value: 0.7095940226374052.[0m
    [32m[I 2023-04-14 20:07:17,217][0m Trial 40 finished with value: 0.7194841942131924 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 625, 'learning_rate': 0.22742993213225504, 'num_leaves': 110, 'max_depth': 36, 'min_child_samples': 50, 'reg_alpha': 0.013711322387537144, 'reg_lambda': 1.420452133554653, 'subsample': 0.45, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 39 with value: 0.7095940226374052.[0m
    [32m[I 2023-04-14 20:07:17,863][0m Trial 41 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:19,643][0m Trial 42 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:21,483][0m Trial 43 finished with value: 0.708608733923404 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 1725, 'learning_rate': 0.23240661294495385, 'num_leaves': 112, 'max_depth': 46, 'min_child_samples': 90, 'reg_alpha': 0.012952372792597734, 'reg_lambda': 16.249229200039245, 'subsample': 0.6500000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.6000000000000001}. Best is trial 43 with value: 0.708608733923404.[0m
    [32m[I 2023-04-14 20:07:23,858][0m Trial 44 finished with value: 0.7099554024640764 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 1725, 'learning_rate': 0.19160146035159226, 'num_leaves': 102, 'max_depth': 56, 'min_child_samples': 70, 'reg_alpha': 0.009956359791921148, 'reg_lambda': 1.1144678896270823, 'subsample': 0.7000000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.6000000000000001}. Best is trial 43 with value: 0.708608733923404.[0m
    [32m[I 2023-04-14 20:07:24,095][0m Trial 45 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:24,365][0m Trial 46 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:24,584][0m Trial 47 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:27,199][0m Trial 48 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:27,426][0m Trial 49 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:27,692][0m Trial 50 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:31,388][0m Trial 51 finished with value: 0.7154269263154419 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 325, 'learning_rate': 0.2248723970812301, 'num_leaves': 113, 'max_depth': 51, 'min_child_samples': 50, 'reg_alpha': 0.013159808028320716, 'reg_lambda': 1.370747992392315, 'subsample': 0.6500000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 43 with value: 0.708608733923404.[0m
    [32m[I 2023-04-14 20:07:31,739][0m Trial 52 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:33,557][0m Trial 53 finished with value: 0.7072158217779939 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 1325, 'learning_rate': 0.19165060072186244, 'num_leaves': 100, 'max_depth': 49, 'min_child_samples': 90, 'reg_alpha': 0.032988695657414115, 'reg_lambda': 10.414997496878858, 'subsample': 0.6000000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.6000000000000001}. Best is trial 53 with value: 0.7072158217779939.[0m
    [32m[I 2023-04-14 20:07:33,807][0m Trial 54 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:34,267][0m Trial 55 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:34,507][0m Trial 56 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:36,541][0m Trial 57 finished with value: 0.7099976953763563 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 925, 'learning_rate': 0.22268061046953183, 'num_leaves': 108, 'max_depth': 58, 'min_child_samples': 90, 'reg_alpha': 0.23090180094538526, 'reg_lambda': 3.248104504178605, 'subsample': 0.6500000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.6000000000000001}. Best is trial 53 with value: 0.7072158217779939.[0m
    [32m[I 2023-04-14 20:07:36,973][0m Trial 58 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:37,201][0m Trial 59 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:38,924][0m Trial 60 finished with value: 0.7169899573009177 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 1525, 'learning_rate': 0.21799352082707246, 'num_leaves': 157, 'max_depth': 57, 'min_child_samples': 90, 'reg_alpha': 0.05127324367384014, 'reg_lambda': 13.398694033398872, 'subsample': 0.55, 'subsample_freq': 1, 'colsample_bytree': 0.6000000000000001}. Best is trial 53 with value: 0.7072158217779939.[0m
    [32m[I 2023-04-14 20:07:41,100][0m Trial 61 finished with value: 0.7056102350569884 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 325, 'learning_rate': 0.22788363311207221, 'num_leaves': 112, 'max_depth': 65, 'min_child_samples': 70, 'reg_alpha': 0.025438244095786443, 'reg_lambda': 0.5000648107653262, 'subsample': 0.6500000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 61 with value: 0.7056102350569884.[0m
    [32m[I 2023-04-14 20:07:41,581][0m Trial 62 finished with value: 0.7239192244367315 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 25, 'learning_rate': 0.24565987271944334, 'num_leaves': 94, 'max_depth': 66, 'min_child_samples': 130, 'reg_alpha': 0.020968214250815496, 'reg_lambda': 0.4240380211174552, 'subsample': 0.6000000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.6000000000000001}. Best is trial 61 with value: 0.7056102350569884.[0m
    [32m[I 2023-04-14 20:07:43,426][0m Trial 63 finished with value: 0.7093494479468163 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 925, 'learning_rate': 0.23324676955762208, 'num_leaves': 82, 'max_depth': 71, 'min_child_samples': 70, 'reg_alpha': 0.031843341985698485, 'reg_lambda': 0.26268642977424467, 'subsample': 0.5, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 61 with value: 0.7056102350569884.[0m
    [32m[I 2023-04-14 20:07:43,879][0m Trial 64 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:52,223][0m Trial 65 finished with value: 0.8486469177356406 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 725, 'learning_rate': 0.20948638586957838, 'num_leaves': 99, 'max_depth': 71, 'min_child_samples': 10, 'reg_alpha': 0.1052450216986218, 'reg_lambda': 0.2757495260962473, 'subsample': 0.5, 'subsample_freq': 1, 'colsample_bytree': 0.6000000000000001}. Best is trial 61 with value: 0.7056102350569884.[0m
    [32m[I 2023-04-14 20:07:52,468][0m Trial 66 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:56,835][0m Trial 67 finished with value: 0.7439592645517168 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 1325, 'learning_rate': 0.25671941582298513, 'num_leaves': 84, 'max_depth': 65, 'min_child_samples': 30, 'reg_alpha': 0.027791846854451404, 'reg_lambda': 0.10836260113679402, 'subsample': 0.7000000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.6000000000000001}. Best is trial 61 with value: 0.7056102350569884.[0m
    [32m[I 2023-04-14 20:07:57,460][0m Trial 68 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:57,601][0m Trial 69 pruned. Trial was pruned at iteration 59.[0m
    [32m[I 2023-04-14 20:07:57,793][0m Trial 70 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:07:59,831][0m Trial 71 finished with value: 0.7022349235029833 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 2125, 'learning_rate': 0.2514458173401127, 'num_leaves': 128, 'max_depth': 48, 'min_child_samples': 90, 'reg_alpha': 0.0042343839479169615, 'reg_lambda': 8.01369798587411, 'subsample': 0.6500000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.8}. Best is trial 71 with value: 0.7022349235029833.[0m
    [32m[I 2023-04-14 20:08:00,334][0m Trial 72 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:00,711][0m Trial 73 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:05,943][0m Trial 74 finished with value: 0.7693485688758231 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 1225, 'learning_rate': 0.2722009464647596, 'num_leaves': 90, 'max_depth': 55, 'min_child_samples': 30, 'reg_alpha': 0.0031442561062965495, 'reg_lambda': 0.4373880255745155, 'subsample': 0.7500000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.5}. Best is trial 71 with value: 0.7022349235029833.[0m
    [32m[I 2023-04-14 20:08:06,495][0m Trial 75 finished with value: 0.7074483395361082 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 25, 'learning_rate': 0.263099831275711, 'num_leaves': 113, 'max_depth': 41, 'min_child_samples': 90, 'reg_alpha': 0.015483935678601117, 'reg_lambda': 9.90931534590584, 'subsample': 0.6000000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 71 with value: 0.7022349235029833.[0m
    [32m[I 2023-04-14 20:08:06,635][0m Trial 76 pruned. Trial was pruned at iteration 59.[0m
    [32m[I 2023-04-14 20:08:06,767][0m Trial 77 pruned. Trial was pruned at iteration 59.[0m
    [32m[I 2023-04-14 20:08:09,348][0m Trial 78 finished with value: 0.7118259154310951 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 2125, 'learning_rate': 0.27614361186105635, 'num_leaves': 134, 'max_depth': 43, 'min_child_samples': 50, 'reg_alpha': 0.01587475267351817, 'reg_lambda': 4.7008088449571455, 'subsample': 0.6000000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 71 with value: 0.7022349235029833.[0m
    [32m[I 2023-04-14 20:08:09,852][0m Trial 79 finished with value: 0.7250104729197145 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 25, 'learning_rate': 0.24688765669671525, 'num_leaves': 114, 'max_depth': 45, 'min_child_samples': 110, 'reg_alpha': 0.029505142668084742, 'reg_lambda': 41.71008148637968, 'subsample': 0.5, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 71 with value: 0.7022349235029833.[0m
    [32m[I 2023-04-14 20:08:11,752][0m Trial 80 finished with value: 0.7079470509540828 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 1325, 'learning_rate': 0.25498450363462993, 'num_leaves': 127, 'max_depth': 53, 'min_child_samples': 70, 'reg_alpha': 0.004120215083982785, 'reg_lambda': 7.974893974174497, 'subsample': 0.55, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 71 with value: 0.7022349235029833.[0m
    [32m[I 2023-04-14 20:08:12,288][0m Trial 81 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:12,729][0m Trial 82 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:12,969][0m Trial 83 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:13,303][0m Trial 84 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:14,894][0m Trial 85 finished with value: 0.7110322827849218 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 525, 'learning_rate': 0.23126823469087615, 'num_leaves': 139, 'max_depth': 41, 'min_child_samples': 130, 'reg_alpha': 0.005750616933181132, 'reg_lambda': 5.356777983541687, 'subsample': 0.7000000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 71 with value: 0.7022349235029833.[0m
    [32m[I 2023-04-14 20:08:15,460][0m Trial 86 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:15,855][0m Trial 87 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:16,358][0m Trial 88 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:24,983][0m Trial 89 finished with value: 0.7663270522002408 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 725, 'learning_rate': 0.24164052008507922, 'num_leaves': 112, 'max_depth': 33, 'min_child_samples': 10, 'reg_alpha': 0.010028599597696086, 'reg_lambda': 1.889796966532381, 'subsample': 0.5, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 71 with value: 0.7022349235029833.[0m
    [32m[I 2023-04-14 20:08:25,207][0m Trial 90 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:27,006][0m Trial 91 finished with value: 0.7103091043108893 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 925, 'learning_rate': 0.22352539830152418, 'num_leaves': 108, 'max_depth': 56, 'min_child_samples': 90, 'reg_alpha': 0.07786735533808328, 'reg_lambda': 2.6389965875743138, 'subsample': 0.6500000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.6000000000000001}. Best is trial 71 with value: 0.7022349235029833.[0m
    [32m[I 2023-04-14 20:08:27,653][0m Trial 92 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:28,071][0m Trial 93 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:30,001][0m Trial 94 finished with value: 0.7156803258551001 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 2625, 'learning_rate': 0.23366665692150243, 'num_leaves': 118, 'max_depth': 60, 'min_child_samples': 110, 'reg_alpha': 0.007499516510774507, 'reg_lambda': 12.02683267885883, 'subsample': 0.7500000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.5}. Best is trial 71 with value: 0.7022349235029833.[0m
    [32m[I 2023-04-14 20:08:31,845][0m Trial 95 finished with value: 0.7038583636195452 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 1925, 'learning_rate': 0.2449161485875714, 'num_leaves': 106, 'max_depth': 53, 'min_child_samples': 90, 'reg_alpha': 0.015865934606789586, 'reg_lambda': 4.287187945832237, 'subsample': 0.6500000000000001, 'subsample_freq': 1, 'colsample_bytree': 0.7}. Best is trial 71 with value: 0.7022349235029833.[0m
    [32m[I 2023-04-14 20:08:33,120][0m Trial 96 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:33,255][0m Trial 97 pruned. Trial was pruned at iteration 59.[0m
    [32m[I 2023-04-14 20:08:33,500][0m Trial 98 pruned. Trial was pruned at iteration 119.[0m
    [32m[I 2023-04-14 20:08:36,711][0m Trial 99 finished with value: 0.718496606821031 and parameters: {'boosting_type': 'gbdt', 'device_type': 'cpu', 'n_estimators': 3425, 'learning_rate': 0.25476699329382246, 'num_leaves': 78, 'max_depth': 44, 'min_child_samples': 30, 'reg_alpha': 0.006442917820858156, 'reg_lambda': 0.9818132921273436, 'subsample': 0.45, 'subsample_freq': 1, 'colsample_bytree': 0.8}. Best is trial 71 with value: 0.7022349235029833.[0m



```python
#mejores hyperparametros, sin variables explicativas mas alla de variable objetivo retrasada.
params = study.best_params
model = LGBMRegressor(objective="regression",random_state=123,**params)
model.fit(train_windows, train_labels)
```




    LGBMRegressor(colsample_bytree=0.8, device_type='cpu',
                  learning_rate=0.2514458173401127, max_depth=48,
                  min_child_samples=90, n_estimators=2125, num_leaves=128,
                  objective='regression', random_state=123,
                  reg_alpha=0.0042343839479169615, reg_lambda=8.01369798587411,
                  subsample=0.6500000000000001, subsample_freq=1)




```python
preds = model.predict(test_windows)
timeseries_report_model(test_labels, preds, tracker, model_name="Experimento 4, LightGBM",
                        seasonality=1, naive=False)
```

    MAE:  0.6045
    RMSE : 0.8177
    MASE : 1.1928
    R2 : 0.5233
    MAPE : 0.2752



```python
plot_importance(model, max_num_features = 20, height=.9, importance_type="gain", figsize=(14,8));
```


    
![png](iron-ore-data-mining_files/iron-ore-data-mining_48_0.png)
    



```python
plt.figure(figsize=(16,8),dpi=100)
plot_time_series(test_labels.index, test_labels, label="Real Values", start=100)
plot_time_series(test_labels.index, preds,label="LightGBM", start=100)
```


    
![png](iron-ore-data-mining_files/iron-ore-data-mining_49_0.png)
    


### VIF approach


```python
vif_data = df.drop(columns=['date', '%_Silica_Concentrate'])
vif_data.head()
```


```python
vif_df = pd.DataFrame(index=vif_data.columns, columns=['VIF'], data=np.ones((22,1))*1000)
max(vif_df.VIF)
```


```python
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(bcolors.BOLD + "Warning: No active frommets remain. Continue?" + bcolors.ENDC)
```


```python

# Prepare data for VIF
vif_data = df.drop(columns=['date', "%_Iron_Concentrate", '%_Silica_Concentrate'])
vif_data = vif_data.astype(float)

# Create a new DataFrame to hold the VIF results
vif_df = pd.DataFrame(index=vif_data.columns, columns=['VIF_0'], data=np.ones((21,1))*1000)
vif_record_df = vif_df

# Counter
k=0

while True:
    # Loop through each column in the DataFrame
    for i, col in enumerate(vif_df.index.to_list()):

        # Calculate the VIF for the current column
        vif = variance_inflation_factor(np.array(vif_data.values), i)
    
        # Add the VIF value to the DataFrame
        vif_df.loc[col, 'VIF_{}'.format(k)] = vif
    
    # Record the VIF column for each iteration
    vif_record_df = pd.concat([vif_record_df, vif_df], axis=1)
    
    
    # Sort the VIF and remove the > 10
    sorted_vif_df = vif_df.sort_values(by='VIF_{}'.format(k), ascending=False)
    print(sorted_vif_df)
    
    if sorted_vif_df['VIF_{}'.format(k)][0] > 10:
        
        # Print out the removed column
        print(bcolors.BOLD + "Removing {} iteration... {}".format(k, sorted_vif_df.index[0]) + bcolors.ENDC)
        
        vif_df = vif_df.drop(index=sorted_vif_df.index[0], axis=0)
        vif_data = vif_data.drop(columns=sorted_vif_df.index[0])
        k+=1
    
    else:
        break

```


```python
vif_record_df
```


```python
p = sns.barplot(data=vif_record_df, x=vif_record_df.index, y="VIF_0")
p.set_xticklabels(
    labels=vif_df.index.tolist(), rotation=30)
```
