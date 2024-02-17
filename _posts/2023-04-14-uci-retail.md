---
layout: post
title: "UCI-Online Retail Dataset"
author: "Frank Hsiung"
categories: post
tags: [post]
image: UCI-cover.png
---

```python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

%matplotlib inline
```


```python
df = pd.read_excel('Online_Retail.xlsx')
```


```python
df.dtypes
```




    InvoiceNo              object
    StockCode              object
    Description            object
    Quantity                int64
    InvoiceDate    datetime64[ns]
    UnitPrice             float64
    CustomerID            float64
    Country                object
    dtype: object




```python
for col in df.columns:
  if df[col].dtype == np.object0:
    df[col] = df[col].astype(str)
```


```python
len(df['InvoiceNo'].unique())
```




    25900




```python
df['InvoiceNo'].apply(lambda x: re.sub('[^0-9]', '', str(x)))
#df[(~df['InvoiceNo'].str.contains('[a-zA-Z]').isna()) & (df['Quantity']>0)]
```




    0         536365
    1         536365
    2         536365
    3         536365
    4         536365
               ...  
    541904    581587
    541905    581587
    541906    581587
    541907    581587
    541908    581587
    Name: InvoiceNo, Length: 541909, dtype: object




```python
df.head()
```





  <div id="df-ea2a98a4-29aa-48e8-b788-dd97eba293a9">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536365</td>
      <td>84029G</td>
      <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536365</td>
      <td>84029E</td>
      <td>RED WOOLLY HOTTIE WHITE HEART.</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ea2a98a4-29aa-48e8-b788-dd97eba293a9')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ea2a98a4-29aa-48e8-b788-dd97eba293a9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ea2a98a4-29aa-48e8-b788-dd97eba293a9');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df.describe()
```





  <div id="df-541c98cb-2c12-4a1e-a675-7c8a5cb97434">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>541909.000000</td>
      <td>541909.000000</td>
      <td>406829.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.552250</td>
      <td>4.611114</td>
      <td>15287.690570</td>
    </tr>
    <tr>
      <th>std</th>
      <td>218.081158</td>
      <td>96.759853</td>
      <td>1713.600303</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-80995.000000</td>
      <td>-11062.060000</td>
      <td>12346.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.250000</td>
      <td>13953.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.080000</td>
      <td>15152.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10.000000</td>
      <td>4.130000</td>
      <td>16791.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80995.000000</td>
      <td>38970.000000</td>
      <td>18287.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-541c98cb-2c12-4a1e-a675-7c8a5cb97434')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-541c98cb-2c12-4a1e-a675-7c8a5cb97434 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-541c98cb-2c12-4a1e-a675-7c8a5cb97434');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
sns.heatmap(data=df.isnull(), cmap='viridis')
```




    <Axes: >




    
![png](/images/UCI_retail_files/UCI_retail_8_1.png)
    


From the heatmap of missing value, we know there's significant missing on Customer's ID, which may cause potential insufficient on our analysis based on Customer(since we may have problem to groupby CustomerID)

#### Data Processing


```python
# Create InvoicePrice to gather summation of total invoice price
df['InvoicePrice'] = df['UnitPrice']*df['Quantity']
df_gb_CustomerID = df.groupby('CustomerID').mean()['InvoicePrice'].reset_index()
df_gb_CustomerID.rename({'InvoicePrice':'AvgInvoicePrice'}, axis=1, inplace=True)
df = df.merge(df_gb_CustomerID, on='CustomerID', how='left',suffixes=(False, False))
```

    <ipython-input-10-148f12872dcd>:3: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
      df_gb_CustomerID = df.groupby('CustomerID').mean()['InvoicePrice'].reset_index()





```python
# Groupby Country
a = df.groupby(['Country'])[['AvgInvoicePrice']].mean().sort_values('AvgInvoicePrice', ascending=False)
b = df.groupby(['Country'])[['InvoiceNo']].count().sort_values('InvoiceNo', ascending=False)
c = df.groupby(['Country'])[['InvoicePrice']].sum().sort_values('InvoicePrice', ascending=False)

df_gb_Country = a.join(b)
df_gb_Country = df_gb_Country.join(c)
df_gb_Country.rename(columns={'InvoiceNo': 'NumInvoice', 'InvoicePrice':'TotalInvoicePrice'}, inplace=True) 
df_gb_Country
```





  <div id="df-6c72512e-3a1c-41b3-8cf2-a69ebeb78177">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>AvgInvoicePrice</th>
      <th>NumInvoice</th>
      <th>TotalInvoicePrice</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Netherlands</th>
      <td>120.059696</td>
      <td>2371</td>
      <td>284661.540</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>108.540785</td>
      <td>1259</td>
      <td>137077.270</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>98.716816</td>
      <td>358</td>
      <td>35340.620</td>
    </tr>
    <tr>
      <th>Sweden</th>
      <td>79.211926</td>
      <td>462</td>
      <td>36595.910</td>
    </tr>
    <tr>
      <th>Lithuania</th>
      <td>47.458857</td>
      <td>35</td>
      <td>1661.060</td>
    </tr>
    <tr>
      <th>Denmark</th>
      <td>45.721211</td>
      <td>389</td>
      <td>18768.140</td>
    </tr>
    <tr>
      <th>Singapore</th>
      <td>39.827031</td>
      <td>229</td>
      <td>9120.390</td>
    </tr>
    <tr>
      <th>Lebanon</th>
      <td>37.641778</td>
      <td>45</td>
      <td>1693.880</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>35.737500</td>
      <td>32</td>
      <td>1143.600</td>
    </tr>
    <tr>
      <th>EIRE</th>
      <td>33.438239</td>
      <td>8196</td>
      <td>263276.820</td>
    </tr>
    <tr>
      <th>Norway</th>
      <td>32.378877</td>
      <td>1086</td>
      <td>35163.460</td>
    </tr>
    <tr>
      <th>Greece</th>
      <td>32.263836</td>
      <td>146</td>
      <td>4710.520</td>
    </tr>
    <tr>
      <th>Bahrain</th>
      <td>32.258824</td>
      <td>19</td>
      <td>548.400</td>
    </tr>
    <tr>
      <th>Finland</th>
      <td>32.124806</td>
      <td>695</td>
      <td>22326.740</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>29.730770</td>
      <td>2002</td>
      <td>56385.350</td>
    </tr>
    <tr>
      <th>Israel</th>
      <td>27.977000</td>
      <td>297</td>
      <td>7907.820</td>
    </tr>
    <tr>
      <th>United Arab Emirates</th>
      <td>27.974706</td>
      <td>68</td>
      <td>1902.280</td>
    </tr>
    <tr>
      <th>Channel Islands</th>
      <td>26.499063</td>
      <td>758</td>
      <td>20086.290</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>25.987371</td>
      <td>401</td>
      <td>10154.320</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>24.280662</td>
      <td>151</td>
      <td>3666.380</td>
    </tr>
    <tr>
      <th>Iceland</th>
      <td>23.681319</td>
      <td>182</td>
      <td>4310.000</td>
    </tr>
    <tr>
      <th>Czech Republic</th>
      <td>23.590667</td>
      <td>30</td>
      <td>707.720</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>23.348943</td>
      <td>9495</td>
      <td>221698.210</td>
    </tr>
    <tr>
      <th>France</th>
      <td>23.167217</td>
      <td>8557</td>
      <td>197403.900</td>
    </tr>
    <tr>
      <th>Spain</th>
      <td>21.832490</td>
      <td>2533</td>
      <td>54774.580</td>
    </tr>
    <tr>
      <th>European Community</th>
      <td>21.176230</td>
      <td>61</td>
      <td>1291.750</td>
    </tr>
    <tr>
      <th>Poland</th>
      <td>21.152903</td>
      <td>341</td>
      <td>7213.140</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>21.034259</td>
      <td>803</td>
      <td>16890.510</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>20.305015</td>
      <td>2069</td>
      <td>40910.960</td>
    </tr>
    <tr>
      <th>Cyprus</th>
      <td>19.926360</td>
      <td>622</td>
      <td>12946.290</td>
    </tr>
    <tr>
      <th>Malta</th>
      <td>19.728110</td>
      <td>127</td>
      <td>2505.470</td>
    </tr>
    <tr>
      <th>Portugal</th>
      <td>19.635007</td>
      <td>1519</td>
      <td>29367.020</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>18.702086</td>
      <td>495478</td>
      <td>8187806.364</td>
    </tr>
    <tr>
      <th>RSA</th>
      <td>17.281207</td>
      <td>58</td>
      <td>1002.310</td>
    </tr>
    <tr>
      <th>Saudi Arabia</th>
      <td>13.117000</td>
      <td>10</td>
      <td>131.170</td>
    </tr>
    <tr>
      <th>Unspecified</th>
      <td>10.930615</td>
      <td>446</td>
      <td>4749.790</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>5.948179</td>
      <td>291</td>
      <td>1730.920</td>
    </tr>
    <tr>
      <th>Hong Kong</th>
      <td>NaN</td>
      <td>288</td>
      <td>10117.040</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6c72512e-3a1c-41b3-8cf2-a69ebeb78177')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-6c72512e-3a1c-41b3-8cf2-a69ebeb78177 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6c72512e-3a1c-41b3-8cf2-a69ebeb78177');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




We notice there's a problem showing avgerage of Invoice order and sum of Invoice price, therefore we dig in to try to give more clear picture about order from **Hong Kong**.


```python
df[df['Country']=='Hong Kong'].groupby('InvoiceDate').mean()
```

    <ipython-input-12-b8b07b28a4da>:1: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
      df[df['Country']=='Hong Kong'].groupby('InvoiceDate').mean()






  <div id="df-83a4650f-8f15-419a-843e-bd7b904dd556">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>InvoicePrice</th>
      <th>AvgInvoicePrice</th>
    </tr>
    <tr>
      <th>InvoiceDate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-24 14:24:00</th>
      <td>19.666667</td>
      <td>3.137895</td>
      <td>NaN</td>
      <td>42.802807</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-03-15 09:44:00</th>
      <td>-1.000000</td>
      <td>2583.760000</td>
      <td>NaN</td>
      <td>-2583.760000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-03-15 09:50:00</th>
      <td>1.000000</td>
      <td>2583.760000</td>
      <td>NaN</td>
      <td>2583.760000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-04-12 09:28:00</th>
      <td>22.875000</td>
      <td>3.544062</td>
      <td>NaN</td>
      <td>48.113750</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-05-13 14:09:00</th>
      <td>13.569892</td>
      <td>2.343656</td>
      <td>NaN</td>
      <td>21.141505</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-06-22 10:27:00</th>
      <td>33.562500</td>
      <td>3.715000</td>
      <td>NaN</td>
      <td>39.622500</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-08-23 09:36:00</th>
      <td>1.000000</td>
      <td>160.000000</td>
      <td>NaN</td>
      <td>160.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-08-23 09:38:00</th>
      <td>15.312500</td>
      <td>5.307292</td>
      <td>NaN</td>
      <td>55.290625</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-09-19 16:13:00</th>
      <td>-1.000000</td>
      <td>2653.950000</td>
      <td>NaN</td>
      <td>-2653.950000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-09-19 16:14:00</th>
      <td>1.000000</td>
      <td>2653.950000</td>
      <td>NaN</td>
      <td>2653.950000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-10-04 13:43:00</th>
      <td>-1.000000</td>
      <td>10.950000</td>
      <td>NaN</td>
      <td>-10.950000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-10-18 12:17:00</th>
      <td>8.740741</td>
      <td>3.694815</td>
      <td>NaN</td>
      <td>15.682222</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-10-28 08:20:00</th>
      <td>20.857143</td>
      <td>2.678571</td>
      <td>NaN</td>
      <td>44.442857</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-11-14 13:26:00</th>
      <td>-1.000000</td>
      <td>326.100000</td>
      <td>NaN</td>
      <td>-326.100000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-11-14 13:27:00</th>
      <td>1.000000</td>
      <td>326.100000</td>
      <td>NaN</td>
      <td>326.100000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-83a4650f-8f15-419a-843e-bd7b904dd556')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-83a4650f-8f15-419a-843e-bd7b904dd556 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-83a4650f-8f15-419a-843e-bd7b904dd556');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Q1. Which region is generating the highest revenue, and which region is generating the lowest?


```python
# create a 1x3 subplot grid
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

# create a horizontal barplot for each feature and set the axes for each plot
sns.barplot(x=df_gb_Country.head(5).index, y='AvgInvoicePrice', ax=axes[0], color='blue', data=df_gb_Country.head(5))
sns.barplot(x=df_gb_Country.head(5).index, y='NumInvoice', ax=axes[1], color='orange', data=df_gb_Country.head(5))
sns.barplot(x=df_gb_Country.head(5).index, y='TotalInvoicePrice', ax=axes[2], color='green', data=df_gb_Country.head(5))
plt.subplots_adjust(wspace=0.4)
axes[0].set_xticks(range(5))
axes[0].set_xticklabels(df_gb_Country.head(5).index.tolist(), rotation=30)
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(df_gb_Country.head(5).index.tolist(), rotation=30)
axes[2].set_xticks(range(5))
axes[2].set_xticklabels(df_gb_Country.head(5).index.tolist(), rotation=30)
```




    [Text(0, 0, 'Netherlands'),
     Text(1, 0, 'Australia'),
     Text(2, 0, 'Japan'),
     Text(3, 0, 'Sweden'),
     Text(4, 0, 'Lithuania')]




    
![png](/images/UCI_retail_files/UCI_retail_17_1.png)
    



```python
# create a 1x3 subplot grid
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,5))

#  Sort by Totalavenue
df_gb_Country = df_gb_Country.sort_values('TotalInvoicePrice', ascending=False)

# create a horizontal barplot for each feature and set the axes for each plot
sns.barplot(x=df_gb_Country.head(15).index, y='AvgInvoicePrice', ax=axes[0], color='blue', data=df_gb_Country.head(15))
sns.barplot(x=df_gb_Country.head(15).index, y='NumInvoice', ax=axes[1], color='orange', data=df_gb_Country.head(15))
sns.barplot(x=df_gb_Country.head(15).index, y='TotalInvoicePrice', ax=axes[2], color='green', data=df_gb_Country.head(15))
plt.subplots_adjust(wspace=0.4, hspace=1)
plt.suptitle("Top15 AvgInvoicePrice Country", )
axes[0].set_xlabel('')
axes[0].set_xticks(range(15))
axes[0].set_xticklabels(df_gb_Country.head(15).index.tolist(), rotation=30)
axes[1].set_xlabel('')
axes[1].set_xticks(range(15))
axes[1].set_xticklabels(df_gb_Country.head(15).index.tolist(), rotation=30)
axes[2].set_xticks(range(15))
axes[2].set_xticklabels(df_gb_Country.head(15).index.tolist(), rotation=30)
```




    [Text(0, 0, 'United Kingdom'),
     Text(1, 0, 'Netherlands'),
     Text(2, 0, 'EIRE'),
     Text(3, 0, 'Germany'),
     Text(4, 0, 'France'),
     Text(5, 0, 'Australia'),
     Text(6, 0, 'Switzerland'),
     Text(7, 0, 'Spain'),
     Text(8, 0, 'Belgium'),
     Text(9, 0, 'Sweden'),
     Text(10, 0, 'Japan'),
     Text(11, 0, 'Norway'),
     Text(12, 0, 'Portugal'),
     Text(13, 0, 'Finland'),
     Text(14, 0, 'Channel Islands')]




    
![png](/images/UCI_retail_files/UCI_retail_18_1.png)
    


#### Countries high in Total Revenue: **United Kingdom**, **Netherlands**, **Ireland(EIRE)**, **Germany**, **France**


```python
# create a 1x3 subplot grid
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,5))

# create a horizontal barplot for each feature and set the axes for each plot
sns.barplot(x=df_gb_Country.head(15).index, y='AvgInvoicePrice', ax=axes[0], color='blue', data=df_gb_Country.head(15))
sns.barplot(x=df_gb_Country.head(15).index, y='NumInvoice', ax=axes[1], color='orange', data=df_gb_Country.head(15))
sns.barplot(x=df_gb_Country.head(15).index, y='TotalInvoicePrice', ax=axes[2], color='green', data=df_gb_Country.head(15))
plt.subplots_adjust(wspace=0.4, hspace=1)
plt.suptitle("Top15 AvgInvoicePrice Country", )
axes[0].set_xlabel('')
axes[0].set_xticks(range(15))
axes[0].set_xticklabels(df_gb_Country.head(15).index.tolist(), rotation=30)
axes[1].set_xlabel('')
axes[1].set_xticks(range(15))
axes[1].set_xticklabels(df_gb_Country.head(15).index.tolist(), rotation=30)
axes[2].set_xticks(range(15))
axes[2].set_xticklabels(df_gb_Country.head(15).index.tolist(), rotation=30)

```




    [Text(0, 0, 'United Kingdom'),
     Text(1, 0, 'Netherlands'),
     Text(2, 0, 'EIRE'),
     Text(3, 0, 'Germany'),
     Text(4, 0, 'France'),
     Text(5, 0, 'Australia'),
     Text(6, 0, 'Switzerland'),
     Text(7, 0, 'Spain'),
     Text(8, 0, 'Belgium'),
     Text(9, 0, 'Sweden'),
     Text(10, 0, 'Japan'),
     Text(11, 0, 'Norway'),
     Text(12, 0, 'Portugal'),
     Text(13, 0, 'Finland'),
     Text(14, 0, 'Channel Islands')]




    
![png](/images/UCI_retail_files/UCI_retail_20_1.png)
    


#### Countries high in Avg Order: **Netherlandas**, **Australia**, **Japan**, **Swedan**, **Lithuania**

We can further see from the graph that the company probability has mroe market share in European countrues like **Netherlands** and **Sweden** and **Ireland(EIRE)**, 

However, for countries that we haven't have larger Invoice order base like **Australia** and **Japan**, we still stike a pretty nice total sales performance from them, which brings up a suggestion to **go further in those country** since it can brings values even with fewer amount of Invoice order meaning higher ROI in expansion strategy.



```python
# create a 1x3 subplot grid
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,5))

# create a horizontal barplot for each feature and set the axes for each plot
sns.barplot(x=df_gb_Country[15:30].index, y='AvgInvoicePrice', ax=axes[0], color='blue', data=df_gb_Country[15:30])
sns.barplot(x=df_gb_Country[15:30].index, y='NumInvoice', ax=axes[1], color='orange', data=df_gb_Country[15:30])
sns.barplot(x=df_gb_Country[15:30].index, y='TotalInvoicePrice', ax=axes[2], color='green', data=df_gb_Country[15:30])
plt.subplots_adjust(wspace=0.4, hspace=1)
plt.suptitle("Top15-30 AvgInvoicePrice Country", )
axes[0].set_xlabel('')
axes[0].set_xticks(range(15))
axes[0].set_xticklabels(df_gb_Country[15:30].index.tolist(), rotation=30)
axes[1].set_xlabel('')
axes[1].set_xticks(range(15))
axes[1].set_xticklabels(df_gb_Country[15:30].index.tolist(), rotation=30)
axes[2].set_xticks(range(15))
axes[2].set_xticklabels(df_gb_Country[15:30].index.tolist(), rotation=30)
```




    [Text(0, 0, 'Denmark'),
     Text(1, 0, 'Italy'),
     Text(2, 0, 'Cyprus'),
     Text(3, 0, 'Austria'),
     Text(4, 0, 'Hong Kong'),
     Text(5, 0, 'Singapore'),
     Text(6, 0, 'Israel'),
     Text(7, 0, 'Poland'),
     Text(8, 0, 'Unspecified'),
     Text(9, 0, 'Greece'),
     Text(10, 0, 'Iceland'),
     Text(11, 0, 'Canada'),
     Text(12, 0, 'Malta'),
     Text(13, 0, 'United Arab Emirates'),
     Text(14, 0, 'USA')]




    
![png](/images/UCI_retail_files/UCI_retail_23_1.png)
    


Combined with last bar plot, we can further drive a conclusion that we should foucs more on **how to increase the average invoice sales per order** within those countries which are **high in number of Invoice order but low in average invoice price** like Germany, France, and Spain

#### We further dive into comparing the difference between those who have potential to expand our business and those with high invoice order number but low in average total price per order.

## Q2. What are the difference between those who have high total profitable niche and and those who have low ones but have relatively huge amount of order?


```python
obs_country = ['Japan', 'Australia']
com_country = ['EIRE', 'Germany', 'Spain']

obs_df = df[df['Country'].isin(obs_country)]
com_df = df[df['Country'].isin(com_country)]

display(obs_df.head(), com_df.head())
```



  <div id="df-e0841a57-010c-486d-b2ad-045f3faa1cb4">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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

    .divScroll
    {
    width: 560px;
    overflow:scroll;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>InvoicePrice</th>
      <th>AvgInvoicePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>197</th>
      <td>536389</td>
      <td>22941</td>
      <td>CHRISTMAS LIGHTS 10 REINDEER</td>
      <td>6</td>
      <td>2010-12-01 10:03:00</td>
      <td>8.50</td>
      <td>12431.0</td>
      <td>Australia</td>
      <td>51.0</td>
      <td>26.734958</td>
    </tr>
    <tr>
      <th>198</th>
      <td>536389</td>
      <td>21622</td>
      <td>VINTAGE UNION JACK CUSHION COVER</td>
      <td>8</td>
      <td>2010-12-01 10:03:00</td>
      <td>4.95</td>
      <td>12431.0</td>
      <td>Australia</td>
      <td>39.6</td>
      <td>26.734958</td>
    </tr>
    <tr>
      <th>199</th>
      <td>536389</td>
      <td>21791</td>
      <td>VINTAGE HEADS AND TAILS CARD GAME</td>
      <td>12</td>
      <td>2010-12-01 10:03:00</td>
      <td>1.25</td>
      <td>12431.0</td>
      <td>Australia</td>
      <td>15.0</td>
      <td>26.734958</td>
    </tr>
    <tr>
      <th>200</th>
      <td>536389</td>
      <td>35004C</td>
      <td>SET OF 3 COLOURED  FLYING DUCKS</td>
      <td>6</td>
      <td>2010-12-01 10:03:00</td>
      <td>5.45</td>
      <td>12431.0</td>
      <td>Australia</td>
      <td>32.7</td>
      <td>26.734958</td>
    </tr>
    <tr>
      <th>201</th>
      <td>536389</td>
      <td>35004G</td>
      <td>SET OF 3 GOLD FLYING DUCKS</td>
      <td>4</td>
      <td>2010-12-01 10:03:00</td>
      <td>6.35</td>
      <td>12431.0</td>
      <td>Australia</td>
      <td>25.4</td>
      <td>26.734958</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e0841a57-010c-486d-b2ad-045f3faa1cb4')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-e0841a57-010c-486d-b2ad-045f3faa1cb4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e0841a57-010c-486d-b2ad-045f3faa1cb4');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





  <div id="df-3cc78c2e-b941-41b0-8e54-278583036366">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>InvoicePrice</th>
      <th>AvgInvoicePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1109</th>
      <td>536527</td>
      <td>22809</td>
      <td>SET OF 6 T-LIGHTS SANTA</td>
      <td>6</td>
      <td>2010-12-01 13:04:00</td>
      <td>2.95</td>
      <td>12662.0</td>
      <td>Germany</td>
      <td>17.7</td>
      <td>16.452931</td>
    </tr>
    <tr>
      <th>1110</th>
      <td>536527</td>
      <td>84347</td>
      <td>ROTATING SILVER ANGELS T-LIGHT HLDR</td>
      <td>6</td>
      <td>2010-12-01 13:04:00</td>
      <td>2.55</td>
      <td>12662.0</td>
      <td>Germany</td>
      <td>15.3</td>
      <td>16.452931</td>
    </tr>
    <tr>
      <th>1111</th>
      <td>536527</td>
      <td>84945</td>
      <td>MULTI COLOUR SILVER T-LIGHT HOLDER</td>
      <td>12</td>
      <td>2010-12-01 13:04:00</td>
      <td>0.85</td>
      <td>12662.0</td>
      <td>Germany</td>
      <td>10.2</td>
      <td>16.452931</td>
    </tr>
    <tr>
      <th>1112</th>
      <td>536527</td>
      <td>22242</td>
      <td>5 HOOK HANGER MAGIC TOADSTOOL</td>
      <td>12</td>
      <td>2010-12-01 13:04:00</td>
      <td>1.65</td>
      <td>12662.0</td>
      <td>Germany</td>
      <td>19.8</td>
      <td>16.452931</td>
    </tr>
    <tr>
      <th>1113</th>
      <td>536527</td>
      <td>22244</td>
      <td>3 HOOK HANGER MAGIC GARDEN</td>
      <td>12</td>
      <td>2010-12-01 13:04:00</td>
      <td>1.95</td>
      <td>12662.0</td>
      <td>Germany</td>
      <td>23.4</td>
      <td>16.452931</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3cc78c2e-b941-41b0-8e54-278583036366')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-3cc78c2e-b941-41b0-8e54-278583036366 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3cc78c2e-b941-41b0-8e54-278583036366');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




```python
display(obs_df.groupby(['Country'])['InvoiceNo','Quantity','InvoicePrice'].mean())
display(com_df.groupby(['Country'])['InvoiceNo','Quantity','InvoicePrice'].mean())


```

    <ipython-input-18-ea689b4ccf19>:1: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
      display(obs_df.groupby(['Country'])['InvoiceNo','Quantity','InvoicePrice'].mean())
    <ipython-input-18-ea689b4ccf19>:1: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
      display(obs_df.groupby(['Country'])['InvoiceNo','Quantity','InvoicePrice'].mean())




  <div id="df-ac20c995-a3a4-40c3-9f7a-6898d7a4d273">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>Quantity</th>
      <th>InvoicePrice</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Australia</th>
      <td>66.444003</td>
      <td>108.877895</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>70.441341</td>
      <td>98.716816</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ac20c995-a3a4-40c3-9f7a-6898d7a4d273')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ac20c995-a3a4-40c3-9f7a-6898d7a4d273 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ac20c995-a3a4-40c3-9f7a-6898d7a4d273');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>



    <ipython-input-18-ea689b4ccf19>:2: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
      display(com_df.groupby(['Country'])['InvoiceNo','Quantity','InvoicePrice'].mean())
    <ipython-input-18-ea689b4ccf19>:2: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
      display(com_df.groupby(['Country'])['InvoiceNo','Quantity','InvoicePrice'].mean())




  <div id="df-518cd0ba-50de-4512-8976-bf109146a699">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>Quantity</th>
      <th>InvoicePrice</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EIRE</th>
      <td>17.403245</td>
      <td>32.122599</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>12.369458</td>
      <td>23.348943</td>
    </tr>
    <tr>
      <th>Spain</th>
      <td>10.589814</td>
      <td>21.624390</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-518cd0ba-50de-4512-8976-bf109146a699')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-518cd0ba-50de-4512-8976-bf109146a699 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-518cd0ba-50de-4512-8976-bf109146a699');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>



So one of the most obvious difference can be drawn here: **The average number of Quantity of products and average Total Invoice order are high in those who has high total profit with just few number of order**.


```python
a = obs_df.groupby(['Country','StockCode'])['Quantity'].sum().reset_index()
a = a.groupby('Country').apply(lambda x: x.nlargest(5, 'Quantity')).reset_index(drop=True)

b = com_df.groupby(['Country','StockCode'])['Quantity'].sum().reset_index()
b = b.groupby('Country').apply(lambda x: x.nlargest(5, 'Quantity')).reset_index(drop=True)

display(a, b)
```



  <div id="df-61261100-9c68-4af2-a5bf-2547361bfa6f">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>Country</th>
      <th>StockCode</th>
      <th>Quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>22492</td>
      <td>2916</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Australia</td>
      <td>23084</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Australia</td>
      <td>21915</td>
      <td>1704</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Australia</td>
      <td>21731</td>
      <td>1344</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Australia</td>
      <td>22630</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Japan</td>
      <td>23084</td>
      <td>3401</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Japan</td>
      <td>22489</td>
      <td>1201</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Japan</td>
      <td>22328</td>
      <td>870</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Japan</td>
      <td>22492</td>
      <td>577</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Japan</td>
      <td>22531</td>
      <td>577</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-61261100-9c68-4af2-a5bf-2547361bfa6f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-61261100-9c68-4af2-a5bf-2547361bfa6f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-61261100-9c68-4af2-a5bf-2547361bfa6f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





  <div id="df-3039593b-b0dc-40f7-b5d9-c1ba2db75abd">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>Country</th>
      <th>StockCode</th>
      <th>Quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EIRE</td>
      <td>22197</td>
      <td>1809</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EIRE</td>
      <td>21212</td>
      <td>1728</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EIRE</td>
      <td>84991</td>
      <td>1536</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EIRE</td>
      <td>21790</td>
      <td>1492</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EIRE</td>
      <td>17084R</td>
      <td>1440</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>22326</td>
      <td>1218</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Germany</td>
      <td>15036</td>
      <td>1164</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Germany</td>
      <td>POST</td>
      <td>1104</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Germany</td>
      <td>20719</td>
      <td>1019</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Germany</td>
      <td>21212</td>
      <td>1002</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Spain</td>
      <td>84997D</td>
      <td>1089</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Spain</td>
      <td>84997C</td>
      <td>1013</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Spain</td>
      <td>20728</td>
      <td>558</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>84879</td>
      <td>417</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Spain</td>
      <td>22384</td>
      <td>406</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3039593b-b0dc-40f7-b5d9-c1ba2db75abd')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-3039593b-b0dc-40f7-b5d9-c1ba2db75abd button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3039593b-b0dc-40f7-b5d9-c1ba2db75abd');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>



Here we know that the distribution of hot items across countries are so different, in the observed countries(**Japan and Australia**), it seems like **23084**, **22492** are popular, while those two items are not there in the top 5 saling record in comparison countries.

Now we further dive deep in what kinds of items are popular across observed countris and comparison countries.


```python

```


```python
display(len(df['StockCode'].unique()))
display(len(df['Description'].unique()))

a = df.groupby(['StockCode']).filter(lambda x: x['Description'].nunique()>1)
a = a.groupby(['StockCode','Description']).sum()
a
```


    4070



    4224


    <ipython-input-20-3beb50f935e1>:5: FutureWarning: The default value of numeric_only in DataFrameGroupBy.sum is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
      a = a.groupby(['StockCode','Description']).sum()






  <div id="df-225a3675-01c2-4e19-a062-abebdb067ffe">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th></th>
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>InvoicePrice</th>
      <th>AvgInvoicePrice</th>
    </tr>
    <tr>
      <th>StockCode</th>
      <th>Description</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">10002</th>
      <th>INFLATABLE POLITICAL GLOBE</th>
      <td>860</td>
      <td>77.15</td>
      <td>723842.0</td>
      <td>759.89</td>
      <td>986.129985</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>177</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">10080</th>
      <th>GROOVY CACTUS INFLATABLE</th>
      <td>303</td>
      <td>9.04</td>
      <td>333014.0</td>
      <td>119.09</td>
      <td>279.914792</td>
    </tr>
    <tr>
      <th>check</th>
      <td>22</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>170</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>gift_0001_10</th>
      <th>nan</th>
      <td>30</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">gift_0001_20</th>
      <th>Dotcomgiftshop Gift Voucher £20.00</th>
      <td>10</td>
      <td>150.38</td>
      <td>0.0</td>
      <td>167.05</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>to push order througha s stock was</th>
      <td>10</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">gift_0001_30</th>
      <th>Dotcomgiftshop Gift Voucher £30.00</th>
      <td>7</td>
      <td>175.53</td>
      <td>0.0</td>
      <td>175.53</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>30</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>3006 rows × 5 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-225a3675-01c2-4e19-a062-abebdb067ffe')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-225a3675-01c2-4e19-a062-abebdb067ffe button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-225a3675-01c2-4e19-a062-abebdb067ffe');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
a = df[df['StockCode']=='23084'].groupby(['Description','Country']).count()[['Quantity']]
a = a.reset_index().merge(df_gb_Country[['NumInvoice']], on='Country')
a['InvoiceRate'] = round(a['Quantity']/a['NumInvoice'], 3)
a
#a.merge(df_gb_Country[['NumInvoice']], on='Country')
```





  <div id="df-443f10b6-2ba0-4b21-9a76-dcab48fbd25d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>Description</th>
      <th>Country</th>
      <th>Quantity</th>
      <th>NumInvoice</th>
      <th>InvoiceRate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Amazon</td>
      <td>United Kingdom</td>
      <td>1</td>
      <td>495478</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>United Kingdom</td>
      <td>888</td>
      <td>495478</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>add stock to allocate online orders</td>
      <td>United Kingdom</td>
      <td>1</td>
      <td>495478</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>allocate stock for dotcom orders ta</td>
      <td>United Kingdom</td>
      <td>1</td>
      <td>495478</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>for online retail orders</td>
      <td>United Kingdom</td>
      <td>1</td>
      <td>495478</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nan</td>
      <td>United Kingdom</td>
      <td>10</td>
      <td>495478</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>temp adjustment</td>
      <td>United Kingdom</td>
      <td>1</td>
      <td>495478</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>website fixed</td>
      <td>United Kingdom</td>
      <td>1</td>
      <td>495478</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Australia</td>
      <td>6</td>
      <td>1259</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>9</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Belgium</td>
      <td>10</td>
      <td>2069</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>10</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Denmark</td>
      <td>3</td>
      <td>389</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>11</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>EIRE</td>
      <td>8</td>
      <td>8196</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Finland</td>
      <td>4</td>
      <td>695</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>13</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>France</td>
      <td>75</td>
      <td>8557</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>14</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Germany</td>
      <td>23</td>
      <td>9495</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>15</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Greece</td>
      <td>1</td>
      <td>146</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>16</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Iceland</td>
      <td>3</td>
      <td>182</td>
      <td>0.016</td>
    </tr>
    <tr>
      <th>17</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Italy</td>
      <td>2</td>
      <td>803</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>18</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Japan</td>
      <td>5</td>
      <td>358</td>
      <td>0.014</td>
    </tr>
    <tr>
      <th>19</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Netherlands</td>
      <td>7</td>
      <td>2371</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>20</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Norway</td>
      <td>2</td>
      <td>1086</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>21</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Poland</td>
      <td>1</td>
      <td>341</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>22</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Portugal</td>
      <td>3</td>
      <td>1519</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>23</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Spain</td>
      <td>2</td>
      <td>2533</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>24</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Sweden</td>
      <td>4</td>
      <td>462</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>25</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Switzerland</td>
      <td>2</td>
      <td>2002</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>26</th>
      <td>RABBIT NIGHT LIGHT</td>
      <td>Unspecified</td>
      <td>2</td>
      <td>446</td>
      <td>0.004</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-443f10b6-2ba0-4b21-9a76-dcab48fbd25d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-443f10b6-2ba0-4b21-9a76-dcab48fbd25d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-443f10b6-2ba0-4b21-9a76-dcab48fbd25d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python

a = df[df['Country'].isin(obs_country)].groupby(['Country', 'StockCode'])[['Quantity']].count().reset_index()
a = a.merge(df_gb_Country[['NumInvoice']], on='Country')
a['TotalInvoicePercentage'] = round(a['Quantity']/a['NumInvoice'], 3)

a = a.groupby(['Country']).apply(lambda x: x.nlargest(5, 'TotalInvoicePercentage')).reset_index(drop=True)
display(a)

b = df[df['StockCode'].isin(a.StockCode)].groupby(['StockCode']).apply(lambda x: x.nlargest(1,'Quantity')).reset_index(drop=True)

a = a.merge(b[['StockCode','Description']], on='StockCode')
a.sort_values(['Country', 'TotalInvoicePercentage'], ascending=False)
```



  <div id="df-9cde871c-f716-4c1f-adfa-b274895057f8">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>Country</th>
      <th>StockCode</th>
      <th>Quantity</th>
      <th>NumInvoice</th>
      <th>TotalInvoicePercentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>22720</td>
      <td>10</td>
      <td>1259</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Australia</td>
      <td>20725</td>
      <td>9</td>
      <td>1259</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Australia</td>
      <td>21731</td>
      <td>9</td>
      <td>1259</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Australia</td>
      <td>21915</td>
      <td>7</td>
      <td>1259</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Australia</td>
      <td>22090</td>
      <td>8</td>
      <td>1259</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Japan</td>
      <td>21218</td>
      <td>7</td>
      <td>358</td>
      <td>0.020</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Japan</td>
      <td>22661</td>
      <td>6</td>
      <td>358</td>
      <td>0.017</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Japan</td>
      <td>22489</td>
      <td>5</td>
      <td>358</td>
      <td>0.014</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Japan</td>
      <td>22662</td>
      <td>5</td>
      <td>358</td>
      <td>0.014</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Japan</td>
      <td>23084</td>
      <td>5</td>
      <td>358</td>
      <td>0.014</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9cde871c-f716-4c1f-adfa-b274895057f8')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9cde871c-f716-4c1f-adfa-b274895057f8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9cde871c-f716-4c1f-adfa-b274895057f8');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>







  <div id="df-df46a6a5-4d97-45b9-9064-aea797a31142">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>Country</th>
      <th>StockCode</th>
      <th>Quantity</th>
      <th>NumInvoice</th>
      <th>TotalInvoicePercentage</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Japan</td>
      <td>21218</td>
      <td>7</td>
      <td>358</td>
      <td>0.020</td>
      <td>RED SPOTTY BISCUIT TIN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Japan</td>
      <td>22661</td>
      <td>6</td>
      <td>358</td>
      <td>0.017</td>
      <td>CHARLOTTE BAG DOLLY GIRL DESIGN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Japan</td>
      <td>22489</td>
      <td>5</td>
      <td>358</td>
      <td>0.014</td>
      <td>PACK OF 12 TRADITIONAL CRAYONS</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Japan</td>
      <td>22662</td>
      <td>5</td>
      <td>358</td>
      <td>0.014</td>
      <td>LUNCH BAG DOLLY GIRL DESIGN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Japan</td>
      <td>23084</td>
      <td>5</td>
      <td>358</td>
      <td>0.014</td>
      <td>RABBIT NIGHT LIGHT</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>22720</td>
      <td>10</td>
      <td>1259</td>
      <td>0.008</td>
      <td>SET OF 3 CAKE TINS PANTRY DESIGN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Australia</td>
      <td>20725</td>
      <td>9</td>
      <td>1259</td>
      <td>0.007</td>
      <td>LUNCH BAG RED SPOTTY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Australia</td>
      <td>21731</td>
      <td>9</td>
      <td>1259</td>
      <td>0.007</td>
      <td>RED TOADSTOOL LED NIGHT LIGHT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Australia</td>
      <td>21915</td>
      <td>7</td>
      <td>1259</td>
      <td>0.006</td>
      <td>RED  HARMONICA IN BOX</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Australia</td>
      <td>22090</td>
      <td>8</td>
      <td>1259</td>
      <td>0.006</td>
      <td>PAPER BUNTING RETROSPOT</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-df46a6a5-4d97-45b9-9064-aea797a31142')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-df46a6a5-4d97-45b9-9064-aea797a31142 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-df46a6a5-4d97-45b9-9064-aea797a31142');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




From here, we further know that for high ROI country:

Japan, top five items are **Biscuit tin**, **Charlotte bag**, **Crayons**, **Lunch bag**, **Night light**

Australia, top five items are **Pantry design**, **Lunch bag**, **Night light**, **Harmonica**, **Paper bunting**


```python
a = df[df['Country'].isin(com_country)].groupby(['Country', 'StockCode'])[['Quantity']].count().reset_index()
a = a.merge(df_gb_Country[['NumInvoice']], on='Country')
a['TotalInvoicePercentage'] = round(a['Quantity']/a['NumInvoice'], 3)

a = a.groupby(['Country']).apply(lambda x: x.nlargest(5, 'TotalInvoicePercentage')).reset_index(drop=True)
display(a)

b = df[df['StockCode'].isin(a.StockCode)].groupby(['StockCode']).apply(lambda x: x.nlargest(1,'Quantity')).reset_index(drop=True)

a = a.merge(b[['StockCode','Description']], on='StockCode')
a.sort_values(['Country','TotalInvoicePercentage'], ascending=False)
```



  <div id="df-1a88e0e2-9e08-425e-9a95-18b6bc7d715e">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>Country</th>
      <th>StockCode</th>
      <th>Quantity</th>
      <th>NumInvoice</th>
      <th>TotalInvoicePercentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EIRE</td>
      <td>C2</td>
      <td>108</td>
      <td>8196</td>
      <td>0.013</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EIRE</td>
      <td>22423</td>
      <td>78</td>
      <td>8196</td>
      <td>0.010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EIRE</td>
      <td>22699</td>
      <td>53</td>
      <td>8196</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EIRE</td>
      <td>85123A</td>
      <td>47</td>
      <td>8196</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EIRE</td>
      <td>21790</td>
      <td>44</td>
      <td>8196</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Germany</td>
      <td>POST</td>
      <td>383</td>
      <td>9495</td>
      <td>0.040</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Germany</td>
      <td>22326</td>
      <td>120</td>
      <td>9495</td>
      <td>0.013</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Germany</td>
      <td>22423</td>
      <td>81</td>
      <td>9495</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Germany</td>
      <td>22328</td>
      <td>78</td>
      <td>9495</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Germany</td>
      <td>22554</td>
      <td>67</td>
      <td>9495</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Spain</td>
      <td>POST</td>
      <td>62</td>
      <td>2533</td>
      <td>0.024</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Spain</td>
      <td>22423</td>
      <td>25</td>
      <td>2533</td>
      <td>0.010</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Spain</td>
      <td>22077</td>
      <td>15</td>
      <td>2533</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>22960</td>
      <td>16</td>
      <td>2533</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Spain</td>
      <td>22326</td>
      <td>12</td>
      <td>2533</td>
      <td>0.005</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1a88e0e2-9e08-425e-9a95-18b6bc7d715e')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1a88e0e2-9e08-425e-9a95-18b6bc7d715e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1a88e0e2-9e08-425e-9a95-18b6bc7d715e');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>







  <div id="df-7c73fe06-6f7c-41c4-a741-9d576547838c">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>Country</th>
      <th>StockCode</th>
      <th>Quantity</th>
      <th>NumInvoice</th>
      <th>TotalInvoicePercentage</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Spain</td>
      <td>POST</td>
      <td>62</td>
      <td>2533</td>
      <td>0.024</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spain</td>
      <td>22423</td>
      <td>25</td>
      <td>2533</td>
      <td>0.010</td>
      <td>REGENCY CAKESTAND 3 TIER</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>22077</td>
      <td>15</td>
      <td>2533</td>
      <td>0.006</td>
      <td>6 RIBBONS RUSTIC CHARM</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Spain</td>
      <td>22960</td>
      <td>16</td>
      <td>2533</td>
      <td>0.006</td>
      <td>JAM MAKING SET WITH JARS</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Spain</td>
      <td>22326</td>
      <td>12</td>
      <td>2533</td>
      <td>0.005</td>
      <td>ROUND SNACK BOXES SET OF4 WOODLAND</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Germany</td>
      <td>POST</td>
      <td>383</td>
      <td>9495</td>
      <td>0.040</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Germany</td>
      <td>22326</td>
      <td>120</td>
      <td>9495</td>
      <td>0.013</td>
      <td>ROUND SNACK BOXES SET OF4 WOODLAND</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Germany</td>
      <td>22423</td>
      <td>81</td>
      <td>9495</td>
      <td>0.009</td>
      <td>REGENCY CAKESTAND 3 TIER</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Germany</td>
      <td>22328</td>
      <td>78</td>
      <td>9495</td>
      <td>0.008</td>
      <td>ROUND SNACK BOXES SET OF 4 FRUITS</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Germany</td>
      <td>22554</td>
      <td>67</td>
      <td>9495</td>
      <td>0.007</td>
      <td>PLASTERS IN TIN WOODLAND ANIMALS</td>
    </tr>
    <tr>
      <th>0</th>
      <td>EIRE</td>
      <td>C2</td>
      <td>108</td>
      <td>8196</td>
      <td>0.013</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EIRE</td>
      <td>22423</td>
      <td>78</td>
      <td>8196</td>
      <td>0.010</td>
      <td>REGENCY CAKESTAND 3 TIER</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EIRE</td>
      <td>22699</td>
      <td>53</td>
      <td>8196</td>
      <td>0.006</td>
      <td>ROSES REGENCY TEACUP AND SAUCER</td>
    </tr>
    <tr>
      <th>5</th>
      <td>EIRE</td>
      <td>85123A</td>
      <td>47</td>
      <td>8196</td>
      <td>0.006</td>
      <td>?</td>
    </tr>
    <tr>
      <th>6</th>
      <td>EIRE</td>
      <td>21790</td>
      <td>44</td>
      <td>8196</td>
      <td>0.005</td>
      <td>VINTAGE SNAP CARDS</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7c73fe06-6f7c-41c4-a741-9d576547838c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7c73fe06-6f7c-41c4-a741-9d576547838c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7c73fe06-6f7c-41c4-a741-9d576547838c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




From here, we further know that for lower ROI country:

Spain, top five items are **Regency**, **Rustic charm ribbon**, **Jam making set**, **Snack box**

Australia, top five items are **Snack box**, **Regency**, **Plasters**

EIRE, top five items are **Regency**, **Vintage snap card**

## Q2. What is the monthly trend of revenue, which months have faced the biggest increase/decrease?

## Global:


```python
# Make new df indexed by InvoiceDate
dt_df = df.set_index('InvoiceDate')


# Group the data by month and calculate the total revenue for each month
monthly_revenue = dt_df['InvoicePrice'].resample('M').sum()

# Calculate the percentage change in revenue between consecutive months
monthly_revenue_pct_change = monthly_revenue.pct_change()

# Create a line plot of the monthly revenue
sns.set_style('whitegrid')
sns.lineplot(x=monthly_revenue.index, y=monthly_revenue.values)
sns.despine()
plt.title('Monthly Revenue')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.show()

# Create a bar plot of the percentage change in revenue
sns.set_style('whitegrid')
sns.barplot(x=monthly_revenue_pct_change.index.month_name(),
            y=monthly_revenue_pct_change.values)
sns.despine()
plt.title('Percentage Change in Monthly Revenue')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Percent Change')
plt.show()
```


    
![png](/images/UCI_retail_files/UCI_retail_42_0.png)
    


    /usr/local/lib/python3.9/dist-packages/seaborn/algorithms.py:98: RuntimeWarning: Mean of empty slice
      boot_dist.append(f(*sample, **func_kwargs))



    
![png](/images/UCI_retail_files/UCI_retail_42_2.png)
    


## Local(United Kingdom)


```python
# Group the data by month and calculate the total revenue for each month
monthly_revenue = dt_df[dt_df['Country']=='United Kingdom']['InvoicePrice'].resample('M').sum()

# Calculate the percentage change in revenue between consecutive months
monthly_revenue_pct_change = monthly_revenue.pct_change()

# Create a line plot of the monthly revenue
sns.set_style('whitegrid')
sns.lineplot(x=monthly_revenue.index, y=monthly_revenue.values)
sns.despine()
plt.title('Monthly Revenue - UK')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.show()

# Create a bar plot of the percentage change in revenue
sns.set_style('whitegrid')
sns.barplot(x=monthly_revenue_pct_change.index.month_name(),
            y=monthly_revenue_pct_change.values)
sns.despine()
plt.title('Percentage Change in Monthly Revenue')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Percent Change - UK')
plt.show()
```


    
![png](/images/UCI_retail_files/UCI_retail_44_0.png)
    


    /usr/local/lib/python3.9/dist-packages/seaborn/algorithms.py:98: RuntimeWarning: Mean of empty slice
      boot_dist.append(f(*sample, **func_kwargs))



    
![png](/images/UCI_retail_files/UCI_retail_44_2.png)
    


## Compact comparison


```python
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10,20))

for i, country in enumerate(['Global', 'United Kingdom', 'Netherlands', 'EIRE', 'Germany', 'France']):
  if i==0:
    # Group the data by month and calculate the total revenue for each month
    monthly_revenue = dt_df['InvoicePrice'].resample('M').sum()
  else:
    monthly_revenue = dt_df[dt_df['Country']==country]['InvoicePrice'].resample('M').sum()

  # Calculate the percentage change in revenue between consecutive months
  monthly_revenue_pct_change = monthly_revenue.pct_change()

  # Create a line plot of the monthly revenue
  sns.lineplot(x=monthly_revenue.index, y=monthly_revenue.values, ax=axes[i][0])
  sns.despine(ax=axes[i][0])
  axes[i][0].set_title(f'Monthly Revenue - {country}')
  axes[i][0].set_xticklabels(axes[i][0].get_xticklabels(), rotation=45)
  axes[i][0].set_xlabel('Month')
  axes[i][0].set_ylabel('Revenue')

  # Create a bar plot of the percentage change in revenue
  sns.barplot(x=monthly_revenue_pct_change.index.month_name(),
              y=monthly_revenue_pct_change.values, ax=axes[i][1])
  sns.despine(ax=axes[i][1])
  axes[i][1].set_title('Percentage Change in Monthly Revenue')
  axes[i][1].set_xticklabels(axes[i][1].get_xticklabels(), rotation=45)
  axes[i][1].set_xlabel('Month')
  axes[i][1].set_ylabel('Percent Change - {}'.format(country))

plt.subplots_adjust(wspace=0.4, hspace=2) 
plt.tight_layout()
plt.show() 
```

    <ipython-input-26-6fbbe781f6ee>:17: UserWarning: FixedFormatter should only be used together with FixedLocator
      axes[i][0].set_xticklabels(axes[i][0].get_xticklabels(), rotation=45)
    /usr/local/lib/python3.9/dist-packages/seaborn/algorithms.py:98: RuntimeWarning: Mean of empty slice
      boot_dist.append(f(*sample, **func_kwargs))
    <ipython-input-26-6fbbe781f6ee>:17: UserWarning: FixedFormatter should only be used together with FixedLocator
      axes[i][0].set_xticklabels(axes[i][0].get_xticklabels(), rotation=45)
    /usr/local/lib/python3.9/dist-packages/seaborn/algorithms.py:98: RuntimeWarning: Mean of empty slice
      boot_dist.append(f(*sample, **func_kwargs))
    <ipython-input-26-6fbbe781f6ee>:17: UserWarning: FixedFormatter should only be used together with FixedLocator
      axes[i][0].set_xticklabels(axes[i][0].get_xticklabels(), rotation=45)
    /usr/local/lib/python3.9/dist-packages/seaborn/algorithms.py:98: RuntimeWarning: Mean of empty slice
      boot_dist.append(f(*sample, **func_kwargs))
    <ipython-input-26-6fbbe781f6ee>:17: UserWarning: FixedFormatter should only be used together with FixedLocator
      axes[i][0].set_xticklabels(axes[i][0].get_xticklabels(), rotation=45)
    /usr/local/lib/python3.9/dist-packages/seaborn/algorithms.py:98: RuntimeWarning: Mean of empty slice
      boot_dist.append(f(*sample, **func_kwargs))
    <ipython-input-26-6fbbe781f6ee>:17: UserWarning: FixedFormatter should only be used together with FixedLocator
      axes[i][0].set_xticklabels(axes[i][0].get_xticklabels(), rotation=45)
    /usr/local/lib/python3.9/dist-packages/seaborn/algorithms.py:98: RuntimeWarning: Mean of empty slice
      boot_dist.append(f(*sample, **func_kwargs))
    <ipython-input-26-6fbbe781f6ee>:17: UserWarning: FixedFormatter should only be used together with FixedLocator
      axes[i][0].set_xticklabels(axes[i][0].get_xticklabels(), rotation=45)
    /usr/local/lib/python3.9/dist-packages/seaborn/algorithms.py:98: RuntimeWarning: Mean of empty slice
      boot_dist.append(f(*sample, **func_kwargs))



    
![png](/images/UCI_retail_files/UCI_retail_46_1.png)
    


There's a direct spike in September for almost all top prfiting country except for Ireland, which provide space to dig deeper if interested.

## Q3. Who are the top customers and how much do they contribute to the total revenue? Is the business dependent on these customers or is the customer base diversified?



```python
customer_revenue = df.groupby('CustomerID')['InvoicePrice'].sum().reset_index()

# Sort the data in descending order by revenue
customer_revenue.sort_values(by='InvoicePrice', ascending=False, inplace=True)

# Calculate the percentage of revenue contributed by each customer
revenue_pct = customer_revenue['InvoicePrice'] / customer_revenue['InvoicePrice'].sum() * 100

print('\nPercentage of total revenue contributed by top 10 customers:')
print(revenue_pct)

# Display the top 10 customers and their revenue contribution
print(customer_revenue.head(10))
```

    
    Percentage of total revenue contributed by top 10 customers:
    1703    3.367311
    4233    3.089596
    3758    2.258803
    1895    1.597248
    55      1.490656
              ...   
    125    -0.013566
    3870   -0.014040
    1384   -0.014364
    2236   -0.019186
    3756   -0.051658
    Name: InvoicePrice, Length: 4372, dtype: float64
          CustomerID  InvoicePrice
    1703     14646.0     279489.02
    4233     18102.0     256438.49
    3758     17450.0     187482.17
    1895     14911.0     132572.62
    55       12415.0     123725.45
    1345     14156.0     113384.14
    3801     17511.0      88125.38
    3202     16684.0      65892.08
    1005     13694.0      62653.10
    2192     15311.0      59419.34



```python
customer_revenue
```





  <div id="df-6c1cd9e3-ab40-4918-9075-f9f58c6d2fc7">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th>CustomerID</th>
      <th>InvoicePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1703</th>
      <td>14646.0</td>
      <td>279489.02</td>
    </tr>
    <tr>
      <th>4233</th>
      <td>18102.0</td>
      <td>256438.49</td>
    </tr>
    <tr>
      <th>3758</th>
      <td>17450.0</td>
      <td>187482.17</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>14911.0</td>
      <td>132572.62</td>
    </tr>
    <tr>
      <th>55</th>
      <td>12415.0</td>
      <td>123725.45</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>125</th>
      <td>12503.0</td>
      <td>-1126.00</td>
    </tr>
    <tr>
      <th>3870</th>
      <td>17603.0</td>
      <td>-1165.30</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>14213.0</td>
      <td>-1192.20</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>15369.0</td>
      <td>-1592.49</td>
    </tr>
    <tr>
      <th>3756</th>
      <td>17448.0</td>
      <td>-4287.63</td>
    </tr>
  </tbody>
</table>
<p>4372 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6c1cd9e3-ab40-4918-9075-f9f58c6d2fc7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-6c1cd9e3-ab40-4918-9075-f9f58c6d2fc7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6c1cd9e3-ab40-4918-9075-f9f58c6d2fc7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Q4. What is the percentage of customers who are repeating their orders? Are they ordering the same products or different?


```python
# Count the number of unique customers
unique_customers = df['CustomerID'].nunique()

# Count the number of customers with multiple orders
repeat_customers = (df.groupby('CustomerID').size() > 1).sum()

# Calculate the percentage of repeat customers
repeat_customer_pct = (repeat_customers / unique_customers) * 100

print(f"Percentage of repeat customers: {repeat_customer_pct:.2f}%")
```

    Percentage of repeat customers: 98.19%



```python
# Group the data by CustomerID and StockCode
customer_orders = df.groupby(['CustomerID', 'StockCode']).size()

# Count the number of customers who have ordered the same product multiple times
repeat_product_customers = len(df[df['CustomerID'].isin(customer_orders.reset_index()['CustomerID'].unique())]['CustomerID'].unique())

# Count the number of customers who have ordered multiple products
diversified_customers = (customer_orders.groupby('CustomerID').size() > 1).sum()

print(f"Percentage of customers who are repeating their orders for the same product: {(repeat_product_customers / unique_customers) * 100:.2f}%")
print(f"Percentage of customers who are diversified: {(diversified_customers / unique_customers) * 100:.2f}%")
```

    Percentage of customers who are repeating their orders for the same product: 100.00%
    Percentage of customers who are diversified: 97.67%


## Q5. For the repeat customers, how long does it take for them to place the next order after being delivered the previous one?


```python
# Create a DataFrame of customer order dates
customer_order_dates = df[['CustomerID', 'InvoiceDate']].drop_duplicates()

# Sort the DataFrame by CustomerID and InvoiceDate
customer_order_dates.sort_values(['CustomerID', 'InvoiceDate'], inplace=True)

# Group the DataFrame by CustomerID and calculate the time difference between consecutive orders
customer_order_dates['time_diff'] = customer_order_dates.groupby('CustomerID')['InvoiceDate'].diff()

# Remove the first order for each customer, since there is no previous order to calculate the time difference with
customer_order_dates = customer_order_dates.dropna()

# Display the time differences between consecutive orders for repeat customers
repeat_customers = customer_order_dates['CustomerID'].value_counts()[customer_order_dates['CustomerID'].value_counts() > 1]
mean_time_diff = customer_order_dates[customer_order_dates['CustomerID'].isin(repeat_customers.index)]['time_diff'].mean()

print('Mean time difference between orders for repeat customers:', mean_time_diff)

#for customer in repeat_customers.index:
#    print('Customer ID:', customer)
#    print('Time between consecutive orders:')
#    print(customer_order_dates[customer_order_dates['CustomerID'] == customer]['time_diff'])
```

    Mean time difference between orders for repeat customers: 30 days 03:31:03.576347661



```python
# Group the data by customer ID and sort the orders by delivery date
grouped = df.groupby('CustomerID').apply(lambda x: x.sort_values('InvoiceDate'))

# Calculate the time difference between consecutive orders for each customer
grouped['TimeSinceLastOrder'] = grouped['InvoiceDate'].diff()

# Filter the data to only include repeat customers
grouped[grouped['TimeSinceLastOrder'].notnull()]
```





  <div id="df-0f137c4b-de50-478f-aa18-a663c8639f58">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe {
        display: block;
        width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        margin-bottom: 5px;
        scrollbar-width: thin;
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
      <th></th>
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>InvoicePrice</th>
      <th>AvgInvoicePrice</th>
      <th>TimeSinceLastOrder</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12346.0</th>
      <th>61624</th>
      <td>C541433</td>
      <td>23166</td>
      <td>MEDIUM CERAMIC TOP STORAGE JAR</td>
      <td>-74215</td>
      <td>2011-01-18 10:17:00</td>
      <td>1.04</td>
      <td>12346.0</td>
      <td>United Kingdom</td>
      <td>-77183.60</td>
      <td>0.000000</td>
      <td>0 days 00:16:00</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">12347.0</th>
      <th>14938</th>
      <td>537626</td>
      <td>85116</td>
      <td>BLACK CANDELABRA T-LIGHT HOLDER</td>
      <td>12</td>
      <td>2010-12-07 14:57:00</td>
      <td>2.10</td>
      <td>12347.0</td>
      <td>Iceland</td>
      <td>25.20</td>
      <td>23.681319</td>
      <td>-42 days +04:40:00</td>
    </tr>
    <tr>
      <th>14968</th>
      <td>537626</td>
      <td>20782</td>
      <td>CAMOUFLAGE EAR MUFF HEADPHONES</td>
      <td>6</td>
      <td>2010-12-07 14:57:00</td>
      <td>5.49</td>
      <td>12347.0</td>
      <td>Iceland</td>
      <td>32.94</td>
      <td>23.681319</td>
      <td>0 days 00:00:00</td>
    </tr>
    <tr>
      <th>14967</th>
      <td>537626</td>
      <td>20780</td>
      <td>BLACK EAR MUFF HEADPHONES</td>
      <td>12</td>
      <td>2010-12-07 14:57:00</td>
      <td>4.65</td>
      <td>12347.0</td>
      <td>Iceland</td>
      <td>55.80</td>
      <td>23.681319</td>
      <td>0 days 00:00:00</td>
    </tr>
    <tr>
      <th>14966</th>
      <td>537626</td>
      <td>84558A</td>
      <td>3D DOG PICTURE PLAYING CARDS</td>
      <td>24</td>
      <td>2010-12-07 14:57:00</td>
      <td>2.95</td>
      <td>12347.0</td>
      <td>Iceland</td>
      <td>70.80</td>
      <td>23.681319</td>
      <td>0 days 00:00:00</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">18287.0</th>
      <th>392732</th>
      <td>570715</td>
      <td>21481</td>
      <td>FAWN BLUE HOT WATER BOTTLE</td>
      <td>4</td>
      <td>2011-10-12 10:23:00</td>
      <td>3.75</td>
      <td>18287.0</td>
      <td>United Kingdom</td>
      <td>15.00</td>
      <td>26.246857</td>
      <td>0 days 00:00:00</td>
    </tr>
    <tr>
      <th>392725</th>
      <td>570715</td>
      <td>21824</td>
      <td>PAINTED METAL STAR WITH HOLLY BELLS</td>
      <td>24</td>
      <td>2011-10-12 10:23:00</td>
      <td>0.39</td>
      <td>18287.0</td>
      <td>United Kingdom</td>
      <td>9.36</td>
      <td>26.246857</td>
      <td>0 days 00:00:00</td>
    </tr>
    <tr>
      <th>423940</th>
      <td>573167</td>
      <td>21824</td>
      <td>PAINTED METAL STAR WITH HOLLY BELLS</td>
      <td>48</td>
      <td>2011-10-28 09:29:00</td>
      <td>0.39</td>
      <td>18287.0</td>
      <td>United Kingdom</td>
      <td>18.72</td>
      <td>26.246857</td>
      <td>15 days 23:06:00</td>
    </tr>
    <tr>
      <th>423939</th>
      <td>573167</td>
      <td>23264</td>
      <td>SET OF 3 WOODEN SLEIGH DECORATIONS</td>
      <td>36</td>
      <td>2011-10-28 09:29:00</td>
      <td>1.25</td>
      <td>18287.0</td>
      <td>United Kingdom</td>
      <td>45.00</td>
      <td>26.246857</td>
      <td>0 days 00:00:00</td>
    </tr>
    <tr>
      <th>423941</th>
      <td>573167</td>
      <td>21014</td>
      <td>SWISS CHALET TREE DECORATION</td>
      <td>24</td>
      <td>2011-10-28 09:29:00</td>
      <td>0.29</td>
      <td>18287.0</td>
      <td>United Kingdom</td>
      <td>6.96</td>
      <td>26.246857</td>
      <td>0 days 00:00:00</td>
    </tr>
  </tbody>
</table>
<p>406828 rows × 11 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0f137c4b-de50-478f-aa18-a663c8639f58')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0f137c4b-de50-478f-aa18-a663c8639f58 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0f137c4b-de50-478f-aa18-a663c8639f58');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Q6. What revenue is being generated from the customers who have ordered more than once?


```python
# Create a DataFrame of all orders made by repeat customers
repeat_orders = df[df['CustomerID'].isin(repeat_customers.index)]

# Calculate the total revenue for each repeat customer
customer_revenue = repeat_orders.groupby('InvoiceNo')['InvoicePrice'].sum()

# Sum the revenue generated by all repeat customers
total_revenue_repeat_customers = customer_revenue.sum()

# Sum the revenue generated by all customers
total_revenue_cutomers = df.groupby('InvoiceNo')['InvoicePrice'].sum().sum()

print('Total revenue generated from repeat customers: {}'.format(total_revenue_repeat_customers))
print('Total revenue percentage generated from repeat customers: {}%'.format(total_revenue_repeat_customers/total_revenue_cutomers*100))
```

    Total revenue generated from repeat customers: 7415862.7930000005
    Total revenue percentage generated from repeat customers: 76.07770372409387%



```python
# Group the customer_order_dates DataFrame by CustomerID and count the number of orders for each customer
customer_order_counts = customer_order_dates.groupby('CustomerID')['InvoiceDate'].count().reset_index()

# Rename the InvoiceDate column to order_count
customer_order_counts = customer_order_counts.rename(columns={'InvoiceDate': 'order_count'})

# Sort the customer_order_counts DataFrame by order_count in descending order
customer_order_counts = customer_order_counts.sort_values('order_count', ascending=False)

# Merge the customer_order_counts DataFrame with the df DataFrame to get the total revenue for each customer
customer_revenue = df.groupby('CustomerID')['InvoicePrice'].sum().reset_index()

# Merge the customer_order_counts and customer_revenue DataFrames
customer_summary = pd.merge(customer_order_counts, customer_revenue, on='CustomerID')

# Calculate the contribution to revenue for each customer
customer_summary['revenue_contribution'] = customer_summary['InvoicePrice'] / customer_summary['InvoicePrice'].sum()

# Map country back to summary df
customer_summary = customer_summary.merge(df[['CustomerID','Country']].drop_duplicates(subset='CustomerID', keep='first'), on='CustomerID')

# Display the top 10 customers who have repeated the most and their contribution to revenue
print(customer_summary.head(10))
```

       CustomerID  order_count  InvoicePrice  revenue_contribution         Country
    0     14911.0          247     132572.62              0.016859            EIRE
    1     12748.0          224      29072.10              0.003697  United Kingdom
    2     17841.0          167      40340.78              0.005130  United Kingdom
    3     14606.0          128      11713.85              0.001490  United Kingdom
    4     15311.0          117      59419.34              0.007556  United Kingdom
    5     13089.0          113      57385.88              0.007298  United Kingdom
    6     12971.0           85      10930.26              0.001390  United Kingdom
    7     14527.0           84       7711.38              0.000981  United Kingdom
    8     13408.0           76      27487.41              0.003496  United Kingdom
    9     14646.0           76     279489.02              0.035542     Netherlands



```python
top_10_customers = customer_summary.sort_values('InvoicePrice', ascending=False).head(10).reset_index()

sns.barplot(data = top_10_customers, x='CustomerID', y='InvoicePrice')
plt.title('Revenue Generated by Top 10 Repeat Customers')
plt.xlabel('Customer ID')
plt.ylabel('InvoicePrice')
plt.xticks(rotation=45)
plt.show()
```


    
![png](/images/UCI_retail_files/UCI_retail_60_0.png)
    



```python
# Create a DataFrame of customer revenue
customer_revenue = df.groupby('CustomerID')['InvoicePrice'].sum()

# Create a boolean mask for customers who have ordered more than once
repeat_customers_mask = df['CustomerID'].duplicated(keep=False)

# Calculate the revenue generated by repeat customers
repeat_customer_revenue = customer_revenue[repeat_customers_mask]
repeat_customer_revenue
```




    CustomerID
    12346.0       0.00
    12347.0    4310.00
    12348.0    1797.24
    12349.0    1757.55
    12350.0     334.40
                ...   
    18280.0     180.60
    18281.0      80.82
    18282.0     176.60
    18283.0    2094.88
    18287.0    1837.28
    Name: InvoicePrice, Length: 4372, dtype: float64




```python
# Filter out the countries with less than 10 customers
a = customer_summary[(customer_summary.groupby('Country')['CustomerID'].transform('size') > 10) & (customer_summary.groupby('Country')['CustomerID'].transform('size') <= 50)]

# Group by CustomerID
grouped_summary = a.groupby('CustomerID').agg({'order_count': 'sum', 'InvoicePrice': 'sum', 'revenue_contribution': 'sum'}).reset_index()

# Map the country back
a = grouped_summary.merge(customer_summary[['CustomerID','Country']], on='CustomerID')

# Print the grouped summary
print(a.head(10))
```

       CustomerID  order_count  InvoicePrice  revenue_contribution      Country
    0     12356.0            2       2811.43              0.000358     Portugal
    1     12362.0           12       5154.58              0.000655      Belgium
    2     12364.0            3       1313.10              0.000167      Belgium
    3     12371.0            1       1887.96              0.000240  Switzerland
    4     12377.0            1       1628.12              0.000207  Switzerland
    5     12379.0            2        850.29              0.000108      Belgium
    6     12380.0            4       2720.56              0.000346      Belgium
    7     12383.0            5       1839.31              0.000234      Belgium
    8     12384.0            2        566.16              0.000072  Switzerland
    9     12394.0            1       1272.48              0.000162      Belgium



```python
customer_summary

# Scatter plot
sns.scatterplot(data=a.head(1000), x='order_count', y='InvoicePrice', hue='Country')
plt.title('Revenue Generated by Repeat Customers')
plt.xlabel('Number of Repeat Orders')
plt.ylabel('Revenue')

# Set the xticks to integer interval
plt.xticks(range(0, a.head(1000)['order_count'].max()+1, 5))

plt.show()
```


    
![png](/images/UCI_retail_files/UCI_retail_63_0.png)
    



```python

```
