# ERM_model_beta
An Enterprise Risk Management Demo, to showcase the capabilities of Neural Networks in pattern recognition. 

# This is the beginner notebook, for my neural network project

## Credit Risk Modeling : practice

# Note: This notebook provides a snapshot of the code, for the results, graphs, and all outputs, please refer to the notebook in the repository.

I will be working to practice using machine learning protocols, and apply them to real-world data to display the power of machine learning. I will be using the Lending Club dataset, which is a dataset of loans that were given out by the Lending Club. The goal is to predict whether or not a loan will be paid off in full or not. This is a classification problem, and I will be using a neural network to solve it.

## Source : https://www.kaggle.com/datasets/laotse/credit-risk-dataset

## Step 1: Load and Preprocess Data


```python
{
    "python.analysis.extraPaths": [
        "./venv/lib/python3.x/site-packages"
    ]
}
```




    {'python.analysis.extraPaths': ['./venv/lib/python3.x/site-packages']}




```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
#import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# Read the CSVs into DataFrames
train_df = pd.read_csv(Path('Resources/credit_risk_dataset.csv'))

# display the first 5 rows of train_df
display(train_df.head())

display(list(train_df.dtypes[train_df.dtypes == "object"].index))
```


<div>
<style scoped>
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
      <th>person_age</th>
      <th>person_income</th>
      <th>person_home_ownership</th>
      <th>person_emp_length</th>
      <th>loan_intent</th>
      <th>loan_grade</th>
      <th>loan_amnt</th>
      <th>loan_int_rate</th>
      <th>loan_status</th>
      <th>loan_percent_income</th>
      <th>cb_person_default_on_file</th>
      <th>cb_person_cred_hist_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22</td>
      <td>59000</td>
      <td>RENT</td>
      <td>123.0</td>
      <td>PERSONAL</td>
      <td>D</td>
      <td>35000</td>
      <td>16.02</td>
      <td>1</td>
      <td>0.59</td>
      <td>Y</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>9600</td>
      <td>OWN</td>
      <td>5.0</td>
      <td>EDUCATION</td>
      <td>B</td>
      <td>1000</td>
      <td>11.14</td>
      <td>0</td>
      <td>0.10</td>
      <td>N</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>9600</td>
      <td>MORTGAGE</td>
      <td>1.0</td>
      <td>MEDICAL</td>
      <td>C</td>
      <td>5500</td>
      <td>12.87</td>
      <td>1</td>
      <td>0.57</td>
      <td>N</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>65500</td>
      <td>RENT</td>
      <td>4.0</td>
      <td>MEDICAL</td>
      <td>C</td>
      <td>35000</td>
      <td>15.23</td>
      <td>1</td>
      <td>0.53</td>
      <td>N</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>54400</td>
      <td>RENT</td>
      <td>8.0</td>
      <td>MEDICAL</td>
      <td>C</td>
      <td>35000</td>
      <td>14.27</td>
      <td>1</td>
      <td>0.55</td>
      <td>Y</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



    ['person_home_ownership',
     'loan_intent',
     'loan_grade',
     'cb_person_default_on_file']



```python
# Create a list of the columns with categorical variables
categorical_variables = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']
```


```python
enc = OneHotEncoder(sparse=False)

# Let us now use the fit_transform() method to encode the categorical variables 
# that we saved earlier, and then convert the encoded data into a DataFrame.
encoded_data = enc.fit_transform(train_df[categorical_variables])

# Create a DataFrame with the encoded variables
encoded_df = pd.DataFrame(
    encoded_data,
    columns = enc.get_feature_names_out(categorical_variables)
    )

# Display sample data
encoded_df.head()
print("We did it! We have not only turned the non-numeric data into numeric data, but we have also created features for each categorical variable.")
```

    We did it! We have not only turned the non-numeric data into numeric data, but we have also created features for each categorical variable.


    /Users/najibabounasr/opt/anaconda3/lib/python3.10/site-packages/sklearn/preprocessing/_encoders.py:868: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
      warnings.warn(



```python
# display the new DataFrame
display(encoded_df.head())

# use the encoded_df DataFrame to replace the categorical variables in the train_df DataFrame
train_df = train_df.merge(encoded_df,left_index=True,right_index=True).drop(categorical_variables,axis=1)
display(train_df.head())
```


<div>
<style scoped>
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
      <th>person_home_ownership_MORTGAGE</th>
      <th>person_home_ownership_OTHER</th>
      <th>person_home_ownership_OWN</th>
      <th>person_home_ownership_RENT</th>
      <th>loan_intent_DEBTCONSOLIDATION</th>
      <th>loan_intent_EDUCATION</th>
      <th>loan_intent_HOMEIMPROVEMENT</th>
      <th>loan_intent_MEDICAL</th>
      <th>loan_intent_PERSONAL</th>
      <th>loan_intent_VENTURE</th>
      <th>loan_grade_A</th>
      <th>loan_grade_B</th>
      <th>loan_grade_C</th>
      <th>loan_grade_D</th>
      <th>loan_grade_E</th>
      <th>loan_grade_F</th>
      <th>loan_grade_G</th>
      <th>cb_person_default_on_file_N</th>
      <th>cb_person_default_on_file_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
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
      <th>person_age</th>
      <th>person_income</th>
      <th>person_emp_length</th>
      <th>loan_amnt</th>
      <th>loan_int_rate</th>
      <th>loan_status</th>
      <th>loan_percent_income</th>
      <th>cb_person_cred_hist_length</th>
      <th>person_home_ownership_MORTGAGE</th>
      <th>person_home_ownership_OTHER</th>
      <th>...</th>
      <th>loan_intent_VENTURE</th>
      <th>loan_grade_A</th>
      <th>loan_grade_B</th>
      <th>loan_grade_C</th>
      <th>loan_grade_D</th>
      <th>loan_grade_E</th>
      <th>loan_grade_F</th>
      <th>loan_grade_G</th>
      <th>cb_person_default_on_file_N</th>
      <th>cb_person_default_on_file_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22</td>
      <td>59000</td>
      <td>123.0</td>
      <td>35000</td>
      <td>16.02</td>
      <td>1</td>
      <td>0.59</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>9600</td>
      <td>5.0</td>
      <td>1000</td>
      <td>11.14</td>
      <td>0</td>
      <td>0.10</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>9600</td>
      <td>1.0</td>
      <td>5500</td>
      <td>12.87</td>
      <td>1</td>
      <td>0.57</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>65500</td>
      <td>4.0</td>
      <td>35000</td>
      <td>15.23</td>
      <td>1</td>
      <td>0.53</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>54400</td>
      <td>8.0</td>
      <td>35000</td>
      <td>14.27</td>
      <td>1</td>
      <td>0.55</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



```python
#isolate the non-catetgorical variables
non_categorical_variables = list(train_df.dtypes[train_df.dtypes != "object"].index)

#display the non-categorical variables as a dataframe
display(train_df[non_categorical_variables])
```


<div>
<style scoped>
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
      <th>person_age</th>
      <th>person_income</th>
      <th>person_emp_length</th>
      <th>loan_amnt</th>
      <th>loan_int_rate</th>
      <th>loan_status</th>
      <th>loan_percent_income</th>
      <th>cb_person_cred_hist_length</th>
      <th>person_home_ownership_MORTGAGE</th>
      <th>person_home_ownership_OTHER</th>
      <th>...</th>
      <th>loan_intent_VENTURE</th>
      <th>loan_grade_A</th>
      <th>loan_grade_B</th>
      <th>loan_grade_C</th>
      <th>loan_grade_D</th>
      <th>loan_grade_E</th>
      <th>loan_grade_F</th>
      <th>loan_grade_G</th>
      <th>cb_person_default_on_file_N</th>
      <th>cb_person_default_on_file_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22</td>
      <td>59000</td>
      <td>123.0</td>
      <td>35000</td>
      <td>16.02</td>
      <td>1</td>
      <td>0.59</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>9600</td>
      <td>5.0</td>
      <td>1000</td>
      <td>11.14</td>
      <td>0</td>
      <td>0.10</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>9600</td>
      <td>1.0</td>
      <td>5500</td>
      <td>12.87</td>
      <td>1</td>
      <td>0.57</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>65500</td>
      <td>4.0</td>
      <td>35000</td>
      <td>15.23</td>
      <td>1</td>
      <td>0.53</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>54400</td>
      <td>8.0</td>
      <td>35000</td>
      <td>14.27</td>
      <td>1</td>
      <td>0.55</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
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
      <th>32576</th>
      <td>57</td>
      <td>53000</td>
      <td>1.0</td>
      <td>5800</td>
      <td>13.16</td>
      <td>0</td>
      <td>0.11</td>
      <td>30</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32577</th>
      <td>54</td>
      <td>120000</td>
      <td>4.0</td>
      <td>17625</td>
      <td>7.49</td>
      <td>0</td>
      <td>0.15</td>
      <td>19</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32578</th>
      <td>65</td>
      <td>76000</td>
      <td>3.0</td>
      <td>35000</td>
      <td>10.99</td>
      <td>1</td>
      <td>0.46</td>
      <td>28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32579</th>
      <td>56</td>
      <td>150000</td>
      <td>5.0</td>
      <td>15000</td>
      <td>11.48</td>
      <td>0</td>
      <td>0.10</td>
      <td>26</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32580</th>
      <td>66</td>
      <td>42000</td>
      <td>2.0</td>
      <td>6475</td>
      <td>9.99</td>
      <td>0</td>
      <td>0.15</td>
      <td>30</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>32581 rows × 27 columns</p>
</div>



```python
# Search for NaN values in the dataframe
display(train_df.isnull().sum())
```


    person_age                           0
    person_income                        0
    person_emp_length                  895
    loan_amnt                            0
    loan_int_rate                     3116
    loan_status                          0
    loan_percent_income                  0
    cb_person_cred_hist_length           0
    person_home_ownership_MORTGAGE       0
    person_home_ownership_OTHER          0
    person_home_ownership_OWN            0
    person_home_ownership_RENT           0
    loan_intent_DEBTCONSOLIDATION        0
    loan_intent_EDUCATION                0
    loan_intent_HOMEIMPROVEMENT          0
    loan_intent_MEDICAL                  0
    loan_intent_PERSONAL                 0
    loan_intent_VENTURE                  0
    loan_grade_A                         0
    loan_grade_B                         0
    loan_grade_C                         0
    loan_grade_D                         0
    loan_grade_E                         0
    loan_grade_F                         0
    loan_grade_G                         0
    cb_person_default_on_file_N          0
    cb_person_default_on_file_Y          0
    dtype: int64



```python
# We need to understand the distribution of the data in each column with nan values
display(train_df['loan_int_rate'].describe())
display(train_df['person_emp_length'].describe())
# Graph the two columns with nan values
fig, ax = plt.subplots(1,2)
ax[0].hist(train_df['loan_int_rate'].dropna(), bins=30)
ax[0].set_title('loan_int_rate')
ax[1].hist(train_df['person_emp_length'].dropna(), bins=30)
ax[1].set_title('person_emp_length')
plt.show()
```


    count    29465.000000
    mean        11.011695
    std          3.240459
    min          5.420000
    25%          7.900000
    50%         10.990000
    75%         13.470000
    max         23.220000
    Name: loan_int_rate, dtype: float64



    count    31686.000000
    mean         4.789686
    std          4.142630
    min          0.000000
    25%          2.000000
    50%          4.000000
    75%          7.000000
    max        123.000000
    Name: person_emp_length, dtype: float64



    
![png](neural_network_files/neural_network_9_2.png)
    


## Removing NaN values: 

### It is critical that we remove NaN vaues, and as we are working with credit default risk-- a critical area of finance-- we will be holistically evaluating the data. We will start with ensuring that NaN values are not simply removed, but are replaced with either the mean of the column, or with new values based on predictive imputation. *We will be using the latter, as it is more accurate.*


```python
# Data with known loan_int_rate
reg_train_data = train_df.dropna(subset=['loan_int_rate'])

# Data with missing loan_int_rate
predict_data = train_df[train_df['loan_int_rate'].isnull()]
```


```python
# check for missing values in the new dataframe
display(reg_train_data.isnull().sum())
display(reg_train_data.describe())
```


    person_age                          0
    person_income                       0
    person_emp_length                 827
    loan_amnt                           0
    loan_int_rate                       0
    loan_status                         0
    loan_percent_income                 0
    cb_person_cred_hist_length          0
    person_home_ownership_MORTGAGE      0
    person_home_ownership_OTHER         0
    person_home_ownership_OWN           0
    person_home_ownership_RENT          0
    loan_intent_DEBTCONSOLIDATION       0
    loan_intent_EDUCATION               0
    loan_intent_HOMEIMPROVEMENT         0
    loan_intent_MEDICAL                 0
    loan_intent_PERSONAL                0
    loan_intent_VENTURE                 0
    loan_grade_A                        0
    loan_grade_B                        0
    loan_grade_C                        0
    loan_grade_D                        0
    loan_grade_E                        0
    loan_grade_F                        0
    loan_grade_G                        0
    cb_person_default_on_file_N         0
    cb_person_default_on_file_Y         0
    dtype: int64



<div>
<style scoped>
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
      <th>person_age</th>
      <th>person_income</th>
      <th>person_emp_length</th>
      <th>loan_amnt</th>
      <th>loan_int_rate</th>
      <th>loan_status</th>
      <th>loan_percent_income</th>
      <th>cb_person_cred_hist_length</th>
      <th>person_home_ownership_MORTGAGE</th>
      <th>person_home_ownership_OTHER</th>
      <th>...</th>
      <th>loan_intent_VENTURE</th>
      <th>loan_grade_A</th>
      <th>loan_grade_B</th>
      <th>loan_grade_C</th>
      <th>loan_grade_D</th>
      <th>loan_grade_E</th>
      <th>loan_grade_F</th>
      <th>loan_grade_G</th>
      <th>cb_person_default_on_file_N</th>
      <th>cb_person_default_on_file_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>29465.000000</td>
      <td>2.946500e+04</td>
      <td>28638.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>...</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
      <td>29465.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>27.714712</td>
      <td>6.602047e+04</td>
      <td>4.788672</td>
      <td>9584.744612</td>
      <td>11.011695</td>
      <td>0.219379</td>
      <td>0.170110</td>
      <td>5.788257</td>
      <td>0.411403</td>
      <td>0.003190</td>
      <td>...</td>
      <td>0.174885</td>
      <td>0.331716</td>
      <td>0.318853</td>
      <td>0.197794</td>
      <td>0.112472</td>
      <td>0.029900</td>
      <td>0.007263</td>
      <td>0.002002</td>
      <td>0.823078</td>
      <td>0.176922</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.300193</td>
      <td>6.190142e+04</td>
      <td>4.154627</td>
      <td>6316.272282</td>
      <td>3.240459</td>
      <td>0.413833</td>
      <td>0.106879</td>
      <td>4.031987</td>
      <td>0.492096</td>
      <td>0.056393</td>
      <td>...</td>
      <td>0.379876</td>
      <td>0.470837</td>
      <td>0.466040</td>
      <td>0.398343</td>
      <td>0.315952</td>
      <td>0.170314</td>
      <td>0.084914</td>
      <td>0.044704</td>
      <td>0.381609</td>
      <td>0.381609</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>4.000000e+03</td>
      <td>0.000000</td>
      <td>500.000000</td>
      <td>5.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.000000</td>
      <td>3.850000e+04</td>
      <td>2.000000</td>
      <td>5000.000000</td>
      <td>7.900000</td>
      <td>0.000000</td>
      <td>0.090000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>26.000000</td>
      <td>5.500000e+04</td>
      <td>4.000000</td>
      <td>8000.000000</td>
      <td>10.990000</td>
      <td>0.000000</td>
      <td>0.150000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>30.000000</td>
      <td>7.910000e+04</td>
      <td>7.000000</td>
      <td>12250.000000</td>
      <td>13.470000</td>
      <td>0.000000</td>
      <td>0.230000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>144.000000</td>
      <td>6.000000e+06</td>
      <td>123.000000</td>
      <td>35000.000000</td>
      <td>23.220000</td>
      <td>1.000000</td>
      <td>0.830000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 27 columns</p>
</div>


### We have to do another round of perprocessing, as NAN values are not going away. We will be using a predictive imputer to replace the NaN values with values that are predicted by the model. This is a more accurate way of replacing NaN values, as it is based on the data itself, and not just the mean of the column.


```python
# Data where 'person_emp_length' is NOT missing (to train the model)
train_data_for_emp_length = reg_train_data.dropna(subset=['person_emp_length'])

# Data where 'person_emp_length' IS missing (to predict 'person_emp_length')
predict_data_for_emp_length = reg_train_data[reg_train_data['person_emp_length'].isnull()]
```


```python
display(train_data_for_emp_length.describe())
display(predict_data_for_emp_length.describe())
```


<div>
<style scoped>
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
      <th>person_age</th>
      <th>person_income</th>
      <th>person_emp_length</th>
      <th>loan_amnt</th>
      <th>loan_int_rate</th>
      <th>loan_status</th>
      <th>loan_percent_income</th>
      <th>cb_person_cred_hist_length</th>
      <th>person_home_ownership_MORTGAGE</th>
      <th>person_home_ownership_OTHER</th>
      <th>...</th>
      <th>loan_intent_VENTURE</th>
      <th>loan_grade_A</th>
      <th>loan_grade_B</th>
      <th>loan_grade_C</th>
      <th>loan_grade_D</th>
      <th>loan_grade_E</th>
      <th>loan_grade_F</th>
      <th>loan_grade_G</th>
      <th>cb_person_default_on_file_N</th>
      <th>cb_person_default_on_file_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>28638.000000</td>
      <td>2.863800e+04</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>...</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
      <td>28638.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>27.727216</td>
      <td>6.664937e+04</td>
      <td>4.788672</td>
      <td>9656.493121</td>
      <td>11.039867</td>
      <td>0.216600</td>
      <td>0.169488</td>
      <td>5.793736</td>
      <td>0.412075</td>
      <td>0.003282</td>
      <td>...</td>
      <td>0.174628</td>
      <td>0.328305</td>
      <td>0.319540</td>
      <td>0.199001</td>
      <td>0.113416</td>
      <td>0.030379</td>
      <td>0.007298</td>
      <td>0.002060</td>
      <td>0.821810</td>
      <td>0.178190</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.310441</td>
      <td>6.235645e+04</td>
      <td>4.154627</td>
      <td>6329.683361</td>
      <td>3.229372</td>
      <td>0.411935</td>
      <td>0.106393</td>
      <td>4.038483</td>
      <td>0.492217</td>
      <td>0.057199</td>
      <td>...</td>
      <td>0.379655</td>
      <td>0.469605</td>
      <td>0.466307</td>
      <td>0.399256</td>
      <td>0.317106</td>
      <td>0.171631</td>
      <td>0.085117</td>
      <td>0.045343</td>
      <td>0.382679</td>
      <td>0.382679</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>4.000000e+03</td>
      <td>0.000000</td>
      <td>500.000000</td>
      <td>5.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.000000</td>
      <td>3.948000e+04</td>
      <td>2.000000</td>
      <td>5000.000000</td>
      <td>7.900000</td>
      <td>0.000000</td>
      <td>0.090000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>26.000000</td>
      <td>5.595600e+04</td>
      <td>4.000000</td>
      <td>8000.000000</td>
      <td>10.990000</td>
      <td>0.000000</td>
      <td>0.150000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>30.000000</td>
      <td>8.000000e+04</td>
      <td>7.000000</td>
      <td>12500.000000</td>
      <td>13.480000</td>
      <td>0.000000</td>
      <td>0.230000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>144.000000</td>
      <td>6.000000e+06</td>
      <td>123.000000</td>
      <td>35000.000000</td>
      <td>23.220000</td>
      <td>1.000000</td>
      <td>0.830000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 27 columns</p>
</div>



<div>
<style scoped>
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
      <th>person_age</th>
      <th>person_income</th>
      <th>person_emp_length</th>
      <th>loan_amnt</th>
      <th>loan_int_rate</th>
      <th>loan_status</th>
      <th>loan_percent_income</th>
      <th>cb_person_cred_hist_length</th>
      <th>person_home_ownership_MORTGAGE</th>
      <th>person_home_ownership_OTHER</th>
      <th>...</th>
      <th>loan_intent_VENTURE</th>
      <th>loan_grade_A</th>
      <th>loan_grade_B</th>
      <th>loan_grade_C</th>
      <th>loan_grade_D</th>
      <th>loan_grade_E</th>
      <th>loan_grade_F</th>
      <th>loan_grade_G</th>
      <th>cb_person_default_on_file_N</th>
      <th>cb_person_default_on_file_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>827.000000</td>
      <td>827.000000</td>
      <td>0.0</td>
      <td>827.000000</td>
      <td>827.000000</td>
      <td>827.000000</td>
      <td>827.000000</td>
      <td>827.000000</td>
      <td>827.000000</td>
      <td>827.0</td>
      <td>...</td>
      <td>827.000000</td>
      <td>827.000000</td>
      <td>827.000000</td>
      <td>827.000000</td>
      <td>827.000000</td>
      <td>827.000000</td>
      <td>827.000000</td>
      <td>827.0</td>
      <td>827.000000</td>
      <td>827.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>27.281741</td>
      <td>44242.383313</td>
      <td>NaN</td>
      <td>7100.181378</td>
      <td>10.036143</td>
      <td>0.315599</td>
      <td>0.191644</td>
      <td>5.598549</td>
      <td>0.388150</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.183797</td>
      <td>0.449819</td>
      <td>0.295042</td>
      <td>0.155985</td>
      <td>0.079807</td>
      <td>0.013301</td>
      <td>0.006046</td>
      <td>0.0</td>
      <td>0.866989</td>
      <td>0.133011</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.921741</td>
      <td>37250.866484</td>
      <td>NaN</td>
      <td>5263.533372</td>
      <td>3.466981</td>
      <td>0.465035</td>
      <td>0.120651</td>
      <td>3.797654</td>
      <td>0.487624</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.387553</td>
      <td>0.497777</td>
      <td>0.456338</td>
      <td>0.363061</td>
      <td>0.271157</td>
      <td>0.114630</td>
      <td>0.077567</td>
      <td>0.0</td>
      <td>0.339792</td>
      <td>0.339792</td>
    </tr>
    <tr>
      <th>min</th>
      <td>21.000000</td>
      <td>4200.000000</td>
      <td>NaN</td>
      <td>1000.000000</td>
      <td>5.420000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.000000</td>
      <td>24000.000000</td>
      <td>NaN</td>
      <td>3200.000000</td>
      <td>6.990000</td>
      <td>0.000000</td>
      <td>0.095000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>25.000000</td>
      <td>36000.000000</td>
      <td>NaN</td>
      <td>6000.000000</td>
      <td>9.910000</td>
      <td>0.000000</td>
      <td>0.170000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>30.000000</td>
      <td>55500.000000</td>
      <td>NaN</td>
      <td>9775.000000</td>
      <td>12.690000</td>
      <td>1.000000</td>
      <td>0.265000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>70.000000</td>
      <td>648000.000000</td>
      <td>NaN</td>
      <td>35000.000000</td>
      <td>21.360000</td>
      <td>1.000000</td>
      <td>0.650000</td>
      <td>27.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 27 columns</p>
</div>



```python
features = [col for col in train_data_for_emp_length.columns if col not in ('person_emp_length','loan_int_rate')]

# Preparing feature matrices

# Create X matrix
X_train_emp = train_data_for_emp_length[features]
y_train_emp = train_data_for_emp_length['person_emp_length']
X_predict_emp = predict_data_for_emp_length[features]
```


```python
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
regressor_emp = RandomForestRegressor(random_state=42)

# Train the model
regressor_emp.fit(X_train_emp, y_train_emp)
```

```python
# Predict 'person_emp_length' for the data with missing values
predicted_emp_lengths = regressor_emp.predict(X_predict_emp)

# Fill in the missing 'person_emp_length' values in the original DataFrame
reg_train_data.loc[reg_train_data['person_emp_length'].isnull(), 'person_emp_length'] = predicted_emp_lengths
```


```python
grand_total = reg_train_data['person_emp_length'].isnull().sum()
display(print(f"After our Imputation using the Regrissive model, we have now stand with a grand total of: {grand_total} missing values in the person_emp_length column."))

# Data with known loan_int_rate
display((reg_train_data))

```

    After our Imputation using the Regrissive model, we have now stand with a grand total of: 0 missing values in the person_emp_length column.



    None



<div>
<style scoped>
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
      <th>person_age</th>
      <th>person_income</th>
      <th>person_emp_length</th>
      <th>loan_amnt</th>
      <th>loan_int_rate</th>
      <th>loan_status</th>
      <th>loan_percent_income</th>
      <th>cb_person_cred_hist_length</th>
      <th>person_home_ownership_MORTGAGE</th>
      <th>person_home_ownership_OTHER</th>
      <th>...</th>
      <th>loan_intent_VENTURE</th>
      <th>loan_grade_A</th>
      <th>loan_grade_B</th>
      <th>loan_grade_C</th>
      <th>loan_grade_D</th>
      <th>loan_grade_E</th>
      <th>loan_grade_F</th>
      <th>loan_grade_G</th>
      <th>cb_person_default_on_file_N</th>
      <th>cb_person_default_on_file_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22</td>
      <td>59000</td>
      <td>123.0</td>
      <td>35000</td>
      <td>16.02</td>
      <td>1</td>
      <td>0.59</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>9600</td>
      <td>5.0</td>
      <td>1000</td>
      <td>11.14</td>
      <td>0</td>
      <td>0.10</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>9600</td>
      <td>1.0</td>
      <td>5500</td>
      <td>12.87</td>
      <td>1</td>
      <td>0.57</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>65500</td>
      <td>4.0</td>
      <td>35000</td>
      <td>15.23</td>
      <td>1</td>
      <td>0.53</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>54400</td>
      <td>8.0</td>
      <td>35000</td>
      <td>14.27</td>
      <td>1</td>
      <td>0.55</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
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
      <th>32576</th>
      <td>57</td>
      <td>53000</td>
      <td>1.0</td>
      <td>5800</td>
      <td>13.16</td>
      <td>0</td>
      <td>0.11</td>
      <td>30</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32577</th>
      <td>54</td>
      <td>120000</td>
      <td>4.0</td>
      <td>17625</td>
      <td>7.49</td>
      <td>0</td>
      <td>0.15</td>
      <td>19</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32578</th>
      <td>65</td>
      <td>76000</td>
      <td>3.0</td>
      <td>35000</td>
      <td>10.99</td>
      <td>1</td>
      <td>0.46</td>
      <td>28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32579</th>
      <td>56</td>
      <td>150000</td>
      <td>5.0</td>
      <td>15000</td>
      <td>11.48</td>
      <td>0</td>
      <td>0.10</td>
      <td>26</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32580</th>
      <td>66</td>
      <td>42000</td>
      <td>2.0</td>
      <td>6475</td>
      <td>9.99</td>
      <td>0</td>
      <td>0.15</td>
      <td>30</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>29465 rows × 27 columns</p>
</div>



<div>
<style scoped>
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
      <th>person_age</th>
      <th>person_income</th>
      <th>person_emp_length</th>
      <th>loan_amnt</th>
      <th>loan_int_rate</th>
      <th>loan_status</th>
      <th>loan_percent_income</th>
      <th>cb_person_cred_hist_length</th>
      <th>person_home_ownership_MORTGAGE</th>
      <th>person_home_ownership_OTHER</th>
      <th>...</th>
      <th>loan_intent_VENTURE</th>
      <th>loan_grade_A</th>
      <th>loan_grade_B</th>
      <th>loan_grade_C</th>
      <th>loan_grade_D</th>
      <th>loan_grade_E</th>
      <th>loan_grade_F</th>
      <th>loan_grade_G</th>
      <th>cb_person_default_on_file_N</th>
      <th>cb_person_default_on_file_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3116.000000</td>
      <td>3.116000e+03</td>
      <td>3048.000000</td>
      <td>3116.000000</td>
      <td>0.0</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
      <td>...</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
      <td>3116.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>27.922657</td>
      <td>6.658905e+04</td>
      <td>4.799213</td>
      <td>9633.119384</td>
      <td>NaN</td>
      <td>0.206675</td>
      <td>0.171088</td>
      <td>5.955071</td>
      <td>0.424262</td>
      <td>0.004172</td>
      <td>...</td>
      <td>0.181643</td>
      <td>0.321887</td>
      <td>0.338896</td>
      <td>0.202182</td>
      <td>0.100128</td>
      <td>0.026637</td>
      <td>0.008665</td>
      <td>0.001605</td>
      <td>0.829268</td>
      <td>0.170732</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.782386</td>
      <td>6.275810e+04</td>
      <td>4.028823</td>
      <td>6377.661049</td>
      <td>NaN</td>
      <td>0.404985</td>
      <td>0.105870</td>
      <td>4.264216</td>
      <td>0.494310</td>
      <td>0.064467</td>
      <td>...</td>
      <td>0.385612</td>
      <td>0.467275</td>
      <td>0.473410</td>
      <td>0.401692</td>
      <td>0.300219</td>
      <td>0.161045</td>
      <td>0.092696</td>
      <td>0.040032</td>
      <td>0.376335</td>
      <td>0.376335</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>4.888000e+03</td>
      <td>0.000000</td>
      <td>500.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.000000</td>
      <td>3.850000e+04</td>
      <td>2.000000</td>
      <td>5000.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.090000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>26.000000</td>
      <td>5.564000e+04</td>
      <td>4.000000</td>
      <td>8000.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.150000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>30.000000</td>
      <td>8.000000e+04</td>
      <td>7.000000</td>
      <td>12000.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.230000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>123.000000</td>
      <td>1.900000e+06</td>
      <td>28.000000</td>
      <td>35000.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.630000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 27 columns</p>
</div>


# Comparison with earlier data: looking for differences and similarities

### It will be quite clear, that the logistic regression technique had succesfully ridden us of the cumersome and spurious NaN values, while also maintaining the overall shape of the data. Any significant changes in median and mean would have been a cause for concern, but as we can see, the data is still quite similar to the original data.


```python
# We compare train_df with reg_train_data
nan_counts_before = train_df['person_emp_length'].isnull().sum()
nan_counts_after = reg_train_data['person_emp_length'].isnull().sum()

print(f"NaN Counts Before: {nan_counts_before}")
print(f"NaN Counts After: {nan_counts_after}")
```

    NaN Counts Before: 895
    NaN Counts After: 0



```python
# We want to check the means and medians of the two datsets, to see if there is a 
# significant difference in the two datasets
mean_before = train_df['person_emp_length'].mean()
median_before = train_df['person_emp_length'].median()

mean_after = reg_train_data['person_emp_length'].mean()
median_after = reg_train_data['person_emp_length'].median()

print(f"Mean Before: {mean_before}, After: {mean_after}")
print(f"Median Before: {median_before}, After: {median_after}")
```

    Mean Before: 4.789686296787225, After: 4.7781608247880865
    Median Before: 4.0, After: 4.0



```python
plt.figure(figsize=(14, 6))

# Histogram for 'person_emp_length' before imputation
plt.subplot(1, 2, 1)
train_df['person_emp_length'].hist(bins=30, alpha=0.7)
plt.title('Person Employment Length Before Imputation')
plt.xlabel('Employment Length')
plt.ylabel('Frequency')

# Histogram for 'person_emp_length' after imputation
plt.subplot(1, 2, 2)
reg_train_data['person_emp_length'].hist(bins=30, alpha=0.7, color='orange')
plt.title('Person Employment Length After Imputation')
plt.xlabel('Employment Length')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
print("Almost identitical, we can see that the data preprocessing has ben a success!")
```


    
![png](neural_network_files/neural_network_23_0.png)
    



```python
# We replace the old data, with the new data 
train_df = reg_train_data
```

# Continuing along-- after cleaning and clearing up the data


```python
import panel as pn

# Visualize the data

# Create Distribution Plots for each column

train_df.hist(figsize=(20, 20))



```




    array([[<Axes: title={'center': 'person_age'}>,
            <Axes: title={'center': 'person_income'}>,
            <Axes: title={'center': 'person_home_ownership'}>],
           [<Axes: title={'center': 'person_emp_length'}>,
            <Axes: title={'center': 'loan_intent'}>,
            <Axes: title={'center': 'loan_grade'}>],
           [<Axes: title={'center': 'loan_amnt'}>,
            <Axes: title={'center': 'loan_int_rate'}>,
            <Axes: title={'center': 'loan_status'}>],
           [<Axes: title={'center': 'loan_percent_income'}>,
            <Axes: title={'center': 'cb_person_default_on_file'}>,
            <Axes: title={'center': 'cb_person_cred_hist_length'}>]],
          dtype=object)




    
![png](neural_network_files/neural_network_26_1.png)
    



```python
# Lets get rid of the columns that have values greater than 10, as these were originally non-numeric data, 
# which we might not get that much information from.

dist_plot_df = train_df.drop(columns=['person_age', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']) 
```


```python
# Now we can create the distribution plots for each column

dist_plot_df.hist(figsize=(20, 20))

```




    array([[<Axes: title={'center': 'person_income'}>,
            <Axes: title={'center': 'person_home_ownership'}>,
            <Axes: title={'center': 'person_emp_length'}>],
           [<Axes: title={'center': 'loan_intent'}>,
            <Axes: title={'center': 'loan_grade'}>,
            <Axes: title={'center': 'loan_amnt'}>],
           [<Axes: title={'center': 'loan_status'}>,
            <Axes: title={'center': 'cb_person_default_on_file'}>, <Axes: >]],
          dtype=object)




    
![png](neural_network_files/neural_network_28_1.png)
    


# SECOND SECTION : Creating a neural network

## Feature Engineering

#### Important: remember to set random state to 1 for reproducibility

#### Also, I did not know until now that the OneHotEncoder() function is meant to be performed before the train_test_split() function. I will be doing this from now on. New insights are gained everyday. 


```python
# We have to start off by encoding categorical data. 

# STEP 1 
# Use OneHotEncoder to create features for each categorical variable

enc = OneHotEncoder(sparse=False)

# Let us now use the fit_transform() method to encode the categorical variables 
# that we saved earlier, and then convert the encoded data into a DataFrame.
encoded_data = enc.fit_transform(train_df[categorical_variables])

# Create a DataFrame with the encoded variables
encoded_df = pd.DataFrame(
    encoded_data,
    columns = enc.get_feature_names_out(categorical_variables)
    )

# Display sample data
encoded_df.head()
print("We did it! We have not only turned the non-numeric data into numeric data, but we have also created features for each categorical variable.")
```

    We did it! We have not only turned the non-numeric data into numeric data, but we have also created features for each categorical variable.


    /Users/najibabounasr/opt/anaconda3/lib/python3.10/site-packages/sklearn/preprocessing/_encoders.py:868: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
      warnings.warn(



```python

```


```python
# Step 2
# Create the features (X) and target (y) sets
X = train_df.drop(columns=["loan_status"]).values
y = train_df["loan_status"].values

# Create the training and testing datasets
# We will set random_state=1 to make sure we get the same split every time we run the code.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create the scaler instance
X_scaler = StandardScaler()

# Fit the scaler
X_scaler.fit(X_train)

# Scale the data

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```


```python
# display the data, and check for nan values

# display(print(X_train_scaled))

# display(print(X_train_scaled.shape))

# display(print(y_train))

# Assuming X_train_scaled is your NumPy array
nan_count = np.isnan(X_train_scaled).sum()
print(nan_count)
```

    3031


## Creating a Neural Network Model using Keras 

I wasn't sure which NN we will be working with. This was the only model I had learned about, and had the most experience with. I will be using the Keras library to create a neural network model. 

I had watched the introductory video explaining the ReLU functions, and the sigmoid functions. The information has widened my perspective on the neural network model, and will I definitely try and apply the learning from the video in this micro-project.

# *There has ben an issue in development, related to the tensorflow installment.*

- *I am not sure why tensorflow no longer works with my Mac-- if this issue will come in the way of my learning, I will definetely be ready to take steps to resolve any depoendency conflicts. For now, though, I will just use a workaround I have found online. I was taught to use sequential and build the neural network that way, but I will now use sklearn's MLPCLassifier instead*

- I will be using ReLU, now understanding that it's steep activation, or 'firing' similar to neurons (as caught in brain EEG scans) is what makes it both powerful and similar in nature to human neuronal firing and excitation. 


```python
#Step 1. 
# Create a sequential model with Keras
# neuron = Sequential()
# Create a sequential model
```


```python
# Step 1: We use sklearn to create a sequential model

from sklearn.neural_network import MLPClassifier

# Example: Creating a model with two hidden layers, the first with 10 neurons and the second with 5 neurons
model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, activation='relu', solver='adam', random_state=1)

# Step 2: Fit the model to the training data
model.fit(X_train_scaled, y_train)

# Step 3: Evaluate the model using the test data
# Assuming X_test and y_test are your features and labels for testing
score = model.score(X_test, y_test)
print(f"Accuracy: {score}")


```

    Accuracy: 0.777657119587349

# Perfecting the Model:

### We have achieved an unsatisfactory score of 0.7777, which is not good enough.Thankfully, we understand that the weights, number of layers and biases of the neurons must be adjusted to achieve a better score.

- Moving forward, I will continue learning to implement my newfound knowledge of neural networks, to improve this beta model. If anything, I hope this snapshot demonstrates my capabilities at the moment. Although my capabilities are limited, I am confident that, like this model, my skills will improve considerably. 
