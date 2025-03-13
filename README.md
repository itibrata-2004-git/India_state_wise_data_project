import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_excel(r"C:\Users\itibr\Naresh IT\India state wise\India_Statewise_Data.xlsx")
data.head()
State/UT	Population (millions)	GSDP (₹ trillion)	Tax per Head (₹)	Total Revenue (₹ trillion)
0	Andhra Pradesh	53.9	11.3	15320	0.83
1	Arunachal Pradesh	1.6	0.4	11250	0.02
2	Assam	35.5	4.3	13840	0.49
3	Bihar	124.8	7.2	8900	1.11
4	Chhattisgarh	29.4	4.0	12100	0.36
data.isnull().sum()
State/UT                      0
Population (millions)         0
GSDP (₹ trillion)             0
Tax per Head (₹)              0
Total Revenue (₹ trillion)    0
dtype: int64
data.dtypes
State/UT                       object
Population (millions)         float64
GSDP (₹ trillion)             float64
Tax per Head (₹)                int64
Total Revenue (₹ trillion)    float64
dtype: object
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 29 entries, 0 to 28
Data columns (total 5 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   State/UT                    29 non-null     object 
 1   Population (millions)       29 non-null     float64
 2   GSDP (₹ trillion)           29 non-null     float64
 3   Tax per Head (₹)            29 non-null     int64  
 4   Total Revenue (₹ trillion)  29 non-null     float64
dtypes: float64(3), int64(1), object(1)
memory usage: 1.3+ KB
for i in data.columns:
    print(i,':','\n',data[i].unique())
State/UT : 
 ['Andhra Pradesh' 'Arunachal Pradesh' 'Assam' 'Bihar' 'Chhattisgarh'
 'Delhi' 'Goa' 'Gujarat' 'Haryana' 'Himachal Pradesh' 'Jharkhand'
 'Karnataka' 'Kerala' 'Madhya Pradesh' 'Maharashtra' 'Manipur' 'Meghalaya'
 'Mizoram' 'Nagaland' 'Odisha' 'Punjab' 'Rajasthan' 'Sikkim' 'Tamil Nadu'
 'Telangana' 'Tripura' 'Uttar Pradesh' 'Uttarakhand' 'West Bengal']
Population (millions) : 
 [ 53.9   1.6  35.5 124.8  29.4  20.6  69.   29.1   7.4  39.3  68.4  35.3
  87.5 124.5   3.1   3.8   1.2   2.   44.6  30.5  81.2   0.7  77.2  39.1
   4.2 240.9  11.4  99.6]
GSDP (₹ trillion) : 
 [11.3  0.4  4.3  7.2  4.  10.5  1.4 18.8 10.6  2.8  5.3 20.5 10.2 10.
 35.3  0.5  0.8  0.3  0.6  7.8 12.2 24.8 14.5  1.1 20.3  3.9 17.1]
Tax per Head (₹) : 
 [15320 11250 13840  8900 12100 47500 24200 28500 31200 15400 13600 33000
 26700 12900 47800  9900 12300  8500  9400 13450 25900 14900 21800 36600
 32500 14200 10400 27600 23800]
Total Revenue (₹ trillion) : 
 [0.83 0.02 0.49 1.11 0.36 0.98 0.05 1.96 0.91 0.11 0.53 2.26 0.94 1.13
 5.95 0.03 0.01 0.6  0.79 1.21 2.83 1.27 0.06 2.51 0.31 2.37]
data.describe (include='all')
State/UT	Population (millions)	GSDP (₹ trillion)	Tax per Head (₹)	Total Revenue (₹ trillion)
count	29	29.000000	29.000000	29.000000	29.000000
unique	29	NaN	NaN	NaN	NaN
top	Andhra Pradesh	NaN	NaN	NaN	NaN
freq	1	NaN	NaN	NaN	NaN
mean	NaN	47.151724	9.103448	20808.965517	1.024483
std	NaN	52.772873	8.669218	11206.781539	1.261248
min	NaN	0.700000	0.300000	8500.000000	0.010000
25%	NaN	4.200000	1.400000	12300.000000	0.060000
50%	NaN	35.300000	7.200000	15320.000000	0.790000
75%	NaN	69.000000	12.200000	27600.000000	1.210000
max	NaN	240.900000	35.300000	47800.000000	5.950000
df = pd.DataFrame(data)
print(data.columns)
Index(['State/UT', 'Population (millions)', 'GSDP (₹ trillion)',
       'Tax per Head (₹)', 'Total Revenue (₹ trillion)'],
      dtype='object')
sns.histplot(df["Population (millions)"], bins=10, kde=True)
plt.title("Population Distribution of Indian States")
plt.show()
No description has been provided for this image
sns.boxplot(x=df["Tax per Head (₹)"])
plt.title("Boxplot of Tax per Head in Indian States")
plt.show()
No description has been provided for this image
sns.scatterplot(x=df["Population (millions)"], y=df["GSDP (₹ trillion)"], hue=df["Total Revenue (₹ trillion)"], palette="coolwarm", size=df["Tax per Head (₹)"])
plt.title("GSDP vs Population (Size based on Tax per Head)")
plt.show()
No description has been provided for this image
plt.figure(figsize=(12, 6))
sns.barplot(x="Total Revenue (₹ trillion)", y="State/UT", data=df, palette="coolwarm")
plt.title("State-wise Total Revenue")
plt.show()
C:\Users\itibr\AppData\Local\Temp\ipykernel_21440\3767583942.py:2: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x="Total Revenue (₹ trillion)", y="State/UT", data=df, palette="coolwarm")
No description has been provided for this image
sns.pairplot(df[["Population (millions)", "GSDP (₹ trillion)", "Tax per Head (₹)", "Total Revenue (₹ trillion)"]])
plt.show()
No description has been provided for this image
# Select only numeric columns for correlation
numeric_columns = df[["Population (millions)", "GSDP (₹ trillion)", "Tax per Head (₹)", "Total Revenue (₹ trillion)"]]

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_columns.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Correlation Between Features")
plt.show()
No description has been provided for this image
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Example DataFrame
df = pd.DataFrame({
    "Population (millions)": [53.9, 1.6, 35.5, 124.8, 29.4],
    "GSDP (₹ trillion)": [11.3, 0.4, 4.3, 7.2, 4.0]
})

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(df[["Population (millions)", "GSDP (₹ trillion)"]])

# Convert to DataFrame for better readability
scaled_df = pd.DataFrame(scaled_data, columns=["Population (millions)", "GSDP (₹ trillion)"])

print(scaled_df)
   Population (millions)  GSDP (₹ trillion)
0               0.117302           1.610328
1              -1.145025          -1.384992
2              -0.326805          -0.313272
3               1.828564           0.483648
4              -0.474036          -0.395712
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Example DataFrame
df = pd.DataFrame({
    "Population (millions)": [53.9, 1.6, 35.5, 124.8, 29.4],
    "GSDP (₹ trillion)": [11.3, 0.4, 4.3, 7.2, 4.0]
})

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(df[["Population (millions)", "GSDP (₹ trillion)"]])

# Convert to DataFrame for better readability
scaled_df = pd.DataFrame(scaled_data, columns=["Population (millions)", "GSDP (₹ trillion)"])

print(scaled_df)
   Population (millions)  GSDP (₹ trillion)
0               0.424513           1.000000
1               0.000000           0.000000
2               0.275162           0.357798
3               1.000000           0.623853
4               0.225649           0.330275
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Example data
X = df[["Population (millions)"]]  # Independent variable
y = df["GSDP (₹ trillion)"]  # Dependent variabl
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

  LinearRegression?i
LinearRegression()
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R² Score
r2 = r2_score(y_test, y_pred)

# Print both metrics
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
Mean Squared Error: 23.05807669292166
R² Score: nan
C:\ProgramData\anaconda3\Lib\site-packages\sklearn\metrics\_regression.py:1211: UndefinedMetricWarning: R^2 score is not well-defined with less than two samples.
  warnings.warn(msg, UndefinedMetricWarning)
import joblib

# Save the trained model to a file
joblib.dump(model, 'gdp_prediction_model.pkl')
['gdp_prediction_model.pkl']
# Load the saved model
model = joblib.load('gdp_prediction_model.pkl')

# Make predictions
y_pred = model.predict(X_test)
import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('gdp_prediction_model.pkl')

# Streamlit app layout
st.title("GSDP Prediction Model")

# User input form
population = st.number_input("Enter the Population (in millions):", min_value=0.0)

# When the user presses the button to predict
if st.button("Predict"):
    # Reshape the input for prediction
    input_features = np.array([[population]])

    # Make prediction
    prediction = model.predict(input_features)

    # Display the result
    st.write(f"Predicted GSDP (₹ trillion): {prediction[0]}")
2025-03-13 16:28:42.172 
  Warning: to view this Streamlit app on a browser, run it with the following
  command:

    streamlit run C:\ProgramData\anaconda3\Lib\site-packages\ipykernel_launcher.py [ARGUMENTS]
2025-03-13 16:28:42.173 Session state does not function when running a script without `streamlit run`
