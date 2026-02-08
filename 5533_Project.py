#1
import pandas as pd

# Load the CSV file
sales_data = pd.read_csv("Sales_csv.csv")

print("\n===== DATASET INFORMATION =====")
sales_data.info()

print("\n===== DISPLAY HEAD =====")
print(sales_data.head(10))

print("\n===== DISPLAY DESCRIPTION =====")
print(sales_data.describe())

#2
import pandas as pd

data = pd.read_csv("Sales_csv.csv")

# Store rows with missing values (BEFORE cleaning)
missing_rows = data[data.isnull().any(axis=1)]

print("\n===== BEFORE CLEANING (MEAN) =====")
print(missing_rows)

# Fill missing values with MEAN
data_mean = data.fillna(data.mean(numeric_only=True))

# Show ONLY those same rows AFTER cleaning (MEAN)
print("\n===== SAME ROWS AFTER CLEANING (MEAN) =====")
print(data_mean.loc[missing_rows.index])

# Store rows with missing values (BEFORE cleaning)
missing_rows = data[data.isnull().any(axis=1)]

print("\n===== BEFORE CLEANING (MEDIAN) =====")
print(missing_rows)

# Fill missing values with MEDIAN
data_median = data.fillna(data.median(numeric_only=True))

# Show ONLY those same rows AFTER cleaning (MEDIAN)
print("\n===== SAME ROWS AFTER CLEANING (MEDIAN) =====")
print(data_median.loc[missing_rows.index])

#3
import pandas as pd
data = pd.read_csv("Sales_csv.csv")

print("\n===== ORIGINAL DATA (FIRST 10 ROWS) =====")
print(data.head(10))

print("\n===== DUPLICATE RECORDS (BEFORE REMOVAL) =====")
duplicates = data[data.duplicated()]

print(duplicates)
print("Number of duplicate records:", duplicates.shape[0])

data_no_duplicates = data.drop_duplicates()

print("\n===== AFTER REMOVING DUPLICATES =====")
print(data_no_duplicates.head(10))

print("\nNumber of records after removing duplicates:", data_no_duplicates.shape[0])

#4
import pandas as pd

# Load dataset
data = pd.read_csv("Sales_csv.csv")

print("\n===== SALES DATA (BEFORE OUTLIER DETECTION) =====")
print(data['Sales'].head(10))

# Calculate IQR
Q1 = data['Sales'].quantile(0.25)
Q3 = data['Sales'].quantile(0.75)

IQR = Q3 - Q1

# Define limits
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Find outliers
outliers = data[(data['Sales'] < lower_limit) | (data['Sales'] > upper_limit)]

print("\n===== DETECTED OUTLIERS =====")
print(outliers[['Sales']])
print("\nNumber of outliers:", outliers.shape[0])

data_no_outliers = data[(data['Sales'] >= lower_limit) & (data['Sales'] <= upper_limit)]

print("\n===== AFTER REMOVING OUTLIERS =====")
print(data_no_outliers['Sales'].head(10))

#5
import pandas as pd

# Load dataset
data = pd.read_csv("Sales_csv.csv")

print("\n===== CATEGORY VALUES (BEFORE CLEANING) =====")
print(data['Category'].head(10))

# Convert to lowercase and remove spaces
data['Category_clean'] = data['Category'].str.lower().str.strip()

print("\n===== UNIQUE CATEGORY VALUES (AFTER CLEANING) =====")
print(data['Category_clean'].head(10).unique())

#6
import pandas as pd

# Load dataset
data = pd.read_csv("Sales_csv.csv")

print("\n===== COLUMN NAMES (BEFORE RENAMING) =====")
print(data.columns)

data_renamed = data.rename(columns={
    'Order Date': 'RENAMED Order Date',
    'Ship Date': 'RENAMED Ship Date',
    'Product Name': 'RENAMED Product Name'
})

print("\n===== COLUMN NAMES (AFTER RENAMING) =====")
print(data_renamed.columns)

#7
import pandas as pd

data = pd.read_csv("Sales_csv.csv")

print("\n===== BEFORE CONVERSION =====")
print(data.dtypes)

# Convert Order Date to datetime
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d-%b-%y')
data['Ship Date'] = pd.to_datetime(data['Ship Date'], format='%d-%b-%y')

print("\n===== AFTER CONVERSION =====")
print(data.dtypes)

#8
import pandas as pd

data = pd.read_csv("Sales_csv.csv")

print("\n===== BEFORE SCALING =====")
print(data['Sales'].head(10))

# Scaling Sales column
data['Sales_scaled'] = (data['Sales'] - data['Sales'].mean()) / data['Sales'].std()

print("\n===== AFTER SCALING =====")
print(data[['Sales','Sales_scaled']].head(10))

#9
import pandas as pd

data = pd.read_csv("Sales_csv.csv")

print("\n===== BEFORE NORMALIZATION =====")
print(data['Profit'].head(10))

# Normalize Profit column
data['Profit_normalized'] = (data['Profit'] - data['Profit'].min()) / (data['Profit'].max() - data['Profit'].min())

print("\n===== AFTER NORMALIZATION =====")
print(data[['Profit','Profit_normalized']].head(10))

#10
import pandas as pd

data = pd.read_csv("Sales_csv.csv")

print("\n===== BEFORE BINNING =====")
print(data['Sales'].head(10))

# Create bins
bins = [0, 10, 50, 70, data['Sales'].max()]
labels = ['Low','Medium','High','Very High']

data['Sales_Group'] = pd.cut(data['Sales'], bins=bins, labels=labels)

print("\n===== AFTER BINNING =====")
print(data[['Sales','Sales_Group']].head(10))

#11
import pandas as pd

data = pd.read_csv("Sales_csv.csv")

print("\n===== BEFORE AGGREGATION =====")
print(data[['Category','Sales']].head(10))

# Aggregate total sales by category
agg_data = data.groupby('Category')['Sales'].sum()

print("\n===== AFTER AGGREGATION =====")
print(agg_data)

#12
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Sales_csv.csv")

print("\n===== DATA USED FOR BAR CHART =====")
print(data[['Category','Sales']].head())

sample = data.head()
bar_data = sample.groupby('Category')['Sales'].sum()

print("\n===== AGGREGATED DATA FOR BAR CHART =====")
print(bar_data)

plt.figure()
bar_data.plot(kind='bar')
plt.title("Bar Chart of Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.show()

#13
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Sales_csv.csv")

print("\n===== DATA USED FOR SCATTER PLOT (FIRST 10 ROWS) =====")
print(data[['Sales','Profit']].head(10))

plt.figure()
plt.scatter(data['Sales'], data['Profit'])
plt.title("Scatter Plot of Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()

#14
import pandas as pd

data = pd.read_csv("Sales_csv.csv")

print("\n===== BEFORE ONE HOT ENCODING =====")
print(data[['Category']].head(10))

# Apply One Hot Encoding
one_hot = pd.get_dummies(data['Category'])

# Combine with original data
data_onehot = pd.concat([data, one_hot], axis=1)

print("\n===== AFTER ONE HOT ENCODING =====")
print(data_onehot[['Category'] + list(one_hot.columns)].head(10))

#15
import pandas as pd

data = pd.read_csv("Sales_csv.csv")

print("\n===== BEFORE DUMMY VARIABLE CREATION =====")
print(data[['Segment']].head(10))

# Create Dummy Variables
dummy = pd.get_dummies(data['Segment'], drop_first=True)

# Combine with original data
data_dummy = pd.concat([data, dummy], axis=1)

print("\n===== AFTER DUMMY VARIABLE CREATION =====")
print(data_dummy[['Segment'] + list(dummy.columns)].head(10))

#16
import pandas as pd

data = pd.read_csv("Sales_csv.csv")

print("\n===== BEFORE LABEL ENCODING =====")
print(data[['Ship Mode']].head(10))

# Label Encoding using pandas
data['Ship_Mode_Label'] = data['Ship Mode'].astype('category').cat.codes

print("\n===== AFTER LABEL ENCODING =====")
print(data[['Ship Mode','Ship_Mode_Label']].head(10))

#17
import pandas as pd

data = pd.read_csv("Sales_csv.csv")

print("\n===== BEFORE LOWER CASING =====")
print(data['Product Name'].head(6))

# Convert text to lowercase
data['Product_Name_lower'] = data['Product Name'].str.lower()

print("\n===== AFTER LOWER CASING =====")
print(data[['Product Name','Product_Name_lower']].head(10))

#18
import pandas as pd
import string

data = pd.read_csv("Sales_csv.csv")

print("\n===== BEFORE REMOVING PUNCTUATION =====")
print(data['Product Name'].head(6))

# Remove punctuation
data['Product_Name_no_punct'] = data['Product Name'].str.replace('[{}]'.format(string.punctuation), '', regex=True)

print("\n===== AFTER REMOVING PUNCTUATION =====")
print(data[['Product Name','Product_Name_no_punct']].head(10))

#19
import pandas as pd

data = pd.read_csv("Sales_csv.csv")

print("\n===== BEFORE TOKENIZATION =====")
print(data['Product Name'].head(6))

# Tokenization (split words)
data['Product_Name_tokens'] = data['Product Name'].str.split()

print("\n===== AFTER TOKENIZATION =====")
print(data[['Product Name','Product_Name_tokens']].head(6))

#20
import pandas as pd

data = pd.read_csv("Sales_csv.csv")

print("\n===== BEFORE FEATURE SCALING =====")
print(data[['Sales','Profit','Quantity']].head(10))

# Apply Standard Scaling
data['Sales_scaled'] = (data['Sales'] - data['Sales'].mean()) / data['Sales'].std()
data['Profit_scaled'] = (data['Profit'] - data['Profit'].mean()) / data['Profit'].std()
data['Quantity_scaled'] = (data['Quantity'] - data['Quantity'].mean()) / data['Quantity'].std()

print("\n===== AFTER FEATURE SCALING =====")
print(data[['Sales','Sales_scaled','Profit','Profit_scaled','Quantity','Quantity_scaled']].head(10))

