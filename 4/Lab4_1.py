import pandas as pd

# Define column names
column_names = ["Name", "Age", "Score"]

# Load data
df = pd.read_csv("student_data.csv", names=column_names)

#Display the first 5 rows of the DataFrame
print(df.head())

#print(df.columns)

#Print the data types of each column
print(df.dtypes)

#Check for any missing values in the DataFrame
print(df.isnull().sum())

#Statistical Analysis
print("Mean test score:", df['Score'].mean())
print("Median test score:", df['Score'].median())

#Filtering Data
filtered_df = df[df['Age'] > 20]

#Grouping and Aggregation
grouped_df = df.groupby('Age')['Score'].mean()

#Display the resulting DataFrame
print(grouped_df)

#Exporting Data
filtered_df.to_csv("filtered_student_data.csv", index=False)