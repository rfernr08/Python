import pandas as pd
import matplotlib.pyplot as plt

# Load the sales data into a pandas DataFrame
sales_df = pd.read_csv('company_sales_data.csv')

# Create a bar chart
plt.bar(sales_df['month_number'] - 0.2, sales_df['facecream'], width=0.4, label='Face Cream', align='center')
plt.bar(sales_df['month_number'] + 0.2, sales_df['facewash'], width=0.4, label='Face Wash', align='center')

plt.xlabel('Month Number')
plt.ylabel('Sales Units')
plt.title('Sales data')
plt.xticks(sales_df['month_number'])
plt.grid(True, linewidth= 1, linestyle="--")
plt.legend()
plt.show()