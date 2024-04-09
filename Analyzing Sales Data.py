#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# Assume 'sales_data' is a NumPy array containing the dataset

# Step 2: Data cleaning
# Handle missing values, outliers, etc.

# Step 3: Analyze overall sales trends
total_sales = np.sum(sales_data[:, 1])  # Assuming sales amounts are in the second column
average_sales_per_month = np.mean(sales_data[:, 1])  # Assuming date information is available
# Compute other metrics as needed

# Step 4: Analyze sales performance by product category
unique_categories = np.unique(sales_data[:, 2])  # Assuming product categories are in the third column
category_sales = []
for category in unique_categories:
    category_mask = sales_data[:, 2] == category
    category_sales.append((category, np.sum(sales_data[category_mask, 1]), np.mean(sales_data[category_mask, 1])))

# Step 5: Explore relationships
# Perform correlation analysis, regression, or other statistical methods

# Step 6: Visualization
plt.figure(figsize=(10, 6))
plt.bar(unique_categories, [sales[1] for sales in category_sales])
plt.title('Total Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

