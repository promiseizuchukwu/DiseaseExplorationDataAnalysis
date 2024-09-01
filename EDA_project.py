#!/usr/bin/env python
# coding: utf-8

# <img align="center" src="https://iili.io/3wI8gI.png" style="height:90px" style="width:30px"/>

# # EDA Case study
# Your task is to find interesting insights in the data that provides value to the stakeholders and visualize and communicate your insights in a clear manner. Follow to the PPDAC cycle when analyzing and visualizing the data. When you find an insight in a plot you write down key words that follows the setup-conflict-resolution framework.

# ### Examples of questions that could be intresting to investigate:
# How has the buisness been performing over time?
# 
# Wich countries is the biggest and most profitable markets?
# 
# Any time of day that the sales are bigger?
# 
# Cost of shipping compared to sales?
# 
# Please come up with your own ideas that could be intresting to investigate. Don't get stuck on small details that wont bring any insights instead try to think "can this insight lead to the company taking any action that can increase sales, increase profitability or decrease cost?"

# In[5]:


import pandas as pd
import numpy as np 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv('C:\\Users\\X360\\NOD PROJECTS\\orders.csv')


# In[6]:


df


# In[7]:


df.info()


# In[ ]:





# In[8]:


df.describe()


# In[9]:


df.describe(include="O")


# In[10]:


df["Shipping_Country"].value_counts( normalize = True)


# In[11]:


sns.set_palette("Paired")
ax = sns.countplot(data= df, x= "Shipping_Country", order=df["Shipping_Country"].value_counts(ascending = False).index, color = "blue")
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right", fontsize = 10)
plt.title("Sales per Country")                  
plt.tight_layout()


# In[12]:


shipping_c_counts = df["Shipping_Country"].value_counts().sort_values(ascending = False).head(15)


# In[28]:


top_countries = df[df["Shipping_Country"].isin(shipping_c_counts.index)]
top_countries


# In[19]:


sns.set_palette("Paired")
ax1 = sns.countplot(data= top_countries, y= "Shipping_Country", order=top_countries["Shipping_Country"].value_counts(ascending = False).index, color = "pink")
ax1.set_xticks(ax1.get_xticks())
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=10, ha="right", fontsize = 10)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
plt.title("Top Countries")
plt.xlabel("Order Count")
plt.ylabel("Country")
plt.tight_layout()


# In[70]:


top_countries_counts = top_countries["Shipping_Country"].value_counts(ascending=False)
custom_palette = ["pink" if i < 5 else "lightsteelblue" for i in range(len(top_countries_counts))]

sns.set_palette(custom_palette)
ax1 = sns.countplot(data=top_countries, x="Shipping_Country", order=top_countries_counts.index)
ax1.set_yticks(ax1.get_yticks())
ax1.set_yticklabels(ax1.get_yticklabels(), ha="right", fontsize=10)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))

# Manually set colors for each bar
for i, bar in enumerate(ax1.patches):
    if i < 5:
        bar.set_facecolor('cornflowerblue')
    else:
        bar.set_facecolor('lightsteelblue')

plt.title("TOP SHIPPING COUNTRIES", fontsize= 11)
plt.ylabel( "Orders Made in Total")
plt.xlabel("Country")
plt.tight_layout()
plt.savefig("top_count_plot.png", transparent = True)



# In[48]:


filtered_df = df[df['Shipping_Country'].isin(['DE', 'SE', 'GB', 'DK', 'NL'])]

# Calculate count of rows for each country
country_counts = filtered_df['Shipping_Country'].value_counts()

# Calculate total count of rows in Shipping_Country column
total_count = len(df['Shipping_Country'])

# Compute percentage for each country
percentage_DE = round((country_counts['DE'] / total_count) * 100, 2)
percentage_SE = round((country_counts['SE'] / total_count) * 100, 2)
percentage_GB = round((country_counts['GB'] / total_count) * 100, 2)
percentage_DK = round((country_counts['DK'] / total_count) * 100, 2)
percentage_NL = round((country_counts['NL'] / total_count) * 100, 2)

print("Percentage of DE:", percentage_DE)
print("Percentage of SE:", percentage_SE)
print("Percentage of GB:", percentage_GB)
print("Percentage of DK:", percentage_DK)
print("Percentage of NL:", percentage_NL)


# In[55]:


rev_per_count = df.groupby(["Shipping_Country"])[["Profit"]].sum().sort_values(by="Profit", ascending = False).head(15).reset_index()
rev_per_count


# In[73]:


custom_palette = ["cornflowerblue" if i < 6 else "lightsteelblue" for i in range(len(rev_per_count))]
ax13 = sns.barplot(data= rev_per_count, x= "Shipping_Country", y = "Profit", estimator = "sum", errorbar=None)
ax13.set_xticks(ax13.get_xticks())
ax13.set_xticklabels(ax13.get_xticklabels(), ha="right", fontsize = 10)
for i, bar in enumerate(ax13.patches):
    if i < 6:
        bar.set_facecolor('cornflowerblue')
    else:
        bar.set_facecolor('lightsteelblue')
plt.title("PROFIT FOR TOP SELLING COUNTRIES")
plt.ylabel( "Total Profit")
plt.xlabel("Country")
ax13.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))  # Format y-axis ticks as integers without decimal places
plt.tight_layout()
plt.savefig("top_profi_coun.png", transparent = True)



# In[ ]:





# In[18]:


ax2 = sns.barplot(data=df, x="Shipping_Country", y ="Profit", estimator = "sum", errorbar=None, order= rev_per_count.index)
ax2.set_xticks(ax2.get_xticks())
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=50, ha="right", fontsize = 10)
plt.title("Profit per Country") 
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))  # Format y-axis ticks as integers without decimal places
plt.tight_layout()


# In[51]:


top_c_prof = df[df["Shipping_Country"].isin(rev_per_count.index)]
top_c_prof


# In[ ]:





# In[52]:


sns.set_palette("Paired")
ax3 = sns.barplot(data= top_countries, x= "Shipping_Country", y = "Profit", estimator = "sum", order=top_countries["Shipping_Country"].value_counts(ascending = False).index, color = "pink", errorbar=None)
ax3.set_xticks(ax3.get_xticks())
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=20, ha="right", fontsize = 10)
plt.title("PROFIT FOR TOP SELLING COUNTRIES")
plt.xlabel( "Total Profit")
plt.ylabel("Country")
ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))  # Format y-axis ticks as integers without decimal places
plt.tight_layout()


# In[75]:


df["Order_Date"] = pd.to_datetime(df["Order_Date"])


# In[76]:


df.dtypes


# In[77]:


df_19 = df[df['Order_Date'].dt.year == 2019]
monthly_sales_19 = df_19.groupby(pd.Grouper(key="Order_Date", freq="ME"))[["Profit"]].sum()


# In[78]:


ax4 = sns.lineplot(data = monthly_sales_19, x=monthly_sales_19.index, y="Profit", marker="o")
plt.title("PROFIT PER MONTH IN 2019") 
ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))  # Format y-axis ticks as integers without decimal places
plt.grid(True)
plt.tight_layout()


# In[79]:


df_20 = df[df['Order_Date'].dt.year == 2020]
monthly_sales_20 = df_20.groupby(pd.Grouper(key="Order_Date", freq="ME"))[["Profit"]].sum()


# In[80]:


ax5 = sns.lineplot(data = monthly_sales_20, x=monthly_sales_20.index, y="Profit", marker="o")
plt.title("PROFIT PER MONTH IN 2020") 
ax5.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))  # Format y-axis ticks as integers without decimal places
ax5.set_xticks(ax5.get_xticks())
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=40, ha="right", fontsize = 10)
plt.grid(True)
plt.tight_layout()


# In[82]:


monthly_sales = df.groupby(pd.Grouper(key="Order_Date", freq="ME"))[["Profit"]].sum()
monthly_sales


# In[86]:


ax6 = sns.lineplot(data = monthly_sales, x=monthly_sales.index, y="Profit", marker="o", color = "cornflowerblue")
plt.title("PROFIT FOR THE YEARS 2018-2020") 
ax6.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))  # Format y-axis ticks as integers without decimal places
ax6.set_xticks(ax6.get_xticks())
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=20, fontsize = 10)
plt.xlabel( "Date")
plt.ylabel("Total Profit")
plt.grid(True)
plt.tight_layout()
plt.savefig("prof_years_plot.png", transparent = True)


# # AVERAGE PROFIT PER ORDER PER TIME

# In[97]:


avg_profit_over_time


# In[87]:


avg_profit_over_time = df.groupby(pd.Grouper(key='Order_Date', freq='ME'))['Profit'].mean().reset_index()
ax7 = sns.lineplot(data = avg_profit_over_time, x= "Order_Date", y="Profit", marker="o")
plt.title("AVG ORDER PROFIT OVER TIME") 
ax7.set_xticks(ax7.get_xticks())
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=40, ha="right", fontsize = 10)
plt.grid(True)
plt.tight_layout()


# In[98]:


avg_ord_cost_time = df.groupby(pd.Grouper(key='Order_Date', freq='ME'))['Total_amount_excl__VAT'].mean().reset_index()
avg_ord_cost_time


# In[96]:


sns.lineplot(data=avg_profit_over_time, x= "Order_Date", y="Profit", marker='o', label='Average Profit per Order', color="dodgerblue")
sns.lineplot(data=avg_ord_cost_time, x= "Order_Date", y="Total_amount_excl__VAT", marker='o', label='Average Price of Order', color ="salmon")

plt.title('AVERAGE PROFIT AND PRICE OF ORDER OVER TIME')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.xticks(rotation=25)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prof_over_time.png", transparent = True)


# In[100]:


missing_values_count = df.groupby(pd.Grouper(key='Order_Date', freq='ME'))['Profit'].apply(lambda x: x.isnull().sum()).reset_index(name='Missing_Values_Count')
missing_values_count


# In[72]:


sns.lineplot(data=missing_values_count, x='Order_Date', y='Missing_Values_Count', marker='o', label='Count of Missing Values in Profit Column')


# In[127]:


missing_values_count


# In[104]:


missing_values_count = df_20.groupby(pd.Grouper(key='Order_Date', freq='ME'))['Profit'].apply(lambda x: x.isnull().sum()).reset_index(name='Missing_Values_Count')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(missing_values_count['Order_Date'], missing_values_count['Missing_Values_Count'], marker='o', linestyle='-', color = "blueviolet")
plt.title('MISSING VALUES IN PROFIT DATA')
plt.xlabel('Date')
plt.ylabel('Missing Values')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("nullvalues.png", transparent = True)


# In[114]:


top_1_ = df_20[df_20["Shipping_Country"].isin(c_counts.index)]
profit_per_country_20 = top_1_.groupby(['Shipping_Country', pd.Grouper(key='Order_Date', freq='ME')])['Profit'].sum().reset_index()
profit_per_country_20


# In[105]:


c_counts = df["Shipping_Country"].value_counts().sort_values(ascending = False).head(6)
top_10 = df[df["Shipping_Country"].isin(c_counts.index)]
sales_per_country_over_time = top_10.groupby(['Shipping_Country', pd.Grouper(key='Order_Date', freq='ME')])['Order_ID'].count().reset_index()

# Plot the count of sales over time for each shipping country
plt.figure(figsize=(10, 6))
sns.lineplot(data=sales_per_country_over_time, x='Order_Date', y='Order_ID', hue='Shipping_Country', marker='o')
plt.title('SALES PER COUNTRY OVER TIME')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[124]:


palette = sns.color_palette("husl", n_colors=len(c_counts))

# Plot the count of sales over time for each shipping country
plt.figure(figsize=(10, 6))
ax16 = sns.lineplot(data=profit_per_country_20, x='Order_Date', y='Profit', hue='Shipping_Country', palette=palette, marker='o')
plt.title('PROFIT PER COUNTRY OVER 2020')
plt.xlabel('Date')
plt.ylabel('Total Profit')
plt.xticks(rotation=25, ha='right')
ax16.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))  # Format y-axis ticks as integers without decimal places
plt.legend(title='Country', bbox_to_anchor=(1, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("pr2020.png", transparent = True)


# In[107]:


profit_per_country_over_time = top_10.groupby(['Shipping_Country', pd.Grouper(key='Order_Date', freq='ME')])['Profit'].sum().reset_index()

# Plot the count of sales over time for each shipping country
plt.figure(figsize=(10, 6))
sns.lineplot(data=profit_per_country_over_time, x='Order_Date', y='Profit', hue='Shipping_Country', marker='o')
plt.title('Count of Sales per Shipping Country Over Time')
plt.xlabel('Date')
plt.ylabel('Total Profit')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[88]:


orders_per_month = df.groupby(pd.Grouper(key='Order_Date', freq='ME')).size()

# Calculate the average count of orders per month
average_orders_per_month = orders_per_month.mean()

# Plot the average count of orders per month
plt.figure(figsize=(10, 6))
sns.lineplot(data=orders_per_month, marker='o')
plt.axhline(y=average_orders_per_month, color='r', linestyle='--', label=f'Average ({average_orders_per_month:.2f} orders)')
plt.title('Average Count of Orders per Month')
plt.xlabel('Date')
plt.ylabel('Count of Orders')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()


# In[126]:


orders_per_month_2020 = df_20.groupby(pd.Grouper(key='Order_Date', freq='M'))['Order_ID'].nunique().reset_index()

orders_per_month_2020


# In[ ]:


filtered_df2 = df[df['Shipping_Country'].isin(['DE', 'SE', 'GB', 'DK', 'FR'])]

# Calculate count of rows for each country
country_counts = filtered_df2.groupby('Shipping_Country')["Profit"].sum()

# Calculate total count of rows in Shipping_Country column
total_count = sum(df['Profit'])

# Compute percentage for each country
percentage_DE = round((country_counts['DE'] / total_count) * 100, 2)
percentage_SE = round((country_counts['SE'] / total_count) * 100, 2)
percentage_GB = round((country_counts['GB'] / total_count) * 100, 2)
percentage_DK = round((country_counts['DK'] / total_count) * 100, 2)
percentage_NL = round((country_counts['NL'] / total_count) * 100, 2)

print("Percentage of DE:", percentage_DE)
print("Percentage of SE:", percentage_SE)
print("Percentage of GB:", percentage_GB)
print("Percentage of DK:", percentage_DK)
print("Percentage of NL:", percentage_NL)


# In[191]:


filtered_df2 = df[df['Shipping_Country'].isin(['DE', 'SE', 'GB', 'DK', 'FR', 'NL'])]
country_prof = filtered_df2.groupby('Shipping_Country')["Profit"].sum()
total_proff = country_prof.sum()
total_proff


# In[135]:


country_prof


# In[158]:


country_profit = df.groupby('Shipping_Country')['Profit'].sum()

# Calculate total profit across all countries
total_profit = country_profit.sum()

# Calculate the percentage of each country's profit relative to the total profit
country_profit_percentage = ((country_profit / total_profit) * 100).reset_index()

country_profit_percentage.sort_values(by = "Profit", ascending = False).reset_index(drop=True)
country_profit_percentage = country_profit_percentage.rename(columns={"Shipping_Country": "Country", "Profit": "Profit %"}).sort_values(by = "Profit %", ascending = False).reset_index(drop=True)


# In[159]:


country_profit_percentage.index += 1


# In[160]:


country_profit_percentage


# In[170]:


monthly_shipping = df.groupby(pd.Grouper(key="Order_Date", freq="ME"))[["Shipping_fee_excl__VAT"]].mean()

sns.lineplot(monthly_shipping, x='Order_Date', y='Shipping_fee_excl__VAT', marker='o')
plt.title('AVERAGE SHIPPING FEE OVER TIME')
plt.xlabel('Date')
plt.ylabel('Average Shipping Fee')
plt.tight_layout()


# In[164]:


shipping_avg = df.groupby('Shipping_Country')['Shipping_fee_excl__VAT'].mean()

# Group by 'Shipping_Country' and calculate the count of orders for each country
orders_count = df.groupby('Shipping_Country')['Order_ID'].count()

# Merge the two calculated values into a single DataFrame
shipping_orders_avg = pd.DataFrame({'Average_Shipping_Fee': shipping_avg, 'Orders_Count': orders_count}).reset_index()

# Calculate the correlation between the two columns, handling NaN values
correlation = shipping_orders_avg['Average_Shipping_Fee'].corr(shipping_orders_avg['Orders_Count'], method='pearson')


# In[165]:


correlation


# In[176]:


shipping_avg = df.groupby('Shipping_Country')['Shipping_fee_excl__VAT'].mean()

# Group by 'Shipping_Country' and calculate the count of orders for each country
orders_count = df.groupby('Shipping_Country')['Order_ID'].count()

# Merge the two calculated values into a single DataFrame
shipping_orders_avg = pd.DataFrame({'Average_Shipping_Fee': shipping_avg, 'Orders_Count': orders_count}).reset_index()

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(shipping_orders_avg['Average_Shipping_Fee'], shipping_orders_avg['Orders_Count'], color="darkorchid")
plt.title('SHIPPING FEE VS AVERAGE COUNT ORDER')
plt.xlabel('Average Shipping Fee')
plt.ylabel('Orders Count')
plt.grid(True)
plt.tight_layout()
plt.savefig("corr.png", transparent = True)


# In[167]:


# Group by 'Shipping_Country' and calculate the average profit for each country
profit_avg_ = df.groupby('Shipping_Country')['Profit'].mean()

# Merge the two calculated values into a single DataFrame
shipping_profit_avg_ = pd.DataFrame({'Average_Shipping_Fee': shipping_avg, 'Average_Profit': profit_avg_}).reset_index()

# Calculate the correlation between the two columns
correlation2 = shipping_profit_avg_['Average_Shipping_Fee'].corr(shipping_profit_avg_['Average_Profit'])


# In[168]:


correlation2


# In[171]:


plt.figure(figsize=(10, 6))
plt.scatter(shipping_profit_avg_['Average_Shipping_Fee'], shipping_profit_avg_['Average_Profit'], color ="darkorchid")
plt.title('Scatter Plot: Average Shipping Fee vs Average Profit per Country')
plt.xlabel('Average Shipping Fee')
plt.ylabel('Average Profit')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[179]:


df.groupby("Shipping_Country")[["Profit"]].sum().sort_values(by="Profit", ascending = False)


# In[190]:


# Plot the count of sales over time for each shipping country
plt.figure(figsize=(10, 6))
ax20 = sns.lineplot(data=profit_per_country_19 , x='Order_Date', y='Profit', hue='Shipping_Country', marker='o')
plt.title('PROFIT PER COUNTRY')
plt.xlabel('Date')
plt.ylabel('Total Profit')
plt.xticks(rotation=25, ha='right')
ax20.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))  # Format y-axis ticks as integers without decimal places
plt.legend(title='Country', bbox_to_anchor=(1, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[185]:


df_19.groupby()


# In[192]:


profit_per_country_19 = filtered_df2.groupby(['Shipping_Country', pd.Grouper(key='Order_Date', freq='ME')])['Profit'].sum().reset_index()


# In[189]:


profit_per_country_19


# In[194]:


palette = sns.color_palette("husl", n_colors=len(profit_per_country_19['Shipping_Country'].unique()))

# Plot the count of sales over time for each shipping country
plt.figure(figsize=(10, 6))
ax20 = sns.lineplot(data=profit_per_country_19 , x='Order_Date', y='Profit', hue='Shipping_Country', palette=palette, marker='o')
plt.title('PROFIT PER COUNTRY OVER TIME')
plt.xlabel('Date')
plt.ylabel('Total Profit')
plt.xticks(rotation=25, ha='right')
ax20.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))  # Format y-axis ticks as integers without decimal places
plt.legend(title='Country', bbox_to_anchor=(1, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("P_C_T.png", transparent = True)


# In[ ]:




