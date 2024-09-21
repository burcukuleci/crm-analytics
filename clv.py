
############################################
# CUSTOMER LIFETIME VALUE
############################################

# 1. Data Preparation
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (number of customers who purchased more than one / total number of customers)
# 5. Profit Margin (profit_margin =  total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Segmentation
# 9. Functionalization

###############################################################
# 1. Data Preparation
###############################################################

# Dataset
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# The dataset named Online Retail II includes the sales of a UK-based online retail store between 01/12/2009 - 09/12/2011.

# Variables

# InvoiceNo: Invoice number. Unique number for each transaction, i.e. invoice. Cancelled transaction if it starts with C.
#            More than one product may have been purchased with an invoice - then there will be more than one line with this invoice information.
# StockCode: Product code. A unique number for each product.
# Description: Product name.
# Quantity: Number of products. Indicates how many of the products on the invoices were sold.
# InvoiceDate: Invoice date and time.
# UnitPrice: Product price (in pounds)
# CustomerID: Unique customer number
# Country: Country name. Country where the customer lives.


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# read excell file.
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()

# filter out the invoices of returned orders.
df = df[~df["Invoice"].str.contains("C", na=False)]

# consider only positive quantity values.
df = df[(df['Quantity'] > 0)]

# drop missing values - CustomerID, Description
df.isnull().sum()
df.dropna(inplace=True)

# Calculate total price for each item within an invoice.
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

# For each customer:
# Calculate number of unique invoices > "total transaction", "frequency"
# Calculate total amount of item bought
# Total amount of spending > "monetary"
cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity': lambda x: x.sum(),
                                        'TotalPrice': lambda x: x.sum()})

# name the columns.
cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

# frequency == 'total_transaction'
# monetary == 'total_price'



##################################################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
##################################################

cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

##################################################
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
##################################################

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

##################################################
# 4. Repeat Rate & Churn Rate (number of customer purchased more than one / total number of customers)
##################################################

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]

churn_rate = 1 - repeat_rate

##################################################
# 5. Profit Margin (profit_margin =  total_price * 0.10)
##################################################

# profit margin for each customer
cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10

##################################################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
##################################################

cltv_c['customer_value'] = cltv_c['average_order_value'] * cltv_c["purchase_frequency"]

cltv_c.head()

##################################################
# 7. Customer Lifetime Value (CLV = (customer_value / churn_rate) x profit_margin)
##################################################

cltv_c["clv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

# sort CLV values in descending order.
cltv_c.sort_values(by="clv", ascending=False).head()
cltv_c.sort_values(by="clv", ascending=False).tail()

cltv_c.describe()

##################################################
# 8. Segmentation
##################################################

# create 4 segments with CLTV values.
# A = highest CLTV, D = lowest CLTV
cltv_c["segment"] = pd.qcut(cltv_c["clv"], 4, labels=["D", "C", "B", "A"])

# anaylze the segments.
cltv_c.groupby("segment").agg({"count", "mean", "sum"})

# save cltv_c dataframe to an csv file.
cltv_c.to_csv("clv_segment.csv")

# Customer ID
# 18102.00000       A
# 14646.00000       A
# 14156.00000       A
# 14911.00000       A
# 13694.00000       A


##################################################
# Functionalization
##################################################

def create_clv_segment(dataframe, profit=0.10):

    # Data preparation
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe['Quantity'] > 0)]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                                   'Quantity': lambda x: x.sum(),
                                                   'TotalPrice': lambda x: x.sum()})

    # Parameters to calculate CLTV
    cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']
    # avg_order_value
    cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']
    # purchase_frequency
    cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]
    # repeat rate & churn rate
    repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate
    # profit_margin
    cltv_c['profit_margin'] = cltv_c['total_price'] * profit
    # Customer Value
    cltv_c['customer_value'] = (cltv_c['avg_order_value'] * cltv_c["purchase_frequency"])

    # Customer Lifetime Value (CLTV)
    cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']

    # Segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c


df = df_.copy()

clv = create_clv_segment(df)

clv.head()





