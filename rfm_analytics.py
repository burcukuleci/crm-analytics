
###############################################################
# Customer Segmentation with RFM
###############################################################

# 1. Business Problem
# 2. Data Understanding
# 3. Data Preparation
# 4. Calculating RFM Metrics
# 5. Calculating RFM Scores
# 6. Creating & Analysing RFM Segments
# 7. Defining Functions

###############################################################
# 1. Business Problem
###############################################################

# Dataset
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# The dataset named Online Retail II includes the sales of a UK-based online retail store between 01/12/2009 - 09/12/2011.

# Variables
#
# InvoiceNo: Invoice number. Unique number for each transaction, i.e. invoice. Cancelled transaction if it starts with C.
#            More than one product may have been purchased with an invoice - then there will be more than one line with this invoice information.
# StockCode: Product code. A unique number for each product.
# Description: Product name.
# Quantity: Number of products. Indicates how many of the products on the invoices were sold.
# InvoiceDate: Invoice date and time.
# UnitPrice: Product price (in pounds)
# CustomerID: Unique customer number
# Country: Country name. Country where the customer lives.


###############################################################
# 2. Data Understanding
###############################################################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)   # float digit customization

# read the excell file.
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()

df.head()
df.shape

df.info()

#check missing values
df.isnull().sum()
# There seems to be 100 thousand missing values ​​for "Customer ID". Since customer segmentation cannot be done
#    with this missing information, it would be better to completely delete the sales with missing IDs.

# Question:What is the total bill for an invoice?
invoice_df = df[df["Invoice"] == 489434]
invoice_total = invoice_df.apply(lambda row: row["Price"] * row["Quantity"], axis=1).sum()
# lets round the float result
invoice_total =round(invoice_total, 2)

# use the apply function with axis=1 to apply the lambda function to each row of the DataFrame.
# lambda function takes each row as input and calculates the product of 'Price' and 'Quantity'. > row["Price"] * row["Quantity"].

#####
# number of unique invoice?
df["Invoice"].nunique()

# number of unique product?
df["Description"].nunique()

# New Variable: Total spending for a product in an invoice
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Total spending for an invoice
df.groupby("Invoice")["TotalPrice"].sum()
df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()

# Invoices starting with C resulted in negative bill.



###############################################################
# 3. Data Preparation
###############################################################

df.isnull().sum()
# There are missing values in columns "Description" and "Customer ID".

# drop rows with missing values.
df.dropna(inplace=True)

# drop out the invoices starting with C.
df = df[~df["Invoice"].str.contains("C", na=False)]




###############################################################
# 4. Calculating RFM Metrics
###############################################################

# Recency, Frequency, Monetary

# find the latest date in the dataframe.
df["InvoiceDate"].max()   # Timestamp('2010-12-09 20:01:00')

# > Create analysis date. (reference date to calculate recency.) - two days later than max invoice date.
today_date = dt.datetime(2010, 12, 11)
type(today_date)

# > create new dataframe 'rfm' frm 'df' with the same column names.
# For each customer:
#   InvoiceDate: Days passed from last purchase date until the analysis date.
#   Invoice: number of unique invoices of a customer. (shopping frequency)
#   TotalPrice: total price that a customer paid for all the invoices. (monetary)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

# > change the column names.
# InVoiceDate > recency , Invoice > frequency, TotalPrice > monetary
rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T

# Filter out the monetary values of zero.
rfm = rfm[rfm["monetary"] > 0]



###############################################################
# 5. Calculating RFM Scores
###############################################################

# Convert RFM metric values into scores on the scale of 1 to 5.

# RECENCY SCORE
#    customers with latest purchase will be scored higher (5).
#    use pd.qcut() > ascending order, smallest values will be scored highest.
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

# 0-100 (min-max) >  0-20, 20-40, 40-60, 60-80, 80-100

# FREQUENCY SCORE
#    customers who purchased a lot of times will be scored higher (5).
#    use pd.qcut() > ascending order, smallest values will be scored lowest.
#    To make the distribution between the score groups equal, each frequency value is given a unique value by using the rank method.
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# MONETARY SCORE
#    customers who purchased a lot of times will be scored higher (5).
#    use pd.qcut() > ascending order, smallest values will be scored lowest.
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# new score columns' data type is 'category'
rfm.info()


# RFM SCORE
#    merge recency and frequency score values as string.
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))



###############################################################
# 6. Creating & Analysing RFM Segments
###############################################################

# Use RegEx (Regular Expression) to select RFM scores.
# [1-2] : 1 or 2

# > create dictionary 'seg_map' : key=regex, value: segment name
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

# > create 'segment' column.
#    replace string method will match the RFM scores with a segment name.
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)


# analyze the segments.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])


# distribution of the segments.
import matplotlib.pyplot as plt

plt.hist(rfm["segment"])


# > create a new dataframe for "new_customers" segment.
new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

# convert ids to float.
new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)

# export dataframe as a csv file.
new_df.to_csv("new_customers.csv")

# export dataframe as a csv file.
rfm.to_csv("rfm.csv")

# customer id information can be extracted from the interested segment.



###############################################################
# 7. Defining Functions
###############################################################

def create_rfm(dataframe, csv=False):

    # DATA PREPERATION
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # CALCULATING RFM METRICS
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # CALCULATING RFM SCORES
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # RFM SCORE
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    # CUSTOMER SEGMENTATION
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)    # convert ID values to integer.

    # export rfm dataframe as csv file if csv = True.
    if csv:
        rfm.to_csv("rfm.csv")

    return rfm

df = df_.copy()

rfm_new = create_rfm(df, csv=True)
# RFM metric and score values are created for each customer.













