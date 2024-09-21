
##############################################################
# CLV Prediction with BG-NBD and Gamma-Gamma Models
##############################################################

# 1. Data Preparation
# 2. Expected Number of Transaction with BG-NBD Model
# 3. Expected Average Profit with Gamma-Gamma Model
# 4. Calculation of CLV with BG-NBD and Gamma-Gamma Model
# 5. Segmentation with CLV
# 6. Functionality

##############################################################
# 1. Data Preparation
##############################################################

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


import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


# determine the outliers.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# replace outliers with threshold values.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# read the excell file.
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df.head()

# drop missing values.
df.isnull().sum()
df.dropna(inplace=True)

# filter out return invoices.
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

# replace outliers with thresholds for columns of "Quantity" and "Price"
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# calculate total price of an item in an invoice
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

analysis_date = dt.datetime(2011, 12, 11)

#### calculate parameter values for CLV calculation.

# recency: time passed since last purchase of the customer. (week)
# T: time between first purchase of the customer and analysis date. (week)
# frequency: number of transactions. (frequency>1, customers who purchased more than once)
# monetary: average order value

# -- calculation of customer lifetime value (CLV):
# recency : (InvoiceDate.max() - InvoiceDate.min()).days
# T       : (analysis_date - InvoiceDate.min()).days
# frequency: Invoice.nunique()
# monetary : TotalPrice.sum()

clv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                                                        lambda InvoiceDate: (analysis_date - InvoiceDate.min()).days],
                                         'Invoice': lambda Invoice: Invoice.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

#             InvoiceDate             Invoice TotalPrice
#              <lambda_0> <lambda_1> <lambda>   <lambda>
# Customer ID
# 12346.0000            0        326        1   310.4400
# 12347.0000          365        368        7  4310.0000

# drop the name of df column names.
clv_df.columns = clv_df.columns.droplevel(0)

# rename the <lambda> column names.
clv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# calculate monetary as per transaction.
clv_df["monetary"] = clv_df["monetary"] / clv_df["frequency"]

clv_df.describe().T

# choose only the customers who purchased more than once.
clv_df = clv_df[(clv_df['frequency'] > 1)]

# expreess 'recency' and 'T' as weekly.
clv_df["recency"] = clv_df["recency"] / 7
clv_df["T"] = clv_df["T"] / 7



##############################################################
# 2. Expected Number of Transaction with BG-NBD Model
##############################################################

# define the model
bgf = BetaGeoFitter(penalizer_coef=0.001)

# fit the model
bgf.fit(clv_df['frequency'],
        clv_df['recency'],
        clv_df['T'])

# >> Who are the 10 customers we expect to purchase the most from in 1 week?

# use function "conditional_expected_number_of_purchases_up_to_time"
# t: period of week

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        clv_df['frequency'],
                                                        clv_df['recency'],
                                                        clv_df['T']).sort_values(ascending=False).head(10)

# same procedure can be done with sklearn's predict function.
bgf.predict(1,
            clv_df['frequency'],
            clv_df['recency'],
            clv_df['T']).sort_values(ascending=False).head(10)


# add expected purchase number within a week for each customer.
clv_df["expected_purc_1_week"] = bgf.predict(1,
                                              clv_df['frequency'],
                                              clv_df['recency'],
                                              clv_df['T'])


# >> Who are the 10 customers we expect to purchase the most from within a month?

# t: 4 weeks (one month)
bgf.predict(4,
            clv_df['frequency'],
            clv_df['recency'],
            clv_df['T']).sort_values(ascending=False).head(10)

# add expected purchase number within a month for each customer.

clv_df["expected_purc_1_month"] = bgf.predict(4,
                                               clv_df['frequency'],
                                               clv_df['recency'],
                                               clv_df['T'])

# >> What is the expected total sales for a month?

bgf.predict(4,
            clv_df['frequency'],
            clv_df['recency'],
            clv_df['T']).sum()


# >> total number of expected purchases for 3 months

bgf.predict(4 * 3,
            clv_df['frequency'],
            clv_df['recency'],
            clv_df['T']).sum()

# add expected purchase number within 3 month for each customer.

clv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               clv_df['frequency'],
                                               clv_df['recency'],
                                               clv_df['T'])


# >> Analyze expected purchase amounts

plot_period_transactions(bgf)
plt.show()



##############################################################
# 3. Expected Average Profit with Gamma-Gamma Model
##############################################################

# define the model
ggf = GammaGammaFitter(penalizer_coef=0.01)

# fit the model.
ggf.fit(clv_df['frequency'], clv_df['monetary'])

# >> top 10 customers with highest average profits
ggf.conditional_expected_average_profit(clv_df['frequency'],
                                        clv_df['monetary']).sort_values(ascending=False).head(10)

# add expected average profit for each customer
clv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(clv_df['frequency'],
                                                                             clv_df['monetary'])

clv_df.sort_values("expected_average_profit", ascending=False).head(10)



##############################################################
# 4. Calculation of CLV with BG-NBD and Gamma-Gamma Model
##############################################################

# use "customer_lifetime_value" function

clv = ggf.customer_lifetime_value(bgf,
                                   clv_df['frequency'],
                                   clv_df['recency'],
                                   clv_df['T'],
                                   clv_df['monetary'],
                                   time=3,  # 3 months
                                   freq="W",  # week
                                   discount_rate=0.01)

clv.head()

# reset the index - columns: Customer ID, clv
clv = clv.reset_index()

# expected purchase of 3 month for each customer.
clv_df["expected_purc_3_month"] = bgf.predict(4*3,
                                               clv_df['frequency'],
                                               clv_df['recency'],
                                               clv_df['T'])

# merge all data together.
clv_final = clv_df.merge(clv, on="Customer ID", how="left")

clv_final.sort_values(by="clv", ascending=False).head(10)

# low recency/T yaşı (new customer), low frwquency, high monetary - HIGH CLV
# high recency/T, high frequency, low monetary - HIGH CLV
# All valuable customers can be captured with "ggf.customer_lifetime_value"



##############################################################
# 5. Segmentation with CLV
##############################################################

# form 4 segments with pd.qcut() according to clv values.
clv_final["segment"] = pd.qcut(clv_final["clv"], 4, labels=["D", "C", "B", "A"])

clv_final.sort_values(by="clv", ascending=False).head(50)

# analyze the segments
clv_final.groupby("segment").agg({"count", "mean", "sum"})

clv_segment_analysis = clv_final.groupby("segment").agg({"count", "mean", "sum"})



##############################################################
# 6. Functionality
##############################################################

def create_clv_period(dataframe, month=3):
    # 1. Data preparation
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    clv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    # do the correction for the parameters
    clv_df.columns = clv_df.columns.droplevel(0)
    clv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    clv_df["monetary"] = clv_df["monetary"] / clv_df["frequency"]
    clv_df = clv_df[(clv_df['frequency'] > 1)]
    clv_df["recency"] = clv_df["recency"] / 7
    clv_df["T"] = clv_df["T"] / 7

    # 2. BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(clv_df['frequency'],
            clv_df['recency'],
            clv_df['T'])

    # expected purchase
    clv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  clv_df['frequency'],
                                                  clv_df['recency'],
                                                  clv_df['T'])

    clv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   clv_df['frequency'],
                                                   clv_df['recency'],
                                                   clv_df['T'])

    clv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   clv_df['frequency'],
                                                   clv_df['recency'],
                                                   clv_df['T'])

    # 3. GAMMA-GAMMA Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(clv_df['frequency'], clv_df['monetary'])
    clv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(clv_df['frequency'],
                                                                                 clv_df['monetary'])

    # 4. Calculating CLV with BG-NBD and GG models
    clv = ggf.customer_lifetime_value(bgf,
                                       clv_df['frequency'],
                                       clv_df['recency'],
                                       clv_df['T'],
                                       clv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    clv = clv.reset_index()
    clv_final = clv_df.merge(clv, on="Customer ID", how="left")

    # 5. Segmentation
    clv_final["segment"] = pd.qcut(clv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return clv_final

df = df_.copy()

clv_final = create_clv_period(df)
clv_final.to_csv("clv_prediction.csv")




