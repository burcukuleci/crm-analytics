##############################################################
# Customer Lifetime Value(CLV) Prediction with BG-NBD and Gamma-Gamma Models
##############################################################

###############################################################
# Business Problem
###############################################################

# FLO wants to determine a roadmap for sales and marketing activities.

# In order for the company to make a medium-long term plan,
# the potential value (CLV) that existing customers will provide to the company in the future must be estimated.


###############################################################
# Dataset
###############################################################

# The dataset consists of information obtained from the past shopping behaviors of
# customers who made their last purchases as OmniChannel (both online and offline shopping) in 2020 - 2021.

# master_id: unique customer id.
# order_channel : channel of the shopping platform. (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : the channel where the last purchase was made.
# first_order_date : date of the customer's first purchase.
# last_order_date : date of the customer's last purchase.
# last_order_date_online : date of the customer's first purchase on the online platform.
# last_order_date_offline : date of the customer's first purchase on the offline platform.
# order_num_total_ever_online : the total number of purchases made by the customer on the online platform.
# order_num_total_ever_offline : the total number of purchases made by the customer on the offline platform.
# customer_value_total_ever_offline : total amount paid by the customer for offline shopping.
# customer_value_total_ever_online : total amount paid by the customer for offline shopping.
# interested_in_categories_12 : list of categories the customer has shopped in the last 12 months.


###############################################################
# 1. Data Preparation
###############################################################

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None

# Read the csv file and make a copy.
df_ = pd.read_csv("datasets/flo_data_20K.csv")
df = df_.copy()

# Define functions for outlier analysis.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)  # integer
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


# Replace outliers from columns of "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" with thresholds.

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)


# Create new variables for each customer's total purchases/transactions "order_num_total" and total spend "customer_value_total".
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Convert columns of date to date data type.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df.info()



###############################################################
# 2. Creating the parameters for CLV calculation.
###############################################################

# Define analysis date as two days ater maximum order date in the dataset.
df["last_order_date"].max()   # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

# Create new dataframe 'clv_df' with clumns of "customer_id", "recency_clv_weekly", "T_weekly", "frequency" and "monetary_clv_avg".
clv_df = pd.DataFrame()
clv_df["customer_id"] = df["master_id"]

recency_diff = (df["last_order_date"] - df["first_order_date"]).dt.days   # dtype: int

clv_df["recency_clv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7.0
clv_df["T_weekly"] = (analysis_date - df["first_order_date"]).dt.days / 7.0
clv_df["frequency"] = df["order_num_total"]    # total number of transaction
clv_df["monetary_clv_avg"] = df["customer_value_total"] / df["order_num_total"]    # monetary per purchase



###############################################################
# 3. Fitting BG/NBD and Gamma-Gamma Models and Calculating CLV for 6 Months
###############################################################

clv_df.describe()

# Fit BG/NBD model.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(clv_df['frequency'],
        clv_df['recency_clv_weekly'],
        clv_df['T_weekly'])

# Predict the expected sales for 3 months (12 weeks): "exp_sales_3_month"
clv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       clv_df['frequency'],
                                       clv_df['recency_clv_weekly'],
                                       clv_df['T_weekly'])

# Predict the expected sales for 6 months (24 weeks): "exp_sales_6_month"
clv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       clv_df['frequency'],
                                       clv_df['recency_clv_weekly'],
                                       clv_df['T_weekly'])

clv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

clv_df.sort_values("exp_sales_6_month",ascending=False)[:10]



# Fit Gamma-Gamma model.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(clv_df['frequency'],
        clv_df['monetary_clv_avg'])

# Predict expected average value of customers: "exp_average_value"
clv_df["exp_average_value"] = ggf.conditional_expected_average_profit(clv_df['frequency'],
                                                                      clv_df['monetary_clv_avg'])

# Calculate CLV for the customers for 6 months.
clv = ggf.customer_lifetime_value(bgf,
                                   clv_df['frequency'],
                                   clv_df['recency_clv_weekly'],
                                   clv_df['T_weekly'],
                                   clv_df['monetary_clv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

clv_df["clv"] = clv

# Return top 20 customers with the highest CLV.
clv_df.sort_values("clv",ascending=False)[:20]



###############################################################
# 4. Customer Segmentation
###############################################################

# Create 4 segments according to clv. segment D is customers with the lowest clv.
clv_df["clv_segment"] = pd.qcut(clv_df["clv"], 4, labels=["D", "C", "B", "A"])

clv_df.head()

clv_df.groupby(["clv_segment"])["clv"].agg({"count", "mean"})

#               mean  count
# clv_segment
# D            80.34   4987
# C           138.31   4986
# B           199.53   4986
# A           362.32   4986

# --  further segment the "A" group into sub-segments A1 and A2 because there's a greater difference in CLV values between segments A and B.
#     This can provide more granular insights into your most valuable customers and allow for more targeted strategies for each sub-segment.

# First, filter out the A segment
segment_A = clv_df[clv_df["clv_segment"] == "A"]

# Now, create two sub-segments within segment A using qcut
segment_A["clv_sub_segment"] = pd.qcut(segment_A["clv"], 2, labels=["A2", "A1"])

# Update the original dataframe 'clv_df' by merging back the sub-segments
clv_df = clv_df.merge(segment_A[["clv", "clv_sub_segment"]], on="clv", how="left")

# Update the 'clv_segment' column conditionally
clv_df["clv_segment"] = clv_df.apply(
    lambda row: row["clv_sub_segment"] if row["clv_segment"] == "A" else row["clv_segment"], axis=1)

# Clean up the temporary column
clv_df.drop(columns=["clv_sub_segment"], inplace=True)

# analayze the segments.
clv_df.groupby(["clv_segment"])["clv"].agg({"count", "mean"})

#               mean  count
# clv_segment
# A1          452.21   2493
# A2          272.42   2495
# B           199.53   4986
# C           138.31   4986
# D            80.34   4987



###############################################################
# 5. Functionality
###############################################################

def create_clv_df(dataframe):

    # data preparation
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # CLV parameters
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    clv_df = pd.DataFrame()
    clv_df["customer_id"] = dataframe["master_id"]
    clv_df["recency_clv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).dt.days) / 7
    clv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).dt.days) / 7
    clv_df["frequency"] = dataframe["order_num_total"]
    clv_df["monetary_clv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    clv_df = clv_df[(clv_df['frequency'] > 1)]

    # BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(clv_df['frequency'],
            clv_df['recency_clv_weekly'],
            clv_df['T_weekly'])

    # predicting expected sales
    clv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               clv_df['frequency'],
                                               clv_df['recency_clv_weekly'],
                                               clv_df['T_weekly'])

    clv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               clv_df['frequency'],
                                               clv_df['recency_clv_weekly'],
                                               clv_df['T_weekly'])

    # # Gamma-Gamma Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(clv_df['frequency'], clv_df['monetary_clv_avg'])
    clv_df["exp_average_value"] = ggf.conditional_expected_average_profit(clv_df['frequency'],
                                                                          clv_df['monetary_clv_avg'])

    # Predict CLV
    clv = ggf.customer_lifetime_value(bgf,
                                       clv_df['frequency'],
                                       clv_df['recency_clv_weekly'],
                                       clv_df['T_weekly'],
                                       clv_df['monetary_clv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    clv_df["clv"] = clv

    # Customer Segmentation
    clv_df["clv_segment"] = pd.qcut(clv_df["clv"], 4, labels=["D", "C", "B", "A"])

    return clv_df

clv_df = create_clv_df(df)

clv_df.head(10)


