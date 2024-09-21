
###############################################################
# Customer Segmentation with RFM
###############################################################

###############################################################
# Business Problem
###############################################################

# FLO wants to divide its customers into segments and determine marketing strategies according to these segments.
# Customer behaviors will be defined and groups will be created according to these behavior clusters.

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
#### 1. Data Preparation
###############################################################

import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

# Read flo_data_20K.csv file. Create a copy of the dataframe.
df_ = pd.read_csv("datasets/flo_data_20K.csv")
df = df_.copy()
df.head()

# Create new columns for total numer of orders for each custemer as "order_num_total" and total amount of expenses of each customer as "customer_value_total".
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df[ "customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Examine the variable types. Convert the type of variables representing dates to 'date'.
df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Count total number of customers, orders and total amount of spending for each order_channel.
df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total": "sum",
                                 "customer_value_total": "sum"})

# Top 10 customers with the highest spending.
df.sort_values("customer_value_total", ascending=False)["master_id"].head(10)

# Top 10 customers with highest number of orders.
df.sort_values("order_num_total", ascending=False)[:10]

# Define a function for above steps of data preperation.

def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df



###############################################################
#### 2. Calculating RFM Metrics.
###############################################################

# Select the analysis date two days later than the latest order date in the dataframe.
df["last_order_date"].max()    # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

# Create an empty dataframe for customer_id, recency, frequency ve monetary values.
rfm = pd.DataFrame()

# Add customer_id information.
rfm["customer_id"] = df["master_id"]

# df["last_order_date"] zaten her bir müşteri için ayrı ayrı girilmiş, groupby yapmama gerek yok
type(analysis_date)

# RECENCY - the number of days from the last purchase date to the analysis date.
rfm["recency"] = (analysis_date - df["last_order_date"]).dt.days

# FREQUENCY - total number of orders for a customer
rfm["frequency"] = df["order_num_total"]

# MONETARY - total amount of spending for a customer
rfm["monetary"] = df["customer_value_total"]



###############################################################
#### 3. Calculating RF and RFM Scores
###############################################################

# Convert RFM metric values into scores on a scale of 1 to 5 using pd.qcut().

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Define 'RF_SCORE' by merging 'recency_score' and 'frequency_score'.
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

# Define 'RF_SCORE' by merging 'recency_score', 'frequency_score' and 'monetary_score'.
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

rfm.head()
rfm.info()



###############################################################
#### 4. Defining RF Scores as Segments
###############################################################

# Create a dictionary with key-value pairs as RegEx and segment names.
# RegEx is used to choose RF scores based on the defined rule.

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

# Create segment column using 'seg_map' dictionary.
rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head()

# number of customers in each segment.
rfm["segment"].value_counts()



###############################################################
#### 5. Segment Analysis
###############################################################

# --- Mean of recency, frequency and monetary for each segment.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

#                          recency       frequency       monetary
#                        mean count      mean count     mean count
# segment
# about_to_sleep       113.79  1629      2.40  1629   359.01  1629
# at_Risk              241.61  3131      4.47  3131   646.61  3131
# cant_loose           235.44  1200     10.70  1200  1474.47  1200
# champions             17.11  1932      8.93  1932  1406.63  1932
# hibernating          247.95  3604      2.39  3604   366.27  3604
# loyal_customers       82.59  3361      8.37  3361  1216.82  3361
# need_attention       113.83   823      3.73   823   562.14   823
# new_customers         17.92   680      2.00   680   339.96   680
# potential_loyalists   37.16  2938      3.30  2938   533.18  2938
# promising             58.92   647      2.00   647   335.67   647


# --- With the help of RFM analysis, find the customers in the relevant profile for 2 cases and save the customer IDs to csv files.

# a. FLO is adding a new women's shoe brand to its portfolio.
#    The product prices of the new brand are above general customer preferences.
#    For this reason, it is desired to contact customers with a profile that will be interested in the brand's promotion and product sales.
#    It is planned that these customers will be loyal (loyal_customers) and customers shopped from the women's (KADIN) category.
#    Save target customer ID's to "new_brand_target_ids" csv file.

# choose customer ids from 'champions' and 'loyal_customers' segments.
target_segment_customer_ids = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]["customer_id"]

# choose customer ids from "target_segment_customer_ids" that shopped from 'KADIN' category.
customer_ids = df[(df["master_id"].isin(target_segment_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]

# save target customer ids to a csv file.
customer_ids.to_csv("new_brand_target_ids.csv", index=False)



# b. A discount of nearly 40% is planned for men's (ERKEK) and children's (COCUK) products.
#    This discount is intended to specifically target customers who are good customers in the past but have not shopped for a long time
#    (cant_loose, hibernating) and new customers (new_customers) who are interested in the relevant categories.
#    Save target customer ID's to "discount_target_customer_ids" csv file.

# categories
df["interested_in_categories_12"].unique()

# choose ids from the target segments.
target_segment_customer_ids = rfm[rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"])]["customer_id"]

#  choose customer ids from "target_segment_customer_ids" that shopped from 'ERKEK' or 'COCUK' categories.
cust_ids = df[(df["master_id"].isin(target_segment_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]

# save target customer ids to a csv file.
cust_ids.to_csv("discount_target_customer_ids.csv", index=False)



###############################################################
####  6. Functionalization
###############################################################

# Define a function including above steps.
def create_rfm(dataframe):
    # Data preperation
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)


    # Calculating RFM metrics
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    rfm = pd.DataFrame()
    rfm["customer_id"] = dataframe["master_id"]
    rfm["recency"] = (analysis_date - dataframe["last_order_date"]).dt.days
    rfm["frequency"] = dataframe["order_num_total"]
    rfm["monetary"] = dataframe["customer_value_total"]

    # Calculating RF and RFM scores.
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))


    # Customer segmentation.
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

    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

    return rfm[["customer_id", "recency", "frequency", "monetary", "RF_SCORE", "RFM_SCORE", "segment"]]

rfm_df = create_rfm(df)



