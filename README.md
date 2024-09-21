# CUSTOMER RELATIONSHIP MANAGEMENT ANALYTICS

This repository contains Python code files for model development for each recommendation model type and three recommendation system projects.

***Note: The codes are from the Miuul 'Recommendation Systems' course. Some parts of the codes are modified by me.***

- Clone the 'Recommendation Systems' repository using your terminal or git bash.

```
git clone https://github.com/burcukuleci/

```
- Download all required packages using the requirements.txt file by running the below command in the terminal.

```
pip install -r requirements.txt
```

- All required data files (except 'rating.csv' and 'armut.csv') are in *datasets* directory. 

***Note: This README.md file provides short information for each Python file. Separate markdown files explain the code and the project in detail. Please refer to those markdown files for detailed information.***

**OUTLINE**

1. [RFM Analytics](#rfm-analytics)
2. [Case Study: FLO RFM Analytics](#case-study-flo-rfm-analytics)
3. [Customer Lifetime Value](#customer-lifetime-value)
4. [CLTV Prediction](#cltv-prediction)
5. [Case Study: FLO CLTV Prediction](#case-study-flo-cltv-prediction)
6. [](#)

--

Customer Relationship Management (CRM) Analytics deals with analyzing customer data, getting to know customers better, dividing customers into segments, making segment-specific business decisions, and developing customer churn models to retain customers.

**Key Performance Indicator (KPI)**s are mathematical indicators used to evaluate the performance of companies, departments or employees.

- **Customer Acquisition Rate**: The percentage of customers acquired in a given time period.
- **Customer Retention Rate**: The percentage of customers who remain loyal over a specific period.
- **Customer Churn Rate**: The percentage of customers who stop doing business with a company over a specific time period.
- **Conversion Rate**: The percentage of potential customers that take a desired action.


## RFM Analytics

> *python file*:  [rfm_analytics.py](rfm_analytics.py)

- Aim: Customer segmentation using RFM score.

- Method: Group customers acording to RFM score. Calculate RFM score from R,F,M score values using related column data.

- dataset: online_retail_II.xlsx [data link](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) 

RFM stands for *Recency*, *Frequency*, and *Monetary* value, and it's a customer segmentation technique to understand and prioritize the customer base.

- **Recency(R)** refers to how recently a customer has made a purchase. Customer who made a purchase more recently will have a higher recency score.

- **Frequency(F)** measures how often a customer makes a purchase over a given period. Customers who purchase frequently are generally more loyal and valuable to a business.

- **Monetary(M)** value refers to the total amount of money a customer has spent with the business over a defined time frame. High-spending customers are often considered more valuable because they contribute more to the business's revenue.

**CUSTOMER SEGMENTATION WITH RFM**

Customer segmentation is dividing customers into groups based on their purchasing habits. Different business strategies can be developed specific to groups. 

*Segmentation steps*: 


1. Calculate RFM metrics.

- Recency: The date of the last purchase. More recent purchases (lower values) score higher recency score. 
- Frequency: The total number of purchases. More frequent buyers (high values) score higher frequency score.
- Monetary: The total amount spent. Buyers spend higher score higher monetary score.

```python
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

# > change the column names.
# InVoiceDate > recency , Invoice > frequency, TotalPrice > monetary
rfm.columns = ['recency', 'frequency', 'monetary']
```

2. Calculating RFM scores on scale from 1 to 5.

RFM metrics need to be converted to RFM scores to be comparable both internally and among each other.

- Recency score: Customers with more recent purchases (lower values) will have higher recency score. 
- Frequency score: More frequent buyers (high values) will ahve higher frequency score.
- Monetary score: Customers that spent higher will have higher monetary score.

```python
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

```

3. Convert metrics into a RFM score. 

- RFM score: score is obtained by merging recency and frequency score values as string. e.g. RFM score is 43 with recency score of 4 and frequency score of 3.

```python
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
```

4. Group the customers using RFM scores.

Using regEx to match RFM scores with a segment name. 'seg_map' dictionary is used to create regEx-segment key-value pairs.

![Customer Segments with RF scores](https://miro.medium.com/v2/resize:fit:803/0*fZYhZ9srVEhmkMbZ.png)

```python
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
```


## Case Study: FLO RFM Analytics

> *python file*:  [flo_rfm.py](flo_rfm.py)

- Aim: Customer segmentation with RFM.

- dataset: flo_data_20K.csv (private data)

***Outline***:

1. Data Preparation
2. Calculating RFM Metrics.
3. Calculating RF and RFM Scores
4. Defining RF Scores as Segments
5. Segment Analysis
6. Functionalization

## Customer Lifetime Value (CLV)

> *python file*:  [clv.py](clv.py)

- Aim: Calculate Customer Lifetime Value (CLV) for each customer.

- Method: First, calculate all the necessary parameters. Then, calculate CLV using these parameter values. 

- dataset: online_retail_II.xlsx [data link](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) 

Customer Lifetime Value (CLV) refers to the total amount of revenue a business can expect from a single customer throughout their relationship with the company. It helps businesses understand the long-term value of their customers and make informed decisions about acquiring and retaining them.

**CLV = (Customer Value / Churn Rate) * Profit Margin**

- **Churn Rate**: Rate of the customers who purchased only once.

- **Profit Margin**:  The company profits from customers' purchases. The customer's profit margin is obtained by multiplying the customer's revenue by the profit percentage.

- **Customer Value** = Average Order Value * Purchase Frequency
- **Average Order Value** = Total Price / Total Transaction
- **Purchase Frequency** = Total Transaction / Total Number Customers
- **Churn Rate** = 1 - Repeat Rate
- **Repeat Rate** = number of customers who purchase more than once / total number of customers
- **Profit Margin** = Total Price * 0.10

Customers are divided into segments by calculating CLV values ​​for each customer and then grouping them according to these values.

## CLV Prediction

> *python file*:  [clv_prediction.py](clv_prediction.py)

- Aim: CLV Prediction with BG-NBD and Gamma-Gamma Models.

- Method: *BetaGeoFitter* and *GammaGammaFitter* models from **lifetimes** library. 

- dataset: online_retail_II.xlsx [data link](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) 


1. Data Preparation

2. Expected Number of Transaction with BG-NBD Model

```python
# define the model
bgf = BetaGeoFitter(penalizer_coef=0.001)

# fit the model
bgf.fit(clv_df['frequency'],
        clv_df['recency'],
        clv_df['T'])

# add expected purchase number within a week for each customer.
clv_df["expected_purc_1_week"] = bgf.predict(1,
                                              clv_df['frequency'],
                                              clv_df['recency'],
                                              clv_df['T'])
```

- Analyze expected purchase amounts, actual data vs. model prediction.

```python
plot_period_transactions(bgf)
plt.show()
```

![Transaction](file:///C:/Users/esrae/Downloads/Figure_1.png)

3. Expected Average Profit with Gamma-Gamma Model

```python
# define the model
ggf = GammaGammaFitter(penalizer_coef=0.01)

# fit the model.
ggf.fit(clv_df['frequency'], clv_df['monetary'])

clv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(clv_df['frequency'],
                                                                             clv_df['monetary'])
```

4. Calculation of CLV with BG-NBD and Gamma-Gamma Model

```python
clv = ggf.customer_lifetime_value(bgf,
                                   clv_df['frequency'],
                                   clv_df['recency'],
                                   clv_df['T'],
                                   clv_df['monetary'],
                                   time=3,    # 3 months
                                   freq="W",  # week
                                   discount_rate=0.01)

# reset the index - columns: Customer ID, clv
clv = clv.reset_index()
```

5. Segmentation with CLV

```python
clv_final = clv_df.merge(clv, on="Customer ID", how="left")

# form 4 segments with pd.qcut() according to clv values.
clv_final["segment"] = pd.qcut(clv_final["clv"], 4, labels=["D", "C", "B", "A"])
```
6. Functionality

## Case Study: FLO CLV Prediction

> *python file*:  [flo_clv.py](flo_clv.py)

- Aim: Predicting CLV for 6 months to make long-term plan for sales and marketing activities.

- Method: *BetaGeoFitter* and *GammaGammaFitter* models from **lifetimes** library. 

- dataset: flo_data_20K.csv (private data)

1. Data Preparation

2. Creating the parameters for CLV calculation.

```python
clv_df["recency_clv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7.0
clv_df["T_weekly"] = (analysis_date - df["first_order_date"]).dt.days / 7.0
clv_df["frequency"] = df["order_num_total"]    # total number of transaction
clv_df["monetary_clv_avg"] = df["customer_value_total"] / df["order_num_total"]    # monetary per purchase
```

3. Fitting BG/NBD and Gamma-Gamma Models and Calculating CLV for 6 Months

```python
# Fit BG/NBD model.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(clv_df['frequency'],
        clv_df['recency_clv_weekly'],
        clv_df['T_weekly'])

# Fit Gamma-Gamma model.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(clv_df['frequency'],
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
```

4. Customer Segmentation

```python
# Create 4 segments according to clv. segment D is customers with the lowest clv.
clv_df["clv_segment"] = pd.qcut(clv_df["clv"], 4, labels=["D", "C", "B", "A"])
```

```
              mean  count
clv_segment              
D            80.34   4987
C           138.31   4986
B           199.53   4986
A           362.32   4986
```

- Segment A (High Profile) > parties, gala, sample submission, pre-promotion, workshops
- Segment D (Low Profile)  > product campaigns, raffles, incentive discounts for those who bring customers


5. Functionality


