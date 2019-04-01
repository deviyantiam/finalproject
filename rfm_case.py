import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import findspark
findspark.init('/usr/local/Cellar/apache-spark/2.4.0/libexec')

from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .master("local") \
    .appName("RFM Analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


df = spark.read.format('com.databricks.spark.csv').\
    options(header='true', \
    inferschema='true').\
    load("onlineretail.csv",header=True)

df.show(5)
df.printSchema()
print('\n Shape =', (df.count(), len(df.columns)))

from pyspark.sql import functions as F
## Check for missing value
ax=plt.subplot(111)
sns.heatmap(df.select([F.isnull(c) for c in df.columns]).collect(),cbar=False,cmap='Greys_r')
ax.set_xticklabels(df.columns, rotation='vertical',fontsize=7)
plt.title('Missing Value Occurrence')
plt.show()

## Delete rows with null value
from pyspark.sql.functions import count
def data_count(inp):
    inp.agg(*[count(i).alias(i) for i in inp.columns]).show()  
data_count(df) # raw data
df_new = df.dropna(how='any') 
data_count(df_new) # after we drop rows with null values

## Change datetime format
from pyspark.sql.functions import to_utc_timestamp, unix_timestamp, lit, datediff, col 
time_format = "MM/dd/yy HH:mm"
df_new = df_new.withColumn('NewInvoiceDate', to_utc_timestamp(unix_timestamp(col('InvoiceDate'),time_format).cast('timestamp'),'UTC'))  
df_new.show(5)

## Calculate total price and create the column
from pyspark.sql.functions import round
df_new = df_new.withColumn('TotalPrice', round( df.Quantity * df.UnitPrice, 2 ))

## Calculate time diff
from pyspark.sql.functions import mean, min, max, sum
date_max = df_new.select(max('NewInvoiceDate')).toPandas()

## Calculate duration
df_new = df_new.withColumn('Duration', datediff(lit(date_max.iloc[0][0]), 'NewInvoiceDate')) 
df_new.show(5)

## Calculate rfm
recency = df_new.groupBy('CustomerID').agg(min('Duration').alias('Recency'))
frequency = df_new.groupBy('CustomerID', 'InvoiceNo').count().groupBy('CustomerID').agg(count("*").alias("Frequency"))
monetary = df_new.groupBy('CustomerID').agg(round(sum('TotalPrice'), 2).alias('Monetary'))
rfm = recency.join(frequency,'CustomerID', how = 'inner').join(monetary,'CustomerID', how = 'inner')
rfm.show(5)
rfm.describe().show()
# # rfm.toPandas().to_csv('rfm_value.csv',index=False)

## Cutting point
def describe_quantile(df_in, columns):
    quantiles = [25, 50, 75]
    quars = np.transpose([np.percentile(df_in.select(x).collect(),quantiles) for x in columns])
    quars = pd.DataFrame(quars, columns=columns)
    quars['quantile'] = [str(p) + '%' for p in quantiles]
    new_df = quars
    new_df = new_df.set_index('quantile')
    new_df = new_df.round(2)
    return new_df
cols = ['Recency','Frequency','Monetary']
df_quar = describe_quantile(rfm,cols)
print(df_quar)

def RScore(x):
    rs=0
    for i in range(3):
        if x <= df_quar.iloc[i,0]:
            rs=i+1
            break
        elif x > df_quar.iloc[2,0]:
            rs=4
            break
    return rs

def FScore(x):
    fs=4
    for i in range(3):
        if x <= df_quar.iloc[i,1]:
            fs-=i
            break
        elif x > df_quar.iloc[2,1]:
            fs=1
            break
    return fs
    
def MScore(x):
    ms=4
    for i in range(3):
        if x <= df_quar.iloc[i,2]:
            ms-=i
            break
        elif x > df_quar.iloc[2,2]:
            ms=1
            break
    return ms
    
from pyspark.sql.functions import udf 
from pyspark.sql.types import IntegerType
R_udf = udf(lambda x: RScore(x), IntegerType())
F_udf = udf(lambda x: FScore(x), IntegerType())
M_udf = udf(lambda x: MScore(x), IntegerType())


##segmentation
rfm_seg=rfm.withColumn('r_seg',R_udf('Recency'))
# rfm_seg.show(5)
rfm_seg=rfm_seg.withColumn('f_seg',F_udf('Frequency'))
# rfm_seg.show(5)
rfm_seg=rfm_seg.withColumn('m_seg',M_udf('Monetary'))
# rfm_seg.show(5)
rfm_seg=rfm_seg.withColumn('RFMScore',F.concat(F.col('r_seg'),F.col('f_seg'),F.col('m_seg')))
rfm_seg.sort(F.col('RFMScore')).show(5)
# # rfm_seg.toPandas().to_csv('rfm_score.csv',index=False)

## Calculate mean 
# rfm_seg.groupBy('RFMScore').agg({'Recency':'mean',\
#     'Frequency': 'mean','Monetary': 'mean'})\
#     .sort(F.col('RFMScore')).show(5)

# grp = 'RFMScore'
# num_cols = ['Recency','Frequency','Monetary']
# df_input = rfm_seg
# def quantile_agg(df_in,gr,colm):
#     qua=df_in.groupBy(gr).agg(*[mean(F.col(i)) for i in colm]).sort(F.col(gr))
#     return qua
# quantile_grouped = quantile_agg(df_input,grp,num_cols)
# quantile_grouped.show(5)
# # quantile_grouped.toPandas().to_csv('quantile_grouped.csv',index=False)#output_dir+'quantile_grouped.csv')

## prepare the data in vector dense
from pyspark.ml.linalg import Vectors
def transData(data):
    return data.rdd.map(lambda r: [r[0],Vectors.dense(r[1:])]).toDF(['CustomerID','rfm']) #Return a new RDD by applying a function to each element of this RDD.
transformed=transData(rfm)
transformed.show(5)
## normalization
from pyspark.ml.feature import MinMaxScaler
scaler = MinMaxScaler(inputCol="rfm",outputCol="features")
scalerModel = scaler.fit(transformed)
scaledData = scalerModel.transform(transformed)
scaledData.show(5,False) # results will not be truncated

###ML
## find optimal parameter
from pyspark.ml.clustering import KMeans
cost = np.zeros(10)
for k in range(2,10):
    kmeans = KMeans().setK(k)\
        .setSeed(1) \
        .setFeaturesCol("features")\
        .setPredictionCol("cluster")
    model = kmeans.fit(scaledData)
    cost[k] = model.computeCost(scaledData)

## plot elbow
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,10),cost[2:10], marker = "o",color='indianred')
plt.title('Elbow Method for Optimal K')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('K')
ax.set_ylabel('Cost')
plt.show()


# ## Silhouette method
from pyspark.ml.evaluation import ClusteringEvaluator
k_min=3
k_max=10
k_lst = np.arange(k_min, k_max+1)
silh_lst = []
for k in k_lst:
# Trains a k-means model
  kmeans = KMeans().setK(k).setSeed(int(np.random.randint(100, size=1)))
  model = kmeans.fit(scaledData)
# Make predictions
  predictions = model.transform(scaledData)
# Evaluate clustering by computing Silhouette score
  evaluator = ClusteringEvaluator()
  silhouette = evaluator.evaluate(predictions)
  silh_lst.append(silhouette)
silhouette = pd.DataFrame(list(zip(k_lst,silh_lst)),columns = ['k','silhouette'])
spark.createDataFrame(silhouette).show()

# ### Kmeans
k = 3
kmeans = KMeans().setK(k).setSeed(1)
model = kmeans.fit(scaledData)
# Make predictions
predictions = model.transform(scaledData)
predictions.show(5,False)
# # predictions.toPandas().to_csv('kmeans_rfm.csv',index=False)

## extract scaled rfm to different columns
def extract(row):
    return (row.CustomerID, ) + tuple(row.features.toArray().tolist()) +(row.prediction,)
d_sc=predictions.rdd.map(extract).toDF(["CustomerID",'r_sca','f_sca','m_sca','prediction']) 
d_sc.show(5)
from pyspark.sql.functions import *
df1 = d_sc.alias('df1')
df2 = rfm_seg.alias('df2')
df_baru=df1.join(df2, df1.CustomerID == df2.CustomerID).select('df1.*','df2.RFMScore')
df_baru.show(5)
# # df_baru.toPandas().to_csv('rfmkmean.csv',index=False)

