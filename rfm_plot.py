from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import findspark
findspark.init('/usr/local/Cellar/apache-spark/2.4.0/libexec')

from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .master("local") \
    .appName("RFM Analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_baru = spark.read.format('com.databricks.spark.csv').\
    options(header='true', \
    inferschema='true').\
    load("rfmkmean.csv",header=True)

df_p=df_baru.rdd.sample(False,0.5,seed=1000)
df_p=spark.createDataFrame(df_p)

def XScore(x):
    xs=0
    if x == 111:
        xs=0
    elif x in (112,113,114,212,213,214,312,313,314,412,413,414):
        xs=1
    elif x in (121,131,141,211,221,231,241,321,331,341,421,431,441):
        xs=2
    elif x == 331:
        xs=3
    elif x == 411:
        xs=4
    elif x == 444:
        xs=5
    else:
        xs=6
    return xs

from pyspark.sql.functions import udf 
from pyspark.sql.types import IntegerType
X_udf = udf(lambda x: XScore(x), IntegerType())

df_p=df_p.withColumn('rfm_class',X_udf('RFMScore'))
df_p.show(5)
df_plot=df_p.select('*').toPandas()
# # df_plot.to_csv('rfm_kmeans_pred.csv',index=False)

'''
Plot raw data; From the figure shown, there are some data that suprisingly dispersed a little too far from others, better exclude them
to get better data visualization
'''
j=plt.subplot2grid((1,1),(0,0),projection='3d')
j.scatter(xs=df_plot['r_sca'],ys=df_plot['f_sca'],zs=df_plot['m_sca'],s=30, c='indianred')
j.set_xlabel('R')
j.set_ylabel('F')
j.set_zlabel('M')
plt.legend()
# j.set_ylim3d(0,0.2)
# j.set_zlim3d(0,0.2)
plt.title('Raw Data').set_position([.5, 1])
plt.show()

j=plt.subplot2grid((1,1),(0,0),projection='3d')
j.scatter(xs=df_plot['r_sca'],ys=df_plot['f_sca'],zs=df_plot['m_sca'],s=30, c='indianred')
j.set_xlabel('R')
j.set_ylabel('F')
j.set_zlabel('M')
plt.legend()
j.set_ylim3d(0,0.2)
j.set_zlim3d(0,0.2)
plt.title('Raw Data (Subtracted)').set_position([.5, 1])
plt.show()
'''
PLOTTING RFM_Analysis

if we  want to make the chart full length, turn ylim/xlim command to comment
'''
color_plot={0:'maroon',1:'indianred',2:'darkgray',3:'mediumvioletred',4:'goldenrod',5:'teal',6:'dimgray'}
cust_label={0:'Best Customers',1:'Loyal Customers',2:'Big Spenders',3:'Almost Lost',4:'Lost Customers',5:'Lost Cheap',6:'Others'}

for c in color_plot: # iterate over color dictionary keys
        df_temp = df_plot[df_plot['rfm_class'] == c]
        plt.scatter(x = 'r_sca', y = 'f_sca', 
            data = df_temp,  
            color=color_plot[c],label=cust_label[c])
plt.xlabel('R')
plt.ylabel('F')
plt.legend()
plt.title('Cluster of Customers - RF (Subtracted)')
plt.ylim(0,0.2)
plt.show()

for c in color_plot: # iterate over color dictionary keys
        df_temp = df_plot[df_plot['rfm_class'] == c]
        plt.scatter(x = 'r_sca', y = 'm_sca', 
            data = df_temp,  
            color=color_plot[c],label=cust_label[c])
plt.xlabel('R')
plt.ylabel('M')
plt.legend()
plt.title('Cluster of Customers - RM (Subtracted)')
plt.ylim(0,0.2)
plt.show()

for c in color_plot: # iterate over color dictionary keys
        df_temp = df_plot[df_plot['rfm_class'] == c]
        plt.scatter(x = 'f_sca', y = 'm_sca', 
            data = df_temp,  
            color=color_plot[c],label=cust_label[c])
plt.xlabel('F')
plt.ylabel('M')
plt.legend()
plt.title('Cluster of Customers - FM (Subtracted)')
plt.xlim(0,0.2)
plt.ylim(0,0.2)
plt.show()

j=plt.subplot2grid((1,1),(0,0),projection='3d')
for c in color_plot: # iterate over color dictionary keys
    df_temp = df_plot[df_plot['rfm_class'] == c]
    print('number of segment',cust_label[c],'is',len(df_temp))
    j.scatter(xs=df_temp['r_sca'],ys=df_temp['f_sca'],zs=df_temp['m_sca'],s=30, c=color_plot[c],label=cust_label[c])
j.set_xlabel('R')
j.set_ylabel('F')
j.set_zlabel('M')
j.set_ylim3d(0,0.2)
j.set_zlim3d(0,0.2)
plt.title('Cluster of Customers based on RFM Analysis - Subtracted').set_position([.5, 1])
plt.legend()
plt.show()

j=plt.subplot2grid((1,1),(0,0),projection='3d')
for c in color_plot: # iterate over color dictionary keys
    df_temp = df_plot[df_plot['rfm_class'] == c]
    j.scatter(xs=df_temp['r_sca'],ys=df_temp['f_sca'],zs=df_temp['m_sca'],s=30, c=color_plot[c],label=cust_label[c])
j.set_xlabel('R')
j.set_ylabel('F')
j.set_zlabel('M')
plt.legend()
# j.set_ylim3d(0,0.2)
# j.set_zlim3d(0,0.2)
plt.title('Cluster of Customers based on RFM Analysis').set_position([.5, 1])
plt.show()

'''
PLOT RFM ANALYSIS WITH LESS DATA
'''
# df_pp=df_baru.rdd.sample(False,0.02,seed=1000)
# df_pp=spark.createDataFrame(df_pp)
# df_pp=df_pp.withColumn('rfm_class',X_udf('RFMScore'))
# df_pp.show(5)
# df_plot2=df_pp.select('*').toPandas()

# color_plot={6:'dimgray',0:'maroon',1:'indianred',2:'darkgray',3:'mediumvioletred',4:'goldenrod',5:'teal'}
# cust_label={6:'Others',0:'Best Customers',1:'Loyal Customers',2:'Big Spenders',3:'Almost Lost',4:'Lost Customers',5:'Lost Cheap'}

# for c in color_plot: # iterate over color dictionary keys
#         df_temp = df_plot2[df_plot2['rfm_class'] == c]
#         plt.scatter(x = 'r_sca', y = 'f_sca', 
#             data = df_temp,  
#             color=color_plot[c],label=cust_label[c])
# plt.xlabel('R')
# plt.ylabel('F')
# plt.legend()
# plt.title('Cluster of Customers - RF (Subtracted)')
# plt.ylim(0,0.05)
# plt.show()

# for c in color_plot: # iterate over color dictionary keys
#         df_temp = df_plot2[df_plot2['rfm_class'] == c]
#         plt.scatter(x = 'r_sca', y = 'm_sca', 
#             data = df_temp,  
#             color=color_plot[c],label=cust_label[c])
# plt.xlabel('R')
# plt.ylabel('M')
# plt.legend()
# plt.title('Cluster of Customers - RM (Subtracted)')
# plt.ylim(0,0.05)
# plt.show()

# for c in color_plot: # iterate over color dictionary keys
#         df_temp = df_plot2[df_plot2['rfm_class'] == c]
#         plt.scatter(x = 'f_sca', y = 'm_sca', 
#             data = df_temp,  
#             color=color_plot[c],label=cust_label[c])
# plt.xlabel('F')
# plt.ylabel('M')
# plt.legend()
# plt.title('Cluster of Customers - FM (Subtracted)')
# plt.xlim(0,0.05)
# plt.ylim(0,0.05)
# plt.show()


'''
PLOTTING K-MEANS

if we  want to make the chart full length, turn ylim/xlim command to comment
'''
color_plot={0:'maroon',1:'dimgray',2:'darkgray'}
cust_label={0:'Cluster 1',1:'Cluster 2',2:'Cluster 3'}

for c in color_plot: # iterate over color dictionary keys
        df_temp = df_plot[df_plot['prediction'] == c]
        plt.scatter(x = 'r_sca', y = 'f_sca', 
            data = df_temp,  
            color=color_plot[c],label=cust_label[c])
plt.xlabel('R')
plt.ylabel('F')
plt.legend()
plt.title('Cluster of Customers using Kmeans - RF (Subtracted)')
plt.ylim(0,0.2)
plt.show()

for c in color_plot: # iterate over color dictionary keys
        df_temp = df_plot[df_plot['prediction'] == c]
        plt.scatter(x = 'r_sca', y = 'm_sca', 
            data = df_temp,  
            color=color_plot[c],label=cust_label[c])
plt.xlabel('R')
plt.ylabel('M')
plt.legend()
plt.title('Cluster of Customers using Kmeans - RM (Subtracted)')
plt.ylim(0,0.2)
plt.show()

for c in color_plot: # iterate over color dictionary keys
        df_temp = df_plot[df_plot['prediction'] == c]
        plt.scatter(x = 'f_sca', y = 'm_sca', 
            data = df_temp,  
            color=color_plot[c],label=cust_label[c])
plt.xlabel('F')
plt.ylabel('M')
plt.legend()
plt.title('Cluster of Customers using Kmeans - FM (Subtracted)')
plt.ylim(0,0.2)
plt.xlim(0,0.2)
plt.show()

## 3D PLOTTING KMEANS
j=plt.subplot2grid((1,1),(0,0),projection='3d')
for c in color_plot: # iterate over color dictionary keys
    df_temp = df_plot[df_plot['prediction'] == c]
    j.scatter(xs=df_temp['r_sca'],ys=df_temp['f_sca'],zs=df_temp['m_sca'],s=30, c=color_plot[c],label=cust_label[c])
j.set_xlabel('R')
j.set_ylabel('F')
j.set_zlabel('M')
plt.legend()
# j.set_ylim3d(0,0.2)
# j.set_zlim3d(0,0.2)
plt.title('Cluster of Customers using Kmeans').set_position([.5, 1])
plt.show()

j=plt.subplot2grid((1,1),(0,0),projection='3d')
for c in color_plot: # iterate over color dictionary keys
    df_temp = df_plot[df_plot['prediction'] == c]
    print('number of',cust_label[c],'is',len(df_temp))
    j.scatter(xs=df_temp['r_sca'],ys=df_temp['f_sca'],zs=df_temp['m_sca'],s=30, c=color_plot[c],label=cust_label[c])
j.set_xlabel('R')
j.set_ylabel('F')
j.set_zlabel('M')
j.set_ylim3d(0,0.2)
j.set_zlim3d(0,0.2)
plt.title('Cluster of Customers using Kmeans - Subtracted').set_position([.5, 1])
plt.legend()
plt.show()


