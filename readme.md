please download dataset
[here](https://drive.google.com/file/d/1j4Nwrno_X5DVMp89i026yxwbZklw0Csp/view?usp=sharing)

and in case you need a little help, I'm writing a list that can serve you as a guideline.
- rfm_case.py: a case study for implementing K-means algorithm, including trying to cluster based on RFM Analysis (quantile)
- rfm_plot.py: filled with a set of commandlines to visualize the result
- rfm_score.csv: the result of RFM-analysis ; the data contains CustomerID,Recency,Frequency,Monetary,r_seg,f_seg,m_seg,RFMScore 
- rfmkmean.csv: the result of K-means and join with rfm_score.csv; CustomerID,r_sca,f_sca,m_sca,prediction,RFMScore
- rfm_kmeans_pred.csv: the final result where we put all data that we may need together; CustomerID,r_sca,f_sca,m_sca,prediction,RFMScore,rfm_class
