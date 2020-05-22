# Databricks notebook source
# imports
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler, CountVectorizerModel, Tokenizer, NGram
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassificationModel
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity 

# COMMAND ----------

# MAGIC %md
# MAGIC # Search for master identities to match identities in transactions
# MAGIC ## INSTRUCTIONS
# MAGIC 
# MAGIC 
# MAGIC 1. __Put the common name of the person you are looking for in the "Player Name" input box. Optionally select values for "Position" and "Year playing" as well__
# MAGIC 2. __Review the `yearID` and `position` values of the resulting records in the "Matching Transactions" table__
# MAGIC 3. __If you find a transaction record that you'd like to search the master ID table for, then__
# MAGIC   1. __Put the `position` value from the record in the "Matching Transactions" table in the "Position" input box__
# MAGIC   2. __Put the `yearID` value from the record in the "Transactions" table in the "Year playing" input box__
# MAGIC   3. __When you are ready to conduct the search, toggle the "Activate Search" box to "Yes". When all results are updated in the "Recommended Master Identities" table, toggle the "Activate Search" box back to "No"__

# COMMAND ----------

dbutils.widgets.text("Player Name", "")
dbutils.widgets.dropdown("Position","P",["P","C","1B","2B","3B","SS","OF"])
dbutils.widgets.dropdown("Year playing","1980",[str(y) for y in range(1870,2020)])
dbutils.widgets.dropdown('Activate Search',"No",["No","Yes"])

# COMMAND ----------

# MAGIC %sql use tim_lortz_databricks_com_identities

# COMMAND ----------

transactions = spark.table("fielding_gold").persist()

# COMMAND ----------

# MAGIC %md ## Matching Transactions

# COMMAND ----------

display(
spark.sql("""SELECT ID, firstLastCommon, yearID, position FROM fielding_gold 
WHERE firstLastCommon == \'{}\'""".format(dbutils.widgets.get('Player Name'))))

# COMMAND ----------

master = spark.table("master_gold").persist()

# COMMAND ----------

model_path = "dbfs:/home/tim.lortz@databricks.com/identities/gbtModel"
model = GBTClassificationModel.load(model_path)

# COMMAND ----------

transactions.count()
master.count()

# COMMAND ----------

# feature pipeline
# indexer = StringIndexer(inputCol="position", outputCol="position_index",handleInvalid="keep")
# encoder = OneHotEncoderEstimator(inputCols=["position_index"], outputCols=["position_categories"])
# inputCols=["token_set_sim", "nGram_cos_sim", "embedding_cos_sim", "position_categories","weight","height"]
# assembler = VectorAssembler(
#     inputCols=inputCols,
#     outputCol="features")
# feature_pipeline = Pipeline(stages=[indexer,encoder,assembler])
feature_pipeline = PipelineModel.load("dbfs:/home/tim.lortz@databricks.com/identities/feature_pipeline")

# COMMAND ----------

# helper functions
udf_cos_sim = F.udf(lambda x,y: float(cosine_similarity([x,y])[0][1]))

udf_token_set_score = F.udf(lambda x,y: float(fuzz.token_set_ratio(x,y)/100))

# def add_cosine_similarity(df,col1,col2,group):
#   results = df.withColumn(group+'_cos_sim',udf_cos_sim(F.col(col1),F.col(col2)).cast("double"))
#   return results

def sparse_to_array(v):
  v = DenseVector(v)
  new_array = list([float(x) for x in v])
  return new_array

sparse_to_array_udf = F.udf(sparse_to_array, ArrayType(DoubleType()))

def insert_comparison_features(df):
  new_df = df.na.drop()\
  .withColumn('year_dist',(F.col('yearID')-F.col('debutYear'))/F.lit(150))\
  .withColumn('token_set_sim',udf_token_set_score(F.col('firstLastGiven'),F.col('firstLastCommon')).cast("double"))\
  .withColumn('nGram_cos_sim',udf_cos_sim(F.col('firstLastGiven_nGram_frequencies'),F.col('firstLastCommon_nGram_frequencies')).cast("double"))\
  .withColumn('embedding_cos_sim',udf_cos_sim(F.col('firstLastGiven_embedding'),F.col('firstLastCommon_embedding')).cast("double"))#\
#   .drop('ID')
  return new_df

def join_compare_transform_predict():
  sample_df = transactions\
    .filter(F.col('firstLastCommon')==F.lit(dbutils.widgets.get("Player Name")))\
    .filter(F.col('yearID')==F.lit(dbutils.widgets.get("Year playing")))\
    .filter(F.col('position')==F.lit(dbutils.widgets.get("Position")))
  joined_df = master.crossJoin(F.broadcast(sample_df.withColumnRenamed('ID','trxID')))
  featured_df = insert_comparison_features(joined_df)
  staged_df = feature_pipeline.transform(featured_df)
  predicted_df = model.transform(staged_df).withColumn('prob_match',sparse_to_array_udf(F.col('probability'))[1])
  return predicted_df

# COMMAND ----------

# helper functions written instead for a pandas/sklearn-based approach
udf_cos_sim = F.udf(lambda x,y: float(cosine_similarity([x,y])[0][1]))

udf_token_set_score = F.udf(lambda x,y: float(fuzz.token_set_ratio(x,y)/100))

# def add_cosine_similarity(df,col1,col2,group):
#   results = df.withColumn(group+'_cos_sim',udf_cos_sim(F.col(col1),F.col(col2)).cast("double"))
#   return results

def sparse_to_array(v):
  v = DenseVector(v)
  new_array = list([float(x) for x in v])
  return new_array

sparse_to_array_udf = F.udf(sparse_to_array, ArrayType(DoubleType()))

def insert_comparison_features(df):
  new_df = df.na.drop()\
  .withColumn('year_dist',(F.col('yearID')-F.col('debutYear'))/F.lit(150))\
  .withColumn('token_set_sim',udf_token_set_score(F.col('firstLastGiven'),F.col('firstLastCommon')).cast("double"))\
  .withColumn('nGram_cos_sim',udf_cos_sim(F.col('firstLastGiven_nGram_frequencies'),F.col('firstLastCommon_nGram_frequencies')).cast("double"))\
  .withColumn('embedding_cos_sim',udf_cos_sim(F.col('firstLastGiven_embedding'),F.col('firstLastCommon_embedding')).cast("double"))#\
#   .drop('ID')
  return new_df

def join_compare_transform_predict():
  sample_df = transactions\
    .filter(F.col('firstLastCommon')==F.lit(dbutils.widgets.get("Player Name")))\
    .filter(F.col('yearID')==F.lit(dbutils.widgets.get("Year playing")))\
    .filter(F.col('position')==F.lit(dbutils.widgets.get("Position")))
  joined_df = master.crossJoin(F.broadcast(sample_df.withColumnRenamed('ID','trxID')))
  featured_df = insert_comparison_features(joined_df)
  staged_df = feature_pipeline.transform(featured_df)
  predicted_df = model.transform(staged_df).withColumn('prob_match',sparse_to_array_udf(F.col('probability'))[1])
  return predicted_df

# COMMAND ----------

sample_pd = transactions\
  .filter(F.col('firstLastCommon')==F.lit(dbutils.widgets.get("Player Name")))\
  .filter(F.col('yearID')==F.lit(dbutils.widgets.get("Year playing")))\
  .filter(F.col('position')==F.lit(dbutils.widgets.get("Position")))\
  .withColumnRenamed('ID','trxID')\
  .toPandas()

# COMMAND ----------

master_pd = master.toPandas()

# COMMAND ----------

featured_pd = pd.merge(master_pd.assign(key=0), sample_pd.assign(key=0), on='key').drop('key', axis=1)

# featured_pd = pd.concat([master_pd,sample_pd],axis=1)

# COMMAND ----------

featured_pd['year_dist'] = (featured_pd['yearID']-featured_pd['debutYear'])/float(150) 
featured_pd['token_set_sim'] = featured_pd.apply(lambda x: float(fuzz.token_set_ratio(x['firstLastGiven'],x['firstLastCommon'])/100),axis=1)
featured_pd['nGram_cos_sim'] = featured_pd.apply(lambda x: cosine_similarity([x['firstLastGiven_nGram_frequencies'],x['firstLastCommon_nGram_frequencies']])[0][1],axis=1)
featured_pd['embedding_cos_sim'] = featured_pd.apply(lambda x: cosine_similarity([x['firstLastGiven_embedding'],x['firstLastCommon_embedding']])[0][1],axis=1)

# COMMAND ----------

featured_pd.head()

# COMMAND ----------

# MAGIC %md ## Recommended Master Identities

# COMMAND ----------

if dbutils.widgets.get('Activate Search')=='Yes':
  matches = join_compare_transform_predict()\
    .select('ID','playerID','firstLastGiven','height','weight','debutYear','year_dist','token_set_sim','ngram_cos_sim','embedding_cos_sim','prob_match','prediction')\
    .filter(F.col('prob_match')>F.lit(0.9))\
    .orderBy('prob_match',ascending=False)
  display(matches)