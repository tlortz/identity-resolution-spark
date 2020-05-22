# Databricks notebook source
# MAGIC %md ## Fit a model that learns when two records match
# MAGIC 
# MAGIC The table we'll use has fields that come from both a master ID table and a transaction-type table.
# MAGIC 
# MAGIC The master ID fields:
# MAGIC - firstLastGiven
# MAGIC - weight
# MAGIC - height
# MAGIC - debutYear
# MAGIC 
# MAGIC The transaction fields:
# MAGIC - firstLastCommon
# MAGIC - POS
# MAGIC - yearID

# COMMAND ----------

# MAGIC %sql use tim_lortz_databricks_com_identities

# COMMAND ----------

import mlflow
import mlflow.spark

# COMMAND ----------

train_test_raw = spark.table("train_test_enriched")

# COMMAND ----------

# MAGIC %md Tweaks to make:
# MAGIC - use `helpers` notebook to set username, DBFS paths, database
# MAGIC - add mlflow integration, log pipeline and two models (one with year, one without)
# MAGIC - create a linking dashboard, where both tables can be filtered by name, and links are formed on ID numbers via widgets and committed to a table (might need to be in separate dashboards)
# MAGIC - also a linking view that combines the linking table with the master and transaction fact tables

# COMMAND ----------

# MAGIC %run "/All Shared/Helpers/python_tags"

# COMMAND ----------

print_ctx_tags()

# COMMAND ----------

get_metastore_username_prefix()

# COMMAND ----------

display(train_test_raw)

# COMMAND ----------

# MAGIC %md There are some null values in the `weight` and `height` fields. These will cause errors downstream if we don't rectify them now

# COMMAND ----------

print("{0} original records, of which {1} do not have nulls".format(train_test_raw.count(),train_test_raw.na.drop().count()))

# COMMAND ----------

# MAGIC %md Since it's a relatively small number - less than 5%, we'll just drop any rows with nulls

# COMMAND ----------

# MAGIC %md #### Feature Engineering
# MAGIC 
# MAGIC Two main flavors of feature engineering to do:
# MAGIC 1. numeric comparisons between fields (year differences between `yearID` and `debutYear`, string similarity between `firstLastGiven` and `firstLastCommon`)
# MAGIC 2. string indexing on `POS`

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler, CountVectorizerModel, Tokenizer, NGram
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.types import IntegerType

# COMMAND ----------

# MAGIC %md Name comparison metric 1: [token set ratio](https://github.com/seatgeek/fuzzywuzzy#token-set-ratio) (word similarity)

# COMMAND ----------

from fuzzywuzzy import fuzz
udf_token_set_score = F.udf(lambda x,y: float(fuzz.token_set_ratio(x,y)/100))

# COMMAND ----------

# MAGIC %md Name comparison metric 2: cosine similarity (character, n-gram and embedding similarity)

# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity 
udf_cos_sim = F.udf(lambda x,y: float(cosine_similarity([x,y])[0][1]))

# COMMAND ----------

# MAGIC %md Wrap the cosine similarity transformations & computations in a single function

# COMMAND ----------

def add_cosine_similarity(df,col1,col2,group):
  results = df.withColumn(group+'_cos_sim',udf_cos_sim(F.col(col1),F.col(col2)).cast("double"))
  return results

# COMMAND ----------

train_test_featured = train_test_raw\
  .na.drop()\
  .withColumn('year_dist',(F.col('yearID')-F.col('debutYear'))/F.lit(150))\
  .withColumn('token_set_sim',udf_token_set_score(F.col('firstLastGiven'),F.col('firstLastCommon')).cast("double"))\
  .withColumn('nGram_cos_sim',udf_cos_sim(F.col('firstLastGiven_nGram_frequencies'),F.col('firstLastCommon_nGram_frequencies')).cast("double"))\
  .withColumn('embedding_cos_sim',udf_cos_sim(F.col('firstLastGiven_embedding'),F.col('firstLastCommon_embedding')).cast("double"))\
  .withColumn('label',F.col('matched').cast(IntegerType()))\
  .drop('ID')\
  .persist()

# COMMAND ----------

display(train_test_featured)

# COMMAND ----------

train_test_featured.count()

# COMMAND ----------

# MAGIC %md Add in string indexer for player fielding position

# COMMAND ----------

indexer = StringIndexer(inputCol="position", outputCol="position_index",handleInvalid="keep")
train_test_indexed = indexer.fit(train_test_featured).transform(train_test_featured)

# COMMAND ----------

encoder = OneHotEncoderEstimator(inputCols=["position_index"], outputCols=["position_categories"])
train_test_encoded = encoder.fit(train_test_indexed).transform(train_test_indexed)

# COMMAND ----------

display(train_test_encoded)

# COMMAND ----------

inputCols=["year_dist", "token_set_sim", "nGram_cos_sim", "embedding_cos_sim", "position_categories","weight","height"]
# inputCols=["token_set_sim", "nGram_cos_sim", "embedding_cos_sim", "position_categories","weight","height"]

assembler = VectorAssembler(
    inputCols=inputCols,
    outputCol="features")

# COMMAND ----------

# train_test_staged = assembler.transform(train_test_indexed)
train_test_staged = assembler.transform(train_test_encoded).persist()

# COMMAND ----------

train_test_staged.write.format('delta').save("dbfs:/home/tim.lortz@databricks.com/identities/train_test_staged")

# COMMAND ----------

# MAGIC %md Put all the transformations into a pipeline

# COMMAND ----------

from pyspark.ml import Pipeline

feature_pipeline = Pipeline(stages=[indexer,encoder,assembler]).fit(train_test_featured)

# COMMAND ----------

feature_pipeline.save("dbfs:/home/tim.lortz@databricks.com/identities/feature_pipeline")

# COMMAND ----------

(train,test) = train_test_staged.randomSplit([0.7, 0.3],seed=42)

# COMMAND ----------

display(train)

# COMMAND ----------

train.count()

# COMMAND ----------

# MAGIC %md ### Create a model and evaluate accuracy
# MAGIC 
# MAGIC For simplicity and robustness, let's start with a GBT model (xgboost might be a good follow-on activity)

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=20)

# COMMAND ----------

model_gbt = gbt.fit(train)

# COMMAND ----------

model_gbt2 = gbt2.fit(train)

# COMMAND ----------

train_fitted = model_gbt.transform(train)

# COMMAND ----------

train_fitted2 = model_gbt2.transform(train)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()

# COMMAND ----------

evaluator.evaluate(train_fitted2)

# COMMAND ----------

model_gbt.featureImportances

# COMMAND ----------

model_gbt2.featureImportances

# COMMAND ----------

model_gbt2.write().overwrite().save("dbfs:/home/tim.lortz@databricks.com/identities/gbtModel")

# COMMAND ----------

dbutils.fs.ls("dbfs:/home/tim.lortz@databricks.com/identities/gbtModel")

# COMMAND ----------

# MAGIC %md Let's also look at test set accuracy

# COMMAND ----------

test_fitted = model_gbt.transform(test)

# COMMAND ----------

evaluator.evaluate(test_fitted)

# COMMAND ----------

display(test_fitted)

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.ml.linalg import DenseVector

def sparse_to_array(v):
  v = DenseVector(v)
  new_array = list([float(x) for x in v])
  return new_array

sparse_to_array_udf = F.udf(sparse_to_array, ArrayType(DoubleType()))

# COMMAND ----------

display(test_fitted.withColumn('prob_match',sparse_to_array_udf(F.col('probability'))[1]))

# COMMAND ----------

# MAGIC %md ### Match new "transactions" with "master" identities

# COMMAND ----------

masterDF = spark.table('master_gold')
display(masterDF)

# COMMAND ----------

trxDF = spark.table('fielding_gold')
display(trxDF)

# COMMAND ----------

# MAGIC %md Change this process so that
# MAGIC 1. The user specifies a name, year, position
# MAGIC 2. The trx table filters down to that combination
# MAGIC 3. Cross-join the filtered trx table and the master table
# MAGIC 4. Create the three string similarity metrics and the year distance
# MAGIC 5. Pass the new table through the feature pipeline (indexing/encoding)
# MAGIC 6. Score the new table & present results in decreasing probability

# COMMAND ----------

pipeline = Pipeline(stages=[indexer,encoder,assembler])

# COMMAND ----------

def merge_inference_filter(sampleDF,threshold=0.9):
  # cross join the tables
  joined = masterDF.crossJoin(F.broadcast(sampleDF))
  # add the features 
  joined_featured = feature_pipeline.transform(
    add_cosine_similarity(
      joined\
        .na.drop()\
        .withColumn('year_dist',(F.col('yearID')-F.col('debutYear'))/F.lit(150))\
        .withColumn('token_set_sim',udf_token_set_score(F.col('firstLastGiven'),F.col('firstLastCommon')).cast("double"))\
        .drop('ID'),\
      "firstLastGiven","firstLastCommon"
    )
  )
  
  # run the model
  joined_fitted = model_gbt.transform(joined_featured)
  # filter down results to those above the min probability
  # select just the key fields
  # sort by the sample_df entries and master IDs by descending probability
  final_results = joined_fitted\
    .withColumn('prob_match',sparse_to_array_udf(F.col('probability'))[1])\
    .filter(F.col('prob_match') >= F.lit(threshold))\
    .select('firstLastGiven','height','weight','debutYear','firstLastCommon','yearID','POS','year_dist','token_set_sim','cos_sim','prediction','prob_match')\
    .orderBy('firstLastCommon','prob_match',ascending=False)
  return final_results

# COMMAND ----------

import pandas as pd
sample_pd = pd.DataFrame({'firstLastCommon':['Gerrit Cole','Derek Jeter','Bo Jackson','Babe Ruth','Babe Ruth','Ty Cobb'],'yearID':[2018,2001,1987,1927,1918,1905],'POS':['P','SS','OF','RF','P','OF']})
sample_pd.head

# COMMAND ----------

import pandas as pd
sample_pd = pd.DataFrame({'firstLastCommon':['Max Scherzer','Jason Varitek'],'yearID':[2018,2001],'POS':['P','C']})
sample_pd.head

# COMMAND ----------

results = merge_inference_filter(spark.createDataFrame(sample_pd),threshold=0.9)

# COMMAND ----------

display(results)

# COMMAND ----------

# MAGIC %md __Next steps:__
# MAGIC 
# MAGIC - feature importances - do POS, height, weight even matter?
# MAGIC - explore other name comparison metrics, e.g. BERT embeddings
# MAGIC - explore other models, e.g. xgboost or even DNN
# MAGIC - once the model is reliable, package it up along with all the upstream transformations in MLflow
# MAGIC - create a process to link transaction IDs to master IDs if either
# MAGIC   - the match probability exceeds a (carefully designed) threshold
# MAGIC   - an adjudicator approves the match
# MAGIC - expose the master IDs with all transactions in a person-centric reporting tool, a linked graph, etc.