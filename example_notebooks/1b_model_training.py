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

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ### Data loading

# COMMAND ----------

#%run "/All Shared/Helpers/python_tags"

# COMMAND ----------

# MAGIC %run "../python_tags"

# COMMAND ----------

spark.sql("USE {}".format(get_metastore_username_prefix()+"_identities"))

# COMMAND ----------

# MAGIC %sql SHOW TABLES

# COMMAND ----------

train_test_raw = spark.table("train_test_enriched").repartition(32).persist()

# COMMAND ----------

display(train_test_raw)

# COMMAND ----------

# MAGIC %md ### Feature Engineering
# MAGIC 
# MAGIC Two main flavors of feature engineering to do:
# MAGIC 1. numeric comparisons between fields (year differences between `yearID` and `debutYear`, string similarity between `firstLastGiven` and `firstLastCommon`)
# MAGIC 2. string indexing on `POS`

# COMMAND ----------

# MAGIC %run ../compare_features

# COMMAND ----------

comparison_mapping = {'name_string':{'featureType':'string','inputCols':['firstLastCommon','firstLastGiven'],'metrics':['all']},
                      'name_bigram':{'featureType':'vector','inputCols':['firstLastCommon_bigram_frequencies','firstLastGiven_bigram_frequencies'],'metrics':['cs']},
                      'name_embedding':{'featureType':'vector','inputCols':['firstLastCommon_bert_embedding','firstLastGiven_bert_embedding'],'metrics':['cs']}}
feature_comparison = string_feature_comparison(comparison_mapping)

# COMMAND ----------

train_test_comparisons = feature_comparison.transform(train_test_raw).persist()

# COMMAND ----------

display(train_test_comparisons)

# COMMAND ----------

display(train_test_comparisons.groupBy('matched').count())

# COMMAND ----------

# MAGIC %md Since these comparisons took so long to compute, we'll save them off for re-use after the current session

# COMMAND ----------

dbutils.fs.ls(get_user_home_folder_path()+"identities/")

# COMMAND ----------

train_test_comparisons.write.partitionBy("matched").format("delta").save(get_user_home_folder_path()+"identities/train_test_comparisons")

# COMMAND ----------

spark.sql("""
CREATE TABLE IF NOT EXISTS train_test_comparisons
USING DELTA
LOCATION \'{}\'""".format(get_user_home_folder_path()+"identities/train_test_comparisons"))

# COMMAND ----------

# MAGIC %sql select * from train_test_comparisons limit 5

# COMMAND ----------

# MAGIC %md #### Package up the standard feature transformations (e.g. indexer, vector assemblers in a pipeline)

# COMMAND ----------

train_test_comparisons = spark.table('train_test_comparisons')\
  .withColumn('year_diff',F.col('yearID')-F.col('debutYear'))\
  .withColumn('matched',F.col('matched').cast('integer'))\
  .persist()
display(train_test_comparisons)

# COMMAND ----------

# MAGIC %md Add in string indexer for player fielding position

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.pipeline import Pipeline
indexer = StringIndexer(inputCol="position", outputCol="position_index",handleInvalid="keep")
label_indexer = StringIndexer(inputCol="matched", outputCol="label",handleInvalid="keep")
encoder = OneHotEncoderEstimator(inputCols=["position_index"], outputCols=["position_categories"])
feature_cols = ['weight','height','year_diff','position_categories','name_string_token_set_ratio','name_string_jaro_winkler_similarity','name_bigram_cosine_similarity','name_embedding_cosine_similarity']
assembler = VectorAssembler(inputCols=feature_cols,outputCol='features')
feature_pipeline = Pipeline(stages=[indexer,label_indexer,encoder,assembler]).fit(train_test_comparisons)

# COMMAND ----------

train_test = feature_pipeline.transform(train_test_comparisons).persist()

# COMMAND ----------

display(train_test.orderBy('ID').drop('firstLastGiven_bert_embedding','firstLastGiven_bigram_frequencies','firstLastCommon_bert_embedding','firstLastCommon_bigram_frequencies'))

# COMMAND ----------

# dbutils.fs.rm(get_user_home_folder_path()+"identities/feature_pipeline",True)

# COMMAND ----------

feature_pipeline.save(get_user_home_folder_path()+"identities/feature_pipeline")

# COMMAND ----------

(train,test) = train_test.randomSplit([0.7, 0.3],seed=42)

# COMMAND ----------

display(train)

# COMMAND ----------

train.count()

# COMMAND ----------

# MAGIC %md ### Create a model and evaluate accuracy
# MAGIC 
# MAGIC For simplicity and robustness, let's start with a GBT model (xgboost might be a good follow-on activity)

# COMMAND ----------

import mlflow
from mlflow import spark as mlflow_spark
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=20)
evaluator = BinaryClassificationEvaluator()

# COMMAND ----------

model = gbt.fit(train)
train_predictions = model.transform(train)

# COMMAND ----------

train_accuracy = evaluator.evaluate(train_predictions)

# COMMAND ----------

train_accuracy

# COMMAND ----------

# MAGIC %md Package up all the steps needed to get a readable table of feature importance values from a fitted pyspark.ml GBT model

# COMMAND ----------

from pyspark.ml.linalg import DenseVector
import pandas as pd

def sparse_to_array(v):
  v = DenseVector(v)
  new_array = list([float(x) for x in v])
  return new_array

def clean_label_feature_importances(fitted_model,training_df):
  importance_list = sparse_to_array(fitted_model.featureImportances)
  importance_df = pd.DataFrame({'importance':importances}).reset_index()
  numeric_metadata = training_df.select("features").schema[0].metadata.get('ml_attr').get('attrs').get('numeric')
  binary_metadata = training_df.select("features").schema[0].metadata.get('ml_attr').get('attrs').get('binary')
  merge_list = numeric_metadata + binary_metadata
  feature_df = pd.DataFrame(merge_list)
  return pd.merge(feature_df,importance_df,left_on='idx',right_on='index').drop('idx',axis=1)

# COMMAND ----------

clean_label_feature_importances(model,train).head(20)

# COMMAND ----------

display(train_predictions.drop('firstLastGiven_bert_embedding','firstLastCommon_bert_embedding'))

# COMMAND ----------

display(train_predictions.groupBy('prediction').count())

# COMMAND ----------

train_accuracy

# COMMAND ----------

# MAGIC %md Use `MLflow` to track models, 

# COMMAND ----------

import tempfile

# COMMAND ----------

with mlflow.start_run():
  model = gbt.fit(train)
  train_predictions = model.transform(train)
  test_predictions = model.transform(test)
  train_accuracy = evaluator.evaluate(train_predictions)
  test_accuracy = evaluator.evaluate(test_predictions)
  mlflow.log_param("model_type","GradientBoostedTree")
  mlflow.log_param("max_iterations",gbt.getMaxIter())
  mlflow.log_param("max_depth",gbt.getMaxDepth())
  mlflow.log_metric("train_auc",train_accuracy)
  mlflow.log_metric("test_auc",test_accuracy)
  mlflow_spark.log_model(model,"gbt_model")
  importance = clean_label_feature_importances(model,train)
  # Log importances using a temporary file
  temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
  temp_name = temp.name
  try:
    importance.to_csv(temp_name, index=False)
    mlflow.log_artifact(temp_name, "feature-importance.csv")
  finally:
    temp.close() # Delete the temp file

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