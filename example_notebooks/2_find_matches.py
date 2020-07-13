# Databricks notebook source
# imports
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

#%run "/All Shared/Helpers/python_tags"

# COMMAND ----------

# MAGIC %run "../python_tags"

# COMMAND ----------

spark.sql("use {}".format(get_metastore_username_prefix()+"_identities"))

# COMMAND ----------

# transactions = spark.table("fielding_gold").persist()
transactions = spark.sql("SELECT * FROM fielding_gold").persist()

# COMMAND ----------

# MAGIC %md ## Matching Transactions

# COMMAND ----------

display(
spark.sql("""SELECT ID, firstLastCommon, yearID, position FROM fielding_gold
WHERE firstLastCommon == \'{}\' ORDER BY yearID, position""".format(dbutils.widgets.get('Player Name'))))

# COMMAND ----------

# master = spark.table("master_gold").persist()
master = spark.sql("SELECT * FROM master_gold").persist()

# COMMAND ----------

transactions.count()
master.count()

# COMMAND ----------

# MAGIC %md Use `MLflow` to retrieve the model we trained in the previous step. The experiment number in this case was 1950292

# COMMAND ----------

from mlflow.tracking import MlflowClient
experimentID = 1950292
runID = 'd22247e59404470880696459957bcb48'

artifactURL = MlflowClient().get_experiment(experimentID).artifact_location
modelURL = f"{artifactURL}/{runID}/artifacts/gbt_model/sparkml"
# model = GBTClassificationModel.load(modelURL)
model = PipelineModel.load(modelURL)

# COMMAND ----------

feature_pipeline = PipelineModel.load(get_user_home_folder_path() + "identities/feature_pipeline")

# COMMAND ----------

# MAGIC %md following [MinHashLSH](https://spark.apache.org/docs/latest/ml-features.html#minhash-for-jaccard-distance) + [Approximate Nearest Neighbor Search](https://spark.apache.org/docs/latest/ml-features.html#approximate-nearest-neighbor-search) to make the search on a smaller space

# COMMAND ----------

from pyspark.ml.feature import MinHashLSH

mh = MinHashLSH(inputCol="firstLastGiven_bigram_frequencies", outputCol="hashes", numHashTables=5)
model_mh = mh.fit(master)
master_hashed = model_mh.transform(master).persist()

# COMMAND ----------

# MAGIC %run ../compare_features

# COMMAND ----------

comparison_mapping = {'name_string':{'featureType':'string','inputCols':['firstLastCommon','firstLastGiven'],'metrics':['all']},
                      'name_bigram':{'featureType':'vector','inputCols':['firstLastCommon_bigram_frequencies','firstLastGiven_bigram_frequencies'],'metrics':['cs']},
                      'name_embedding':{'featureType':'vector','inputCols':['firstLastCommon_bert_embedding','firstLastGiven_bert_embedding'],'metrics':['cs']}}
feature_comparison = string_feature_comparison(comparison_mapping)

# COMMAND ----------

# helper functions

def sparse_to_array(v):
  v = DenseVector(v)
  new_array = list([float(x) for x in v])
  return new_array

sparse_to_array_udf = F.udf(sparse_to_array, ArrayType(DoubleType()))

def insert_comparison_features(df):
  new_df = feature_comparison.transform(
    df.na.drop()\
      .withColumn('year_diff',F.col('yearID')-F.col('debutYear'))
  )
#   .withColumn('year_dist',(F.col('yearID')-F.col('debutYear'))/F.lit(150))\
#   .withColumn('token_set_sim',udf_token_set_score(F.col('firstLastGiven'),F.col('firstLastCommon')).cast("double"))\
#   .withColumn('nGram_cos_sim',udf_cos_sim(F.col('firstLastGiven_nGram_frequencies'),F.col('firstLastCommon_nGram_frequencies')).cast("double"))\
#   .withColumn('embedding_cos_sim',udf_cos_sim(F.col('firstLastGiven_embedding'),F.col('firstLastCommon_embedding')).cast("double"))#\
  return new_df

def join_compare_transform_predict():
  sample_df = transactions\
    .filter(F.col('firstLastCommon')==F.lit(dbutils.widgets.get("Player Name")))\
    .filter(F.col('yearID')==F.lit(dbutils.widgets.get("Year playing")))\
    .filter(F.col('position')==F.lit(dbutils.widgets.get("Position")))
  
  key = sample_df.select('firstLastCommon_bigram_frequencies').toPandas().firstLastCommon_bigram_frequencies[0]
  master_subset = model_mh.approxNearestNeighbors(master_hashed, key, 100).drop('hashes','distCol')#.drop('distCol')
#   joined_df = master.crossJoin(F.broadcast(sample_df.withColumnRenamed('ID','trxID')))
  joined_df = master_subset.crossJoin(F.broadcast(sample_df.withColumnRenamed('ID','trxID')))
  featured_df = insert_comparison_features(joined_df)
  staged_df = feature_pipeline.transform(featured_df)
  predicted_df = model.transform(staged_df).withColumn('prob_match',sparse_to_array_udf(F.col('probability'))[1])
  return predicted_df

# COMMAND ----------

# MAGIC %md ## Recommended Master Identities

# COMMAND ----------

if dbutils.widgets.get('Activate Search')=='Yes':
  matches = join_compare_transform_predict()\
    .select('ID','playerID','firstLastGiven','height','weight','debutYear','year_diff','name_string_token_set_ratio','name_string_jaro_winkler_similarity',\
            'name_bigram_cosine_similarity','name_embedding_cosine_similarity','prob_match','prediction')\
    .filter(F.col('prob_match')>F.lit(0.9))\
    .orderBy('prob_match',ascending=False)
  display(matches)