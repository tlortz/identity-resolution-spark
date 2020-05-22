# Databricks notebook source
# MAGIC %md Dependencies:
# MAGIC - Libraries
# MAGIC   - fuzzywuzzy (Python)
# MAGIC   - python-Levenshtein (Python)
# MAGIC   - spark-nlp (Python)
# MAGIC   - Lahman (R)
# MAGIC   - JohnSnowLabs:spark-nlp:2.4.5 (Maven)
# MAGIC - spark configs
# MAGIC   - `spark.databricks.session.share true`

# COMMAND ----------

# MAGIC %run ./entity_tranforms

# COMMAND ----------



# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler, CountVectorizerModel, Tokenizer, NGram
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import Vectors, VectorUDT

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline

# COMMAND ----------

# MAGIC %md Setup metastore and DBFS locations

# COMMAND ----------

# MAGIC %sql CREATE DATABASE IF NOT EXISTS tim_lortz_databricks_com_identities; 
# MAGIC use tim_lortz_databricks_com_identities

# COMMAND ----------

#%fs mkdirs dbfs:/home/tim.lortz@databricks.com/identities

# COMMAND ----------

# MAGIC %fs ls dbfs:/home/tim.lortz@databricks.com/identities

# COMMAND ----------

# MAGIC %md #### Get key biographical info from People table to be used as master IDs

# COMMAND ----------

# MAGIC %r
# MAGIC library(dplyr)
# MAGIC master <- (Lahman::People) %>%
# MAGIC   dplyr::mutate(ID=rownames(.),firstLastCommon=paste0(nameFirst," ",nameLast), firstLastGiven=paste0(nameGiven," ",nameLast),debutYear=lubridate::year(debut)) %>%
# MAGIC   dplyr::select(ID,playerID,nameFirst,nameLast,nameGiven,firstLastCommon,firstLastGiven,weight,height,debut,debutYear)

# COMMAND ----------

# MAGIC %r
# MAGIC head(master)

# COMMAND ----------

# MAGIC %r
# MAGIC ## Register as a table in SparkSQL
# MAGIC masterDF <- SparkR::createDataFrame(master)
# MAGIC SparkR::createOrReplaceTempView(masterDF, "master_raw")
# MAGIC display(masterDF)

# COMMAND ----------

# MAGIC %r
# MAGIC library(SparkR)
# MAGIC ## Register as a table in SparkSQL
# MAGIC name_lookup <- dropna(select(masterDF,c("ID","playerID","firstLastCommon","firstLastGiven")))
# MAGIC SparkR::createOrReplaceTempView(name_lookup, "name_lookup")
# MAGIC display(name_lookup)

# COMMAND ----------

# MAGIC %r
# MAGIC master_silver <- dropna(select(masterDF,c("ID","playerID","firstLastGiven","height","weight","debutYear")))
# MAGIC SparkR::createOrReplaceTempView(master_silver, "master_silver")

# COMMAND ----------

# MAGIC %sql SELECT * FROM master_silver LIMIT 5

# COMMAND ----------

# MAGIC %md Use the `Fielding` table as subsequent "transactions" that need to be compared against known IDs. We'll pretend that we don't have the `playerID` field

# COMMAND ----------

# MAGIC %r 
# MAGIC fielding <- (Lahman::Fielding) %>%
# MAGIC   dplyr::mutate(ID=rownames(.)) %>%
# MAGIC   dplyr::filter(G>3) %>%
# MAGIC   dplyr::select(ID,playerID,yearID,POS)
# MAGIC 
# MAGIC head(fielding)

# COMMAND ----------

# MAGIC %r
# MAGIC ## Register as a table in SparkSQL
# MAGIC fieldingDF <- SparkR::createDataFrame(fielding)
# MAGIC SparkR::createOrReplaceTempView(fieldingDF, "fielding_raw")
# MAGIC display(fieldingDF)

# COMMAND ----------

# MAGIC %r
# MAGIC nrow(fieldingDF)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW fielding_silver AS
# MAGIC SELECT fielding_raw.ID, name_lookup.firstLastCommon, yearID, POS as position FROM name_lookup
# MAGIC INNER JOIN fielding_raw
# MAGIC WHERE name_lookup.playerID = fielding_raw.playerID

# COMMAND ----------

# MAGIC %md #### Add computationally expensive features to master and fielding tables
# MAGIC 
# MAGIC - BERT embeddings
# MAGIC - Count-vectorized 2-grams

# COMMAND ----------

sparknlp.start()

# COMMAND ----------

# explain_document_pipeline = PretrainedPipeline("explain_document_ml",lang="en")

# COMMAND ----------

# master_silver_annotated = explain_document_pipeline.transform(spark.table("master_silver").withColumn('text',F.col('firstLastGiven')))

# COMMAND ----------

# display(master_silver_annotated)

# COMMAND ----------

# bert_embeddings = BertEmbeddings.pretrained('bert_base_uncased')\
#           .setInputCols(["document", "token"])\
#           .setOutputCol("embeddings")

# COMMAND ----------

# get word-level embeddings
# master_silver_token_embeddings = bert_embeddings.transform(master_silver_annotated)

# COMMAND ----------

# roll up word-level embeddings to the "document" (i.e. the full name)
# document_embeddings = SentenceEmbeddings() \
#   .setInputCols(["document", "embeddings"]) \
#   .setOutputCol("name_embeddings") \
#   .setPoolingStrategy("AVERAGE")

# to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())

# COMMAND ----------

# master_silver_name_embeddings = document_embeddings.transform(master_silver_token_embeddings)\
#   .withColumn('embedding',F.element_at(F.col('name_embeddings.embeddings'),1))\
#   .withColumn('firstLastGiven_embedding',to_vector(F.col('embedding')))\
#   .drop('text','document','sentence','token','spell','lemmas','stems','pos','embeddings','name_embeddings','embedding')\
#   .persist()

# COMMAND ----------

# display(master_silver_name_embeddings)

# COMMAND ----------

# MAGIC %md Now create the count-vectorized 2-grams

# COMMAND ----------

# import string
# letters = [l for l in list(string.ascii_lowercase)]
# numbers=[str(d) for d in range(10)]
# vocab = letters + numbers

# import itertools
# vocab_2gram = [e[0]+e[1]+e[2] for e in itertools.product(*[vocab, [" "], vocab])]
# def create_ngram_counts(df,input_col):
#   ngram = NGram(n=2, inputCol=input_col+"_chars", outputCol=input_col+"_nGrams")
#   new_df = ngram.transform(df.withColumn(input_col+"_chars",F.split(F.lower(F.col(input_col)),"")))
#   cv = CountVectorizerModel.from_vocabulary(vocab_2gram,inputCol=input_col+"_nGrams", outputCol=input_col+"_nGram_frequencies")
#   return cv.transform(new_df).drop(input_col+"_nGrams",input_col+"_chars")

# COMMAND ----------

# master_gold = create_ngram_counts(master_silver_name_embeddings,"firstLastGiven").persist()

# COMMAND ----------

feature_generator = entity_feature_generator(transform_mapping={'firstLastGiven':['bert','bigram']})

# COMMAND ----------

master_gold = feature_generator.fit_transform(df=spark.table('master_silver')).persist()

# COMMAND ----------

master_gold.count()

# COMMAND ----------

display(master_gold)

# COMMAND ----------

# MAGIC %md __Next steps__:
# MAGIC - Review Luke's work and avoid duplication
# MAGIC - Replace the existing tables with new schema
# MAGIC - Build a class to automate comparisons between two cross-joined dataframes (e.g. a config dict to compare columns pairwise and return a cosine similarity score)
# MAGIC   - Call that class in the model training and search notebooks
# MAGIC - Re-train the model on a dataset that has pairs intentionally chosen closer together in time
# MAGIC - Evaluate the new model and look again at feature importances
# MAGIC - Overhaul the notebooks to use only the new methods - make them very clean
# MAGIC - Create a new repo in my GitHub
# MAGIC - Clone it locally
# MAGIC - Download workspace into the local clone, commit changes
# MAGIC - Make sure that all notebooks leverage %run "/All Shared/Helpers/python_tags" correctly, with paths and names built off the tags

# COMMAND ----------

master_gold.write.format("delta").save("dbfs:/home/tim.lortz@databricks.com/identities/master_gold")

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE TABLE master_gold
# MAGIC USING DELTA
# MAGIC LOCATION "dbfs:/home/tim.lortz@databricks.com/identities/master_gold"

# COMMAND ----------

# MAGIC %sql describe master_gold

# COMMAND ----------

# MAGIC %md Now repeat for the `fielding` table

# COMMAND ----------

fielding_silver_annotated = explain_document_pipeline.transform(spark.table("fielding_silver").withColumn('text',F.col('firstLastCommon')))

# COMMAND ----------

fielding_silver_token_embeddings = bert_embeddings.transform(fielding_silver_annotated)

# COMMAND ----------

fielding_silver_name_embeddings = document_embeddings.transform(fielding_silver_token_embeddings)\
  .withColumn('embedding',F.element_at(F.col('name_embeddings.embeddings'),1))\
  .withColumn('embedding',to_vector(F.col('embedding')))\
  .drop('text','document','sentence','token','spell','lemmas','stems','pos','embeddings','name_embeddings')\
  .persist()

# COMMAND ----------

display(fielding_silver_name_embeddings)

# COMMAND ----------

fielding_gold = create_ngram_counts(fielding_silver_name_embeddings,"firstLastCommon").persist()

# COMMAND ----------

fielding_gold.count()

# COMMAND ----------

# fielding_gold.write.format("delta").save("dbfs:/home/tim.lortz@databricks.com/identities/fielding_gold")

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE TABLE IF NOT EXISTS fielding_gold
# MAGIC USING DELTA
# MAGIC LOCATION "dbfs:/home/tim.lortz@databricks.com/identities/fielding_gold"

# COMMAND ----------

# MAGIC %sql SHOW TABLES

# COMMAND ----------

# MAGIC %sql SELECT * FROM fielding_gold LIMIT 3

# COMMAND ----------

# MAGIC %md #### Create a table to be used for training & testing the identity resolution model

# COMMAND ----------

# MAGIC %sql SELECT * FROM name_lookup LIMIT 3

# COMMAND ----------

# MAGIC %md Create an equal number of matched and non-matched records from `master` and `fielding`

# COMMAND ----------

matched_names = spark.table("name_lookup").select("firstLastCommon","firstLastGiven").sample(.25,seed=42)

# COMMAND ----------

matched_names.count()

# COMMAND ----------

master_raw = spark.table('master_raw')
fielding_raw = spark.table('fielding_raw')

# COMMAND ----------

a = fielding_raw.join(master_raw.select('playerID','firstLastCommon'),on='playerID')

#.drop('ID')\
pairs_raw = a.withColumnRenamed('playerID','playerID_1')\
    .crossJoin(F.broadcast(master_raw.withColumnRenamed('playerID','playerID_2').drop('firstLastCommon')))\
    .withColumn("matched",(F.col('playerID_1')==F.col('playerID_2')))
display(pairs_raw)

# COMMAND ----------

pairs_raw.count()

# COMMAND ----------

matched_set = pairs_raw.filter(F.col("matched"))
matched_count = matched_set.count()
total_count = pairs_raw.count()
unmatched_count = total_count - matched_count
balance_sample_ratio = matched_count / unmatched_count
unmatched_set = pairs_raw.filter(~ F.col("matched")).sample(False, balance_sample_ratio, 42)

# COMMAND ----------

train_test_raw = matched_set.union(unmatched_set)\
  .select('yearID','POS','ID','firstLastCommon','firstLastGiven','weight','height','debutYear','matched')\
  .withColumnRenamed('POS','position')\
  .persist()

# COMMAND ----------

train_test_raw.count()

# COMMAND ----------

fielding_gold.printSchema()

# COMMAND ----------

train_test_enriched = train_test_raw\
  .join(master_gold.select('ID','embedding','firstLastGiven_nGram_frequencies'),train_test_raw.ID == master_gold.ID,"inner")\
  .withColumnRenamed('embedding','firstLastGiven_embedding')\
  .drop(master_gold.ID)\
  .join(fielding_gold,\
        (train_test_raw.firstLastCommon == fielding_gold.firstLastCommon) & \
        (train_test_raw.yearID == fielding_gold.yearID) & \
        (train_test_raw.position == fielding_gold.position),\
       "inner")\
  .withColumnRenamed("embedding",'firstLastCommon_embedding')\
  .drop(fielding_gold.ID)\
  .drop(fielding_gold.firstLastCommon)\
  .drop(fielding_gold.yearID)\
  .drop(fielding_gold.position)

# COMMAND ----------

display(train_test_enriched.orderBy('weight',ascending=False))

# COMMAND ----------

train_test_enriched.write.format("delta").save("dbfs:/home/tim.lortz@databricks.com/identities/train_test_enriched")

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE TABLE train_test_enriched
# MAGIC USING DELTA
# MAGIC LOCATION "dbfs:/home/tim.lortz@databricks.com/identities/train_test_enriched"

# COMMAND ----------

train_test_enriched.createOrReplaceTempView("train_test_enriched")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) FROM train_test_enriched

# COMMAND ----------

# MAGIC %md Create the table to be used for links later on

# COMMAND ----------

# MAGIC %sql 
# MAGIC use tim_lortz_databricks_com_identities

# COMMAND ----------

# MAGIC %sql DROP TABLE matches

# COMMAND ----------

# MAGIC %sql show tables

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE TABLE IF NOT EXISTS matches (
# MAGIC   id_master STRING,
# MAGIC   id_transaction STRING,
# MAGIC   date_created TIMESTAMP,
# MAGIC   added_by STRING)
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/home/tim.lortz@databricks.com/identities/matches'

# COMMAND ----------

# MAGIC %sql DESCRIBE HISTORY matches

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE VIEW matches_platinum AS
# MAGIC SELECT master_gold.firstLastGiven as GivenName,
# MAGIC       fielding_gold.firstLastCommon as Name,
# MAGIC       master_gold.debutYear as DebutYear,
# MAGIC       fielding_gold.yearID as Year,
# MAGIC       fielding_gold.position as Position, 
# MAGIC       master_gold.height as Height,
# MAGIC       master_gold.weight as Weight,
# MAGIC       matches.date_created as DateCreated
# MAGIC FROM matches 
# MAGIC INNER JOIN master_gold ON master_gold.ID = matches.id_master
# MAGIC INNER JOIN fielding_gold ON fielding_gold.ID = matches.id_transaction;