# Databricks notebook source
# MAGIC %md Dependencies:
# MAGIC - Libraries
# MAGIC   - fuzzywuzzy (Python)
# MAGIC   - python-Levenshtein (Python)
# MAGIC   - spark-nlp==2.4.5 (Python)
# MAGIC   - Lahman (R)
# MAGIC   - com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5 (Maven)
# MAGIC - spark configs
# MAGIC   - `spark.databricks.session.share true`

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

#%run "/All Shared/Helpers/python_tags"

# COMMAND ----------

# MAGIC %run "../python_tags"

# COMMAND ----------

db_root_path = get_user_home_folder_path() + 'identities/'
db_name = get_metastore_username_prefix() + '_identities'

# COMMAND ----------

db_root_path

# COMMAND ----------

db_name

# COMMAND ----------

# MAGIC %md Setup metastore and DBFS locations

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(db_name)); 
spark.sql("use {}".format(db_name))

# COMMAND ----------

dbutils.fs.mkdirs(db_root_path)

# COMMAND ----------

display(dbutils.fs.ls(db_root_path))

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
# MAGIC name_lookup <- dropna(SparkR::select(masterDF,c("ID","playerID","firstLastCommon","firstLastGiven")))
# MAGIC SparkR::createOrReplaceTempView(name_lookup, "name_lookup")
# MAGIC display(name_lookup)

# COMMAND ----------

# MAGIC %r
# MAGIC master_silver <- dropna(SparkR::select(masterDF,c("ID","playerID","firstLastGiven","height","weight","debutYear")))
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
# MAGIC - Count-vectorized bigrams

# COMMAND ----------

# MAGIC %md Use the `string_transforms` utility class included with this project

# COMMAND ----------

# MAGIC %run ../string_transforms

# COMMAND ----------

feature_generator = string_feature_generator(transform_mapping={'firstLastGiven':['bert','bigram']})

# COMMAND ----------

master_gold = feature_generator.fit_transform(df=spark.table('master_silver')).persist()

# COMMAND ----------

master_gold.count()

# COMMAND ----------

display(master_gold)

# COMMAND ----------

# MAGIC %sql SHOW TABLES

# COMMAND ----------

master_gold.write.format("delta").mode("append").option("mergeSchema", "true").save(db_root_path+"master_gold/")

# COMMAND ----------

spark.sql("""
CREATE TABLE IF NOT EXISTS master_gold
USING DELTA
LOCATION \'{}\'""".format(db_root_path+"master_gold/"))

# COMMAND ----------

# MAGIC %sql describe master_gold

# COMMAND ----------

# MAGIC %md Now repeat for the `fielding` table

# COMMAND ----------

feature_generator_fielding = string_feature_generator(transform_mapping={'firstLastCommon':['bert','bigram']})

# COMMAND ----------

fielding_gold = feature_generator_fielding.fit_transform(spark.table("fielding_silver")).persist()

# COMMAND ----------

fielding_gold.count()

# COMMAND ----------

fielding_gold.write.format("delta").mode("append").option("mergeSchema", "true").save(db_root_path+"fielding_gold/")

# COMMAND ----------

spark.sql("""
CREATE TABLE IF NOT EXISTS fielding_gold
USING DELTA
LOCATION \'{}\'""".format(db_root_path+"fielding_gold/"))

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

# make sure we don't try to match fielding records with players from the future, and try to 
# reduce the number of pairs with extreme distances between yearID and debutYear so that 
# the model doesn't overfit on the year gap
pairs_raw = a.withColumnRenamed('playerID','playerID_1')\
    .crossJoin(F.broadcast(master_raw.withColumnRenamed('playerID','playerID_2').drop('firstLastCommon')))\
    .withColumn("matched",(F.col('playerID_1')==F.col('playerID_2')))\
    .filter(F.col('yearID') >= F.col('debutYear'))\
    .withColumn("year_proximity",(F.lit(1.0)-(F.col('yearID')-F.col('debutYear'))/F.lit(150)))\
    .withColumn("keep",F.col("year_proximity") > F.rand(seed=1))\
    .filter(F.col("keep"))\
    .drop('year_proximity','keep')\
    .persist()
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
  .select('yearID','POS','master_raw.ID','firstLastCommon','firstLastGiven','weight','height','debutYear','matched')\
  .withColumnRenamed('POS','position')\
  .persist()

# COMMAND ----------

train_test_raw.count()

# COMMAND ----------

fielding_gold = spark.table('fielding_gold')
fielding_gold.printSchema()

# COMMAND ----------

master_gold = spark.table('master_gold')
master_gold.printSchema()

# COMMAND ----------

train_test_enriched = train_test_raw\
  .join(master_gold.select('ID','firstLastGiven_bert_embedding','firstLastGiven_bigram_frequencies'),train_test_raw.ID == master_gold.ID,"inner")\
  .drop(master_gold.ID)\
  .join(fielding_gold.drop('ID'),\
        (train_test_raw.firstLastCommon == fielding_gold.firstLastCommon) & \
        (train_test_raw.yearID == fielding_gold.yearID) & \
        (train_test_raw.position == fielding_gold.position),\
       "inner")\
  .drop(fielding_gold.firstLastCommon)\
  .drop(fielding_gold.yearID)\
  .drop(fielding_gold.position)

# COMMAND ----------

display(train_test_enriched.orderBy('weight',ascending=False))

# COMMAND ----------

train_test_enriched.write.format("delta").save(db_root_path + "train_test_enriched")

# COMMAND ----------

spark.sql("""
CREATE TABLE IF NOT EXISTS train_test_enriched
USING DELTA
LOCATION \'{}\'
""".format(db_root_path + "train_test_enriched"))

# COMMAND ----------

train_test_enriched.createOrReplaceTempView("train_test_enriched")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) FROM train_test_enriched

# COMMAND ----------

# MAGIC %md Create the table to be used for links later on

# COMMAND ----------

spark.sql("use {}".format(get_metastore_username_prefix() + "_identities"))

# COMMAND ----------

# MAGIC %sql show tables

# COMMAND ----------

spark.sql("""
CREATE TABLE IF NOT EXISTS matches (
  id_master STRING,
  id_transaction STRING,
  date_created TIMESTAMP,
  added_by STRING)
USING DELTA
LOCATION \'{}\'
""".format(get_user_home_folder_path() + 'identities/matches'))

# COMMAND ----------

# MAGIC %sql DESCRIBE HISTORY matches

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE VIEW matches_platinum AS
# MAGIC SELECT DISTINCT master_gold.firstLastGiven as GivenName,
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

# COMMAND ----------

# MAGIC %sql select * from matches_platinum limit 5