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

train_test_raw = spark.table("train_test_raw")

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

# MAGIC %md Name comparison metric 2: cosine similarity (character similarity)

# COMMAND ----------

# in order to have consistent vectors for distance comparison, create a universal vocabulary rather than inferring an independent one for each column
import string
letters = [l for l in list(string.ascii_lowercase)]
numbers=[str(d) for d in range(10)]
vocab = letters + numbers

def create_letter_counts(df,input_col):
  new_df = df.withColumn(input_col+"_vec",F.split(F.lower(F.col(input_col)),""))
  cv = CountVectorizerModel.from_vocabulary(vocab,inputCol=input_col+"_vec", outputCol=input_col+"_features")
  return cv.transform(new_df).drop(input_col+"_vec")

# COMMAND ----------

import itertools
vocab_2gram = [e[0]+e[1]+e[2] for e in itertools.product(*[vocab, [" "], vocab])]
def create_ngram_counts(df,input_col):
  ngram = NGram(n=2, inputCol=input_col+"_chars", outputCol=input_col+"_nGrams")
  new_df = ngram.transform(df.withColumn(input_col+"_chars",F.split(F.lower(F.col(input_col)),"")))
  cv = CountVectorizerModel.from_vocabulary(vocab_2gram,inputCol=input_col+"_nGrams", outputCol=input_col+"_features")
  return cv.transform(new_df).drop(input_col+"_nGrams",input_col+"_chars")

# COMMAND ----------

# MAGIC %md And the custom distance metric for cosine similarity

# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity 
udf_cos_sim = F.udf(lambda x,y: float(cosine_similarity([x,y])[0][1]))

# COMMAND ----------

# MAGIC %md Wrap the cosine similarity transformations & computations in a single function

# COMMAND ----------

def add_cosine_similarity(df,nameCol1,nameCol2):
#   df1 = create_letter_counts(df,nameCol1)
  df1 = create_ngram_counts(df,nameCol1)
#   df2 = create_letter_counts(df1,nameCol2)
  df2 = create_ngram_counts(df1,nameCol2)
  results = df2.withColumn('cos_sim',udf_cos_sim(F.col(nameCol1+"_features"),F.col(nameCol2+"_features")).cast("double"))\
    .drop(nameCol1+"_features",nameCol2+"_features")
  return results

# COMMAND ----------

train_test_featured = train_test_raw\
  .na.drop()\
  .withColumn('year_dist',(F.col('yearID')-F.col('debutYear'))/F.lit(150))\
  .withColumn('token_set_sim',udf_token_set_score(F.col('firstLastGiven'),F.col('firstLastCommon')).cast("double"))\
  .withColumn('label',F.col('matched').cast(IntegerType()))\
  .drop('ID')

train_test_featured = add_cosine_similarity(train_test_featured,"firstLastGiven","firstLastCommon")

# COMMAND ----------

display(train_test_featured)

# COMMAND ----------

# MAGIC %md __Can we leverage BERT embeddings from spark-nlp to get more context from names?__

# COMMAND ----------

# MAGIC %md Following [JSL tutorial](https://towardsdatascience.com/text-classification-in-spark-nlp-with-bert-and-universal-sentence-encoders-e644d618ca32)

# COMMAND ----------

import sparknlp
sparknlp.start()

# COMMAND ----------

from sparknlp.base import *
from sparknlp.annotator import *
import pandas as pd
print("Spark NLP version", sparknlp.version())
print("Apache Spark version:", spark.version)

# COMMAND ----------

ner_bert = NerDLModel.pretrained('ner_dl_bert')\
  .setInputCols(['document','token'])\
  .setOutputCols('embeddings')

# COMMAND ----------

glove_embeddings = WordEmbeddingsModel.pretrained('glove_100d')\
  .setInputCols(['document','token'])\
  .setOutputCols('embeddings')

# COMMAND ----------



# COMMAND ----------

document = DocumentAssembler()\
    .setInputCol("firstLastGiven")\
    .setOutputCol("document")
    
# we can also use sentence detector here 
# if we want to train on and get predictions for each sentence# downloading pretrained embeddings
use = UniversalSentenceEncoder.pretrained()\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")# the classes/labels/categories are in category columnclasssifierdl = ClassifierDLApproach()\
  .setInputCols(["sentence_embeddings"])\
  .setOutputCol("class")\
  .setLabelColumn("category")\
  .setMaxEpochs(5)\
  .setEnableOutputLogs(True)use_clf_pipeline = Pipeline(
    stages = [
        document,
        use,
        classsifierdl
    ])

# COMMAND ----------

# MAGIC %md Resume original workflow...

# COMMAND ----------

indexer = StringIndexer(inputCol="POS", outputCol="POSIndex",handleInvalid="keep")
train_test_indexed = indexer.fit(train_test_featured).transform(train_test_featured)

# COMMAND ----------

encoder = OneHotEncoderEstimator(inputCols=["POSIndex"], outputCols=["POSCategories"])
train_test_encoded = encoder.fit(train_test_indexed).transform(train_test_indexed)

# COMMAND ----------

display(train_test_encoded)

# COMMAND ----------

inputCols=["year_dist", "token_set_sim", "cos_sim", "POSCategories","weight","height"]

assembler = VectorAssembler(
    inputCols=inputCols,
    outputCol="features")

# train_test_staged = assembler.transform(train_test_indexed)
train_test_staged = assembler.transform(train_test_encoded).persist()

# COMMAND ----------

# MAGIC %md Put all the transformations into a pipeline

# COMMAND ----------

from pyspark.ml import Pipeline

feature_pipeline = Pipeline(stages=[indexer,encoder,assembler]).fit(train_test_featured)

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

gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

# COMMAND ----------

model_gbt = gbt.fit(train)

# COMMAND ----------

train_fitted = model_gbt.transform(train)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()

# COMMAND ----------

evaluator.evaluate(train_fitted)

# COMMAND ----------

model_gbt.featureImportances

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

# MAGIC %md #### Try with xgboost instead - the gbt model is relying almost exclusively on token set scores

# COMMAND ----------

train.createOrReplaceTempView("train")
test.createOrReplaceTempView("test")

# COMMAND ----------

# MAGIC %scala
# MAGIC val train = spark.table("train")
# MAGIC val test = spark.table("test")

# COMMAND ----------

# MAGIC %scala
# MAGIC val xgbParam = Map("eta" -> 0.1f,
# MAGIC       "max_depth" -> 3,
# MAGIC       "num_class" -> 2,
# MAGIC       "num_round" -> 100,
# MAGIC       "num_workers" -> 8)

# COMMAND ----------

# MAGIC %scala
# MAGIC import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel,XGBoostClassifier}
# MAGIC val xgbClassifier = new XGBoostClassifier(xgbParam).setFeaturesCol("features").setLabelCol("label")
# MAGIC val model_xgb = xgbClassifier.fit(train)

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
# MAGIC 
# MAGIC val evaluator = new BinaryClassificationEvaluator()

# COMMAND ----------

# MAGIC %scala
# MAGIC evaluator.evaluate(model_xgb.transform(train))

# COMMAND ----------

# MAGIC %md ### Match new "transactions" with "master" identities

# COMMAND ----------

masterDF = spark.table('master_silver')
display(masterDF)

# COMMAND ----------

trxDF = spark.table('fielding_silver')
display(trxDF)

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