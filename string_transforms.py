# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.ml.feature import CountVectorizerModel, Tokenizer, NGram
from pyspark.ml.linalg import Vectors, VectorUDT

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline

import string
import itertools

sparknlp.start()

# COMMAND ----------

# MAGIC %run "/All Shared/Helpers/python_tags"

# COMMAND ----------

class string_feature_generator():
  """ transformer that takes a Spark data frame with one or more columns of entities in raw text
      and returns the data frame with one or more ML preprocessing methods to one or more of the 
      string entity columns
      
      the transform_mapping parameter should be a Python dict that maps column names to the
      desired transformations, e.g.
      {'entityCol1':["bigram","bert"], 'entityCol2':['bigram']}
  """
  
  def __init__(self,transform_mapping,\
               suffixes={'character':'_character_frequencies','bigram':'_bigram_frequencies','bert':'_bert_embedding'}):
    self.transform_mapping = transform_mapping
    self.suffixes=suffixes
    self.embeddings_needed=[]
    self.bert_embeddings=None
    self.explain_document_pipeline=None
    self.vocab_bigram=None
    for m in self.transform_mapping.keys():
      vals = self.transform_mapping[m]
      if type(vals) == list:
        for v in vals:
          self.embeddings_needed.append(v)
      else:
        self.embeddings_needed.append(vals)
    self.embeddings_needed = set(self.embeddings_needed)
    
    if 'bert' in self.embeddings_needed:
#       self.explain_document_pipeline = PretrainedPipeline("explain_document_ml",lang="en",disk_location="{}spark-nlp/explain_document_ml/".format(get_user_home_folder_path()))
      self.explain_document_pipeline = PretrainedPipeline("explain_document_ml",lang="en")
      self.bert_embeddings = BertEmbeddings.pretrained('bert_base_uncased')\
        .setInputCols(["document", "token"])\
        .setOutputCol("embeddings")
    
    if 'bigram' in self.embeddings_needed:
      letters = [l for l in list(string.ascii_lowercase)]
      numbers=[str(d) for d in range(10)]
      vocab = letters + numbers
      self.vocab_bigram = [e[0]+e[1]+e[2] for e in itertools.product(*[vocab, [" "], vocab])]
      
  
  def _fit_bert(self,df,col):
    to_vector = F.udf(lambda a: Vectors.dense(a), VectorUDT())
    df_annotated = self.explain_document_pipeline.transform(df.withColumn('text',F.col(col)))\
                      .drop('spell','lemmas','stems','pos')
    df_embeddings = self.bert_embeddings.transform(df_annotated)
    document_embeddings = SentenceEmbeddings() \
      .setInputCols(["document", "embeddings"]) \
      .setOutputCol("doc_embedding") \
      .setPoolingStrategy("AVERAGE")
    df_entity_embeddings = document_embeddings.transform(df_embeddings)\
      .withColumn('embedding_vector',F.element_at(F.col('doc_embedding.embeddings'),1))\
      .withColumn(col+self.suffixes['bert'],to_vector(F.col('embedding_vector')))\
      .drop('embeddings','doc_embedding','embedding_vector','text','document','sentence','token')
    return df_entity_embeddings
  
  def _fit_bigram_frequencies(self,df,col):
    ngram = NGram(n=2, inputCol=col+"_chars", outputCol=col+'_nGrams')
    new_df = ngram.transform(df.withColumn(col+"_chars",F.split(F.lower(F.col(col)),"")))
    cv = CountVectorizerModel.from_vocabulary(self.vocab_bigram,inputCol=col+"_nGrams", outputCol=col+self.suffixes['bigram'])
    return cv.transform(new_df).drop(col+"_nGrams",col+"_chars")
    
  def fit_transform(self,df):
    for col in self.transform_mapping.keys():
      transforms_needed = self.transform_mapping[col]
      if type(transforms_needed) != list:
        transforms_needed = list(transforms_needed)
      for t in transforms_needed:
        print("Transforming column {} with {} method".format(col,t))
        if t == 'bert':
          df = self._fit_bert(df,col)
        if t == 'bigram':
          df = self._fit_bigram_frequencies(df,col)
    return df

# COMMAND ----------

# import pandas as pd
# sample_df = spark.createDataFrame(pd.DataFrame({'name':['John Doe','John Do','Jane Doe']}))

# COMMAND ----------

# feature_generator = string_feature_generator(transform_mapping={'name':['bert','bigram']})

# COMMAND ----------

# display(feature_generator.fit_transform(sample_df))