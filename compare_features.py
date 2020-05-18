# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler, CountVectorizerModel, Tokenizer, NGram
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.types import IntegerType

from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity 
from Levenshtein import jaro_winkler

# COMMAND ----------

# embed comparison metrics in Spark UDFs
udf_token_set_score = F.udf(lambda x,y: float(fuzz.token_set_ratio(x,y)/100))
udf_cos_sim = F.udf(lambda x,y: float(cosine_similarity([x,y])[0][1]))
udf_jaro_winkler = F.udf(lambda x,y: float(jaro_winkler(x,y)))
# Jaccard?

# COMMAND ----------

class string_feature_comparison():
  """ transformer that takes a Spark data frame with two or more columns of entities in string or vector representation
      and returns the data frame with columns added for all the specified comparisons between the entity
      vector columns
      
      the transform_mapping parameter should be a Python dict that maps column names to the
      desired transformations, e.g.
      {'output_col_base1':{'featureType':'string/vector','inputCols':['stringCol1','stringCol2'],'metrics':['all'/'cs'/'ts'/'jw']},...}
      
      For 'featureType' == 'string', the available metrics are 'ts' (token set ratio, 'jw' (jaro winkler similarity) or 'all' (both)
      For 'featureType' == 'vector', the only available metric is 'cs' (cosine similarity)
      Inputs for 'featureType' == 'vector' can be created with the string_transforms.fit_transform() method in this project
  """
  
  def __init__(self,comparison_mapping,\
               suffixes={'cs':'_cosine_similarity','ts':'_token_set_ratio','jw':'_jaro_winkler_similarity'}):
    self.comparison_mapping = comparison_mapping
    self.suffixes=suffixes

  def transform(self,df):
    for output_base in self.comparison_mapping.keys():
      if self.comparison_mapping[output_base]['featureType'] == 'string':
        for metric in self.comparison_mapping[output_base]['metrics']:
          if metric == 'all':
            df = df.withColumn(output_base+self.suffixes['ts'],udf_token_set_score(F.col(self.comparison_mapping[output_base]['inputCols'][0]),\
                                                                              F.col(self.comparison_mapping[output_base]['inputCols'][1])))\
              .withColumn(output_base+self.suffixes['jw'],udf_jaro_winkler(F.col(self.comparison_mapping[output_base]['inputCols'][0]),\
                                                                              F.col(self.comparison_mapping[output_base]['inputCols'][1])))
          elif metric == 'ts': #token set ratio
            df = df.withColumn(output_base+self.suffixes['ts'],udf_token_set_score(F.col(self.comparison_mapping[output_base]['inputCols'][0]),\
                                                                              F.col(self.comparison_mapping[output_base]['inputCols'][1])))
          elif metric == 'jw': # jaro-winkler
            df = df.withColumn(output_base+self.suffixes['jw'],udf_jaro_winkler(F.col(self.comparison_mapping[output_base]['inputCols'][0]),\
                                                                              F.col(self.comparison_mapping[output_base]['inputCols'][1])))
          else:
            print("Metric must be a list containing one or more of 'ts', 'jw', 'all'")
      elif self.comparison_mapping[output_base]['featureType'] == 'vector':
        for metric in self.comparison_mapping[output_base]['metrics']:
          if metric == 'cs': #cosine similarity
            df = df.withColumn(output_base+self.suffixes['cs'],udf_cos_sim(F.col(self.comparison_mapping[output_base]['inputCols'][0]),\
                                                                                F.col(self.comparison_mapping[output_base]['inputCols'][1])))
          else:
            print("Metric must be a list containing one or more of 'cs', ...")
    return df

# COMMAND ----------

# EXAMPLE 1: only string comparisons - no vector comparisons
# import pandas as pd
# sample_df = spark.createDataFrame(pd.DataFrame({'name1':['John Doe','Jane Doe'],'name2':['John Do','Jean Doe']}))
# comparison_mapping = {'name':{'featureType':'string','inputCols':['name1','name2'],'metrics':['all']}}
# feature_comparison = string_feature_comparison(comparison_mapping)
# display(feature_comparison.transform(sample_df))

# COMMAND ----------

# EXAMPLE 2: do comparisons on vectorized representations of entity strings. 
# First, load the string_transforms class
# %run ./string_transforms

# COMMAND ----------

# import pandas as pd
# sample_df = spark.createDataFrame(pd.DataFrame({'name1':['John Doe','Jane Doe'],'name2':['John Do','Jean Doe']}))
# sample_df_featured = feature_generator.fit_transform(sample_df)
# comparison_mapping = {'name_string':{'featureType':'string','inputCols':['name1','name2'],'metrics':['all']},
#                       'name_bigram':{'featureType':'vector','inputCols':['name1_bigram_frequencies','name2_bigram_frequencies'],'metrics':['cs']},
#                       'name_embedding':{'featureType':'vector','inputCols':['name1_bert_embedding','name2_bert_embedding'],'metrics':['cs']}}
# feature_comparison = string_feature_comparison(comparison_mapping)
# display(feature_comparison.transform(sample_df_featured))