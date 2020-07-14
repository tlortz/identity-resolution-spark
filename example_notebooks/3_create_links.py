# Databricks notebook source
# MAGIC %md
# MAGIC # Linking identities seen in transactions with master identities
# MAGIC ## INSTRUCTIONS
# MAGIC 
# MAGIC 
# MAGIC 1. __Put the common name of the person you are looking for in the "Common Name" input box__
# MAGIC 2. __Put the full, given name of the person in the "Given Name" box__
# MAGIC 3. __Compare the records in the "Master Identities" and "Transactions" tables. If you find a match between a master record and transaction record that you'd like to register, then__
# MAGIC   1. __Put the `ID` value from the record in the "Master Identities" table in the "Master ID" input box__
# MAGIC   2. __Put the `ID` value from the record in the "Transactions" table in the "Transaction ID" input box__
# MAGIC   3. __When you are ready to register the match, toggle the "Register" box to "Yes". When all results are updated, toggle the "Register" box back to "No"__

# COMMAND ----------

from datetime import datetime
import pandas as pd
from pyspark.sql.types import TimestampType
from pyspark.sql import functions as F
from delta.tables import *

# COMMAND ----------

#%run "/All Shared/Helpers/python_tags"

# COMMAND ----------

# MAGIC %run "../python_tags"

# COMMAND ----------

spark.sql("use {}".format(get_metastore_username_prefix() + "_identities"))
display(spark.sql("show tables"))

# COMMAND ----------

dbutils.widgets.text('Master ID',defaultValue="")
dbutils.widgets.text('Transaction ID',defaultValue="")

# COMMAND ----------

dbutils.widgets.text('Given Name',defaultValue="")
dbutils.widgets.text('Common Name',defaultValue="")
dbutils.widgets.dropdown('Register',"No",["No","Yes"])

# COMMAND ----------

master_ids = set(spark.table('master_gold').select('ID').toPandas().ID.tolist())
transaction_ids = set(spark.table('fielding_gold').select('ID').toPandas().ID.tolist())

# COMMAND ----------

# MAGIC %md ## Master Identities

# COMMAND ----------

display(
spark.sql("""SELECT ID, playerID, firstLastGiven, height, weight, debutYear FROM master_gold
WHERE firstLastGiven == \'{}\'""".format(dbutils.widgets.get('Given Name'))))

# COMMAND ----------

# MAGIC %md ## Transactions

# COMMAND ----------

display(
spark.sql("""SELECT ID, firstLastCommon, yearID, position FROM fielding_gold
WHERE firstLastCommon == \'{}\'""".format(dbutils.widgets.get('Common Name'))))

# COMMAND ----------

new_record = spark.createDataFrame(pd.DataFrame({'id_master':[dbutils.widgets.get('Master ID')],\
                                                  'id_transaction': [dbutils.widgets.get('Transaction ID')],\
                                                  'date_created':[datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S")],\
                                                  'added_by':[get_login_email()]}))\
  .withColumn('date_created',F.col('date_created').cast(TimestampType()))

# COMMAND ----------

if dbutils.widgets.get('Master ID') in master_ids and dbutils.widgets.get('Transaction ID') in transaction_ids and dbutils.widgets.get('Register')=="Yes":
#   new_record.write.format("delta").mode("append").save("dbfs:/home/tim.lortz@databricks.com/identities/matches")
# use whenNotMatchedInsert to avoid writing duplicate matches
  deltaTable = DeltaTable.forPath(spark, get_user_home_folder_path() + "identities/matches")
  deltaTable.alias("master").merge(
    new_record.alias("updates"),
    "master.id_master = updates.id_master" and "master.id_transaction = updates.id_transaction") \
  .whenNotMatchedInsert(values =
    {
      "id_master": "updates.id_master",
      "id_transaction": "updates.id_transaction",
      "date_created": "updates.date_created",
      "added_by": "updates.added_by"
    }
  ) \
  .execute()

# COMMAND ----------

# MAGIC %sql SELECT * FROM matches ORDER BY id_master, id_transaction, date_created

# COMMAND ----------

matches_platinum = spark.table('matches_platinum')

# COMMAND ----------

# MAGIC %md ## Transaction - Master Identity Links

# COMMAND ----------

display(
  spark.table('matches_platinum').where((F.col('GivenName')==F.lit(dbutils.widgets.get('Given Name'))) | (F.col('Name')==F.lit(dbutils.widgets.get('Common Name'))))
)

# COMMAND ----------

# MAGIC %sql SELECT * FROM matches