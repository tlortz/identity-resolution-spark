# Databricks notebook source
# call this via %run convention to get the following in the calling notebook:
# - variables
# - friendly context output 
# - helper functions for conventions used

# COMMAND ----------

from sys import version
  
ctx_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
extra_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext()

ctx = {
  'tags':ctx_tags,
  'extra_ctx': extra_ctx,
  'login_email': ctx_tags.get("user").get(),
  'databricks_release': ctx_tags.get("clientBranchName").get(),
  'databricks_runtime_version': ctx_tags.get("sparkVersion").get(),
  'default_notebook_language': ctx_tags.get("notebookLanguage").get(),
  'notebook_url': extra_ctx.get("api_url").get() + "/#notebook/" + ctx_tags.get("notebookId").get(),
  'notebook_path': extra_ctx.get("notebook_path").get(),
  'cluster_id': ctx_tags.get("clusterId").get(),
  'python_version': version[:5]
}

print("""defined dict ctx""")

# COMMAND ----------

def print_ctx_tags():
  print("--- Notebook Metadata ---")
  print(f"login_email: {ctx['login_email']}")
  print(f"databricks release: {ctx['databricks_release']}")
  print(f"databricks runtime version: {ctx['databricks_runtime_version']}")
  print(f"default notebook language: {ctx['default_notebook_language']}")
  print(f"notebook url: {ctx['notebook_url']}")
  print(f"notebook path: {ctx['notebook_path']}")
  print(f"cluster id: {ctx['cluster_id']}")
  print(f"python version: {ctx['python_version']}")
  
def get_login_email():
  return ctx['login_email']

def get_username_prefix():
  return get_login_email().split("@")[0]

def get_metastore_username_prefix():
  import re
  return re.sub("[^A-Za-z0-9]", "_", get_username_prefix())

def get_user_home_folder_name(): 
  return get_login_email()

def get_user_home_folder_path(fuse=False):
  return ("dbfs:/", "/dbfs/")[fuse] + "home/" + get_user_home_folder_name() + "/"

def get_user_filestore_folder_name(): 
  return get_login_email()

def get_user_filestore_folder_path(fuse=False):
  return ("dbfs:/", "/dbfs/")[fuse] + "FileStore/" + get_user_filestore_folder_name() + "/"

def get_user_tmp_folder_name(): 
  return get_login_email()

def get_user_tmp_folder_path(fuse=False):
  return ("dbfs:/", "/dbfs/")[fuse] + "tmp/" + get_user_tmp_folder_name() + "/"

print("""--- python functions (assumes `ctx`) ---
print_ctx_tags()
get_login_email(): string
get_username_prefix(): string
get_metastore_username_prefix(): string
...
get_user_home_folder_name(): string
get_user_home_folder_path(fuse=False): string
...
get_user_tmp_folder_name(): string
get_user_tmp_folder_path(fuse=False): string
...
get_user_filestore_folder_name(): string
get_user_filestore_folder_path(fuse=False): string
""")