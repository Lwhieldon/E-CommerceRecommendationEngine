# Databricks notebook source
# MAGIC %md 
# MAGIC <img  src="https://emerj.com/wp-content/uploads/2018/10/recommendation-engines-for-fashion-6-current-applications-6.jpg" width="500" align="center">
# MAGIC 
# MAGIC # Fashion & Beauty E-Commerce Recommendation Engine
# MAGIC 
# MAGIC Using machine learning to increase sales, provide ease of information overload, and customize shopping experiences which leads to happy customers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Abstract
# MAGIC 
# MAGIC As e-commerce continues to grow in every industry, businesses are looking to AI & Machine Learning to address customer satisfaction, information overload, and key business insights into their customer segments. In my final  project as part of [UMBC's DATA603: Platforms for Big Data Processing](https://catalog.umbc.edu/preview_program.php?catoid=27&poid=5016), Spring 2021 class, I will build a content-based recommendation engine to enable **related product**-type of recommendations when a customer is actively using an e-commerce website. To help demonstrate how content-based recommendation engines can be deployed to help solve a business problem, this project is intended to provide a step-by-step guide on buidling an e-commerce prescriptive recommender system for a fictitious fashion and beauty e-commerce brand. By deploying this content-based recommendation engine, it will help this fashion brand better identify its customer segments and allow its customers to have a personalized experience while accessing the e-commerce website. Furthermore, this supports the fashion brand's ability to competitively price its products based on consumers' behaviors to enable an increase in overall profit and revenue streams.
# MAGIC 
# MAGIC The data used in this project is curated by Jianmo Ni (Ph.D. in Computer Science from University of California San Diego and NLP Researcher at Google), containing a collection of [2018 Amazon product details & reviews](http://deepyeti.ucsd.edu/jianmo/amazon/). From this dataset, I will be using a subset for fashion and beauty products, containing metadata describing the products as well as reviews from real-world customers who have purchased the products. This notebook is intended to demonstrate how to build a recommendation engine using the data one would typically have available from running an e-commerce website, and then applying natural language processing (NLP) techniques on product titles, categories, descriptions, and customer reviews. Upon the initial NLP analysis, I will provide a mechanicsm to deploy a content-based recommendation engine using the exciting features within Pyspark API & Databricks platform, constructing pipelines for both product and user based recommendations.
# MAGIC 
# MAGIC Additionally, I will demonstrate the output of these pipelines by building similarity functions (in scala because of limitations in Pyspark in Databricks) directly in the Databricks notebook, which will provide a dataset showing similar products tagged to a product and profile. This data can then be handed off to web developers at the fictitious fashion and beauty brand to add the engine directly onto the webpage. For the purposes of this project, I have created a Tableau dashboard (hosted on Tableau public) that will give a demo of the recommendation engine's capabilities as if it were deployed on a fashion brand's e-commerce website.
# MAGIC 
# MAGIC Lastly, I will provide potential opportunities for further enhancements that can be made to the recommendation engines by suggesting further study in collaborative and demographic based filtering methods.
# MAGIC 
# MAGIC **Note**:This notebook should be run on a Databricks ML 7.3+ cluster.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
from pyspark import keyword_only
from pyspark.sql.functions import count,  min, max, sum,instr, monotonically_increasing_id, pandas_udf, lit, countDistinct, col, array_join, expr,  explode, struct, collect_list
from pyspark.sql import DataFrame
from delta.tables import *
import pandas as pd
import gzip
import shutil
import os
import requests
import html
import nltk
from typing import Iterator
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, Normalizer, RegexTokenizer, BucketedRandomProjectionLSH, SQLTransformer, HashingTF,  Word2Vec
from pyspark.ml.linalg import Vector, Vectors
from pyspark.ml.stat import Summarizer
from pyspark.ml.clustering import LDA, KMeans, BisectingKMeans, GaussianMixture
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
import mlflow.spark

# COMMAND ----------

# MAGIC %md 
# MAGIC # Part 1: Data Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Download & Decompress Files
# MAGIC 
# MAGIC The basic building block of this type of recommender is product data.  These data may include information about the manufacturer, price, materials, country of origin, etc. and are typically accompanied by friendly product names and descriptions.  For the purposes of this project I will focus on making use of the unstructured information found in product titles and descriptions as well as product category information.
# MAGIC 
# MAGIC The dataset used is the [2018 Amazon reviews dataset](http://deepyeti.ucsd.edu/jianmo/amazon/).  It consists of several files representing both user-generated reviews and product metadata. Focusing on the 5-core subset of data in which all users and items have at least 5 reviews, I download the gzip-compressed files to the gz folder associated and identified as /mnt/reviews before decompressing the metadata JSON files to a folder named metadata and reviews JSON files to a folder named reviews [1]:
# MAGIC 
# MAGIC <img src='https://leelearningtest1.s3.amazonaws.com/reviews_folder_structure2.png' width=250>

# COMMAND ----------

# DBTITLE 1,Download Configuration
# directories for data files
download_path = '/mnt/reviews/bronze/gz'
metadata_path = '/mnt/reviews/bronze/metadata'
reviews_path = '/mnt/reviews/bronze/reviews'

perform_download = False # set to True to redownload the gzip files

# COMMAND ----------

# DBTITLE 1,Files to Download
file_urls_to_download = [
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/AMAZON_FASHION.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_AMAZON_FASHION.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/All_Beauty.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_All_Beauty.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Clothing_Shoes_and_Jewelry.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Clothing_Shoes_and_Jewelry.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Luxury_Beauty.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Luxury_Beauty.json.gz'
  ]

# COMMAND ----------

# DBTITLE 1,Reset Directories for Downloads
if perform_download:

  # clean up directories from prior runs
  try:
    dbutils.fs.rm(download_path, recurse=True)
  except:
    pass
  dbutils.fs.mkdirs(download_path)

  try:
    dbutils.fs.rm(metadata_path, recurse=True)
  except:
    pass
  dbutils.fs.mkdirs(metadata_path)

  try:
    dbutils.fs.rm(reviews_path, recurse=True)
  except:
    pass
  dbutils.fs.mkdirs(reviews_path)

# COMMAND ----------

# DBTITLE 1,Download & Decompress Files
if perform_download:
  
  # for each file to download:
  for file_url in file_urls_to_download:
    
    print(file_url)
    
    # extract file names from the url
    gz_file_name = file_url.split('/')[-1]
    json_file_name = gz_file_name[:-3]
    
    # determine where to place unzipped json
    if 'meta_' in json_file_name:
      json_path = metadata_path
    else:
      json_path = reviews_path

    # download the gzipped file
    request = requests.get(file_url)
    with open('/dbfs' + download_path + '/' + gz_file_name, 'wb') as f:
      f.write(request.content)

    # decompress the file
    with gzip.open('/dbfs' + download_path + '/' + gz_file_name, 'rb') as f_in:
      with open('/dbfs' + json_path + '/' + json_file_name, 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)

# COMMAND ----------

# MAGIC %md
# MAGIC Now I verify the decompressed JSON files in the metadata and reviews folders

# COMMAND ----------

# DBTITLE 1,List Metadata Files
display(
  dbutils.fs.ls(metadata_path)
  )

# COMMAND ----------

# DBTITLE 1,List Review Files
display(
  dbutils.fs.ls(reviews_path)
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 2: Prep Metadata
# MAGIC 
# MAGIC With the metadata files in place, I extract the relevant information from the documents and make the information more easily queriable.  
# MAGIC 
# MAGIC In reviewing the metadata files, it appears the brand, category, title, description, and image fields along with each product's unique identifier, (*asin* field), will be of useful for the project exercise performed. There is a ton of other information available but I will limit the attributes assigned to the metadata to these columns:

# COMMAND ----------

# DBTITLE 1,Prepare Environment for Data
_ = spark.sql('DROP DATABASE IF EXISTS reviews CASCADE')
_ = spark.sql('CREATE DATABASE reviews')

# COMMAND ----------

# DBTITLE 1,Extract Common Elements from Metadata JSON
# common elements of interest from json docs (only import ones actually used later)
metadata_common_schema = StructType([
	StructField('asin', StringType()),
	StructField('category', ArrayType(StringType())),
	StructField('description', ArrayType(StringType())),
	StructField('title', StringType()),
    StructField('image',ArrayType(StringType()))
	])

# read json to dataframe
raw_metadata = (
  spark
    .read
    .json(
      metadata_path,
      schema=metadata_common_schema
      )
    )

display(raw_metadata)

# COMMAND ----------

# MAGIC %md A [notebook](https://colab.research.google.com/drive/1Zv6MARGQcrBbLHyjPVVMZVnRWsRnVMpV) made available by the data host indicates some entries may be invalid and should be removed. These records are identified by the presence of the *getTime* JavaScript method call in the *title* field:

# COMMAND ----------

# DBTITLE 1,Eliminate Unnecessary Records
# remove bad records and add ID for deduplication work
metadata = (
  raw_metadata
    .filter( instr(raw_metadata.title, 'getTime')==0 ) # unformatted title
    )

metadata.count()

# COMMAND ----------

# MAGIC %md The dataset also contains a few duplicate entries based on the *asin* value:

# COMMAND ----------

# DBTITLE 1,Number of ASINs with More Than One Record
# count number of records per ASIN value
(
  metadata
  .groupBy('asin')  
    .agg(count('*').alias('recs'))
  .filter('recs > 1')
  ).count()

# COMMAND ----------

# MAGIC %md Using an artificial id, I will eliminate the duplicates by arbitrarily selecting one record for each ASIN to remain in the dataset.  Notice I am caching the dataframe within which this id is defined in order to fix its value.  Otherwise, the value generated by *monotonically_increasing_id()* will be inconsistent during the self-join:

# COMMAND ----------

# DBTITLE 1,Deduplicate Dataset
# add id to enable de-duplication and more efficient lookups (cache to fix id values)
metadata_with_dupes = (
  metadata
    .withColumn('id', monotonically_increasing_id())
  ).cache()

# locate first entry for each asin
first_asin = (
  metadata_with_dupes
    .groupBy('asin')
      .agg(min('id').alias('id'))
  )

# join to eliminate unmatched entries
deduped_metadata = (
  metadata_with_dupes
    .join(first_asin, on='id', how='leftsemi')
  )

deduped_metadata.count()

# COMMAND ----------

# DBTITLE 1,Verify Duplicates Eliminated
# should return 0 if no duplicates
(
  deduped_metadata
  .groupBy('asin')
    .agg(count('*').alias('recs'))
  .filter('recs > 1')
  ).count()

# COMMAND ----------

# MAGIC %md To make the next data processing steps easier to perform, I will persist the deduplicated data to storage.  By using Delta Lake as the storage format, I am enabling a set of data modification statements which I will employ later:

# COMMAND ----------

# DBTITLE 1,Persist Deduplicated Data
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS reviews.metadata')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/silver/metadata', ignore_errors=True)

# persist as delta table
(
  deduped_metadata
   .repartition(sc.defaultParallelism * 4)
   .write
   .format('delta')
   .mode('overwrite')
   .save('/mnt/reviews/silver/metadata')
  )

# make table queriable
_ = spark.sql('''
  CREATE TABLE IF NOT EXISTS reviews.metadata
  USING DELTA
  LOCATION '/mnt/reviews/silver/metadata'
  ''')

# show data
display(
  spark.table('reviews.metadata')
  )

# COMMAND ----------

# DBTITLE 1,Drop Cached Data
_ = metadata_with_dupes.unpersist()

# COMMAND ----------

# MAGIC %md With the deduplicated data in place, I will now turn to cleansing the fields in the metadata.  Taking a closer look at the *description* field, I can see there are unescaped HTML characters and full HTML tags which I need to clean up.  I can see some similar cleansing is needed for the *title* and the *category* fields.  With the *category* field, it appears that the category hierarchy breaks down at a level where an HTML tag is encountered.  For that field, I will truncate the category information at the point a tag is discovered. Additionally, I will only keep the first record within the *image* field array.
# MAGIC 
# MAGIC To make this code easier to implement, I'll make use of a pandas UDF and update the data in place:

# COMMAND ----------

# DBTITLE 1,Define Function to Unescape HTML
# pandas function to unescape HTML characters
@pandas_udf(StringType())
def unescape_html(text: pd.Series) -> pd.Series:
  return text.apply(html.unescape)

# register function for use with SQL
_ = spark.udf.register('unescape_html', unescape_html)

# COMMAND ----------

# DBTITLE 1,Cleanse Titles
# MAGIC %sql
# MAGIC 
# MAGIC MERGE INTO reviews.metadata x
# MAGIC USING ( 
# MAGIC   SELECT -- remove HTML from feature array
# MAGIC     a.id,
# MAGIC     unescape_html(
# MAGIC       REGEXP_REPLACE(a.title, '<.+?>', '')
# MAGIC       ) as title
# MAGIC   FROM reviews.metadata a
# MAGIC   WHERE a.title RLIKE '<.+?>|&\\w+;'  -- contains html tags & chars
# MAGIC   ) y
# MAGIC ON x.id = y.id
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET x.title = y.title

# COMMAND ----------

# DBTITLE 1,Cleanse Descriptions
# MAGIC %sql
# MAGIC 
# MAGIC MERGE INTO reviews.metadata x
# MAGIC USING ( 
# MAGIC   SELECT -- remove HTML from feature array
# MAGIC     a.id,
# MAGIC     COLLECT_LIST( 
# MAGIC       unescape_html(
# MAGIC         REGEXP_REPLACE(b.d, '<.+?>', '')
# MAGIC         )
# MAGIC       ) as description
# MAGIC   FROM reviews.metadata a
# MAGIC   LATERAL VIEW explode(a.description) b as d
# MAGIC   WHERE b.d RLIKE '<.+?>|&\\w+;'  -- contains html tags & chars
# MAGIC   GROUP BY a.id
# MAGIC   ) y
# MAGIC ON x.id = y.id
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET x.description = y.description

# COMMAND ----------

# DBTITLE 1,Cleanse Categories
# MAGIC %sql
# MAGIC 
# MAGIC MERGE INTO reviews.metadata x
# MAGIC USING ( 
# MAGIC   SELECT  -- only keep elements prior to html
# MAGIC     m.id,
# MAGIC     COLLECT_LIST( unescape_html(o.c) ) as category
# MAGIC   FROM reviews.metadata m
# MAGIC   INNER JOIN (
# MAGIC     SELECT -- find first occurance of html in categories
# MAGIC       a.id,
# MAGIC       MIN(b.index) as first_bad_index
# MAGIC     FROM reviews.metadata a
# MAGIC     LATERAL VIEW posexplode(a.category) b as index,c
# MAGIC     WHERE b.c RLIKE '<.+?>'  -- contains html tags
# MAGIC     GROUP BY a.id
# MAGIC     ) n 
# MAGIC     ON m.id=n.id
# MAGIC   LATERAL VIEW posexplode(m.category) o as index,c
# MAGIC   WHERE o.index < n.first_bad_index
# MAGIC   GROUP BY m.id
# MAGIC   ) y
# MAGIC   ON x.id=y.id
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET x.category=y.category

# COMMAND ----------

# DBTITLE 1,Cleanse Images
# MAGIC %sql
# MAGIC 
# MAGIC MERGE INTO reviews.metadata x
# MAGIC USING ( 
# MAGIC   SELECT  -- only first record in array
# MAGIC     m.id,
# MAGIC     SPLIT(m.image[0],',')  as image
# MAGIC   FROM reviews.metadata m
# MAGIC    ) y
# MAGIC   ON x.id=y.id
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET x.image=y.image

# COMMAND ----------

# MAGIC %md
# MAGIC Since I am certain there are no operations being performed on metadata table that take longer than the retention interval, I will turn off this safety check by setting the Apache Spark configuration property spark.databricks.delta.retentionDurationCheck.enabled to false. I then set the VACUUM to retain for 0 hours to help clean up the files underneath the table.

# COMMAND ----------

# DBTITLE 1,Cleanup Delta Table
spark.conf.set('spark.databricks.delta.retentionDurationCheck.enabled', False)
_ = spark.sql('VACUUM reviews.metadata RETAIN 0 HOURS')

# COMMAND ----------

# MAGIC %md Now I can see how the cleansed metadata appears:

# COMMAND ----------

# DBTITLE 1,Review Cleansed Metadata
display(
  spark.table('reviews.metadata')
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 3: Prep Reviews
# MAGIC 
# MAGIC As with the metadata files, there are only a limited number of fields in the reviews JSON documents relevant to the needs of this project. I'll retrieve the ASIN for each product as well as the reviewer's ID and the time of their review.  Fields such as whether a purchase was verified or the number of other users who found the review could be useful but I'll leave them out of this analysis for now:

# COMMAND ----------

# DBTITLE 1,Extract Common Elements from Reviews JSON
# common elements of interest from json docs
reviews_common_schema = StructType([
  StructField('asin', StringType()),
  StructField('overall', DoubleType()),
  StructField('reviewerID', StringType()),
  StructField('unixReviewTime', LongType())
  ])

# read json to dataframe
reviews = (
  spark
    .read
    .json(
      reviews_path,
      schema=reviews_common_schema
      )
    )

# present data for review
display(
  reviews
  )

# COMMAND ----------

# MAGIC %md A quick check for duplicates finds that there is little clean up to do.  While maybe not truly duplicates, if a user submits multiple reviews on a single product, I want to take the latest of these as their go-forward review. From there, I want to make sure the system is not capturing multiple records for that same, last date and time:

# COMMAND ----------

# tack on sequential ID
reviews_with_duplicates = (
  reviews.withColumn('rid', monotonically_increasing_id())
  ).cache() # cache to fix the id in place


# locate last product review by reviewer
last_review_date = (
  reviews_with_duplicates
    .groupBy('asin', 'reviewerID')
      .agg(max('unixReviewTime').alias('unixReviewTime'))
  )

# deal with possible multiple entries on a given date
last_review = (
  reviews_with_duplicates
    .join(last_review_date, on=['asin','reviewerID','unixReviewTime'], how='leftsemi')
    .groupBy('asin','reviewerID')
      .agg(min('rid').alias('rid'))
    )

# locate last product review by a user
deduped_reviews = (reviews_with_duplicates
  .join(last_review, on=['asin','reviewerID','rid'], how='leftsemi')
  .drop('rid')
  )

display(deduped_reviews)

# COMMAND ----------

# MAGIC %md Now I persist the data to a Delta Lake table before proceeding. We persis the table so that it is fault tolerant and we won't have any inconsistencies in the underlying data within the tables:

# COMMAND ----------

# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS reviews.reviews')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/silver/reviews', ignore_errors=True)

# persist reviews as delta table
(
  deduped_reviews
   .repartition(sc.defaultParallelism * 4)
   .write 
   .format('delta')
   .mode('overwrite')
   .save('/mnt/reviews/silver/reviews')
  )

# make table queriable
_ = spark.sql('''
  CREATE TABLE IF NOT EXISTS reviews.reviews
  USING DELTA
  LOCATION '/mnt/reviews/silver/reviews'
  ''')

# COMMAND ----------

# DBTITLE 1,Drop Cached Dataset
_ = reviews_with_duplicates.unpersist()

# COMMAND ----------

# MAGIC %md To make joining to the product metadata easier, I add the product ID generated earlier with the metadata to the reviews table:

# COMMAND ----------

# DBTITLE 1,Update Reviews with Product IDs
_ = spark.sql('ALTER TABLE reviews.reviews ADD COLUMNS (product_id long)')

# retrieve asin-to-id map
ids = spark.table('reviews.metadata').select('asin','id')

# access reviews table in prep for merge
reviews = DeltaTable.forName(spark, 'reviews.reviews')

# perform merge to update ID
( reviews.alias('reviews')
    .merge(
      ids.alias('metadata'),
      condition='reviews.asin=metadata.asin'
      )
    .whenMatchedUpdate(set={'product_id':'metadata.id'})
).execute()

# display updated records
display(
  spark.table('reviews.reviews')
  )

# COMMAND ----------

# MAGIC %md ## Step 4: Filter Metadata by Reviews
# MAGIC 
# MAGIC For the purposes of this project to demonstrate the ability for product reviewers (that we will use as the basis for our profiles in the engine) to see similar products, I only want to keep complimentary products in the metadata files. I will now filter the metadata so that we only capture these products.

# COMMAND ----------

_ = spark.sql('DELETE FROM reviews.metadata as metadata WHERE NOT EXISTS (SELECT product_id from reviews.reviews as reviews where metadata.id = reviews.product_id )')

# COMMAND ----------

# MAGIC %md now let's check to see if our delete statement performed successfully. If so, no records should return.

# COMMAND ----------

check = spark.sql('SELECT distinct id from reviews.metadata as metadata LEFT JOIN reviews.reviews as reviews ON metadata.id = reviews.product_id where reviews.product_id is null')

display(check)

# COMMAND ----------

# DBTITLE 1,Cleanup Delta Table
spark.conf.set('spark.databricks.delta.retentionDurationCheck.enabled', False)
_ = spark.sql('VACUUM reviews.reviews RETAIN 0 HOURS')

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2a: Determine Title Similarity

# COMMAND ----------

# MAGIC %md
# MAGIC Now I will examine how features may be extracted from product titles in order to calculate similarities between products. These similarities will be used as the basis for making **Related Product** recommendations:
# MAGIC 
# MAGIC <img src="https://leelearningtest1.s3.amazonaws.com/1_j7kFOQgi6V-DvnSLHcUYHg.png" width="600">

# COMMAND ----------

# MAGIC %md **NOTE** The cluster with which this notebook is run should be created using a cluster-scoped initialization script which installs the NLTK WordNet corpus and Averaged Perceptron Tagger.  The below cell can be used to generate such a script to associate it with the cluster **before** running any code that depends upon it:

# COMMAND ----------

# DBTITLE 1,Generate Cluster Init Script
dbutils.fs.mkdirs('dbfs:/databricks/scripts/')

dbutils.fs.put(
  '/databricks/scripts/install-nltk-downloads.sh',
  '''#!/bin/bash\n/databricks/python/bin/python -m pip install nltk\n/databricks/python/bin/python -m pip install mlflow\n/databricks/python/bin/python -m nltk.downloader all''', 
  True
  )
  
# show script content
print(
  dbutils.fs.head('dbfs:/databricks/scripts/install-nltk-downloads.sh')
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 1: Prepare Title Data
# MAGIC 
# MAGIC The goal in this project is to recommend highly similar products and to identify such products based on product names.  This information is captured in the *title* field within this dataset:

# COMMAND ----------

# DBTITLE 1,Retrieve Titles
# retrieve titles for each product
titles = (
  spark
    .table('reviews.metadata')
    .filter('title Is Not Null')
    .select('id', 'asin', 'title')
  )

num_of_titles = titles.count()
num_of_titles

# COMMAND ----------

# present titles
display(
  titles
  )

# COMMAND ----------

# MAGIC %md In order to enable similarity comparisons between titles, I will employ a fairly straightforward word-based comparison where each word is weighted relative to its occurrence in the title and it's overall occurrence across all titles.  These weights are often calculated using *term-frequency - inverse document frequency* (*TF-IDF*) scores.[2]
# MAGIC 
# MAGIC As a first step in calculating TF-IDF scores, we need to split out the words in our titles and move them to a consistent case (i.e. present tense).  This is done here using the RegexTokenizer in PySpark:
# MAGIC 
# MAGIC <img src='https://leelearningtest1.s3.amazonaws.com/tokenization.png'>
# MAGIC 
# MAGIC Note that I am leveraging RegexTokenizer in Pyspark instead of a simple Tokenizer that splits text on white-space since it allows me to better deal with things like stray punctuation characters:

# COMMAND ----------

# DBTITLE 1,Retrieve Words from Titles
# split titles into words
tokenizer = RegexTokenizer(
    minTokenLength=2, 
    pattern='[^\w\d]',  # split on "not a word and not a digit"
    inputCol='title', 
    outputCol='words'
    )

title_words = tokenizer.transform(titles)

display(title_words)

# COMMAND ----------

# MAGIC %md With the words split out, I need to figure out how to deal with common word variations such as singular vs. plural forms and tense verbs (i.e. present, past & future), and other word-form variations which may cause very similar words to be seen separately.
# MAGIC 
# MAGIC One technique for this is known as stemming.[3]  With stemming, common word suffixes are dropped in order to truncate a word to its root (stem).  While effective, stemming lacks knowledge about how a word is used and how words with non-standard forms, *i.e. man vs. men*, might be related. Using a slightly more sophisticated technique known as lemmatization [4], I can better standardize word forms:
# MAGIC 
# MAGIC <img src='https://leelearningtest1.s3.amazonaws.com/main-qimg-250c86c2671ae3f4c4ad13191570f036.png' width = '700'>
# MAGIC 
# MAGIC There are a variety of libraries I can use to convert words to *lemmata* (plural of *lemma*).  NLTK [5] is one of the more popular of these and is pre-installed in most Databricks ML runtime environments.  Still, to perform lemmatization with NLTK, I install a tagged corpus and part of speech (POS) predictor on each worker node in our cluster (hence the init script).
# MAGIC 
# MAGIC Here, I am using the WordNet corpus [6] to provide context for the words.  I use this context to not only standardize words but to eliminate words that are not commonly used as adjectives, nouns, verbs or adverbs, the parts of speech that typically carry the most information.  
# MAGIC 
# MAGIC **NOTE** I'm implementing the lemmatization logic using a pandas UDF with an *iterator of series to iterator of series* type.  This is useful in scenarios where expensive initialization takes place.

# COMMAND ----------

# DBTITLE 1,Standardize Words
# declare the udf
@pandas_udf(ArrayType(StringType()))
def lemmatize_words(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    
    # setup wordnet part-of-speech info
    pos_corpus = nltk.corpus.wordnet
    pos_dict = {
      'J': nltk.corpus.wordnet.ADJ,
      'N': nltk.corpus.wordnet.NOUN,
      'V': nltk.corpus.wordnet.VERB,
      'R': nltk.corpus.wordnet.ADV
      }
    
    # initialize lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()
  
    def get_lemmata(words):
      
      # determine part of speech for each word
      pos_words = nltk.pos_tag(words.tolist())
      
      # lemmatize words in relevant parts of speech
      lemmata = []
      for word, pos in pos_words:
          # just use first char of part of speech
          pos = pos[0]
          
          # if part of speech of interest, lemmatize
          if pos in pos_dict:
            lemmata += [lemmatizer.lemmatize(word, pos_dict[pos])] 
            
      return lemmata
  
    # for each set of words from the iterator
    for words in iterator:
        yield words.apply(get_lemmata)

# use function to convert words to lemmata
title_lemmata = (
  title_words
    .withColumn(
      'lemmata',
      lemmatize_words('words') 
      )
    .select('id','asin', 'title', 'lemmata')
    ).cache()

display(title_lemmata)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 2: Calculate TF-IDF Scores
# MAGIC 
# MAGIC With the data prepared, I can now proceed with the calculation of the *term-frequency* portion of our TF-IDF calculation.  Here, I take a simple count of the occurrence of words within a title.  Because titles are typically succinct, I would expect that most words will occur only once in a given title.  To avoid counting more rare words that will not help with a similarity comparison, I limit the number of words to count to the top 262,144 words across all titles.  This is the default configuration of the word counters in Spark but I am explicitly assigning this value in the code to make it clear that a limit is in place.
# MAGIC 
# MAGIC The CountVectorizer performs the count through a brute-force exercise which works fine for this project's intention.

# COMMAND ----------

# DBTITLE 1,Count Word Occurrences in Titles
# count word occurences
title_tf = (
  CountVectorizer(
    inputCol='lemmata', 
    outputCol='tf',
    vocabSize=262144  # top n words to consider
    )
    .fit(title_lemmata)
    .transform(title_lemmata)
    )

display(title_tf.select('id','asin','lemmata','tf'))

# COMMAND ----------

# MAGIC %md Now I calculate the *inverse document frequency* (IDF) for the words in the titles. As a word is used more and more frequently across titles, it's IDF score will decrease logarithmically indicating it carries less and less differentiating information.  The raw IDF scores are typically multiplied against the TF scores to produce the desired TF-IDF score. The formula for this calculation is listed below:
# MAGIC 
# MAGIC <img src="https://leelearningtest1.s3.amazonaws.com/review_tfidf.png" width="500">
# MAGIC 
# MAGIC All of is tackled through the IDF transform:

# COMMAND ----------

# DBTITLE 1,Calculate TF-IDF Scores
# calculate tf-idf scores
title_tfidf = (
  IDF(inputCol='tf', outputCol='tfidf')
    .fit(title_tf)
    .transform(title_tf)
  )

display(
  title_tfidf.select('id','asin','lemmata','tfidf')
  )

# COMMAND ----------

# MAGIC %md We need to normalize the return TF-IDF scores (i.e. the Values in the tfidf fiels) because they are not normalized and could throw off our recommendation engine.  With similarity calculations based on the distance calculations, I apply an L2-normalization for this project.  To do this, I use the Normalizer function to transform to TF-IDF scores:

# COMMAND ----------

# DBTITLE 1,Normalize the TF-IDF Values
# apply normalizer
title_tfidf_norm = (
  Normalizer(inputCol='tfidf', outputCol='tfidf_norm', p=2.0)
    .transform(title_tfidf)
  )

display(title_tfidf_norm.select('id','asin','lemmata','tfidf','tfidf_norm'))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 3: Identify Products with Similar Titles
# MAGIC 
# MAGIC I now have features to calculate similarities between titles. **Local Sensitive Hashing** is a technique to limit the comparing of products by looking at those that are most likely to be similar to one another. In LSH, we define a false positive as a pair of distant input features (with d(p,q)≥r2) which are hashed into the same bucket, and we define a false negative as a pair of nearby features (with d(p,q)≤r1) which are hashed into different buckets.
# MAGIC 
# MAGIC Note the output columns (outputCol = 'hash') is Seq[Vector] where the dimension of the array equals numHashTables, and the dimensions of the vectors are set to 1.[7]

# COMMAND ----------

# DBTITLE 1,Apply LSH to Titles
# configure lsh
bucket_length = 0.0001
lsh_tables = 5

# fit the algorithm to the dataset 
fitted_lsh = (
  BucketedRandomProjectionLSH(
    inputCol = 'tfidf_norm', 
    outputCol = 'hash', 
    numHashTables = lsh_tables, 
    bucketLength = bucket_length
    ).fit(title_tfidf_norm)
  )

# assign LSH buckets to users
hashed_vectors = (
  fitted_lsh
    .transform(title_tfidf_norm)
    ).cache()

display(
  hashed_vectors.select('id','asin','title','tfidf_norm','hash')
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC Using LSH, I sort titles into buckets of approximately similar values.  While not perfect, it can locate similar products reasonably well:

# COMMAND ----------

# retrieve data for example product
sample_product = hashed_vectors.filter('asin==\'B00NJP2GAK\'') 
                                       
display(
  sample_product.select('id','asin','title','tfidf_norm','hash')
  )                                     

# COMMAND ----------

number_of_titles = 100

# retrieve n nearest customers 
similar_k_titles = (
  fitted_lsh.approxNearestNeighbors(
    hashed_vectors, 
    sample_product.collect()[0]['tfidf_norm'], 
    number_of_titles, 
    distCol='distance'
    )
    .withColumn('raw_sim', expr('1/(1+distance)'))
    .withColumn('min_score', expr('1/(1+sqrt(2))'))
    .withColumn('similarity', expr('(raw_sim - min_score)/(1-min_score)'))
    .select('id', 'asin', 'title', 'distance', 'similarity')
  )
  
display(similar_k_titles)

# COMMAND ----------

# MAGIC %md 
# MAGIC From a quick review of the data, I observe many items with similar titles. I examine the products within the category in which the sample product resides to get a better sense of how complete my approach is; however, there's a degree of subjectivity in this evaluation that is difficult to avoid. As with other NLP projects, this is a very common challenge when analyzing recommendations where there is no "baseline truth" to compare against (e.g. Product rating or a purchase event).  What I do in this project is adjust settings to establish a reasonably good set of recommendations, performing a limited test with real customers to see how they respond to the suggestions from the model.

# COMMAND ----------

# DBTITLE 1,Drop Cached Datasets
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2b: Determine Description Similarity
# MAGIC 
# MAGIC Now I will examine how features may be extracted from product descriptions in order to calculate similarities between products. These similarities will be used as the basis for making **Related products** recommendations:

# COMMAND ----------

# MAGIC 
# MAGIC %md 
# MAGIC ## Step 1: Prepare Description Data
# MAGIC 
# MAGIC Titles are useful, but carry limited information.  Longer descriptions (as found in the *description* field of this dataset) provide quite a bit more detail which I can use to identify similar items.
# MAGIC 
# MAGIC But having more words with which to make a comparison means more data to process and more complexity to deal with.  To help with that, I employ one of several dimension reduction techniques which attempt to condense text into a much more narrow set of *topics* or *concepts* which can then be used as the basis for comparison.
# MAGIC 
# MAGIC To get started exploring this direction, let's flatten the array that holds our description data and tokenize the words in the text as we did earlier:

# COMMAND ----------

# DBTITLE 1,Retrieve Descriptions
# retrieve descriptions for each product
descriptions = (
  spark
    .table('reviews.metadata')
    .filter('size(description) > 0')
    .withColumn('descript', array_join('description',' '))
    .selectExpr('id', 'asin', 'descript as description')
  )

num_of_descriptions = descriptions.count()
num_of_descriptions

# COMMAND ----------

# present descriptions
display(
  descriptions
  )

# COMMAND ----------

# DBTITLE 1,Retrieve Words from Descriptions
# split titles into words
tokenizer = RegexTokenizer(
    minTokenLength=2, 
    pattern='[^\w\d]',  # split on "not a word and not a digit"
    inputCol='description', 
    outputCol='words'
    )

description_words = tokenizer.transform(descriptions)

display(description_words)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 2a: Extract LDA Topic Features
# MAGIC 
# MAGIC Let's now explore the use of Latent Dirichlet Allocation (LDA) to reduce the descriptions to a condensed set of topics:
# MAGIC 
# MAGIC <img src="https://leelearningtest1.s3.amazonaws.com/sample_words.gif" >
# MAGIC 
# MAGIC While the math behind LDA can get complex, the technique itself is fairly easy to understand.  In a nutshell, I examine the co-occurrence of words in the description text.  "Clusters" of words that occur with each other with some regularity represent *topics*, though to a human such topics might be a little challenging to comprehend. Still, I score each description based on its alignment with each of the topics discovered across the descriptions.  Those per-topic scores then provide the basis for locating similar documents in the dataset. [8]
# MAGIC 
# MAGIC To perform the LDA calculations, I first standardize the words in the descriptions using lemmatization, just as I did with the title data:

# COMMAND ----------

# declare the udf
@pandas_udf(ArrayType(StringType()))
def lemmatize_words(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    
    # setup wordnet part-of-speech info
    pos_corpus = nltk.corpus.wordnet
    pos_dict = {
      'J': nltk.corpus.wordnet.ADJ,
      'N': nltk.corpus.wordnet.NOUN,
      'V': nltk.corpus.wordnet.VERB,
      'R': nltk.corpus.wordnet.ADV
      }
    
    # initialize lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()
  
    def get_lemmata(words):
      
      # determine part of speech for each word
      pos_words = nltk.pos_tag(words.tolist())
      
      # lemmatize words in relevant parts of speech
      lemmata = []
      for word, pos in pos_words:
          # just use first char of part of speech
          pos = pos[0]
          
          # if part of speech of interest, lemmatize
          if pos in pos_dict:
            lemmata += [lemmatizer.lemmatize(word, pos_dict[pos])] 
            
      return lemmata
  
    # for each set of words from the iterator
    for words in iterator:
        yield words.apply(get_lemmata)

# use function to convert words to lemmata
description_lemmata = (
  description_words
    .withColumn(
      'lemmata',
      lemmatize_words('words') 
      )
    .select('id','asin', 'description', 'lemmata')
    ).cache()

display(description_lemmata)

# COMMAND ----------

# MAGIC %md 
# MAGIC I now count the occurrence of the words in the dataset.  Because I am working with many more words than were found in the titles, I use the HashingTF transform to do this work.  
# MAGIC 
# MAGIC While the HashingTF and CountVectorizer appear to be performing the same work, the HashingTF transform is using a short-cut to speed up the process. Instead of creating a distinct list of words from across all the words in the descriptions, calculating term-frequency scores for each, and then limiting the resulting vector to the top occurring of those words, the HashingTF transform hashes each word to an integer value and uses that hashed value as the word's index in the output vector. Instead of creating a word-lookup table and performing a count against it, I simply calculate a hash and add one to the value in the associated index position.  The trade off is that hash collisions will occur so that there will be situations where two unrelated words are counted as if they were the same.  Understanding the tradeoff of accuracy here, I accept the possibility of a little sloppiness in order to pick up substantial performance gains, and so I choose HashingTF to transform the data:

# COMMAND ----------

# DBTITLE 1,Count Words
# get word counts from descriptions
description_tf = (
  HashingTF(
    inputCol='lemmata', 
    outputCol='tf',
    numFeatures=262144  # top n words to consider
    )
    .transform(description_lemmata)
  )

display(description_tf)

# COMMAND ----------

# MAGIC %md 
# MAGIC With word counts in place, I apply LDA to define topics and score the descriptions relative to these.  Given that this is a training use case and as such a smaller cluster has been deployed, this step takes approx ~20 minutes to complete.  With that in mind, notice that the model is *learning* the LDA topics using a 25% random sample of the overall dataset which will allow me to arrive at valid topics while limiting computation time:

# COMMAND ----------

# DBTITLE 1,Apply LDA
# identify LDA topics & score descriptions against these
description_lda = (
  LDA(
    k=100, 
    maxIter=20,
    featuresCol='tf',
    optimizer='online' # use the online optimizer for scalability
    )
    .fit(description_tf.sample(False, 0.25)) # train on a random sample of the data
    .transform(description_tf) # transform all the data
  )

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/tmp/description_lda', ignore_errors=True)

# persist as delta table
(
  description_lda
   .write
   .format('delta')
   .mode('overwrite')
   .save('/mnt/reviews/tmp/description_lda')
  )

display(
  spark.table('DELTA.`/mnt/reviews/tmp/description_lda`').select('id','asin','description','topicDistribution')
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC The LDA scores now serve as features with which I evaluate description similarities.  As in Part 2a: Determine Title Similarities, I normalize these before proceeding with that step:

# COMMAND ----------

# DBTITLE 1,Normalize LDA Features
description_lda = spark.table('DELTA.`/mnt/reviews/tmp/description_lda`')

description_lda_norm = (
  Normalizer(inputCol='topicDistribution', outputCol='features', p=2.0)
    .transform(description_lda)
  ).cache()

display(description_lda_norm.select('id','asin','description', 'features'))

# COMMAND ----------

# MAGIC %md 
# MAGIC Before looking at how to calculate similarities using normalized features, I examine another dimension reduction technique.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 2b: Extract Word2Vec *Concept* Features
# MAGIC 
# MAGIC LDA scores the descriptions on their relationship to discovered topics using words found anywhere in the description. In other words, the sequencing of words in a description is not taken into consideration. 
# MAGIC 
# MAGIC Word2Vec on the other hand examines word proximity to get at *concepts* within the descriptions:
# MAGIC 
# MAGIC <img src="https://leelearningtest1.s3.amazonaws.com/vectors.gif">
# MAGIC 
# MAGIC **NOTE** Word2Vec does not require any preprocessing other than tokenization.  Also, note that Word2Vec can take a while to run so I am fitting the model on only 25% of the data.

# COMMAND ----------

# DBTITLE 1,Apply Word2Vec
description_words = description_words.cache()

# generate w2v set
description_w2v =(
  Word2Vec(
    vectorSize=100,
    minCount=3,              # min num of word occurances required for consideration
    maxSentenceLength=1000,  # max num of words in description to consider
    numPartitions=sc.defaultParallelism*10,
    maxIter=20,
    inputCol='words',
    outputCol='w2v'
    )
    .fit(description_words.sample(False, 0.25))
    .transform(description_words)
  )

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/tmp/description_w2v', ignore_errors=True)

# persist as delta table
(
  description_w2v
   .write
   .format('delta')
   .mode('overwrite')
   .save('/mnt/reviews/tmp/description_w2v')
  )

display(
  spark.table('DELTA.`/mnt/reviews/tmp/description_w2v`').select('id','asin','description','w2v')
  )

# COMMAND ----------

# MAGIC %md As with the LDA-derived features, the Word2Vec features require normalization:

# COMMAND ----------

# DBTITLE 1,Normalize Word2Vec
description_w2v = spark.table('DELTA.`/mnt/reviews/tmp/description_w2v`')

description_w2v_norm = (
  Normalizer(inputCol='w2v', outputCol='features', p=2.0)
    .transform(description_w2v)
  ).cache()

display(description_w2v_norm.select('id','asin','description', 'w2v', 'features'))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 3: Calculate Description Similarities
# MAGIC 
# MAGIC Instead of using LSH like we did with the product titles, I use k-means clustering on the LDA/Word2Vec features to find similar items.
# MAGIC 
# MAGIC Using k-means clustering, I assign products to clusters based on feature similarity.  I then use cluster assignment to limit the searches for similar products to those found within a cluster (much like to limit similarity search to products in a shared bucket using LSH).  In Spark there are two options: Traditional k-means and bisecting k-means. Either is applicable but I will use bisecting k-means for this project.  I then use traditional elbow techniques to identify an optimal number of clusters though consideration of query performance against the result set is also important.  Here, I opt for 50 clusters as this seems to provide a reasonable split on the data.  It's important to consider the application of clustering in this project as an approximate technique, much like LSH:
# MAGIC 
# MAGIC **NOTE** The clustered/bucketed data is being persisted to Delta Lake for re-use in a later notebook. In addition, we are persisting the clustering model using mlflow for similar re-use.

# COMMAND ----------

# DBTITLE 1,Assign Descriptions to Clusters
clustering_model = (
  BisectingKMeans(
    k=50,
    featuresCol='features', 
    predictionCol='bucket',
    maxIter=100,
    minDivisibleClusterSize=100000
    )
    .fit(description_w2v_norm.sample(False, 0.25))
  )

descriptions_clustered = (
  clustering_model
    .transform(description_w2v_norm)
  )

# persist the clustering model for next notebook
with mlflow.start_run():
  mlflow.spark.log_model(
    clustering_model, 
    'model',
    registered_model_name='description_clust'
    )
  
# persist this data for the next notebook
shutil.rmtree('/dbfs/mnt/reviews/tmp/description_sim', ignore_errors=True)
(
  descriptions_clustered
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/reviews/tmp/description_sim')
  )

display(
  descriptions_clustered
    .groupBy('bucket')
    .agg(count('*').alias('descriptions'))
    .orderBy('bucket')
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC With the data assigned to a cluster, locating similar products is fairly straightforward.  I perform an exhaustive comparison within each cluster/bucket.
# MAGIC 
# MAGIC When using LSH, the calculation of the Euclidean distance between vectors was performed for me.  Here, I will need to perform the distance calculation using a custom function. 
# MAGIC 
# MAGIC Euclidean distance calculations between two vectors are fairly easy to perform.  However, the pandas UDFs used before does not have a means to accept a vector in its native format.  But Scala does.  By registering our Scala function for the calculation of Euclidean distance between two vectors with the Spark SQL engine, we can easily apply our function to data in a Spark DataFrame using Python as will be demonstrated in a later cell:

# COMMAND ----------

# DBTITLE 1,Define Function for Distance Calculations
# MAGIC %scala
# MAGIC 
# MAGIC import math._
# MAGIC import org.apache.spark.ml.linalg.{Vector, Vectors}
# MAGIC 
# MAGIC val euclidean_distance = udf { (v1: Vector, v2: Vector) =>
# MAGIC     sqrt(Vectors.sqdist(v1, v2))
# MAGIC }
# MAGIC 
# MAGIC spark.udf.register("euclidean_distance", euclidean_distance)

# COMMAND ----------

# MAGIC %md And now I can make the recommendations, limiting comparisons to those items in the same bucket (cluster) as the sample product:
# MAGIC 
# MAGIC **NOTE** I use the same sample product as was used in the last notebook.

# COMMAND ----------

# DBTITLE 1,Retrieve Sample Product
sample_product = descriptions_clustered.filter('asin==\'B001H32N3Q\'')

display(sample_product)

# COMMAND ----------

# DBTITLE 1,Retrieve Similar Descriptions
display(
  descriptions_clustered
    .withColumnRenamed('features', 'features_b')
    .join(sample_product.withColumnRenamed('features', 'features_a'), on='bucket', how='inner')  # join on bucket/cluster
    .withColumn('distance', expr('euclidean_distance(features_a, features_b)')) # calculate distance
    .withColumn('raw_sim', expr('1/(1+distance)')) # convert distance to similarity score
    .withColumn('min_score', expr('1/(1+sqrt(2))'))
    .withColumn('similarity', expr('(raw_sim - min_score)/(1-min_score)'))
    .select(descriptions_clustered.id, descriptions_clustered.asin, descriptions_clustered.description, 'distance', 'similarity')
    .orderBy('distance', ascending=True)
    .limit(100) # get top 100 recommendations
    .select(descriptions_clustered.id, descriptions_clustered.asin, descriptions_clustered.description, 'distance', 'similarity')
    )

# COMMAND ----------

# DBTITLE 1,Drop Cached Datasets
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()

# COMMAND ----------

# MAGIC %md Unlike the TF-IDF scored titles, the basis for matching descriptions is a little harder to intuit from a simple viewing of the data.  The notion of *topics* or *concepts* are a bit more elusive than simple word-count derived scores.  Still, a review of the descriptions gives a sense as to why some descriptions are considered more similar than others. To constrain things a bit further, I might consider limiting the LDA and Word2Vec feature generation to a smaller number of words from the beginning of the description as this is the part of the text most likely to directly tie into the key aspects of the product.  With Word2Vec, this is done through a simple argument setting.  With LDA, I would need to add a step to truncate the tokenized words before performing lemmatization.

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2c: Determine Category Similarity
# MAGIC 
# MAGIC Now I will examine how features may be extracted from product categories in order to calculate similarities between products. These similarities will be used as the basis for making **Related products** recommendations:

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 1: Retrieve & Prepare Category Data
# MAGIC 
# MAGIC In an ideal scenario, the product hierarchy would be organized in a consistent manner such that every entry rolled up to a shared root node.  Such a structure, known as a tree, would allow to calculate product similarities based on the position of products within the tree structure.  Thabet Slimani provides a helpful review of various similarity metrics that could be used with trees [9].
# MAGIC 
# MAGIC Unfortunately, the product category captured in this dataset does not form a tree.  Instead, there is inconsistencies where a child category might be positioned under a parent category with one product with their positions reversed in another or where a child category might leap over a parent to roll directly into a shared ancestor.  The problems with the category hierarchy in this dataset are not easily corrected so I cannot treat this data as a tree.  That said, I can still apply some fun engineering techniques to assess similarity.
# MAGIC 
# MAGIC For future expansions of this project, I would like to recommend a master data management solution that can help provide more effective hierarchical & workable structures over time. But for now to get started with the category data, I first retrieve a few values and examine its structure:

# COMMAND ----------

# DBTITLE 1,Retrieve Category Data
# retrieve descriptions for each product
categories = (
  spark
    .table('reviews.metadata')
    .filter('size(category) > 0')
    .select('id', 'asin', 'category')
  )

num_of_categories = categories.count()
num_of_categories

# COMMAND ----------

display(
  categories
  )

# COMMAND ----------

# MAGIC %md The category data is organized as an array where the array index indicates the level in the hierarchy.  Appending ancestry information to each level's name allows me to uniquely identify each level within the overall structure so that categories with a common name but residing under different parents or other ancestors may be distinguishable from one another. I'll tackle this using a pandas UDF:
# MAGIC 
# MAGIC As I am finding many levels where the level name is more like a product or feature description, I am not sure if this is valid data or represents a misparsing of data from the source website.  With that being said, I will limit the category hierarchy to 10 levels max and break the hierarchy should I encounter a level name with more than 100 characters to avoid this data.  These are arbitrary settings that can be tweaked in future iterations or removed entirely if the fashion & beauty brand decides to leverage a master data management process.

# COMMAND ----------

# DBTITLE 1,Rename Categories to Include Ancestry
@pandas_udf(ArrayType(StringType()))
def cleanse_category(array_series: pd.Series) -> pd.Series:
  
  def cleanse(array):
    delim = '|'
    level_name = ''
    ret = []
    
    for a in array[:10]:  # limit to 10 levels
      if len(a) > 100:     # stop if level name more than max chars
        break
      else:
        level_name += a.lower() + '|'
        ret += [level_name]
    return ret
          
  return array_series.apply(cleanse)


categories = (
  spark
    .table('reviews.metadata')
    .filter('size(category)>0')
    .select('id','asin','category')
    .withColumn('category_clean', cleanse_category('category'))
  )

display(categories)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 2: Calculate Category Similarities
# MAGIC 
# MAGIC With the level names adjusted, I one-hot encode the category levels for each product.  I use the CountVectorizer used with the TF-IDF calculations to tackle this, setting it's *binary* argument to *True* so that all output is either 0 or 1 (which it should be anyway).  As before, this transform will limit the entries to some top number of frequently occurring values:

# COMMAND ----------

# DBTITLE 1,One-Hot Encode the Categories
categories_encoded = (
  HashingTF(
    inputCol='category_clean', 
    outputCol='category_onehot',
    binary=True
    )
    .transform(categories)
    .select('id','asin','category','category_onehot')
    .cache()
  )

display(
  categories_encoded
  )

# COMMAND ----------

# MAGIC %md As before, I divide records into buckets.  Because of how I assembled the categories, I know the primary categories are similar (it's a fashion and beauty brand so the top level category isn't enough to tell me a true story on categories) so to avoid overlap I group the products into buckets based on the second member of the categories array and see where that leads me:

# COMMAND ----------

# DBTITLE 1,Group Items Based on 2nd Tier Parent
roots = (
  categories_encoded
    .withColumn('root', expr('category[1]'))
    .groupBy('root')
      .agg(count('*').alias('members'))
    .withColumn('bucket', monotonically_increasing_id())
  )

categories_clustered = (
  categories_encoded
    .withColumn('root', expr('category[1]'))
    .join(roots, on='root', how='inner')
    .select('id','asin','category','category_onehot','bucket')
  )

display(roots.orderBy('members'))

# COMMAND ----------

# MAGIC %md In the real world, the categories would be more aligned to the fashion & beauty e-commerce platform (so we probably would not see categories like 'Men','Boys', and 'Baby') but for now I will not perform any further transformations as this gives us a general idea of the categories for which our products can be grouped. As an extension to this project, I will recommend a more thorough examination of product categories, partnering with the fashion & beauty brand to build categories for which is important to their business but for now I leave it as is.
# MAGIC 
# MAGIC I can now check all of the product categories for similarity by using a simple Jaccard similarity score. This is done by dividing the number of overlapping levels by the distinct number of levels between two products. The below image gives a basic example of how the Jaccard similarity score is derived using a simple example about books and user tastes on them:
# MAGIC 
# MAGIC <img src="https://leelearningtest1.s3.amazonaws.com/jaccard.png" width ='700'>
# MAGIC 
# MAGIC As before, I perform this work using a Scala function which I register with Spark SQL so that I can make use of it against the DataFrame defined in Python:

# COMMAND ----------

# DBTITLE 1,Define Function for Jaccard Similarity Calculation
# MAGIC %scala
# MAGIC 
# MAGIC import math._
# MAGIC import org.apache.spark.ml.linalg.SparseVector
# MAGIC 
# MAGIC val jaccard_similarity = udf { (v1: SparseVector, v2: SparseVector) =>
# MAGIC   val indices1 = v1.indices.toSet
# MAGIC   val indices2 = v2.indices.toSet
# MAGIC   val intersection = indices1.intersect(indices2)
# MAGIC   val union = indices1.union(indices2)
# MAGIC   intersection.size.toDouble / union.size.toDouble
# MAGIC }
# MAGIC 
# MAGIC spark.udf.register("jaccard_similarity", jaccard_similarity)

# COMMAND ----------

# MAGIC %md Let's return to our sample product to see how this is working:
# MAGIC 
# MAGIC **NOTE** We are using the same sample product as was used in the prior notebooks.

# COMMAND ----------

# DBTITLE 1,Retrieve Sample Product
sample_product = categories_clustered.filter("asin=='B001H32N3Q'")
display(sample_product)

# COMMAND ----------

# DBTITLE 1,Find Similar Products
display(
  categories_clustered
    .withColumnRenamed('category_onehot', 'features_b')
    .join(sample_product.withColumnRenamed('category_onehot', 'features_a'), on='bucket', how='inner')
    .withColumn('similarity', expr('jaccard_similarity(features_a, features_b)'))
    .orderBy('similarity', ascending=False)
    .limit(100)
    .select(categories_clustered.id, categories_clustered.asin, categories_clustered.category, 'similarity')
    )

# COMMAND ----------

# DBTITLE 1,Drop Cached Datasets
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()

# COMMAND ----------

# MAGIC %md Please proceed to notebook part 2 for the completion of the project 
