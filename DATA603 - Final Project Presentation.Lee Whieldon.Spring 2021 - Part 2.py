# Databricks notebook source
# MAGIC %md 
# MAGIC <img  src="https://emerj.com/wp-content/uploads/2018/10/recommendation-engines-for-fashion-6-current-applications-6.jpg" width="500" align="center">
# MAGIC 
# MAGIC # Fashion & Beauty E-Commerce Recommendation Engine
# MAGIC 
# MAGIC Using machine learning to increase sales, provide ease of information overload, and customize shopping experiences which leads to happy customers.
# MAGIC 
# MAGIC With our product titles, descriptions, and categories cleansed & ready for our recommendation engine, let's now explore how we leverage the reviews to extract profiles of the Fashion & Beauty brand's customers.
# MAGIC 
# MAGIC This is Part 2 of the Databricks notebooks being used for this project

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
# MAGIC # Part 3: User Profile Recommendations
# MAGIC I now examine how user ratings can be added to a **content-based** recommendation engine. By using feedback in the form of ratings assigned to products, I can start building a profile reflecting a user who interacts with the e-commerce platform and build an engine that can show the user other products they might be interested in based on the ratings they have given to other products.
# MAGIC 
# MAGIC <img src="https://leelearningtest1.s3.amazonaws.com/Collaborative_filtering.gif">

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 1: Retrieve Product Profiles
# MAGIC 
# MAGIC So far, I have made the content-based recommendations based exclusively on product features. This provides me the ability to identify items similar to others when the goal is supporting the fashion & beauty brand enablement of customers to explore product alternatives directly on the e-commerce platform.
# MAGIC 
# MAGIC The recommender I am building is a little different in that I want it to learn from a profile's product preferences based on the kinds of features likely to resonate with a customer.  This allows me to recommend a much wider variety of products but still do so with consideration of the kinds of features likely to appeal to that customer. I feel that especially in the fashion & beauty industry, it is imperative to understand your customers as tastes and attitude towards products changes so quickly.
# MAGIC 
# MAGIC To get started with this recommender, I will use the Word2Vec features along with their cluster/bucket assignment that was generated in part 2b, step 3:

# COMMAND ----------

# DBTITLE 1,Retrieve Products with Features
# retrieve featurized product data
product_features = (
  spark
    .table('DELTA.`/mnt/reviews/tmp/description_sim`').selectExpr('id','bucket','w2v','features')
    .join( # join to product metadata to allow easier review of recommendations
        spark.table('reviews.metadata').select('id','asin','title','category','description'),
        on='id',
        how='inner'
      )
    .select('id', 'asin', 'w2v', 'features', 'title', 'description', 'category', 'bucket')
  )

display(
  product_features
  ) 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 2: Assemble User Profiles
# MAGIC 
# MAGIC Now I need to construct a weighted average of the features preferred by each customer based on the product reviews provided.  While different than testing a machine learning model, I will use the same naming convention by separating reviews into exclusive training & testing sets to check customer preferences.  The training set will consist of the last two ratings by a user while the testing set will consist of all ratings by a user:

# COMMAND ----------

# DBTITLE 1,Separate Reviews into Training & Testing Sets
# retrieve user reviews numbered most recent to oldest
sequenced_reviews = (
  spark
    .sql('''
  WITH validReviews AS (
    SELECT
      reviewerID,
      product_id,
      overall as rating,
      unixReviewTime
    FROM reviews.reviews
    WHERE product_id IS NOT NULL
    )
  SELECT
    a.reviewerID,
    a.product_id,
    a.rating,
    row_number() OVER (PARTITION BY a.reviewerID ORDER BY a.unixReviewTime DESC) as seq_id
  FROM validReviews a
  LEFT SEMI JOIN (SELECT reviewerID FROM validReviews GROUP BY reviewerID HAVING COUNT(*)>=5) b
    ON a.reviewerID=b.reviewerID
    ''')
  )

# get last two ratings as training
reviews_train = (
  sequenced_reviews
    .filter('seq_id <= 2')
    .select('reviewerID', 'product_id', 'rating')
  )

# get all but last two ratings as testing
reviews_test = (
  sequenced_reviews
    .filter('seq_id > 2')
    .select('reviewerID', 'product_id', 'rating')
  )

display(
  reviews_test
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC Now I grab the count of reviews for all product users have rated, using the rating number to determine if we have any skews in the data.  Like with most e-commerce websites, the majority of products go unrated: I can safeley assume that a user will give a rating when they are extremely pleased or extremely dissatisfied with a product so I expect ratings to be skewed a bit towards the ends of the scale:

# COMMAND ----------

display(
  reviews_test
    .groupBy('rating')
      .agg( count('*').alias('instances'))
    .orderBy('rating')
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC Sure enough, the ratings are skewed. For the purposes of this project, I will only look at users who have given a highly favorable rating (rating = 4 or 5) as I can assume that if they rated the product high, they have purchased the product and this can give us a good indcator of the other types of products they might buy in the future to support our recommendation engine.
# MAGIC 
# MAGIC I perform the weighted averaging against a feature vector, using the Summarizer transformer in pyspark.ml.stat. This allows me to perform simple aggregations against vectors with little code and includes support for weighted means.  Notice that for this, I am making use of the Word2Vec features in their pre-normalized state which means I'll need to apply normalization after they are averaged:

# COMMAND ----------

# DBTITLE 1,Assemble User-Profiles
# calculate weighted averages on product features for each user
user_profiles_test = (
  product_features
    .join(
      reviews_test.filter('rating >= 4'),  # limit ratings to 4s and 5s as discussed above
      on=[product_features.id==reviews_test.product_id], 
      how='inner'
      )
    .groupBy('reviewerID')
      .agg( 
        Summarizer.mean(col('w2v'), weightCol=col('rating')).alias('w2v')  
        )
  )

user_profiles_test_norm = (
  Normalizer(inputCol='w2v', outputCol='features', p=2.0)
    .transform(user_profiles_test)
  ).cache()


display(
  user_profiles_test_norm
  )

# COMMAND ----------

# MAGIC %md Now I have one feature vector for each user representing the weighted feature preferences for that user.  I can think of this vector as representing the *ideal product* for each user (based on his or her ratings of products).  The goal will now be to find products similar to this product. I assign each feature vector to a cluster/bucket. The bucketed profiles are persisted for later use:

# COMMAND ----------

# DBTITLE 1,Assign Cluster/Bucket & Persist User-Profiles
# retrieve model from mlflow
cluster_model = mlflow.spark.load_model(
    model_uri='models:/description_clust/None'
  )

# assign profiles to clusters/buckets
user_profiles_test_clustered = (
  cluster_model.transform(user_profiles_test_norm)
  )

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/gold/user_profiles_test', ignore_errors=True)

# persist dataset as delta table
(
  user_profiles_test_clustered
   .write
   .format('parquet')
   .mode('overwrite')
   .partitionBy('bucket')
   .save('/mnt/reviews/gold/user_profiles_test')
  )

display(
  spark.table('PARQUET.`/mnt/reviews/gold/user_profiles_test`')
  )

# COMMAND ----------

# DBTITLE 1,Examine Distribution by Bucket
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   bucket,
# MAGIC   count(*) as profiles
# MAGIC FROM PARQUET.`/mnt/reviews/gold/user_profiles_test`
# MAGIC GROUP BY bucket
# MAGIC ORDER BY bucket

# COMMAND ----------

# MAGIC %md While some buckets are greater than others, we have a pretty good variety of profiles assigned to each bucket so this gives us enough data to work with. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 3: Build & Evaluate Recommendations
# MAGIC 
# MAGIC I now have products with features and the user-profiles representing their product preferences.  To find products our engine might recommend, I calculate similarities between the products and these user-preferences.
# MAGIC 
# MAGIC To enable evaluation of these recommendations, I limit the users to a small random sample and calculate a weighted mean percent score. 

# COMMAND ----------

# DBTITLE 1,Define Function for Distance Calculation
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

# DBTITLE 1,Take Random Sample of Reviewers
test_profiles= spark.table('PARQUET.`/mnt/reviews/gold/user_profiles_test`').sample(False, 0.01)

test_profiles.count()

# COMMAND ----------

# DBTITLE 1,Determine Recommendations for Selected Users
# make recommendations for sampled reviewers
recommendations = (
  product_features
    .hint('skew','bucket') # hint to ensure join is balanced
    .withColumnRenamed('features', 'features_b')
    .join( test_profiles.withColumnRenamed('features', 'features_a'), on='bucket', how='inner') # join products to profiles on buckets
    .withColumn('distance', expr('euclidean_distance(features_a, features_b)')) # calculate similarity
    .withColumn('raw_sim', expr('1/(1+distance)'))
    .withColumn('min_score', expr('1/(1+sqrt(2))'))
    .withColumn('similarity', expr('(raw_sim - min_score)/(1-min_score)'))
    .select('reviewerID', 'id', 'similarity')
    .withColumn('rank_ui', expr('percent_rank() OVER(PARTITION BY reviewerID ORDER BY similarity DESC)')) # calculate percent rank for recommendations
    )

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/gold/user_profile_recommendations', ignore_errors=True)

# persist dataset as delta table
(
  recommendations
   .write
   .format('delta')
   .mode('overwrite')
   .save('/mnt/reviews/gold/user_profile_recommendations')
  )

# COMMAND ----------

# DBTITLE 1,Show Selection of Results for One Reviewer
# we are retrieving a subset of recommendations for one user so that the range of rank_ui values is more visible
display(
  spark
    .table('DELTA.`/mnt/reviews/gold/user_profile_recommendations`')
    .join( spark.table('DELTA.`/mnt/reviews/gold/user_profile_recommendations`').limit(1), on='reviewerID', how='left_semi' )
    .sample(False,0.01) 
    .orderBy('rank_ui', ascending=True)
  )

# COMMAND ----------

# MAGIC %md And now we can evaluate our model to see if it's successful in identifying where customers makes their next output:

# COMMAND ----------

# retreive recommendations generated in prior step
recommendations = spark.table('DELTA.`/mnt/reviews/gold/user_profile_recommendations`')

# calculate evaluation metric
display(
  reviews_train
  .join(
    recommendations, 
    on=[reviews_train.product_id==recommendations.id, reviews_train.reviewerID==recommendations.reviewerID], 
    how='inner'
    )
  .withColumn('weighted_r', recommendations.rank_ui * reviews_train.rating)
  .groupBy()
    .agg( sum('weighted_r').alias('numerator'),
          sum('rating').alias('denominator')
        )
  .withColumn('mean_percent_rank', expr('numerator / denominator'))
  .select('mean_percent_rank')
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC #### What does this Precision Score Mean?
# MAGIC At first glance, **23%** precision would not appear to be a great score but let's talk through how to interpret the score: We are looking at the precision that our recommendation engine will provide an *accurate* product to the customer as they are perusing the e-commerce webpage and shopping. As explored in Li, Wang, Knox, & Padmanabhan's experiment where they created a recommendation engine using hierarchical clustering before providing a recommendation, their precision, on average, was around 29%, AND outperformed a typical K-means clustering technique which had an average score of 18%.[11] 
# MAGIC 
# MAGIC Thus, 23% is a pretty good start for a recommendation engine of this nature; note also that the precision score will improve overtime as we continue to track and monitor the fashion & beauty brand's customer base.

# COMMAND ----------

# MAGIC %md 
# MAGIC # Part 4: Deploying a Content-Based Recommendation Engine
# MAGIC 
# MAGIC This section will bring all the prior parts & steps together and demonstrates how a content-base recommendation might be operationalized and handed off to web developers building and maintaing the e-commerce website to incorporate into the site.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 1: Assemble Product Feature Pipelines
# MAGIC 
# MAGIC The content recommender I am building uses 3 feature sets: The **title**, **description** & **category** fields in the product metadata. We can leverage pipelines to organize each of these feature sets we created through a series of various transformations that must be applied in a specific sequence. Once the pipelines are built, I then fit and persist for re-use between cycles. I first build the title transformation pipeline.
# MAGIC 
# MAGIC Each pipeline will consist of the following:</p>
# MAGIC 1. Tokenize the title to produce bag of words for the model
# MAGIC 2. Lemmatize the bag of words
# MAGIC 3. Calculate token frequency
# MAGIC 4. Calculate TF-IDF scores
# MAGIC 5. Normalize the TF-IDF scores
# MAGIC 
# MAGIC Most all of the above steps can be performed using standard mlflow pipelines; however, lemmatization step will need a custom transformer that I will build. 

# COMMAND ----------

# DBTITLE 1,Define Custom Transform for Lemmatization
class NLTKWordNetLemmatizer(
  Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
  ):
 
  @keyword_only
  def __init__(self, inputCol=None, outputCol=None):
    super(NLTKWordNetLemmatizer, self).__init__()
    kwargs = self._input_kwargs
    self.setParams(**kwargs)
  
  @keyword_only
  def setParams(self, inputCol=None, outputCol=None):
    kwargs = self._input_kwargs
    return self._set(**kwargs)

  def setInputCol(self, value):
    return self._set(inputCol=value)
    
  def setOutputCol(self, value):
    return self._set(outputCol=value)
  
  def _transform(self, dataset):
    
    # copy-paste of previously defined UDF
    # =========================================
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
    # =========================================
    
    t = ArrayType(StringType())
    out_col = self.getOutputCol()
    in_col = dataset[self.getInputCol()]
    
    return dataset.withColumn(out_col, lemmatize_words(in_col))

# COMMAND ----------

# MAGIC %md Now I construct the titles pipeline and use the custom transformer above inside it.

# COMMAND ----------

# DBTITLE 1,Assemble Titles Pipeline
# step 1: tokenize the title
title_tokenizer = RegexTokenizer(
  minTokenLength=2, 
  pattern='[^\w\d]', 
  inputCol='title', 
  outputCol='words'
  )

# step 2: lemmatize the word tokens
title_lemmatizer = NLTKWordNetLemmatizer(
  inputCol='words',
  outputCol='lemmata'
  )

# step 3: calculate term-frequencies
title_tf = CountVectorizer(
  inputCol='lemmata', 
  outputCol='tf',
  vocabSize=262144
  )
  
# step 4: calculate inverse document frequencies
title_tfidf = IDF(
  inputCol='tf', 
  outputCol='tfidf'
  )
  
# step 5: normalize tf-idf scores
title_normalizer = Normalizer(
  inputCol='tfidf', 
  outputCol='tfidf_norm', 
  p=2.0
  )

# step 6: assign titles to buckets
title_lsh = BucketedRandomProjectionLSH(
  inputCol = 'tfidf_norm', 
  outputCol = 'hash', 
  numHashTables = 5, 
  bucketLength = 0.0001
  )

# assemble pipeline
title_pipeline = Pipeline(stages=[
  title_tokenizer,
  title_lemmatizer,
  title_tf,
  title_tfidf,
  title_normalizer,
  title_lsh
  ])

# COMMAND ----------

# MAGIC %md With the title pipeline in place, I will fit the pipeline and perform a transformation just to verify it works as expected.  

# COMMAND ----------

# DBTITLE 1,Fit & Test Titles Pipeline
# retrieve titles
titles = (
  spark
    .table('reviews.metadata')
    .repartition(sc.defaultParallelism)
    .filter('title Is Not Null')
    .select('id', 'asin', 'title')
  )

# fit pipeline to titles
title_fitted_pipeline = title_pipeline.fit(titles)

# present transformed data for validation
display(
  title_fitted_pipeline.transform(titles.limit(1000))
  )

# COMMAND ----------

# MAGIC %md Now I will save the title pipeline by using the **mlflow registry**. mlflow registry is a great tool for MLOps where models can be testing and moved between testing to productionizing states. While demonstrating a full mlOps pipeline deployment is not in scope for this project, I wanted to try out the mlflow feature in Databricks to test out the feature and share a typical use case of handing off this pipeline to a client like the fashion & beauty brand. 

# COMMAND ----------

# DBTITLE 1,Persist Fitted Titles Pipeline
# persist pipeline
with mlflow.start_run():
  mlflow.spark.log_model(
    title_fitted_pipeline, 
    'pipeline',
    registered_model_name='title_fitted_pipeline'
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Now I create a pipeline for the description data.  I am using use the Word2Vec functionality.  The pipeline will consist of the following steps:</p>
# MAGIC 
# MAGIC 1. Flatten description from array to string
# MAGIC 2. Tokenize the description
# MAGIC 2. Apply Word2Vec transformation
# MAGIC 3. Normalize Word2Vec scores
# MAGIC 4. Apply Gaussian Mixture as clustering algorithm
# MAGIC 
# MAGIC **Note:** While I applied a different clustering technique in Part 1 of the project series, the Databricks' Pipeline had difficulty leveraging Kmeans clustering. As such, I leveraged Gaussian Mixture instead and that seems to work pretty well
# MAGIC 
# MAGIC I do not need to create custom transformers for this pipeline, but I will use the SQL transformer as a first step:

# COMMAND ----------

# DBTITLE 1,Assemble Descriptions Pipeline
# step 1: flatten the description to form a single string
descript_flattener = SQLTransformer(
    statement = '''
    SELECT id, array_join(description, ' ') as description 
    FROM __THIS__ 
    WHERE size(description)>0'''
    )

# step 2: split description into tokens
descript_tokenizer = RegexTokenizer(
    minTokenLength=2, 
    pattern='[^\w\d]', 
    inputCol='description', 
    outputCol='words'
    )

# step 3: convert tokens into concepts
descript_w2v =  Word2Vec(
    vectorSize=100,
    minCount=3,              
    maxSentenceLength=200,  
    numPartitions=sc.defaultParallelism*10,
    maxIter=20,
    inputCol='words',
    outputCol='w2v'
    )
  
# step 4: normalize concept scores
descript_normalizer = Normalizer(
    inputCol='w2v', 
    outputCol='features', 
    p=2.0
    )

# step 5: assign titles to buckets -- 
descript_cluster = GaussianMixture(
    k=10,
    featuresCol='features', 
  predictionCol='bucket',
    maxIter=20
    )


# assemble the pipeline
descript_pipeline = Pipeline(stages=[
    descript_flattener,
    descript_tokenizer,
    descript_w2v,
    descript_normalizer,
    descript_cluster
    ])

# COMMAND ----------

# DBTITLE 1,Fit & Test Descriptions Pipeline
# retrieve descriptions
descriptions = (
    spark
      .table('reviews.metadata')
      .select('id','description')
      .repartition(sc.defaultParallelism)
      .sample(False, 0.25)
)
# fit pipeline
descript_fitted_pipeline = descript_pipeline.fit(descriptions)

# verify transformation
display(
  descript_fitted_pipeline.transform(descriptions.limit(1000))
  )

# COMMAND ----------

# DBTITLE 1,Persist Fitted Descriptions Pipeline
# persist pipeline
with mlflow.start_run():
  mlflow.spark.log_model(
    descript_fitted_pipeline, 
    'pipeline',
    registered_model_name='descript_fitted_pipeline'
    )

# COMMAND ----------

# MAGIC %md And now I build the categories pipeline.  This pipeline will perform the following steps:</p>
# MAGIC 
# MAGIC 1. Alter category levels to include lineage
# MAGIC 2. Perform one-hot encoding of category levels
# MAGIC 3. Identify the root member
# MAGIC 
# MAGIC The first of these steps is handled through a custom transformer. The last of these steps is handled through a SQL transformer:

# COMMAND ----------

# DBTITLE 1,Define Custom Transformer
# define custom transformer to flatten the category names to include lineage
class CategoryFlattener(
  Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
  ):
 
  @keyword_only
  def __init__(self, inputCol=None, outputCol=None):
    super(CategoryFlattener, self).__init__()
    kwargs = self._input_kwargs
    self.setParams(**kwargs)
  
  @keyword_only
  def setParams(self, inputCol=None, outputCol=None):
    kwargs = self._input_kwargs
    return self._set(**kwargs)

  def setInputCol(self, value):
    return self._set(inputCol=value)
    
  def setOutputCol(self, value):
    return self._set(outputCol=value)
  
  def _transform(self, dataset):
    
    # copy-paste of previously defined UDF
    # =========================================
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
    # =========================================
    
    t = ArrayType(StringType())
    out_col = self.getOutputCol()
    in_col = dataset[self.getInputCol()]
    
    return dataset.withColumn(out_col, cleanse_category(in_col))

# COMMAND ----------

# DBTITLE 1,Assemble Categories Pipeline
# step 1: flatten hierarchy
category_flattener = CategoryFlattener(
    inputCol='category',
    outputCol='category_clean'
    )

# step 2: one-hot encode hierarchy values
category_onehot = HashingTF(
    inputCol='category_clean', 
    outputCol='category_onehot',
    binary=True
    )

# step 3: assign bucket
category_cluster = SQLTransformer(
    statement='SELECT id, category[0] as bucket, category, category_clean, category_onehot FROM __THIS__ WHERE size(category)>0'
    )

# assemble pipeline
category_pipeline = Pipeline(stages=[
    category_flattener,
    category_onehot,
    category_cluster
    ])

# COMMAND ----------

# DBTITLE 1,Fit & Test Categories Pipeline
# retrieve categories
categories = (
    spark
      .table('reviews.metadata')
      .select('id','category')
      .repartition(sc.defaultParallelism)
    )

# fit pipeline
category_fitted_pipeline = category_pipeline.fit(categories)

# test the transformation
display(
  category_fitted_pipeline.transform(categories.limit(1000))
  )

# COMMAND ----------

# DBTITLE 1,Persist Fitted Categories Pipeline
# persist pipeline
with mlflow.start_run():
  mlflow.spark.log_model(
    category_fitted_pipeline, 
    'pipeline',
    registered_model_name='category_fitted_pipeline'
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 2: Build Features for Products
# MAGIC 
# MAGIC With each pipelines defined, trained & persisted, I generate features to build recommendations.   I create a features tables with the new information when rolling out a new pipeline.  I then rerun pipelines as new products are added to the e-commerce website or when metadata tagged to a product changes.
# MAGIC 
# MAGIC Given the limitations on date attributes within the data, I will create a method instead for identifying changes on each feature set and capturing those updates.
# MAGIC 
# MAGIC I will first retrieve the pipeline objects from mlflow and this can be done easily by grabbing them from the mlflow registry. I grab the last version of each pipeline currently residing registry. To be a true partner with the fashion & beauty brand, I will recommend a more robust application of the mlflow registry, encouraging them to utilizing stages of moving these pipelines from test to production tags accordingly. But for the purposes of this project, I left off tags to productionize the pipelines.

# COMMAND ----------

# DBTITLE 1,Attempt to Retrieve Titles Pipeline
# retrieve pipeline from registry
retrieved_title_pipeline = mlflow.spark.load_model(
    model_uri='models:/title_fitted_pipeline/None'  
    )

# COMMAND ----------

# MAGIC %md To avoid a [*leaky pipeline* problem](https://rebeccabilbro.github.io/module-main-has-no-attribute/), I create a class definition to retrieve the pipeline object and force the top-level environment to recognize it. 
# MAGIC 
# MAGIC Databricks provides a more elegant solution but given the time constraints of completing this project at the semester deadline, I use the below class definition to force the environment to see the custom transformer in this notebook.

# COMMAND ----------

# DBTITLE 1,Force Environment to See Custom Transformer in This Notebook
# ensure top-level environment sees class definition (which must be included in this notebook)
m = __import__('__main__')
setattr(m, 'NLTKWordNetLemmatizer', NLTKWordNetLemmatizer)

# COMMAND ----------

# DBTITLE 1,Retrieve Titles Pipeline
# retrieve pipeline from registry
retrieved_title_pipeline = mlflow.spark.load_model(
    model_uri='models:/title_fitted_pipeline/None'
    )

# COMMAND ----------

# DBTITLE 1,Retrieve Descriptions Pipeline
# retrieve pipeline from registry
retrieved_descript_pipeline = mlflow.spark.load_model(
    model_uri='models:/descript_fitted_pipeline/None'
    )

# COMMAND ----------

# DBTITLE 1,Retrieve Categories Pipeline
# ensure top-level environment sees class definition (which must be included in this notebook)
m = __import__('__main__')
setattr(m, 'CategoryFlattener', CategoryFlattener)

# retrieve pipeline from registry
retrieved_category_pipeline = mlflow.spark.load_model(
    model_uri='models:/category_fitted_pipeline/None'
    )

# COMMAND ----------

# MAGIC %md Now, I can generate features for the products.  I create an empty dataframe with a schema as would be expected to be created by my pipelines. I will append this to the target table so that I can build a DeltaLake table with a schema I can perform a merge operation later.  I do this instead of other data definition techniques as the vector user-defined type is not recognized in such a statement.  Creating the DeltaLake table in this manner ensures that I have a target schema for the merge but that I do not modify any data if such a schema is already in place:

# COMMAND ----------

# DBTITLE 1,Create Title Features DeltaLake Object
# retreive an empty dataframe
titles_dummy = (
  spark
    .table('reviews.metadata')
    .filter('1==2')
    .select('id','title')
  )

# transform empty dataframe to get transformed structure
title_dummy_features = (
  retrieved_title_pipeline
    .transform(titles_dummy)
    .selectExpr('id','tfidf_norm as features','hash')
  )

# persist the empty dataframe to laydown the schema (if not already in place)
(
title_dummy_features
  .write
  .format('delta')
  .mode('append')
  .save('/mnt/reviews/gold/title_pipeline_features')
  )

# COMMAND ----------

# MAGIC %md Now I move the real data updates into the schema:

# COMMAND ----------

# DBTITLE 1,Transform Titles to Features
# retrieve titles to process for features
titles_to_process = (
  spark
    .table('reviews.metadata')
    .select('id','title')
  )

# generate features
title_features = (
  retrieved_title_pipeline
    .transform(titles_to_process)
    .selectExpr('id','tfidf_norm as features','hash')
  )

# COMMAND ----------

# DBTITLE 1,Merge Title Features
# access reviews table in prep for merge
target = DeltaTable.forPath(spark, '/mnt/reviews/gold/title_pipeline_features')

# perform merge on target table
( target.alias('target')
    .merge(
      title_features.alias('source'),
      condition='target.id=source.id'
      )
    .whenMatchedUpdate(set={'features':'source.features', 'hash':'source.hash'})
    .whenNotMatchedInsertAll()
  ).execute()

# display features data
display(
  spark.table('DELTA.`/mnt/reviews/gold/title_pipeline_features`')
  )

# COMMAND ----------

# MAGIC %md I now do the same process to the description & category features:

# COMMAND ----------

# DBTITLE 1,Generate Description Features
# retreive an empty dataframe
descript_dummy = (
  spark
    .table('reviews.metadata')
    .filter('1==2')
    .select('id','description')
  )

# transform empty dataframe to get transformed structure
descript_dummy_features = (
  retrieved_descript_pipeline
    .transform(descript_dummy)
    .selectExpr('id','w2v','features','bucket')
  )

# persist the empty dataframe to laydown the schema (if not already in place)
(
descript_dummy_features
  .write
  .format('delta')
  .mode('append')
  .save('/mnt/reviews/gold/descript_pipeline_features')
  )

# retrieve titles to process for features
descriptions_to_process = (
  spark
    .table('reviews.metadata')
    #.repartition(sc.defaultParallelism * 16)
    .select('id','description')
  )

# generate features
description_features = (
  retrieved_descript_pipeline
    .transform(descriptions_to_process)
    .selectExpr('id','w2v','features','bucket')
  )

# access reviews table in prep for merge
target = DeltaTable.forPath(spark, '/mnt/reviews/gold/descript_pipeline_features')

# perform merge on target table
( target.alias('target')
    .merge(
      description_features.alias('source'),
      condition='target.id=source.id'
      )
    .whenMatchedUpdate(set={'w2v':'source.w2v', 'features':'source.features', 'bucket':'source.bucket'})
    .whenNotMatchedInsertAll()
  ).execute()

# display features data
display(
  spark.table('DELTA.`/mnt/reviews/gold/descript_pipeline_features`')
  )

# COMMAND ----------

# DBTITLE 1,Examine Distribution by Bucket
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   bucket,
# MAGIC   count(*) as products
# MAGIC FROM DELTA.`/mnt/reviews/gold/descript_pipeline_features`
# MAGIC GROUP BY bucket
# MAGIC ORDER BY bucket

# COMMAND ----------

# DBTITLE 1,Generate Category Features
# retreive an empty dataframe
category_dummy = (
  spark
    .table('reviews.metadata')
    .filter('1==2')
    .select('id','category')
  )

# transform empty dataframe to get transformed structure
category_dummy_features = (
  retrieved_category_pipeline
    .transform(category_dummy)
    .selectExpr('id','category_onehot as features','bucket')
  )

# persist the empty dataframe to laydown the schema (if not already in place)
(
category_dummy_features
  .write
  .format('delta')
  .mode('append')
  .save('/mnt/reviews/gold/category_pipeline_features')
  )

# retrieve categories to process for features
category_to_process = (
  spark
    .table('reviews.metadata')
    #.repartition(sc.defaultParallelism * 16)
    .select('id','category')
  )

# generate features
category_features = (
  retrieved_category_pipeline
    .transform(category_to_process)
    .selectExpr('id','category_onehot as features','bucket')
  )

# access reviews table in prep for merge
target = DeltaTable.forPath(spark, '/mnt/reviews/gold/category_pipeline_features')

# perform merge on target table
( target.alias('target')
    .merge(
      category_features.alias('source'),
      condition='target.id=source.id'
      )
    .whenMatchedUpdate(set={'features':'source.features', 'bucket':'source.bucket'})
    .whenNotMatchedInsertAll()
  ).execute()

# display features data
display(
  spark.table('DELTA.`/mnt/reviews/gold/category_pipeline_features`')
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 3: Generate User-Profiles
# MAGIC 
# MAGIC To generate user-profiles, I won't assemble a pipeline because the transformatioins are much more easily implemented using a DataFrame. I use some kind of change detection to limit the number of profiles needed to generate on a given cycle.  

# COMMAND ----------

user_profiles_raw = (
  spark
    .sql('''
      SELECT
        a.reviewerID,
        a.overall as rating,
        b.w2v
      FROM reviews.reviews a
      INNER JOIN DELTA.`/mnt/reviews/gold/descript_pipeline_features` b
        ON a.product_id=b.id
      WHERE a.overall >= 4 AND a.product_id Is Not Null
    ''')
  .groupBy('reviewerID')
    .agg(
      Summarizer.mean(col('w2v'), weightCol=col('rating')).alias('w2v') 
      )
  )

# COMMAND ----------

# MAGIC %md Note the Normalizer transform has no dependency on training. I can create one as needed, instead of being concerned about persisting one for re-use:

# COMMAND ----------

# DBTITLE 1,Normalize Profile Scores
user_profiles_norm = (
  Normalizer(inputCol='w2v', outputCol='features', p=2.0)
    .transform(user_profiles_raw)
   )

# COMMAND ----------

# MAGIC %md Unlike the Normalizer, the clustering model used to assign profiles to buckets does depend on prior training. The model retrieved is associated with the description features accessed in the first step of our user-profile creation. I access that model directly from the pipeline, by-passing the transformations leading into it:

# COMMAND ----------

# DBTITLE 1,Assign Profiles to Bucket
# retrieve pipeline from registry
retrieved_descript_pipeline = mlflow.spark.load_model(
    model_uri='models:/descript_fitted_pipeline/None'
    )

# assign profiles to clusters/buckets
user_profiles = (
  retrieved_descript_pipeline.stages[-1].transform(user_profiles_norm)
  )

# COMMAND ----------

# MAGIC %md I save user-profiles and their bucket/cluster assignments:

# COMMAND ----------

# DBTITLE 1,Save Profiles 
# retreive an empty dataframe
user_profiles_dummy_features = (
  user_profiles
    .filter('1==2')
    .select('reviewerID','features','bucket')
  )

# transform empty dataframe to get transformed structure
(
user_profiles_dummy_features
  .write
  .format('delta')
  .mode('append')
  .save('/mnt/reviews/gold/user_profile_pipeline_features')
  )

# persist the empty dataframe to laydown the schema (if not already in place)
target = DeltaTable.forPath(spark, '/mnt/reviews/gold/user_profile_pipeline_features')

# perform merge on target table
( target.alias('target')
    .merge(
      user_profiles.alias('source'),
      condition='target.reviewerID=source.reviewerID'
      )
    .whenMatchedUpdate(set={'features':'source.features', 'bucket':'source.bucket'})
    .whenNotMatchedInsertAll()
  ).execute()

# display features data
display(
  spark.table('DELTA.`/mnt/reviews/gold/user_profile_pipeline_features`')
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 4: Create Product-Based Recommendations
# MAGIC 
# MAGIC I can now create a method to publish recommendations to the "application" layer to hand off to developers who develop the content on the e-commerce website. I am creating a fictitious process that keeps the features in place and updates them with jobs using the pipelines created. For the purposes of this project, I will not go into detail on performance improvements that could be made to the pipelines. Instead, I will focus on logic required to pre-generate recommendations and save them in a table in Databricks.

# COMMAND ----------

# MAGIC %md My first step here is determining for which products I will be making recommendations.  I limit the processing to 10,000 products in the *Clothing, Shoes & Jewelry* category.  In the real world, I might need to process more or fewer products in order to work our way efficiently through product backlogs or to keep up with changes in our product portfolio.
# MAGIC 
# MAGIC I look at the products for which the engine is making recommendations as **a-product**s and the products that make up potential recommendations as **b-product**s.  The b-products are limited here to products within the same high-level product category, *i.e.* *Clothing, Shoes & Jewelry*, which, again, will need further review for a full production-ready state for all categories for the fashion & beauty brands products:

# COMMAND ----------

## parallelize downstream work
#spark.conf.set('spark.sql.shuffle.partitions', sc.defaultParallelism * 10)

# products we ae making recommendations for
product_a =  (
  spark
    .table('DELTA.`/mnt/reviews/gold/category_pipeline_features`')
    .filter("bucket = 'Clothing, Shoes & Jewelry'")
    .limit(10000) # we would typically have some logic here to identify products that we need to generate recommendations for
    .selectExpr('id as id_a')
  ).cache()

# recommendation candidates to consider
product_b = (
  spark
    .table('DELTA.`/mnt/reviews/gold/category_pipeline_features`')
    .filter("bucket = 'Clothing, Shoes & Jewelry'")
    .selectExpr('id as id_b')
  ).cache()

# COMMAND ----------

# MAGIC %md I now calculate similarities using the content-derived features.  I am making this process as separate from the feature generation steps above, but I will leverage the already retrieved pipelines in the code below to avoid repeating that step.
# MAGIC 
# MAGIC Notice that I set limits on similarity scores and on distance between products (in the LSH lookup).  These limits are implemented to reduce the volume of data headed into the final recommendation scoring calculation:

# COMMAND ----------

# DBTITLE 1,Define Similarity Functions
# MAGIC %scala
# MAGIC 
# MAGIC import math._
# MAGIC import org.apache.spark.ml.linalg.{Vector, Vectors, SparseVector}
# MAGIC 
# MAGIC // function for jaccard similarity calc
# MAGIC // from https://github.com/karlhigley/spark-neighbors/blob/master/src/main/scala/com/github/karlhigley/spark/neighbors/linalg/DistanceMeasure.scala
# MAGIC val jaccard_similarity = udf { (v1: SparseVector, v2: SparseVector) =>
# MAGIC   val indices1 = v1.indices.toSet
# MAGIC   val indices2 = v2.indices.toSet
# MAGIC   val intersection = indices1.intersect(indices2)
# MAGIC   val union = indices1.union(indices2)
# MAGIC   intersection.size.toDouble / union.size.toDouble
# MAGIC }
# MAGIC spark.udf.register("jaccard_similarity", jaccard_similarity)
# MAGIC 
# MAGIC // function for euclidean distance derived similarity
# MAGIC val euclidean_similarity = udf { (v1: Vector, v2: Vector) =>
# MAGIC   val distance = sqrt(Vectors.sqdist(v1, v2))
# MAGIC   val rawSimilarity = 1 / (1+distance)
# MAGIC   val minScore = 1 / (1+sqrt(2))
# MAGIC   (rawSimilarity - minScore)/(1 - minScore)
# MAGIC }
# MAGIC spark.udf.register("euclidean_similarity", euclidean_similarity)

# COMMAND ----------

# DBTITLE 1,Calculate Category Similarities
# categories for products for which we wish to make recommendations
a = (
  spark
    .table('DELTA.`/mnt/reviews/gold/category_pipeline_features`')
    .join(product_a, on=[col('id')==product_a.id_a], how='left_semi')
    .withColumn('features_a', col('features'))
    .withColumn('id_a',col('id'))
    .drop('features','id')
    )

# categories for products which will be considered for recommendations
b = (
  spark
    .table('DELTA.`/mnt/reviews/gold/category_pipeline_features`')
    .join(product_b, on=[col('id')==product_b.id_b], how='left_semi')
    .withColumn('features_b', col('features'))
    .withColumn('id_b',col('id'))
    .drop('features','id')
    )

# similarity results
category_similarity = (
  a.crossJoin(b)
    .withColumn('similarity', expr('jaccard_similarity(features_a, features_b)'))
    .select('id_a', 'id_b', 'similarity')
    )

display(category_similarity)

# COMMAND ----------

# DBTITLE 1,Calculate Title Similarities
# categories for products for which we wish to make recommendations
a = (
  spark
    .table('DELTA.`/mnt/reviews/gold/title_pipeline_features`')
    .join(product_a, on=[col('id')==product_a.id_a], how='left_semi')
    .selectExpr('id as id_a','features as tfidf_norm')
  )

# categories for products which will be considered for recommendations
b = (
  spark
    .table('DELTA.`/mnt/reviews/gold/title_pipeline_features`')
    .join(product_b, on=[col('id')==product_b.id_b], how='left_semi')
    .selectExpr('id as id_b','features as tfidf_norm')
  )

# retrieve fitted LSH model from pipeline
fitted_lsh = retrieved_title_pipeline.stages[-1]

# similarity results
title_similarity = (
  fitted_lsh.approxSimilarityJoin(
      a,  
      b,
      threshold = 1.4, # this is pretty high so that we can be more inclusive.  Setting it at or above sqrt(2) will bring back every product
      distCol='distance'
      )
    .withColumn('raw_sim', expr('1/(1+distance)'))
    .withColumn('min_score', expr('1/(1+sqrt(2))'))
    .withColumn('similarity', expr('(raw_sim - min_score)/(1-min_score)'))
    .selectExpr('datasetA.id_a as id_a', 'datasetB.id_b as id_b', 'similarity')
    .filter('similarity > 0.01') # set a lower limit for consideration
    )

display(title_similarity)

# COMMAND ----------

# DBTITLE 1,Calculate Description Similarities
# categories for products for which we wish to make recommendations
a = (
 spark
   .table('DELTA.`/mnt/reviews/gold/descript_pipeline_features`')
   .join(product_a, on=[col('id')==product_a.id_a], how='left_semi')
   .selectExpr('id as id_a', 'features as features_a', 'bucket')
 )

# categories for products which will be considered for recommendations
b = (
 spark
   .table('DELTA.`/mnt/reviews/gold/descript_pipeline_features`')
   .join(product_b, on=[col('id')==product_b.id_b], how='left_semi')
   .selectExpr('id as id_b', 'features as features_b', 'bucket')
 )

# caculate similarities
description_similarity = (
 a  
   .hint('skew','bucket')
   .join(b, on=[a.bucket==b.bucket], how='inner')
   .withColumn('similarity', expr('euclidean_similarity(features_a, features_b)'))
   .select('id_a','id_b','similarity')
   .filter('similarity > 0.01') # set a lower limit for consideration
 )

display(description_similarity)

# COMMAND ----------

# MAGIC %md With similarity scores calculated leveraging title and category data, we can combine these scores to create a final score.  Buckets to limit the number of comparisons is being performed in the steps above; there are still a lot of products to potentially return here.  A simple function is used to limit products to a top N number of products:

# COMMAND ----------

# DBTITLE 1,Generate Recommendations
sqlContext.setConf("spark.sql.shuffle.partitions",4500)
# function to get top N product b's for a given product a
def get_top_products( data ):
  '''the incoming dataset is expected to have the following structure: id_a, id_b, score'''  
  
  rows_to_return = 11 # limit to top 10 products (+1 for self)
  
  return data.sort_values(by=['score'], ascending=False).iloc[0:rows_to_return] # might be faster ways to implement this sort/trunc

# combine similarity scores and get top N highest scoring products
recommendations = (
  category_similarity
    .join(title_similarity, on=['id_a', 'id_b'], how='inner')
    .withColumn('score', category_similarity.similarity * title_similarity.similarity)
    .select(category_similarity.id_a, category_similarity.id_b, 'score')  
    .groupBy('id_a')
      .applyInPandas(
        get_top_products, 
        schema='''
          id_a long,
          id_b long,
          score double
          ''')
    )

display(recommendations)

# COMMAND ----------

# DBTITLE 1,Clean Up Cached Datasets
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 4b: Create Profile-Based Recommendations
# MAGIC 
# MAGIC As with the content-only recommenders, there are numerous factors that go into deciding whether recommendations should be pre-computed and cached or calculated dynamically.  Instead of sorting out the strategies one might take with regards to that, I define the logic required to assemble the recommendations from the feature sets as they currently are recorded.
# MAGIC 
# MAGIC Here, we'll limit the generation of recommendations to a limited number of users and constrain our recommendations to the top 5 highest scoring items:

# COMMAND ----------

# DBTITLE 1,Define Similarity Function
# MAGIC %scala
# MAGIC 
# MAGIC import math._
# MAGIC import org.apache.spark.ml.linalg.{Vector, Vectors, SparseVector}
# MAGIC 
# MAGIC // function for euclidean distance derived similarity
# MAGIC val euclidean_similarity = udf { (v1: Vector, v2: Vector) =>
# MAGIC   val distance = sqrt(Vectors.sqdist(v1, v2))
# MAGIC   val rawSimilarity = 1 / (1+distance)
# MAGIC   val minScore = 1 / (1+sqrt(2))
# MAGIC   (rawSimilarity - minScore)/(1 - minScore)
# MAGIC }
# MAGIC spark.udf.register("euclidean_similarity", euclidean_similarity)

# COMMAND ----------

# DBTITLE 1,Retrieve Datasets for Recommendations
products = (
  spark
    .table('DELTA.`/mnt/reviews/gold/descript_pipeline_features`')
    .withColumnRenamed('features', 'features_b')
    .cache()
    )
products.count() # force caching to complete

user_profiles = (
  spark
    .table('DELTA.`/mnt/reviews/gold/user_profile_pipeline_features`')
    .withColumnRenamed('features', 'features_a')
    .sample(False, 0.0001)
    ).cache()

# see how profiles distributed between buckets
display(
  user_profiles
    .groupBy('bucket')
      .agg(count('*'))
  )

# COMMAND ----------

# DBTITLE 1,Generate Recommendations
sqlContext.setConf("spark.sql.autoBroadcastJoinThreshold",-1)

# make recommendations for sampled reviewers
recommendations = (
  products
    .hint('skew','bucket') # hint to ensure join is balanced
    .join( 
      user_profiles, 
      on='bucket', 
      how='inner'
      ) # join products to profiles on buckets
    .withColumn('score', expr('euclidean_similarity(features_a, features_b)')) # calculate similarity
    .withColumn('seq', expr('row_number() OVER(PARTITION BY reviewerID ORDER BY score DESC)'))
    .filter('seq <= 5')
    .selectExpr('reviewerID', 'id as product_id', 'seq')
    )

display(
  recommendations
  )

# COMMAND ----------

# DBTITLE 1,Clean Up Cached Datasets
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()

# COMMAND ----------

# MAGIC %md # Part 5: Visualize Recommendation Engine 

# COMMAND ----------

# MAGIC %md
# MAGIC Now I can demonstrate through the use of a Tableau Public Dashboard how the product and profile-based recommendation engines might be leveraged on an e-commerce platform as a way to articulate future benefit to the fashion & beauty brand. By exporting the output of both the profile & product recommendation engines, we can visualize how a user might see the recommendations from the engine when selecting a product or by changing profiles (this is called **Customer ID** in the Tableau Dashboard).

# COMMAND ----------

# MAGIC %scala
# MAGIC displayHTML(s"""<div class='tableauPlaceholder' id='viz1619797304302' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;ProductRecommendationEngineVisual&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='ProductRecommendationEngineVisual&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pr&#47;ProductRecommendationEngineVisual&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1619797304302');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='900px';vizElement.style.height='927px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 6: Outcomes & Conclusions of Study

# COMMAND ----------

# MAGIC %md
# MAGIC As demonstrated in this study, profitable opportunities and insights can arise when successfully deploying and maintaining a recommendation engine on a brand's e-commerce website. From content-base profile identification to a more personalized shopping experience for customers actively accessing the site, the fashion & beauty brand can benefit immensely from conclusions of this study.  To continue operationalizing this recommender system and to obtain full benefits of this solution, I suggest the following nexts step to the fashion & beauty brand:
# MAGIC 
# MAGIC 1. **Review & Condense Succinct Product Categories**: As with most e-commerce platforms, it is critical maintain hierarchical groupings/categories of products so that your organization can perform more advanced analysis on the best products being sold on the e-commerce website as well as a better enablement of organizing products for the consumer to access directly on the site. As a future study, I recommend a more thorough examination of product categories and cleaning up the category tags assigned to the products, thereby partnering with the fashion & beauty brand to build categories for which is important to their business.
# MAGIC 
# MAGIC 2. **Master Data Management**: In order to continue to maintain order and a clear, effective pipeline for the recommendation engine to operate successfully, I recommend a SaaS master data management solution to help group similar products into categories more efficiently, as well as maintain an active repository of product ids to avoid duplication of products as new products are added to their inventory.  Many cloud-based technology platforms, like Azure and AWS, offer such platforms as a service (PaaS) that can easily integrate with the Databrick's pipelines and datasets as maintained.
# MAGIC 
# MAGIC 3. **Enhanced Profile Recommendation with User Trend Modeling**: In this study, we looked at the total inventory of reviewers as our basis for creating our profile-based recommendation engine. As a further ehancement, I recommend that the fashion & beauty brand store time-based interactions of their customers on the site so that they can see how their customers tastes & interests change overtime. This will then provide a more accurate depiction of their customers interests. Bagher, R. C., Hassanpour, H., & Mashayekhi, H. explore this is their study and had favorable results versus a non-trend profile based model. [10]
# MAGIC 
# MAGIC As with any solution, this recommendation engine will continue to improve overtime as more data is available on the profiles as well as product classifications. 
# MAGIC 
# MAGIC Thank you for taking time to read my project!

# COMMAND ----------

# MAGIC %md # Appendix

# COMMAND ----------

# MAGIC %md
# MAGIC <pre>
# MAGIC Contributors : <a href=https://github.com/Lwhieldon>Lee Whieldon</a>
# MAGIC </pre>
# MAGIC 
# MAGIC <pre>
# MAGIC Languages    : Python
# MAGIC Tools/IDE    : Databricks, 8.1 (includes Apache Spark 3.1.1, Scala 2.12)
# MAGIC Compute      : AWS i3.xlarge (Driver & Workers, 30.5 GB Memory, 4 Cores, 1 DBU); 1 driver, 8 workers      
# MAGIC Libraries    : pyspark, delta, pandas, gzip, shutil, os, requests, html, nltk, typing, mlflow
# MAGIC </pre>
# MAGIC 
# MAGIC <pre>
# MAGIC Assignment Submitted : May 2021
# MAGIC </pre>

# COMMAND ----------

# MAGIC %md
# MAGIC [1] Jianmo Ni, Jiacheng Li, Julian McAuley. Empirical Methods in Natural Language Processing (EMNLP), 2019. http://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf
# MAGIC 
# MAGIC [2] Robertson, Stephen. "Understanding inverse document frequency: on theoretical arguments for IDF." Journal of documentation (2004).
# MAGIC 
# MAGIC [3] https://en.wikipedia.org/wiki/Stemming
# MAGIC 
# MAGIC [4] https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
# MAGIC 
# MAGIC [5] https://buildmedia.readthedocs.org/media/pdf/nltk/latest/nltk.pdf
# MAGIC 
# MAGIC [6] https://www.nltk.org/howto/wordnet.html
# MAGIC 
# MAGIC [7] https://spark.apache.org/docs/latest/ml-features#lsh-algorithms
# MAGIC 
# MAGIC [8] D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent dirichlet allocation,” J. Mach. Learn. Res., vol. 3, pp. 993–1022, 2003.
# MAGIC 
# MAGIC [9] T. Slimani, “Description and evaluation of semantic similarity measures approaches,” ArXiv Prepr. ArXiv13108059, 2013, Available: https://arxiv.org/ftp/arxiv/papers/1310/1310.8059.pdf
# MAGIC 
# MAGIC [10] Bagher, R. C., Hassanpour, H., & Mashayekhi, H. (2017). User trends modeling for a content-based recommender system. Expert Systems with Applications, 87, 209–219. https://doi.org/10.1016/j.eswa.2017.06.020
# MAGIC 
# MAGIC [11] Li, L., Wang, D., Li, T., Knox, D., & Padmanabhan, B. (2011). Scene: A scalable two-stage personalized news recommendation system. Proceedings of the 34th International ACM SIGIR Conference on Research and Development in Information Retrieval, 125–134
