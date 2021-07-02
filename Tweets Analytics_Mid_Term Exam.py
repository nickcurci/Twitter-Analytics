# Databricks notebook source
# MAGIC %md # Mid-term Exam (100 points)
# MAGIC 
# MAGIC This exam will use fitbit tweets I have collected between April 2019 and October 2019 excpet August 2019.
# MAGIC 
# MAGIC Please answer the following five questions using pyspark. **Please try to visualize your results whenever possible. After you are done, export your file as HTML, save it as a zip file and upload it on Blackboard.**
# MAGIC 
# MAGIC 1. What is the number of retweets by week day? Which day has the most retweets?
# MAGIC 2. For all tweets in Japanese (ja), what are the top 5 locations of the users excluding null and Japan.
# MAGIC 3. Find the total number of tweets where the text contains either charge 2 or charge 3 (please check page 96 of the your text for references)
# MAGIC 4. What is the percentage of verified users?
# MAGIC 5. What is average rating of tweets between verified users and non-verified users?

# COMMAND ----------

# MAGIC %md Read the fitbit tweets, select fields to do furhter analysis. Please examine the fields carefully as I have imported additional fields for this exam.
# MAGIC 
# MAGIC **To speed up the processing time, we will only import the tweets in October 2019.**

# COMMAND ----------

# read tweets into a DataFrame

tweets=spark.read.json('/FileStore/tables/tweets/fitbit/2019_10.json')

# Select the fields that are of interest to do further analysis.
from pyspark.sql.functions import col, to_timestamp
tweets_selected=tweets.select(col('created_at').alias('date'), 'lang', 'source', col('id_str').alias('tweet_id'), col('user.screen_name').alias('user_name'), col('user.lang').alias('user_lang'), col('user.location').alias('user_location'), col('user.verified').alias('user_verified'), col('user.followers_count').alias('user_followers'), col('user.friends_count').alias('user_friends'), col('user.created_at').alias('user_joinDate'),col('retweeted_status.user.screen_name').alias('retweet_user'),
col('retweeted_status.reply_count').alias('reply_count'),
col('retweeted_status.retweet_count').alias('retweet_count'),
col('retweeted_status.favorite_count').alias('favorite_count'), 
col('retweeted_status.text').alias('retweet_text'),'text')


# convert tweet date, user joining date from string to timestamp
tweets_selected=tweets_selected.withColumn('date', to_timestamp('date','E MMM dd HH:mm:ss +0000 yyyy'))
tweets_selected=tweets_selected.withColumn('user_joinDate', to_timestamp('user_joinDate','E MMM dd HH:mm:ss +0000 yyyy'))

# create a view to use spark SQL
tweets_selected.createOrReplaceTempView('tweetsT')

# put the data frame into cache

tweets_selected.cache()

# COMMAND ----------

# MAGIC %md Below code will create a rating dataframe storing average rating for each tweet you may need for question 5.

# COMMAND ----------

from pyspark.sql.functions import col, explode, split, instr, avg, isnull, when

#load afinn for sentiment analysis
afinn = spark.read.option("inferSchema", "true").option("header", "true").option("delimiter", '\t')\
 .csv("/FileStore/tables/utilities/afinn.txt")


tweets_withID=tweets_selected.filter(col('lang')=='en').select('tweet_id', 'text', explode(split('text', " ")).alias('word'))

rating=tweets_withID.join(afinn, tweets_withID['word']==afinn['Word'], 'inner').groupBy('tweet_id', 'text').agg(avg('rating').alias('rating'))

rating.printSchema()

# COMMAND ----------

# MAGIC %md 1. what is the number of **RETWEETS** by week day? Which day has the most retweets?

# COMMAND ----------

from pyspark.sql.functions import *
display(tweets_selected.groupBy(date_format('date', 'E').alias('Weekday')).agg(sum('retweet_count').alias("Total Retweets")).orderBy('Weekday', ascending = True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Thursday has the most retweets

# COMMAND ----------

# MAGIC %md 2.For all tweets in Japanese (ja), what are the top 5 locations of the users excluding null and Japan

# COMMAND ----------

from pyspark.sql.functions import *
display(tweets_selected.select('user_location').filter(col('lang') == 'ja').groupBy('user_location').count().alias('number of tweets').orderBy('count', ascending = False).na.drop().limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## London is the top location with Japanese tweets

# COMMAND ----------

# MAGIC %md 3.Find the total number of tweets where the text contains either charge 2 or charge 3 (please check page 96 of the your text for references)

# COMMAND ----------

from pyspark.sql.functions import *
tweets_selected.select('tweet_id').filter(instr(lower(col('text')), 'charge 2' or 'charge 3' )>=1).count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## There are 312 tweets about the Charge 2 or Charge 3 device

# COMMAND ----------

# MAGIC %md 4.What is the percentage of verified users?

# COMMAND ----------

from pyspark.sql.functions import *
display(tweets_selected.select('user_verified').alias('Verified').groupBy('user_verified').count().orderBy('count', ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 92% of users are NOT verified

# COMMAND ----------

# MAGIC %md 5.What is average rating of tweets between verified users and non-verified users?

# COMMAND ----------

inner_join = tweets_selected.join(rating, tweets_selected.tweet_id == rating.tweet_id)
display(inner_join)

# COMMAND ----------

from pyspark.sql.functions import *
display(inner_join.groupBy('user_verified').agg(avg(rating['rating']).alias('Rating')))

# COMMAND ----------

# MAGIC %md
# MAGIC ## There is not a strong difference in sentiment between verified and non verified users

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Lets add in a rating category

# COMMAND ----------

addedratings = inner_join.withColumn('sentiment', when(col('rating')>=1, 'positive').when(col('rating')<=-1, 'Negative').otherwise('neutral')).orderBy('rating', ascending = True)

# COMMAND ----------

display(addedratings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lets count the number of tweets by location and category

# COMMAND ----------

display(addedratings.select('user_location', 'sentiment').groupBy('user_location', 'sentiment').count().alias('number of tweets').orderBy('count', ascending = False).na.drop().limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## London has the most tweets by location and the data shows us that most of the tweets coming from London are positive
