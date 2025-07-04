# etl_jobs.py
from pyspark.sql import SparkSession
from dq_email_decorators import expect_row_count, assert_non_null

spark = SparkSession.builder.appName("quality_with_email").getOrCreate()

@expect_row_count(10)
@assert_non_null(["c1", "c2", "c3", "c4", "c5"])
def load_source_table():
    return (
        spark.read.option("header", True)
             .csv("s3://bucket/raw/my_data.csv")
    )

@expect_row_count(lambda n: n > 1_000_000)
@assert_non_null(["user_id"])
def transform_events():
    return (
        spark.read.parquet("s3://bucket/raw/events/")
             .filter("event_type = 'click'")
    )

df = transform_events()   # job continues; e-mail is sent only if a rule breaks