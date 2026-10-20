# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "ea4fef77-4858-4c38-b041-daf2f384c708",
# META       "default_lakehouse_name": "lh_divvy_bench",
# META       "default_lakehouse_workspace_id": "0cf64660-62b7-4220-a83a-4d4a1cf90511",
# META       "known_lakehouses": [
# META         {
# META           "id": "ea4fef77-4858-4c38-b041-daf2f384c708"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

import logging, sys
from datetime import datetime, timezone

LOG_RECORDS = []  # in-memory buffer; saved to a Delta table in step 7

class ListHandler(logging.Handler):
    """Captures every log record as a dict so we can persist it later."""
    def emit(self, record):
        LOG_RECORDS.append({
            "ts_utc":    datetime.fromtimestamp(record.created, tz=timezone.utc),
            "level":     record.levelname,
            "function":  record.funcName,
            "message":   record.getMessage(),
            "exception": logging.Formatter().formatException(record.exc_info) if record.exc_info else None,
        })

logger = logging.getLogger("nb_error_handling_logging")
logger.setLevel(logging.INFO)
logger.propagate = False      # don't duplicate into Spark's root logger
logger.handlers.clear()       # safe to re-run the cell

console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(funcName)s | %(message)s"))
logger.addHandler(console)
logger.addHandler(ListHandler())

logger.info("Logger initialised")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import time
from pyspark.errors import AnalysisException

# Fabric sample data: Azure Open Datasets – Public Holidays (anonymous access)
spark.conf.set("fs.azure.sas.holidaydatacontainer.azureopendatastorage.blob.core.windows.net", "")
HOLIDAYS_PATH = "wasbs://holidaydatacontainer@azureopendatastorage.blob.core.windows.net/Processed"

def load_parquet(path):
    start = time.time()
    logger.info(f"Reading {path}")
    try:
        df = spark.read.parquet(path)
        rows = df.count()
    except AnalysisException:
        # Expected, recoverable failure (bad path / schema) -> log with traceback, return None
        logger.exception("Read failed: AnalysisException")
        return None
    except Exception:
        # Anything unexpected -> log it, then re-raise so the run fails loudly
        logger.exception("Unexpected failure")
        raise
    else:
        # Runs only if the try block raised nothing
        logger.info(f"Loaded {rows:,} rows")
        return df
    finally:
        # Always runs: success, handled error, or re-raised error
        logger.info(f"Finished in {time.time() - start:.1f}s")

df_holidays = load_parquet(HOLIDAYS_PATH)
display(df_holidays.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import time
from pyspark.errors import AnalysisException

# Fabric sample data: Azure Open Datasets – Public Holidays (anonymous access)
spark.conf.set("fs.azure.sas.holidaydatacontainer.azureopendatastorage.blob.core.windows.net", "")
HOLIDAYS_PATH = "wasbs://holidaydatacontainer@azureopendatastorage.blob.core.windows.net/Processed"

def load_parquet(path):
    start = time.time()
    logger.info(f"Reading {path}")
    try:
        df = spark.read.parquet(path)
        rows = df.count()
    except AnalysisException:
        # Expected, recoverable failure (bad path / schema) -> log with traceback, return None
        logger.exception("Read failed: AnalysisException")
        return None
    except Exception:
        # Anything unexpected -> log it, then re-raise so the run fails loudly
        logger.exception("Unexpected failure")
        raise
    else:
        # Runs only if the try block raised nothing
        logger.info(f"Loaded {rows:,} rows")
        return df
    finally:
        # Always runs: success, handled error, or re-raised error
        logger.info(f"Finished in {time.time() - start:.1f}s")

df_holidays = load_parquet(HOLIDAYS_PATH)
display(df_holidays.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_bad = load_parquet(HOLIDAYS_PATH + "_does_not_exist")

print("Returned:", df_bad)
print("Log records captured so far:", len(LOG_RECORDS))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

class DataQualityError(Exception):
    """Raised when a dataset breaks a business rule (not a technical failure)."""

def check_not_null(df, column, max_null_pct=0.05):
    total = df.count()
    nulls = df.filter(df[column].isNull()).count()
    pct = nulls / total
    if pct > max_null_pct:
        raise DataQualityError(f"{column}: {pct:.1%} nulls (limit {max_null_pct:.0%})")
    return pct

for col in ["countryOrRegion", "isPaidTimeOff"]:
    try:
        pct = check_not_null(df_holidays, col)
    except DataQualityError as e:
        logger.warning(f"DQ check failed -> {e}")
    else:
        logger.info(f"DQ check passed -> {col}: {pct:.1%} nulls")

print("Log records captured so far:", len(LOG_RECORDS))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, TimestampType, StringType

LOG_SCHEMA = StructType([
    StructField("ts_utc",    TimestampType()),
    StructField("level",     StringType()),
    StructField("function",  StringType()),
    StructField("message",   StringType()),
    StructField("exception", StringType()),
])

def flush_logs(table="notebook_run_log"):
    if not LOG_RECORDS:
        return
    try:
        (spark.createDataFrame(LOG_RECORDS, schema=LOG_SCHEMA)
              .withColumn("notebook", F.lit("nb_error_handling_logging"))
              .write.format("delta").mode("append").saveAsTable(table))
    except Exception:
        # Logging must never crash the pipeline -> log the failure, keep the buffer
        logger.exception("Failed to persist logs")
    else:
        print(f"Persisted {len(LOG_RECORDS)} records to {table}")
        LOG_RECORDS.clear()

flush_logs()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# MAGIC %%sql
# MAGIC SELECT ts_utc,
# MAGIC        level,
# MAGIC        function,
# MAGIC        message,
# MAGIC        exception IS NOT NULL AS has_traceback
# MAGIC FROM   notebook_run_log
# MAGIC WHERE  level IN ('ERROR', 'WARNING')
# MAGIC ORDER  BY ts_utc

# METADATA ********************

# META {
# META   "language": "sparksql",
# META   "language_group": "synapse_pyspark"
# META }
