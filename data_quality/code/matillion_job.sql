-- Matillion ETL Job SQL
-- Example: Load raw data into staging tables
COPY INTO STAGING_TABLE_SHIPMENTS
FROM S3 's3://your-bucket/raw/shipments'
CREDENTIALS (ACCESS_KEY_ID 'your-access-key-id', SECRET_ACCESS_KEY 'your-secret-access-key')
DELIMITER ','
IGNOREHEADER 1;

-- Transform data using either matillion or dbt transformations
-- Example: Clean, join, and aggregate data
