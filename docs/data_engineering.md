# Data Engineering

## Architecture
This project will follow the medallion architecture for incremental ETL outlined by [Databricks](https://www.databricks.com/glossary/medallion-architecture).
A quick summary of the medallion architecture:
* There are 3 data layers: Bronze, Silver, and Gold
* Bronze layer contains the raw, as-is data
* Silver layer contains data that has been filtered, transformed, and made available for any projects, use cases, etc. It's the main layer for people to use when wanting to work with data.
* Gold layer contains data that's specific to a project. This includes project specific filtering and transformations.
* As new data is received, an incremental ETL process can be developed to make efficient data loading from Bronze to Silver to Gold layers.

PySpark will be used to load all the parquet files to establish the dataset by combining all the data from 2021 through 2024 for taxicab rides.
In practice, it's common to use SQL to manage tabular data models and implement the medallion architecture.
However, for the sake of this project, data will be maintained in parquet format due to the file sizes and portability of parquet files.
For datasets ranging in the hundreds of GBs and above, a data warehouse / data lake would be the optimal solution to process, manage, and store data.

## Bronze to Silver Layer
When going from the Bronze layer to the Silver layer there will be the following steps:
1. Compute the trip duration and add it as a column.
2. Filtering the data
3. Join in Zone ID information to the data. The Zone ID contains the location ID and the borough where the pickup and drop-off occurred.
4. Perform additional data transformations, such as decomposing timestamps into Day of the Week, pickup hour, etc.

## Data Filtering Rules for Bronze to Silver Layers
The following rules are to be implemented to filter and clean data when going from Bronze layer to Silver layer.
These rules are not meant to be specific for the project, but generally applicable to any project that uses this data.
The goal is to synthesize a dataset that contains valid fares only.
Project specific rules will be developed when transforming data from the Silver layer to the Gold layer.
These rules will be disclosed later.

To determine rules for filtering data, the [data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) for taxicab rides will be used to understand the data.
Some rules can be developed by simply consulting the data dictionary.
Other rules will be derived by doing some basic data analysis on the raw data to identify reasonable filtering rules.
Some of the data analysis-based rules will be set based on judgement alone, as I lack specific domain knowledge of the taxi industry.
The Bronze Data Exploration notebook contains the results of data analysis using PySpark and Azure Synapse.
Please consult that document to see the results of the analysis.

### Vendor Code:
* Must be 1 or 2

### Pickup Time
* Timestamp must be within the time range (2021-01-01 to 2024-12-31)
* Cannot be null

### Dropoff Time
* Timestamp must be within the time range (2021-01-01 to 2024-12-31)
* Cannot be null

### Passenger Count:
* Cannot be null
* Must be > 0 and <= 6

### Trip Distance:
* Cannot be null
* Must be > 0
* Must be <= 30

### PULocationID & DOLocationID:
* Must be between 1 and 263 as integers

### RateCodeID:
* Must be 1 through 6 as integers

### Store and Fwd Flag
* Column will be dropped

### Payment Type:
* Must be 1 or 2.
* 3,4,5,6 correspond to No charge, dispute, unknown, and voided trip, respectively. These will not be considered valid trips.

### Fare Amount:
* Cannot be null
* Must be > 0
* Must be <= 70

### Extra:
* Can be null. Null values will be imputed as $0.00
* Must be >= $0.00
* Must be <= $15.00

### MTA Tax:
* Can be null. Null values will be imputed as $0.00
* Must be $0.00 or $0.50

### Improvement Surcharge:
* Can be null. Null values will be imputed as $0.00
* Must be $0.00, $0.30, or $1.00

### Tip Amount:
* Can be null. Null values will be imputed to $0.00
* Must be >= $0.00

### Tolls Amount:
* Must be > 0
* Must be <= 30

### Total Amount:
* Will be excluded and simply recalculated from other cost fields.

### Congestion Surcharge:
* Can be null. Null values will be imputed as $0.00
* Must be either $0.00 or $2.50

### Airport Fee:
* Must be $0.00, $1.25, or $1.75
* Null values will be filled as $0.00

### Trip Duration Minutes:
* Cannot be null
* Must be > 0
* Must be <= 90

## Silver Layer to Gold Layer
To develop the dataset used to predict fare amount, the following rules will be applied based on domain knowledge and the specifications of the problem statement.
A 3 month block of data will be sliced off from the gold layer dataset.
This 3 month block of data will be used for EDA and initial model training purposes.
If more data is needed, then these activities will be performed using PySpark.

### Fare Amount
* Must be >= 3.00

### RatecodeID
* Must be = 1

### Columns to Keep
Most columns can be dropped due to domain knowledge specifying trip distance and duration are what matter to calculating the fare amount.
An initial approach would be to keep the columns minimal and see how predictive models perform.
More data can be included in a later iteration if desired, but in general less columns lead to smaller file sizes / data sizes which is almost always a good thing.

The columns to keep are:
* trip duration (mins)
* trip distance
* The year and month the pickup occurred

Other columns to consider including if desired:
* congestion surcharge
* tolls amount
* Extra amounts for rush hour or overnight fares