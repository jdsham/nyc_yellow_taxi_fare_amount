# Project Overview

![image](./docs/supporting_materials/taxi_cab.jpg)

This project is about building a predictive model to predict the fare amount for a yellow taxicab ride in NYC.
By simply specifying the trip distance and estimated trip duration, one can use the model to predict the fare amount for the tide.

Note: This project is primarily meant for learning purposes.

## Introduction
If you have used Uber or Lyft before, you know that the app provides an estimated pickup and drop-off time and a predetermined fare amount. 
The rise of ride-hailing apps brought new competition to the taxi-cab industry.
If you are like me, you use Uber or Lyft without thinking about comparing a taxicab ride, mainly due to the ease of use that ride-hailing apps provide.
Often, people will simply compare Uber and Lyft prices to determine which app they should use for their ride.

In New York City there are 13,587 taxi medallions issued by the City -- a place where taxi cabs are frequently used.
There has been news over the past several years regarding how Uber and Lyft have impacted the taxi medallion market in NYC, which means people have been using Uber and Lyft enough to create competition within NYC's taxi industry.
What if an app could be developed that lets a user input their travel details to get an estimated price for calling a taxicab?
Such an app could help consumers make better informed decisions about taking Uber, Lyft, or a taxicab to travel somewhere by car.
So, let us build a model to predict the taxicab fare amount!

## Problem Statement
People in New York City want to compare how much a ride will cost when examining Uber, Lyft, and hailing a yellow taxicab. A predictive model must be developed to provide an estimated fare amount, the cost before taxes, surcharges, tolls, other fees, and tips are added to the total cost. The users should input the most basic information possible to reduce the effort barrier to use the model. Only fares charged under the standard rate will be considered.

## Success Criteria
The model must be able to predict on average the fare amount within $1 of the actual fare amount.
Other metrics can be used to evaluate the predictive model performance, but the dollar amount must be met.

## Use Case
To generate predictions, the user must provide some basic information to the predictive model.
This would be information like distance from pickup to drop-off and the estimated trip duration.
A straightforward way to get this information would be to use Google Maps or a similar navigation product to provide this information.
It is not too far off a stretch to think that someone would do this, considering they would be using a phone to use Uber or Lyft.
By keeping the amount of information limited and easy to obtain, this can help frugal consumers use an application that might save several dollars.

## The Data
We need data to build the model and luckily NYC publishes their Yellow Taxicab fare data publicly online.
The data is published on [this website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) in parquet format.
Data is partitioned into files by month and goes back as far as 2009.
For this project, data from 2021 through 2024 will be used.
A data dictionary in PDF format to accompany the data is available [here](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf).

## Scope of the Project
This project will start with obtaining the published data and work all the way through to building a predictive model (single or ensemble of multiple acceptable models).

Some of the steps are:
1. Obtain the data
2. Establish domain knowledge
3. Data engineering to ingest, clean, and transform data to streamline analytics
4. Exploratory data analysis
5. Defining data preparation
6. Develop initial set of predictive models
7. Identify promising models and optimize hyperparameters
8. Consider an ensemble of the best models and compare against individual models.

There will be a project write-up to discuss findings through each of the steps and a conclusion.
Additionally, code will be published along with the write-up.

## Technical Information
There will be a variety of technical tools used in this project.
The main language of choice will be Python, with Bash and SQL being used if necessary.
Some code will be purposefully leveraging PySpark's API even though the same action could be performed in SQL.

Software Packages:
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Sci-kit Learn
* LightGBM
* XGboost
* Catboost
* MLFlow
* Optuna
* Jupyter
* PySpark

A list of package versions will be provided.

Platforms:
* Azure Data Lake Storage Gen2
* Azure Synapse / PySpark
* Azure Machine Learning
* Local computing

