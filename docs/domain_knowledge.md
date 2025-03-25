# Domain Knowledge for Taxicab Fare Amounts

A critical component of data science is understanding and employing domain knowledge.
Developing a successful predictive model is not a computational inevitability; you must know something about the domain you are working on to increase the chance of success.

Taxicab ride data contain multiple columns. The data dictionary can be found [here](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf).

## Some Basic Information About Fares
Taxicab rides in NYC have the following data reported:
1. The pickup and drop-off timestamp
2. The pickup and drop-off location ID. The location ID is a localized area within a borough.
3. Passenger count
4. Trip distance
5. Pricing information, including fare amount, taxes, fees, tolls, tips.
6. Rate Code ID, which specifies the fare rate that determines the fare amount
7. The vendor ID that supplied the trip record to the city.

Additionally, there is a Zone ID lookup CSV file that contains metadata regarding pickup and location IDs.
Location IDs are localized areas within a borough and there are 263 distinct location IDs across all of NYC's boroughs.
While there are 5 boroughs in NYC:
* Manhattan
* Brooklyn
* Queens
* The Bronx
* Staten Island

The Zone ID lookup files contain 6 boroughs, with the same 5 listed above and then 'EWR' for Newark Airport only.
The Zone ID information must be joined onto the taxicab data to include relevant information.

## What is the Fare Amount?
The fare amount is the base time and distance fare calculated by the taxi meter.
This is the base price the customers must pay before taxes, fees, tolls, and tips are accounted for to make up the total amount.

## What Information Matters for Fare Amount?
Information about how the fare amount is calculated is publicly available online [here](https://www.nyc.gov/site/tlc/passengers/taxi-fare.page#:~:text=Taxi%20Fare%20*%20$3.00%20initial%20charge.%20*,Westchester%2C%20Rockland%2C%20Dutchess%2C%20Orange%20or%20Putnam%20Counties.)

Specifically, details pertaining to fare amount calculations for standard metered fares are as follow:
* $3.00 initial charge.
* Plus 70 cents per 1/5 mile when traveling above 12mph or per 60 seconds in slow traffic or when the vehicle is stopped.
* Plus 50 cents MTA State Surcharge for all trips that end in New York City or Nassau, Suffolk, Westchester, Rockland, Dutchess, Orange or Putnam Counties.
* Plus $1.00 Improvement Surcharge.
* Plus $1.00 overnight surcharge 8pm to 6am.
* Plus $2.50 rush hour surcharge from 4pm to 8pm on weekdays, excluding holidays.
* Plus New York State Congestion Surcharge of $2.50 (Yellow Taxi) or $2.75 (Green Taxi) or 75 cents (any shared ride) for all trips that begin, end or pass through Manhattan south of 96th Street.
* Plus MTA Congestion Pricing toll of 75 cents for Yellow and Green Taxi for the area of Manhattan south of and including 60th Street, excluding the FDR Drive, West Side Highway/Route 9A, and the Hugh L. Carey Tunnel connections to West Street. For more information on the MTA’s Congestion Pricing toll, please visit https://new.mta.info/tolls/congestion-relief-zone/taxi-fhv-tolls
* Plus tips and any tolls.
* There is no charge for extra passengers, luggage or bags, or paying by credit card.
* The on-screen rate message should read: "Rate #01 – Standard City Rate."

What really matters for the fare amount is the distance and trip duration.
Even then, only a proportion of the distance and duration are computed for the fare amount.
Other columns do not directly impact the base fare amount.

## Why Not Include Taxes or Fees for the Predicted Value Returned To the User?
Taxes and additional fees can be added to the predicted fare amount.
Some fees and taxes are universally applied, while others depend upon the time of day and location of pickup and drop-off.
Other fees, such as tolls, also depend upon the route taken.
To provide an estimated total fare amount (minus tips) would require some additional work and potentially additional information provided by the user.
Note that Uber and Lyft users must also pay the same fees.