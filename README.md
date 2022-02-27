## Inspiration
The financial market has been unpredictable, especially so in recent years. We want to help players in the market make informed decision and we want this help to be made accessible such that people of all demographics will be able to make better financial decisions.

## What it does
We have create a data-driven all-in-one web app that suggests similar stocks to any inputted stocks as well as advises stock decisions based on stock data and analyst recommendations. This serves as a comprehensive platform for financial advice that disseminates consolidated information that even the layman will understand.

## How we built it
For the clustering of stocks algorithm, we used the K-means package in Python and passed in the top 100 stocks (S&P 100) which clustered the stocks based on continuous variables that we got through scrapping Yahoo Finance (etc. net revenue, market cap, debt). 

We used web scrapping of finance sites as well as the pandas and matplotlib packages on python to plot graphs to show the information of the stocks. One of the graphs would be the daily High price of the stock and volume plotted against time.

## Challenges we ran into
Initially, we wanted to include the S&P 500 stocks into the ML clustering algorithm, however the data took too much time to download and hence we decided to reduce our scope to the S&P 100. 

## Accomplishments that we're proud of
As we are all pre-university students with little to no experience in coding, we are proud to have came up with a python web app in such a short period of time. Additionally, this is an issue we all have personally experienced (jumping into buying stocks with little to no education on stocks), we hope our hack would be able to solve this problem.

## What we learned
We have learnt basic scrapping of online websites through the use of APIs as well as how to use basic ML packages such as scikit and K-means. We have also experimented with both the Matplotlib and Seaborn packages to build interesting visualizations with data.


## What's next for Drama Club
1. The scope of the clustering can be increased to the S&P 500
2. Include a sentiment analysis tracker through the use of online scrappers of popular social media sites such as Reddit
3. Branch out into different financial instruments such as bonds and crypto to cater to a larger audience
