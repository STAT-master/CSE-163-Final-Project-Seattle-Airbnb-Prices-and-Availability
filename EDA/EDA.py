"""
Arthor: Ruifan Ding
Part: Exploratory
Discription: Explores the Seattle Airbnb datasets for the final project.
"""

import pandas as pd
import matplotlib.pyplot as plt


def dataset_size(calendar, listings, reviews):
    """
    Show the number of rows and columns for each dataset.
    """
    print("Calendar:", calendar.shape)
    print("Listings:", listings.shape)
    print("Reviews:", reviews.shape)


def missing_data(calendar, listings, reviews):
    """
    Find missing values in the datasets.
    """
    print("Calendar missing values:")
    print(calendar.isna().sum())
    print("Listings missing values:")
    print(listings.isna().sum().to_string())
    print("Reviews missing values:")
    print(reviews.isna().sum())


def variables_of_interest(calendar, listings, reviews):
    """
    Show the variables used for the research questions.
    """
    calendar_columns = ["listing_id", "date", "available", "price"]
    listings_columns = ["id", "neighbourhood_cleansed",
                        "room_type", "bedrooms", "accommodates"]
    reviews_columns = ["listing_id", "date"]
    print("Calendar variables:")
    print(calendar[calendar_columns].dtypes)
    print("Listings variables:")
    print(listings[listings_columns].dtypes)
    print("Reviews variables:")
    print(reviews[reviews_columns].dtypes)


def clean_price(price):
    """
    Convert price string into a number.
    """
    return float(price.replace("$", "").replace(",", ""))


def get_month(date):
    """
    Get the month from a date string.
    """
    return date[5:7]


def get_year(date):
    """
    Get the year from a date string.
    """
    return date[:4]


def summary_interest(calendar, listings, reviews):
    """
    Summarize the variables of interest.
    """
    priced = calendar[calendar["date"] < "2017-01-01"]
    priced = priced[priced["price"].notna()].copy()
    priced["price"] = priced["price"].apply(clean_price)
    priced["month"] = priced["date"].apply(get_month)
    print("Daily price:")
    print(priced["price"].describe())
    print("Bedrooms and accommodates:")
    print(listings[["bedrooms", "accommodates"]].describe())
    print("Months:")
    print(priced.groupby("month")["listing_id"].count())
    print("Room types:")
    print(listings.groupby("room_type")["id"].count())
    print("Neighborhoods:")
    neighborhoods = listings.groupby("neighbourhood_cleansed")["id"].count()
    print(neighborhoods.to_string())
    print("Availability:")
    print(calendar.groupby("available")["listing_id"].count())
    past_reviews = reviews[reviews["date"] < "2016-01-04"].copy()
    past_reviews["review_count"] = 1
    review_counts = past_reviews.groupby("listing_id")[
        "review_count"].sum().reset_index()
    listing_data = listings.merge(
        review_counts, left_on="id", right_on="listing_id", how="left"
    )
    listing_data["review_count"] = listing_data["review_count"].fillna(0)
    print("Historical reviews:")
    print(listing_data["review_count"].describe())
    availability = calendar[["listing_id", "available"]].copy()
    availability["available"] = availability["available"] == "t"
    rates = availability.groupby("listing_id")[
        "available"].mean().reset_index()
    listing_data = listing_data.merge(
        rates, left_on="id", right_on="listing_id", how="left"
    )
    print("Availability rate:")
    print(listing_data["available"].describe())


def plot_calendar(calendar):
    """
    Creates two plots about rental prices and availability.
    """
    data = calendar[calendar["date"] < "2017-01-01"].copy()
    data["month"] = data["date"].apply(get_month)
    prices = data.dropna(subset=["price"]).copy()
    prices["price"] = prices["price"].apply(clean_price)
    monthly_prices = prices.groupby("month")["price"].mean()
    fig, ax = plt.subplots()
    monthly_prices.plot(kind="line", ax=ax)
    plt.title("Average Rental Price by Month in 2016")
    plt.xlabel("Month")
    plt.ylabel("Average Price ($)")
    fig.savefig("calendar_price.png")
    data["is_available"] = data["available"] == "t"
    monthly_availability = data.groupby("month")["is_available"].mean()
    fig, ax = plt.subplots()
    monthly_availability.plot(kind="line", ax=ax)
    plt.title("Availability Rate by Month in 2016")
    plt.xlabel("Month")
    plt.ylabel("Availability Rate")
    fig.savefig("calendar_availability.png")


def plot_listings(listings):
    """
    Creates two plots about Airbnb room types and prices.
    """
    room_counts = listings.groupby("room_type")["id"].count()
    fig, ax = plt.subplots()
    room_counts.plot(kind="bar", ax=ax)
    plt.title("Number of Listings by Room Type")
    plt.xlabel("Room Type")
    plt.ylabel("Number of Listings")
    fig.savefig("listings_room_type.png")
    data = listings.copy()
    data["price"] = data["price"].apply(clean_price)
    room_prices = data.groupby("room_type")["price"].mean()
    fig, ax = plt.subplots()
    room_prices.plot(kind="bar", ax=ax)
    plt.title("Average Listing Price by Room Type")
    plt.xlabel("Room Type")
    plt.ylabel("Average Price ($)")
    fig.savefig("listings_price.png")


def plot_reviews(reviews):
    """
    Creates two plots about historical Airbnb reviews.
    """
    data = reviews[reviews["date"] < "2016-01-01"].copy()
    data["year"] = data["date"].apply(get_year)
    yearly_reviews = data.groupby("year")["id"].count()
    fig, ax = plt.subplots()
    yearly_reviews.plot(kind="bar", ax=ax)
    plt.title("Number of Reviews by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Reviews")
    fig.savefig("reviews_year.png")
    review_counts = reviews.groupby("listing_id")["id"].count()
    fig, ax = plt.subplots()
    review_counts.plot(kind="hist", bins=20, ax=ax)
    plt.title("Distribution of Reviews per Listing")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Number of Listings")
    fig.savefig("reviews_distribution.png")


def main():
    """
    Loads the datasets and do EDA.
    """
    calendar = pd.read_csv("calendar.csv")
    listings = pd.read_csv("listings.csv", low_memory=False)
    reviews = pd.read_csv("reviews.csv")
    dataset_size(calendar, listings, reviews)
    missing_data(calendar, listings, reviews)
    variables_of_interest(calendar, listings, reviews)
    summary_interest(calendar, listings, reviews)
    plot_calendar(calendar)
    plot_listings(listings)
    plot_reviews(reviews)


if __name__ == "__main__":
    main()
