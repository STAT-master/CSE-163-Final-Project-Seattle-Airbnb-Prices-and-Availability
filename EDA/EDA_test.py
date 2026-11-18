"""
Author: Ruifan Ding
Part: EDA Testing
Description: Tests the data processing used in the EDA project.
"""

import pandas as pd
import EDA


def test_helpers():
    """
    Tests the price and date conversion functions.
    """
    assert EDA.clean_price("$100.00") == 100.0
    assert EDA.clean_price("$1,250.50") == 1250.5
    assert EDA.get_month("2016-07-15") == "07"
    assert EDA.get_year("2016-07-15") == "2016"


def test_small_data():
    """
    Tests review counts and availability rates using small data.
    """
    calendar = pd.DataFrame({
        "listing_id": [1, 1, 2, 2],
        "available": ["t", "f", "t", "t"]
    })
    calendar["available"] = calendar["available"] == "t"
    rates = calendar.groupby("listing_id")["available"].mean()
    assert rates.loc[1] == 0.5
    assert rates.loc[2] == 1.0

    reviews = pd.DataFrame({
        "listing_id": [1, 1, 2],
        "date": ["2015-10-01", "2016-01-04", "2015-11-01"]
    })
    past_reviews = reviews[reviews["date"] < "2016-01-04"].copy()
    past_reviews["review_count"] = 1

    counts = past_reviews.groupby("listing_id")[
        "review_count"].sum().reset_index()
    listings = pd.DataFrame({"id": [1, 2, 3]})
    combined = listings.merge(
        counts, left_on="id", right_on="listing_id", how="left"
    )
    combined["review_count"] = combined["review_count"].fillna(0)
    assert combined["review_count"].tolist() == [1, 1, 0]


def main():
    """
    Runs all the EDA tests.
    """
    test_helpers()
    test_small_data()
    print("All tests passed!")


if __name__ == "__main__":
    main()
