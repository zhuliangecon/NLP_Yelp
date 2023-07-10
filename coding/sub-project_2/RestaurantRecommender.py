import pandas as pd
import random
from uszipcode import SearchEngine
import numpy as np
import urllib.parse


class RestaurantRecommender:
    def __init__(self):
        self.business_df = pd.read_csv("tip_business.csv")
        self.tips_df = pd.read_csv("tip_text.csv")
        self.keywords = ["pizza", "sushi", "burger", "tacos", "ramen", "steak", "fried chicken", "pasta", "wings"]
        self.custom_pool = []

    def get_nearby_zipcodes(self, zipcode, radius):
        search = SearchEngine()
        zipcodes = search.by_zipcode(zipcode)

        if not zipcodes:
            print(f"Zip code '{zipcode}' not found.")
            return []

        lat, lng = zipcodes.lat, zipcodes.lng
        nearby_zipcodes = search.by_coordinates(lat, lng, radius=radius, returns=0)
        return [zipcode.zipcode for zipcode in nearby_zipcodes]

    def filter_by_zipcode(self, zipcode, radius):
        nearby_zipcodes = self.get_nearby_zipcodes(zipcode, radius)
        if not nearby_zipcodes:
            return []

        nearby_filtered_businesses = self.business_df[self.business_df["postal_code"].isin(nearby_zipcodes)]
        return nearby_filtered_businesses

    def filter_by_keyword(self, keyword, df):
        keyword_filtered_tips = self.tips_df[self.tips_df["text"].str.contains(keyword, case=False)]
        keyword_filtered_businesses = df[df["business_id"].isin(keyword_filtered_tips["business_id"])]
        return keyword_filtered_businesses, keyword_filtered_tips

    def compute_custom_score(self, row):
        return np.log(row["review_count"] + 1) * row["stars"] * row["tips_count"]

    def generate_google_maps_url(self, name, address):
        base_url = "https://www.google.com/maps/search/?api=1&query="
        query = f"{name} {address}"
        url_encoded_query = urllib.parse.quote(query)
        return base_url + url_encoded_query

    def recommender(self, keyword, zipcode, radius, n=3):
        nearby_filtered_businesses = self.filter_by_zipcode(zipcode, radius)
        filtered_by_keyword, keyword_filtered_tips = self.filter_by_keyword(keyword, nearby_filtered_businesses)

        filtered_by_keyword = filtered_by_keyword.copy()
        
        if filtered_by_keyword.empty:
            print("No restaurant found. Try increasing the search radius or using a different keyword.")
            return []
        filtered_by_keyword["tips_count"] = filtered_by_keyword["business_id"].apply(lambda x: len(keyword_filtered_tips[keyword_filtered_tips["business_id"] == x]))
        filtered_by_keyword["custom_score"] = filtered_by_keyword.apply(self.compute_custom_score, axis=1)

        top_n_restaurants = filtered_by_keyword.nlargest(n, "custom_score")

        recommendations = []
        for index, restaurant in top_n_restaurants.iterrows():
            stars = int(restaurant["stars"])
            half_stars = (restaurant["stars"] - stars) == 0.5
            star_str = "★" * stars
            if half_stars:
                star_str += "☆"

            tips_with_keyword = keyword_filtered_tips[keyword_filtered_tips["business_id"] == restaurant["business_id"]]
            tips_with_keyword_count = len(tips_with_keyword)
            random_tips_with_keyword = tips_with_keyword.sample(min(3, tips_with_keyword_count))["text"].tolist()

            location = f"{restaurant['address']}, {restaurant['city']}, {restaurant['state']} {restaurant['postal_code']}"
            google_maps_url = self.generate_google_maps_url(restaurant["name"], location)

            recommendations.append({
                "name": restaurant['name'],
                "stars": star_str,
                "keyword": keyword,
                "tips": random_tips_with_keyword,
                "tips_count": tips_with_keyword_count,
                "location": location,
                "custom_score": restaurant["custom_score"],
                "google_maps_url": google_maps_url
            })

        return recommendations
    
    def draw(self, zipcode, radius, keywords=None):
        if not keywords:
            keywords = self.keywords

        random_keyword = random.choice(keywords)
        recommendations = self.recommender(random_keyword, zipcode, radius)

        if recommendations:
            random_recommendation = random.choice(recommendations)
            return random_recommendation
        else:
            return None
