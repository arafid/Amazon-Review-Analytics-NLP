# MIT 5032 - Create the 300-review project sample
# Run this only if you want to recreate the small project dataset from Reviews.csv.

import pandas as pd

DATA_FILE = "Reviews.csv"

PRODUCTS = [
    "B007JFMH8M", "B002QWP89S", "B003B3OOPA", "B001EO5Q64", "B001RVFERK",
    "B000KV61FC", "B0013A0QXC", "B005K4Q1YA", "B0090X8IPM", "B007Y59HVM"
]


def rating_sentiment(score):
    if score >= 4:
        return "positive"
    elif score == 3:
        return "neutral"
    else:
        return "negative"


# Load only the columns needed for the project.
df = pd.read_csv(DATA_FILE, usecols=["ProductId", "Score", "Text"])
df = df.dropna()
df = df.drop_duplicates(subset=["ProductId", "Text"])
df = df[df["ProductId"].isin(PRODUCTS)].copy()
df["RatingSentiment"] = df["Score"].apply(rating_sentiment)

# Randomly take 10 negative, 10 neutral, and 10 positive reviews per product.
sample_list = []

for product in PRODUCTS:
    product_reviews = df[df["ProductId"] == product]

    for sentiment in ["negative", "neutral", "positive"]:
        group = product_reviews[product_reviews["RatingSentiment"] == sentiment]
        sample = group.sample(n=10, random_state=42)
        sample_list.append(sample)

project_sample = pd.concat(sample_list, ignore_index=True)
project_sample = project_sample[["ProductId", "Score", "Text"]]
project_sample.to_csv("amazon_reviews_project_sample.csv", index=False)

print("Project sample created:", len(project_sample), "reviews")
print(project_sample["Score"].value_counts().sort_index())
