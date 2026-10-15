# ============================================================
# MIT 5032 GROUP PROJECT
# Amazon Customer Review Sentiment & Insight Tool
# Pandas + NLTK VADER
# ============================================================


# -------------------- 1. IMPORT PACKAGES --------------------

import pandas as pd
import matplotlib.pyplot as plt
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords


# Download the NLTK resources we need
nltk.download("vader_lexicon")
nltk.download("stopwords")


# -------------------- 2. LOAD THE DATA --------------------

df = pd.read_csv("amazon_reviews_project_sample.csv")

# Remove missing values and duplicate reviews
df = df.dropna(subset=["ProductId", "Score", "Text"])
df = df.drop_duplicates(subset=["ProductId", "Text"])

print("Reviews loaded:", len(df))


# -------------------- 3. SENTIMENT FROM STAR RATINGS --------------------

# We use the star rating as our comparison or "ground truth"
def rating_sentiment(score):

    if score >= 4:
        return "positive"

    elif score == 3:
        return "neutral"

    else:
        return "negative"


df["RatingSentiment"] = df["Score"].apply(rating_sentiment)


print("\nRating sentiment counts:")
print(df["RatingSentiment"].value_counts())


# -------------------- 4. NLTK VADER SENTIMENT --------------------

sia = SentimentIntensityAnalyzer()


def vader_sentiment(text):

    scores = sia.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        return "positive"

    elif compound <= -0.05:
        return "negative"

    else:
        return "neutral"


# Save the VADER compound score
df["VADER_Compound"] = df["Text"].apply(
    lambda text: sia.polarity_scores(text)["compound"]
)


# Save the VADER sentiment category
df["VADER_Sentiment"] = df["Text"].apply(vader_sentiment)


print("\nExample VADER results:")
print(
    df[
        [
            "Score",
            "RatingSentiment",
            "VADER_Compound",
            "VADER_Sentiment"
        ]
    ].head()
)


# -------------------- 5. EVALUATE THE RESULTS --------------------

# True means the VADER sentiment matched the star-rating sentiment
df["Match"] = (
    df["RatingSentiment"] == df["VADER_Sentiment"]
)


# Calculate overall accuracy
accuracy = df["Match"].mean() * 100


# Save reviews where VADER and the star rating disagree
failures = df[df["Match"] == False].copy()


print("\nOverall VADER accuracy:")
print(round(accuracy, 2), "%")


print("\nNumber of disagreements:")
print(len(failures), "out of", len(df))


print("\nAccuracy by rating sentiment:")

print(
    df.groupby("RatingSentiment")["Match"]
      .mean()
      .mul(100)
      .round(2)
)


# Save the results
df.to_csv(
    "vader_results.csv",
    index=False
)

failures.to_csv(
    "failure_cases.csv",
    index=False
)


# -------------------- 6. TOKENIZE NEGATIVE REVIEWS --------------------

stop_words = set(
    stopwords.words("english")
)


def clean_tokens(text):

    # Make the review lowercase and split it into words
    tokens = wordpunct_tokenize(
        text.lower()
    )

    clean_list = []

    for word in tokens:

        # Keep real words and remove common stopwords
        if word.isalpha() and word not in stop_words:

            clean_list.append(word)

    return clean_list


# -------------------- 7. COMPLAINT THEME WORDS --------------------

# Each dictionary key is a business complaint theme.
# Each value contains words related to that theme.

theme_words = {

    "taste": [
        "taste",
        "tastes",
        "flavor",
        "flavors",
        "sweet",
        "salty",
        "bitter"
    ],

    "packaging": [
        "package",
        "packaging",
        "box",
        "bag",
        "container",
        "seal",
        "broken",
        "damaged"
    ],

    "freshness": [
        "fresh",
        "stale",
        "expired",
        "spoiled",
        "mold",
        "old"
    ],

    "price/value": [
        "price",
        "cost",
        "expensive",
        "cheap",
        "value",
        "money"
    ],

    "shipping": [
        "shipping",
        "delivery",
        "delivered",
        "arrived",
        "ship"
    ],

    "quality": [
        "quality",
        "texture",
        "dry",
        "hard",
        "soft"
    ]
}


# -------------------- 8. FIND THE MAIN THEME --------------------

def find_theme(text):

    tokens = clean_tokens(text)

    best_theme = "other"

    highest_count = 0


    for theme, words in theme_words.items():

        count = 0


        for token in tokens:

            if token in words:

                count += 1


        if count > highest_count:

            highest_count = count

            best_theme = theme


    return best_theme


# We only look for complaint themes in negative reviews
negative_reviews = df[
    df["RatingSentiment"] == "negative"
].copy()


negative_reviews["Theme"] = (
    negative_reviews["Text"].apply(find_theme)
)


print("\nComplaint theme counts:")

print(
    negative_reviews["Theme"].value_counts()
)


# -------------------- 9. TOP COMPLAINT FOR EACH PRODUCT --------------------

product_list = []


for product in negative_reviews["ProductId"].unique():

    product_data = negative_reviews[
        negative_reviews["ProductId"] == product
    ]


    counts = product_data["Theme"].value_counts()


    top_theme = counts.index[0]

    count = counts.iloc[0]


    product_list.append({

        "ProductId": product,

        "TopComplaint": top_theme,

        "Count": count

    })


# Turn the list into a DataFrame
product_insights = pd.DataFrame(
    product_list
)


# -------------------- 10. SUGGESTED SELLER ACTIONS --------------------

actions = {

    "taste":
        "Review customer feedback about taste and flavor.",

    "packaging":
        "Check packaging and sealing problems.",

    "freshness":
        "Check freshness, storage, and expiration issues.",

    "price/value":
        "Review pricing and value concerns.",

    "shipping":
        "Review shipping and delivery handling.",

    "quality":
        "Check product quality and consistency.",

    "other":
        "Read the negative reviews to identify the issue."

}


product_insights["SuggestedAction"] = (
    product_insights["TopComplaint"].map(actions)
)


print("\nTop complaint for each product:")

print(
    product_insights.to_string(index=False)
)


# Save product insights
product_insights.to_csv(
    "product_insights.csv",
    index=False
)


# -------------------- 11. SENTIMENT COMPARISON CHART --------------------

sentiment_counts = pd.DataFrame({

    "Star Rating":
        df["RatingSentiment"].value_counts(),

    "VADER":
        df["VADER_Sentiment"].value_counts()

})


sentiment_counts.plot(
    kind="bar",
    title="Star Rating Sentiment vs. VADER Sentiment"
)


plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    "sentiment_comparison.png"
)

plt.show()


# -------------------- 12. COMPLAINT THEME CHART --------------------

negative_reviews["Theme"].value_counts().plot(

    kind="bar",

    title="Complaint Themes in Negative Reviews"

)


plt.xlabel("Complaint Theme")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    "complaint_themes.png"
)

plt.show()


# -------------------- 13. SIMPLE APPLICATION MENU --------------------

while True:

    print("\nCUSTOMER REVIEW INSIGHT TOOL")

    print("1 - Show sentiment accuracy")

    print("2 - Show product complaint insights")

    print("3 - Show complaint theme counts")

    print("0 - Exit")


    choice = input(
        "Choose an option: "
    )


    if choice == "1":

        print(
            "\nVADER accuracy:",
            round(accuracy, 2),
            "%"
        )

        print(
            "Disagreements:",
            len(failures),
            "out of",
            len(df)
        )


    elif choice == "2":

        print(
            "\nPRODUCT COMPLAINT INSIGHTS"
        )

        print(
            product_insights.to_string(
                index=False
            )
        )


    elif choice == "3":

        print(
            "\nCOMPLAINT THEME COUNTS"
        )

        print(
            negative_reviews[
                "Theme"
            ].value_counts()
        )


    elif choice == "0":

        print(
            "Program finished."
        )

        break


    else:

        print(
            "Invalid option."
        )
