# Technical Case Study: Amazon Customer Review Analytics

## Problem

Star ratings summarize satisfaction, but they do not explain *why* a customer is satisfied or dissatisfied. Sellers and product teams may receive more review text than they can realistically inspect one review at a time.

This project built a Python/NLP workflow to convert unstructured Amazon review text into sentiment labels, failure cases, complaint categories, and product-level insights.

## Data Strategy

The source Amazon Fine Food Reviews dataset contains 568,454 reviews. A direct random sample would be dominated by positive reviews, so the project used stratified random sampling by product and sentiment.

Ten products were selected. For each product, the workflow sampled:

- 10 negative reviews
- 10 neutral reviews
- 10 positive reviews

This created a balanced 300-review evaluation dataset.

## Analytics Pipeline

1. Load and clean review data with Pandas.
2. Remove missing and duplicate rows.
3. Convert star ratings into positive, neutral, and negative comparison labels.
4. Apply NLTK VADER to the review text.
5. Compare VADER sentiment with rating-derived labels.
6. Save disagreement cases for failure analysis.
7. Tokenize negative reviews and remove stop words.
8. Map review tokens to complaint categories with a dictionary-based classifier.
9. Aggregate top complaint themes by product.
10. Export CSV outputs and Matplotlib charts.

## Evaluation

The submitted project run produced 146 matches out of 300 reviews (48.67% agreement).

Class-level agreement:
- Positive: 93%
- Negative: 48%
- Neutral: 5%

The large gap across classes showed that mixed and neutral text is difficult for a simple rule-based sentiment method and that numeric star ratings are not a perfect representation of written sentiment.

## Business Findings

Across 100 negative reviews, the complaint-theme distribution was:

- Taste: 39
- Other: 19
- Packaging: 18
- Quality: 10
- Price / Value: 9
- Freshness: 4
- Shipping: 1

The value of the workflow is not only sentiment classification. It turns open-ended review text into structured signals that can help a seller identify where to investigate product problems first.

## Technical Skills Demonstrated

Python, Pandas, NLTK VADER, NLP, tokenization, stop-word removal, rule-based classification, stratified sampling, class balancing, DataFrame operations, aggregation, exploratory data analysis, error analysis, Matplotlib, CSV output, and business intelligence.

## What I Would Improve Next

A stronger version could use a larger sample, richer domain-specific theme vocabulary, improved neutral/mixed-sentiment handling, and a lightweight dashboard for exploring product-level insights.
