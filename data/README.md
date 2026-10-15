# Data

This repository includes the **300-review balanced project sample** only.

The original Amazon Fine Food Reviews dataset contains 568,454 reviews and is not included here because of its size. To recreate the sample, download the original `Reviews.csv` file from the Amazon Fine Food Reviews dataset source and place it in the working directory before running:

```bash
python src/create_sample.py
```

The sampling script uses 10 selected products and randomly selects 10 negative, 10 neutral, and 10 positive reviews per product with `random_state=42`.
