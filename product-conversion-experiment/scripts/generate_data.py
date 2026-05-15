import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000

df = pd.DataFrame({
    "user_id": range(n),
    "group": np.random.choice(["control", "variant"], n),
    "device_type": np.random.choice(["mobile", "desktop"], n),
    "country": np.random.choice(["US", "JP", "TW"], n),
    "session_time": np.random.normal(300, 50, n),
})

df["converted"] = np.where(
    df["group"] == "control",
    np.random.binomial(1, 0.12, n),
    np.random.binomial(1, 0.15, n)
)

df["revenue"] = df["converted"] * np.random.uniform(50, 200, n)

df.to_csv("data/experiment_data.csv", index=False)


## This is a simple script to generate synthetic A/B testing data for demonstration purposes.
'''
np.random.seed(42)

n = 5000

data = pd.DataFrame({
    "user_id": range(n),
    "group": np.random.choice(["control", "variant"], size=n),
})

# baseline conversion
data["converted"] = np.where(
    data["group"] == "control",
    np.random.binomial(1, 0.12, n),
    np.random.binomial(1, 0.13, n)  # variant slightly better
)

# revenue
data["revenue"] = data["converted"] * np.random.uniform(50, 200, n)

# session time
data["session_time"] = np.random.normal(300, 50, n)

data.to_csv("data/ab_data.csv", index=False)
'''