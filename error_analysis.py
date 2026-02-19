import pandas as pd
import re


INPUT_PATH = "sample_250_high_acc.csv"  # your uploaded predictions
df = pd.read_csv(INPUT_PATH)


df["is_error"] = df["ground-truth"] != df["model output"]
errors = df[df["is_error"]].copy()

total_errors = len(errors)

def word_count(text):
    if pd.isna(text):
        return 0
    # Count words ignoring punctuation
    return len(re.findall(r"\b\w+\b", str(text)))

errors["word_count"] = df["input text"].apply(word_count)


# Error Category 1:
# Short lines (<3 words)
short_line_errors = errors[errors["word_count"] < 3]
short_line_count = len(short_line_errors)


# Error Category 2:
# Phoebe as true speaker
phoebe_errors = errors[errors["ground-truth"] == "Phoebe Buffay"]
phoebe_count = len(phoebe_errors)

# Compute percentages
short_line_percent = (short_line_count / total_errors) * 100 if total_errors > 0 else 0
phoebe_percent = (phoebe_count / total_errors) * 100 if total_errors > 0 else 0

# Print stats
error_distribution = pd.DataFrame({
    "Error Pattern": [
        "Short line (<3 words)",
        "True speaker is Phoebe Buffay"
    ],
    "Error Count": [
        short_line_count,
        phoebe_count
    ],
    "Percent of Total Errors": [
        round(short_line_percent, 2),
        round(phoebe_percent, 2)
    ]
})

print(f"Total errors: {total_errors}")
print()
print(error_distribution)
