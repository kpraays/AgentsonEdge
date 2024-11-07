import pandas as pd

# Load data
df = pd.read_csv(r"C:\Users\aayus\Downloads\output_plots\output_readings\processed_latency.csv")

def func(row):
    val = row["id"].split("_")
    return val[0], val[1]

df.replace(-1, pd.NA, inplace=True)
df[["category", "q_id"]] = df.apply(func, axis=1, result_type="expand")


median_prompt_tokens = df.groupby('category')['prompt_tokens'].median().reset_index()
mean_prompt_tokens = df.groupby('category')['prompt_tokens'].mean().reset_index()


df = df.merge(median_prompt_tokens, on='category', suffixes=('', '_median'))
df = df.merge(mean_prompt_tokens, on='category', suffixes=('', '_mean'))
df['prompt_tokens_mean'] = df['prompt_tokens_mean'].astype('int32')

df['diff'] = abs(df['prompt_tokens'] - df['prompt_tokens_median'])
df_sorted = df.sort_values(['category', 'diff'])
result = df_sorted.drop_duplicates(subset=['category'], keep='first')
output = result[['category', 'id', 'prompt_tokens', 'prompt_tokens_median']]

print(output)
