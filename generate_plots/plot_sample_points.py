import plotly.express as px
import pandas as pd
import os

dest = r"C:\Users\aayus\Downloads\output_plots\plots\workload\sample_points"

# Load data
df = pd.read_csv(r"C:\Users\aayus\Downloads\output_plots\output_readings\processed_sample_points.csv")

# Split 'id' into 'category' and 'q_id'
def func(row):
    val = row["id"].split("_")
    return val[0], val[1]

df.replace(-1, pd.NA, inplace=True)
df[["category", "q_id"]] = df.apply(func, axis=1, result_type="expand")
# df = df.sort_values(axis=0, by=["category", "q_id"])
df['index'] = df.index  # Add an index column to use as time

# Melt the DataFrame to long format
df_melted = df.melt(id_vars=['index', 'category', 'q_id'], 
                    value_vars=['cpu_mem_max', 'gpu_mem_max'],
                    var_name='Type', 
                    value_name='Memory Usage')

# Memory Usage Graph
fig = px.line(df_melted, x='index', y='Memory Usage', color='category', symbol='Type', symbol_sequence=["circle", "cross-open"],
              labels={'Memory Usage': 'Memory Usage (MB)', 'Type': 'Memory Type'}, 
              title='Memory Usage')
fig.update_traces(mode='markers')
fig.show()
fig.write_html(os.path.join(dest, "memory_usage_max.html"))


cpu_gpu_usage = df.melt(id_vars=['index', 'category', 'q_id'], 
                    value_vars=['cpu_per_max', 'gpu_per_max'],
                    var_name='Type', 
                    value_name='CPU/GPU Usage')

# CPU/GPU Usage Graph
fig = px.line(cpu_gpu_usage, x='index', y='CPU/GPU Usage',
              color='category', symbol='Type', symbol_sequence=["circle", "cross-open"],
              labels={'value': 'CPU/GPU Usage (%)', 'variable': 'Type'}, title='CPU/GPU Usage')
fig.update_traces(mode='lines+markers')
fig.show()
fig.write_html(os.path.join(dest, "cpu_gpu_percen_max.html"))

throughput = df.melt(id_vars=['index', 'category', 'q_id'], 
                    value_vars=['generation_throughput' , 'prompt_eval_throughput'],
                    var_name='Type', 
                    value_name='throughput')

# Throughput Graph
fig = px.line(throughput, x='index', y='throughput',
              color='category', symbol='Type', symbol_sequence=["circle", "cross-open"],
              labels={'throughput': 'Throughput (tokens/sec)', 'variable': 'Type'}, title='Throughput')
fig.update_traces(mode='lines+markers')
fig.show()
fig.write_html(os.path.join(dest, "throughput.html"))