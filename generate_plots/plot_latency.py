# Time between tokens across workloads
# Time to first Token across workloads
# Total time by number of tokens


import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import os

dest = r"C:\Users\aayus\Downloads\output_plots\plots\workload\secs"

# Load data
df = pd.read_csv(r"C:\Users\aayus\Downloads\output_plots\output_readings\granular_helper\contextual_task-oriented_same_output_length_total_time\processed_latency_same_output_compare.csv")

# Split 'id' into 'category' and 'q_id'
def func(row):
    val = row["id"].split("_")
    return val[0], val[1]

df.replace(-1, pd.NA, inplace=True)
df[["category", "q_id"]] = df.apply(func, axis=1, result_type="expand")
df['secs'] = df['secs']/ 7200 # convert to hours - more intuitive (readings at 0.5 seconds interval)
df['total_time'] = df['total_time']/ 1000 # convert ms to secs
df['total_tokens'] = df['total_tokens'].astype('int32')
df['prompt_tokens'] = df['prompt_tokens'].astype('int32')
df['TTFT'] = df['TTFT'] / 1000 # convert ms to secs

# Font settings
font_dict = {
    'family': "Arial, sans-serif",  # Font family
    'size': 22,                    # Font size
    'color': "black"               # Font color
}

# We want both axes to start at zero. Using rangemode="tozero"
# Distance between x ticks = 2 hours
multiples = int((df['secs'].max()/ 1)+1)
xticks = [i*1 for i in range(1, multiples+1)] # Excluding zero tick.


df_melted = df.melt(id_vars=['secs', 'category'], 
                    value_vars=['TTFT', 'TBT'],
                    var_name='Type',
                    value_name='TTFT/TBT')

fig = sp.make_subplots(rows=1, cols=2,
                       shared_yaxes=False, horizontal_spacing=0.1)


TTFT = df_melted[df_melted['Type'] == 'TTFT']
TTFT_fig = px.line(TTFT, x='secs', y='TTFT/TBT', color='category', color_discrete_sequence=px.colors.qualitative.Safe)
for trace in TTFT_fig.data:
    fig.add_trace(trace, row=1, col=1)


TBT = df_melted[df_melted['Type'] == 'TBT']
TBT_fig = px.line(TBT, x='secs', y='TTFT/TBT', color='category', color_discrete_sequence=px.colors.qualitative.Safe)
for trace in TBT_fig.data:
    trace.showlegend = False  # Hide legend
    fig.add_trace(trace, row=1, col=2)

fig.update_layout(
    title_text="Latency",
    showlegend=True,  # Show legend only once
    legend_title_text="Category",
    font=font_dict,
    yaxis_showticklabels=True,
    yaxis2_showticklabels=True # add ticks on right hand side plot
)

fig.update_xaxes(rangemode="tozero", title_text="Hours", row=1, col=1, tickvals=xticks)
fig.update_xaxes(rangemode="tozero", title_text="Hours", row=1, col=2, tickvals=xticks)
fig.update_yaxes(rangemode="tozero", title_text="Time to First Token (secs)", row=1, col=1)
fig.update_yaxes(rangemode="tozero", title_text="Time between Tokens (ms)", row=1, col=2)

fig.update_traces(mode="lines")
fig.show()
# fig.write_image(os.path.join(dest, "latency_token_times.png"), width=1920 , height=1080)

#####################################

sorted_df = df.sort_values(axis=0,  by=["category", "prompt_tokens"])
fig = px.line(sorted_df, x='prompt_tokens', y='total_time', color='category', color_discrete_sequence=px.colors.qualitative.Safe)

fig.update_layout(
    title_text="Total Time vs Tokens in prompt",
    showlegend=True,
    font=font_dict)

xticks = [100, 200, 300, 400, 500]
fig.update_xaxes(rangemode="tozero", title_text="Total Tokens in Prompt", tickvals=xticks)
fig.update_yaxes(rangemode="tozero", title_text="Total Time for query (secs)")

fig.update_traces(mode='markers', marker=dict(size=25))
fig.show()
# fig.write_image(os.path.join(dest, "total_tokens_times.png"), width=1920 , height=1080)
