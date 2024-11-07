import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import os

dest = r"C:\Users\aayus\Downloads\output_plots\plots\workload\secs"

# Load data
df = pd.read_csv(r"C:\Users\aayus\Downloads\output_plots\output_readings\processed_secs.csv")

# Split 'id' into 'category' and 'q_id'
def func(row):
    val = row["id"].split("_")
    return val[0], val[1]

df.replace(-1, pd.NA, inplace=True)
df[["category", "q_id"]] = df.apply(func, axis=1, result_type="expand")
df['secs'] = df['secs']/ 7200 # convert to hours - more intuitive (readings at 0.5 seconds interval)
df['cpu_per'] = df['cpu_per']/ 4  # We need average CPU usage.

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
                    value_vars=['cpu_mem', 'gpu_mem'],
                    var_name='Type', 
                    value_name='Memory Usage')

fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('CPU', 'GPU'),
                       shared_yaxes=True, horizontal_spacing=0.1)


# CPU Memory Plot
cpu_mem = df_melted[df_melted['Type'] == 'cpu_mem']
cpu_mem_fig = px.line(cpu_mem, x='secs', y='Memory Usage', color='category', color_discrete_sequence=px.colors.qualitative.Safe)
for trace in cpu_mem_fig.data:
    fig.add_trace(trace, row=1, col=1)

# GPU Memory Plot
gpu_mem = df_melted[df_melted['Type'] == 'gpu_mem']
gpu_mem_fig = px.line(gpu_mem, x='secs', y='Memory Usage', color='category', color_discrete_sequence=px.colors.qualitative.Safe)
for trace in gpu_mem_fig.data:
    trace.showlegend = False  # Hide legend
    fig.add_trace(trace, row=1, col=2)

fig.update_layout(
    title_text="Memory Usage",
    showlegend=True,  # Show legend only once
    legend_title_text="Category",
    font=font_dict,
    yaxis_showticklabels=True,
    yaxis2_showticklabels=True # add ticks on right hand side plot
)

fig.update_xaxes(rangemode="tozero", title_text="Hours", row=1, col=1, tickvals=xticks)
fig.update_xaxes(rangemode="tozero", title_text="Hours", row=1, col=2, tickvals=xticks)
fig.update_yaxes(rangemode="tozero", title_text="CPU Memory Usage (MB)", row=1, col=1)
fig.update_yaxes(rangemode="tozero", title_text="GPU Memory Usage (MB)", row=1, col=2)

fig.update_traces(mode="lines")
fig.show()
fig.write_image(os.path.join(dest, "memory_usage_plot_secs.png"), width=1920 , height=1080)

#####################################

cpu_gpu_usage = df.melt(id_vars=['secs', 'category'], 
                    value_vars=['cpu_per', 'gpu_per'],
                    var_name='Type', 
                    value_name='CPU/GPU Usage')


fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('CPU', 'GPU'),
                       shared_yaxes=True)


cpu_usage = cpu_gpu_usage[cpu_gpu_usage['Type'] == 'cpu_per']
cpu_usage_fig = px.line(cpu_usage, x='secs', y='CPU/GPU Usage', color='category', color_discrete_sequence=px.colors.qualitative.Safe)
for trace in cpu_usage_fig.data:
    fig.add_trace(trace, row=1, col=1)

gpu_usage = cpu_gpu_usage[cpu_gpu_usage['Type'] == 'gpu_per']
gpu_usage_fig = px.line(gpu_usage, x='secs', y='CPU/GPU Usage', color='category', color_discrete_sequence=px.colors.qualitative.Safe)
for trace in gpu_usage_fig.data:
    trace.showlegend = False  # Hide legend
    fig.add_trace(trace, row=1, col=2)


fig.update_layout(
    title_text="CPU/GPU Usage",
    showlegend=True,  # Show legend only once
    legend_title_text="Category",
    font=font_dict,
    yaxis_showticklabels=True,
    yaxis2_showticklabels=True # add ticks on right hand side plot
)

fig.update_xaxes(rangemode="tozero", title_text="Hours", row=1, col=1, tickvals=xticks)
fig.update_xaxes(rangemode="tozero", title_text="Hours", row=1, col=2, tickvals=xticks)
fig.update_yaxes(rangemode="tozero", title_text="Average CPU usage (%)", row=1, col=1)
fig.update_yaxes(rangemode="tozero", title_text="GPU Usage (%)", row=1, col=2)

fig.update_traces(mode="lines")
fig.show()
fig.write_image(os.path.join(dest, "cpu_cpu_percen_plot_secs.png"), width=1920 , height=1080)


#####################################

throughput = df.melt(id_vars=['secs', 'category'], 
                    value_vars=['generation_throughput' , 'prompt_eval_throughput'],
                    var_name='Type', 
                    value_name='throughput')


fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Token generation throughput', 'Prompt eval throughput'),
                       shared_yaxes=True)


generation_throughput = throughput[throughput['Type'] == 'generation_throughput']
generation_throughput_fig = px.line(generation_throughput, x='secs', y='throughput', color='category')
for trace in generation_throughput_fig.data:
    fig.add_trace(trace, row=1, col=1)

prompt_eval_throughput = throughput[throughput['Type'] == 'prompt_eval_throughput']
prompt_eval_throughput_fig = px.line(prompt_eval_throughput, x='secs', y='throughput', color='category')
for trace in prompt_eval_throughput_fig.data:
    trace.showlegend = False  # Hide legend
    fig.add_trace(trace, row=1, col=2)

fig.update_layout(
    title_text="Throughput",
    showlegend=True,  # Show legend only once
    legend_title_text="Category",
    font=font_dict,
    yaxis_showticklabels=True,
    yaxis2_showticklabels=True # add ticks on right hand side plot
)

fig.update_xaxes(rangemode="tozero", title_text="Hours", row=1, col=1, tickvals=xticks)
fig.update_xaxes(rangemode="tozero", title_text="Hours", row=1, col=2, tickvals=xticks)
fig.update_yaxes(rangemode="tozero", title_text="Token generation throughput (tokens/sec)", row=1, col=1)
fig.update_yaxes(rangemode="tozero", title_text="Prompt eval throughput (tokens/sec)", row=1, col=2)

fig.update_traces(mode="lines+markers")
fig.show()
fig.write_image(os.path.join(dest, "throughput.png"), width=1920 , height=1080)