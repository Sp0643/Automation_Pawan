8Meeting Progress Update - Reconciliation Automation Project

1. Current Progress:

We have successfully created the initial draft of the automation workflow.

This draft was presented to the Planning team for review.


2. Feedback and Next Steps:

Based on the feedback, the logic needs further refinement and evolution.

We have identified two potential approaches for improving the workflow:

a. Rebuilding from Scratch:

Exploring the feasibility of creating a new workflow from scratch that aligns better with the evolving requirements.


b. Enhancing the Existing Workflow:

Utilizing the current workflow and improving its efficiency.

Instead of processing 100% of the data through the automated logic, we'll focus on automating 90% of the data efficiently.

The remaining 10% (which may have exceptions or require special handling) will be managed through Data Analysts (DA) to ensure accuracy.



3. Next Steps:

Assess the pros and cons of both approaches.

Collaborate with stakeholders to finalize the most suitable solution.

Continue refining the workflow to align with business requirements.


Please let me know if you'd like any additional details or adjustments to the pointers.



Below is a straight-ahead, no-functions script that reads the activity log and reproduces the key Tableau-style visuals in Plotly.
Just copy-paste into a Jupyter cell or a file like simple_activity_charts.py, change the file path to your CSV/XLSX, install the two libraries (pip install pandas plotly), and run—it will pop each chart in your browser.

# ──────────────────────────────────────────────────────────────────────────────
# VERY SIMPLE PLOTLY CODE (no custom functions)
# ──────────────────────────────────────────────────────────────────────────────
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1 ▸ LOAD DATA ───────────────────────────────────────────────────────────────
df = pd.read_csv("activity_log.csv")              # ⚠️  point to your file
df['TimestampLocal'] = pd.to_datetime(df['TimestampLocal'])
df['Date']          = pd.to_datetime(df['Date']).dt.date   # keep only date

# TimeDelta → seconds (handles "hh:mm:ss" strings or numbers)
if df['TimeDelta'].dtype == object:
    df['TimeDelta'] = (
        pd.to_timedelta(df['TimeDelta'])
          .dt.total_seconds()
          .fillna(0)
          .astype(int)
    )

# 2 ▸ TOP-10 APPLICATIONS (Bar) ───────────────────────────────────────────────
top_apps = (
    df.groupby('APPL_NAME')['TimeDelta']
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .reset_index()
)

fig = px.bar(
    top_apps,
    x='APPL_NAME',
    y='TimeDelta',
    title="Top-10 Applications by Total Time",
    labels={'APPL_NAME': 'Application', 'TimeDelta': 'Seconds'}
)
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# 3 ▸ DAILY USAGE TREND (Line) ────────────────────────────────────────────────
daily_usage = (
    df.groupby(['Date', 'APPL_NAME'])['TimeDelta']
      .sum()
      .reset_index()
)

fig = px.line(
    daily_usage,
    x='Date',
    y='TimeDelta',
    color='APPL_NAME',
    title="Daily Time Spent per Application",
    labels={'TimeDelta': 'Seconds'}
)
fig.show()

# 4 ▸ WEEKDAY × HOUR HEATMAP ──────────────────────────────────────────────────
tmp = df.copy()
tmp['hour']    = tmp['TimestampLocal'].dt.hour
tmp['weekday'] = tmp['TimestampLocal'].dt.day_name()

weekday_order = ['Monday','Tuesday','Wednesday',
                 'Thursday','Friday','Saturday','Sunday']
tmp['weekday'] = pd.Categorical(tmp['weekday'],
                                categories=weekday_order,
                                ordered=True)

heat_data = (
    tmp.groupby(['weekday', 'hour'])['TimeDelta']
       .sum()
       .reset_index()
       .pivot(index='weekday', columns='hour', values='TimeDelta')
       .fillna(0)
)

fig = px.imshow(
    heat_data,
    aspect='auto',
    color_continuous_scale='Blues',
    title="Activity Heat-Map (Seconds)",
)
fig.update_layout(xaxis_title='Hour of Day', yaxis_title='')
fig.show()

# 5 ▸ CLICKSTREAM SANKEY (Prev-App → Next-App) ────────────────────────────────
df_sorted = df.sort_values(['User ID', 'TimestampLocal'])
df_sorted['prev_app'] = df_sorted.groupby('User ID')['APPL_NAME'].shift()

transitions = (
    df_sorted.dropna(subset=['prev_app'])
             .groupby(['prev_app', 'APPL_NAME'])
             .size()
             .reset_index(name='count')
)

min_trans = 300                                     # change threshold if needed
transitions = transitions[transitions['count'] >= min_trans]

labels       = pd.unique(transitions[['prev_app','APPL_NAME']].values.ravel())
label_index  = {label: idx for idx, label in enumerate(labels)}
sources      = transitions['prev_app'].map(label_index)
targets      = transitions['APPL_NAME'].map(label_index)
values       = transitions['count']

fig = go.Figure(go.Sankey(
    node=dict(label=labels, pad=15, thickness=20),
    link=dict(source=sources, target=targets, value=values)
))
fig.update_layout(title_text=f"App Transitions (≥{min_trans} events)")
fig.show()

# 6 ▸ SIMPLE DAILY ANOMALY SCATTER (click count z-score) ─────────────────────
daily_clicks = (
    df.groupby(['User ID', 'Date'])
      .size()
      .reset_index(name='clicks')
)

# z-score per user
daily_clicks['z'] = (
    daily_clicks.groupby('User ID')['clicks']
                .transform(lambda s: (s - s.mean()) / s.std(ddof=0))
)

fig = px.scatter(
    daily_clicks,
    x='Date',
    y='clicks',
    color='z',
    color_continuous_scale='RdBu',
    title='Daily Click Count Anomaly',
    hover_data=['User ID', 'z']
)
fig.show()

What this script does

Section	Chart produced	Insight you get

2	Bar	Which apps eat the most time
3	Line	Trends of time spent on each app day-by-day
4	Heat-map	Peak activity hours for each weekday
5	Sankey	Common paths users take from one app to another
6	Scatter	Days where a user’s click volume deviates strongly from their norm


No custom functions, classes, or tricky abstractions—just plain procedural code you can edit or extend inline. Happy charting!


