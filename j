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


Here’s a straight-through Pandas snippet that adds all three new columns:

import pandas as pd

# df already loaded ----------------------------------------------------------
# ── 1.  make sure TimeDelta is in seconds ------------------------------------
# If TimeDelta is a string like "0:01:35", convert it; otherwise keep as-is.
df['TimeDelta_sec'] = (
    pd.to_timedelta(df['TimeDelta'])      # works for "hh:mm:ss", "m:s", etc.
      .dt.total_seconds()                # → float (seconds)
      .fillna(0)                         # optional: replace NaNs with 0
      .astype(int)                       # make it int if you prefer
)

# ── 2.  derive minutes & hours ----------------------------------------------
df['TimeDelta_minute'] = df['TimeDelta_sec'] / 60          # minutes  (float)
df['TimeDelta_hour']   = df['TimeDelta_sec'] / 3600        # hours    (float)

# ── 3.  week number from timestamp ------------------------------------------
# If TimestampLocal is already datetime64, skip the conversion on the left.
df['week_number'] = (
    pd.to_datetime(df['TimestampLocal'])   # ensure datetime dtype
      .dt.isocalendar()                    # ISO calendar (year, week, day)
      .week                                # take the week component
)

# optional: tidy up -----------------------------------------------------------
df.drop(columns='TimeDelta_sec', inplace=True)  # keep only the new fields

What it adds

New column	Type	Meaning

TimeDelta_minute	float	duration in minutes
TimeDelta_hour	float	duration in hours
week_number	int	ISO week number (1–53) from timestamp


That’s it—run once and the DataFrame has the extra analytics-friendly fields ready for plotting or grouping.



Here is the simplest step-by-step code to convert Excel serial dates (45757.99 style) to real datetime values like 2025-04-10:


---

✅ Step-by-step simple code:

import pandas as pd

# 1. Load all sheets
all_sheets = pd.read_excel("Loretta 1-10 Apr - Copy.xlsx", sheet_name=None)

# 2. Columns to convert
date_cols = ['Date', 'TimestampLocal', 'TimestampUTC']

# 3. Convert in each sheet
for name, df in all_sheets.items():
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit='D', origin='1899-12-30')


---

🧪 To check:

# View first few rows of any sheet
print(all_sheets['M 1-10'].head())


---

Let me know if you want to combine all sheets too!





Here is a full Python code to generate a summary table for each user from your dataset, assuming the following:

Each row represents an activity log with a user.

You have a column User id and a column deltaTime (in seconds or minutes).

You want to:

1. Get total time (converted to hours),


2. Count records,


3. Count working days (if total time > 1 hour),


4. Compute average records/day and average working hours/day.





---

✅ Code

import pandas as pd

# Example input: df = all_sheets['M 1-10'] (already cleaned)

# 1. Convert deltaTime to hours
df['deltaTime_hours'] = df['deltaTime'] / 3600  # adjust if unit is minutes

# 2. Convert date to proper datetime if not already
df['Date'] = pd.to_datetime(df['Date']).dt.date  # remove time part

# 3. Group by user and date to get daily total time per user
daily = df.groupby(['User id', 'Date']).agg(
    daily_hours=('deltaTime_hours', 'sum'),
    records_per_day=('deltaTime_hours', 'count')
).reset_index()

# 4. Working day = days with total > 1 hour
daily['is_working_day'] = daily['daily_hours'] > 1

# 5. Summary per user
summary = daily.groupby('User id').agg(
    total_hours=('daily_hours', 'sum'),
    total_records=('records_per_day', 'sum'),
    working_days=('is_working_day', 'sum'),
).reset_index()

# 6. Add averages
summary['avg_records_per_day'] = summary['total_records'] / summary['working_days']
summary['avg_hours_per_day'] = summary['total_hours'] / summary['working_days']

# 7. Display
import ace_tools as tools; tools.display_dataframe_to_user(name="User Summary Table", dataframe=summary)


---

📌 Output Table Columns:

User id

total_hours – sum of delta time in hours

total_records – number of records for user

working_days – days where deltaTime > 1 hour

avg_records_per_day – total records ÷ working days

avg_hours_per_day – total hours ÷ working days


Let me know if your deltaTime is in minutes instead of seconds — just divide by 60 instead of 3600.

Here is a full Python code to generate a summary table for each user from your dataset, assuming the following:

Each row represents an activity log with a user.

You have a column User id and a column deltaTime (in seconds or minutes).

You want to:

1. Get total time (converted to hours),


2. Count records,


3. Count working days (if total time > 1 hour),


4. Compute average records/day and average working hours/day.





---

✅ Code

import pandas as pd

# Example input: df = all_sheets['M 1-10'] (already cleaned)

# 1. Convert deltaTime to hours
df['deltaTime_hours'] = df['deltaTime'] / 3600  # adjust if unit is minutes

# 2. Convert date to proper datetime if not already
df['Date'] = pd.to_datetime(df['Date']).dt.date  # remove time part

# 3. Group by user and date to get daily total time per user
daily = df.groupby(['User id', 'Date']).agg(
    daily_hours=('deltaTime_hours', 'sum'),
    records_per_day=('deltaTime_hours', 'count')
).reset_index()

# 4. Working day = days with total > 1 hour
daily['is_working_day'] = daily['daily_hours'] > 1

# 5. Summary per user
summary = daily.groupby('User id').agg(
    total_hours=('daily_hours', 'sum'),
    total_records=('records_per_day', 'sum'),
    working_days=('is_working_day', 'sum'),
).reset_index()

# 6. Add averages
summary['avg_records_per_day'] = summary['total_records'] / summary['working_days']
summary['avg_hours_per_day'] = summary['total_hours'] / summary['working_days']

# 7. Display
import ace_tools as tools; tools.display_dataframe_to_user(name="User Summary Table", dataframe=summary)


---

📌 Output Table Columns:

User id

total_hours – sum of delta time in hours

total_records – number of records for user

working_days – days where deltaTime > 1 hour

avg_records_per_day – total records ÷ working days

avg_hours_per_day – total hours ÷ working days


Let me know if your deltaTime is in minutes instead of seconds — just divide by 60 instead of 3600.

