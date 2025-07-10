import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import date
import json

# ---------- Inverter-Specific Preprocessing ----------
def preprocess_inverter_data(df):
    if 'SOURCE_KEY' not in df.columns or 'DATE_TIME' not in df.columns:
        return None

    df['SOURCE_KEY'] = df['SOURCE_KEY'].fillna('UNKNOWN')
    unique_ids = df['SOURCE_KEY'].unique()
    id_mapping = {uid: f"S{i+1}" for i, uid in enumerate(unique_ids)}
    df['SOURCE_ID'] = df['SOURCE_KEY'].map(id_mapping)
    df['SOURCE_ID_NUMBER'] = df['SOURCE_ID'].str.extract(r'(\d+)').astype(int)

    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], errors='coerce')
    df = df.dropna(subset=['DATE_TIME'])

    df.rename(columns={'AC_POWER': 'AC_POWER_OUTPUT', 'DC_POWER': 'DC_POWER_INPUT'}, inplace=True)
    df['AC_POWER_FIXED'] = df['AC_POWER_OUTPUT'] * 10
    df['EFFICIENCY'] = df['AC_POWER_FIXED'] / df['DC_POWER_INPUT']
    df['EFFICIENCY_%'] = df['EFFICIENCY'] * 100
    df['Value'] = df['AC_POWER_FIXED']
    df = df.loc[:, ~df.columns.duplicated()]

    if 'TIME_STAMP' not in df.columns:
        df.rename(columns={'DATE_TIME': 'TIME_STAMP'}, inplace=True)
    df = df[['TIME_STAMP', 'SOURCE_ID', 'SOURCE_ID_NUMBER', 'Value', 'EFFICIENCY_%', 'AC_POWER_FIXED']]
    df = df.sort_values("TIME_STAMP")
    df["time_index"] = range(len(df))
    return df

# ---------- Load & Clean Data ----------
def load_clean_data(file):
    df = pd.read_csv(file)
    st.write("📄 Uploaded Columns:", df.columns.tolist())

    if 'SOURCE_KEY' in df.columns and 'DATE_TIME' in df.columns:
        df = preprocess_inverter_data(df)
        st.success("☀️ Detected inverter data. Preprocessed automatically.")
        return df

    timestamp_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].notna().sum() > 0:
                    timestamp_col = col
                    break
            except:
                continue

    if not timestamp_col:
        for col in df.columns:
            if df[col].dtype in ['int64', 'object']:
                try:
                    df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m', errors='coerce')
                    if df[col].notna().sum() > 0:
                        timestamp_col = col
                        break
                except:
                    continue

    if not timestamp_col:
        st.error("❌ Could not detect a timestamp column. Please include at least one date-like column.")
        return pd.DataFrame()

    df['Timestamp'] = df[timestamp_col]

    value_col = None
    for col in df.columns:
        if col.lower() in ['value', 'output', 'consumption', 'energy', 'acpower_kw']:
            value_col = col
            break

    if not value_col:
        num_cols = df.select_dtypes(include='number').columns
        for col in num_cols:
            value_col = col
            break

    if not value_col:
        st.error("❌ Could not detect a numeric 'Value' column.")
        return pd.DataFrame()

    df = df.dropna(subset=["Timestamp", value_col])
    df['Value'] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=["Value"])
    df = df.sort_values("Timestamp")
    df["time_index"] = range(len(df))
    return df

# ---------- Anomaly Detection ----------
def detect_anomalies(df):
    model = IsolationForest(contamination=0.01, random_state=42)
    df["anomaly"] = model.fit_predict(df[["Value"]]) == -1
    return df

# ---------- Forecasting ----------
def run_forecast(df):
    df = df.copy()
    df["TIME_STAMP"] = pd.to_datetime(df["TIME_STAMP"], errors='coerce')
    df = df.dropna(subset=["TIME_STAMP", "AC_POWER_FIXED"])
    df["time_index"] = range(len(df))

    X = df[["time_index"]]
    y = df["AC_POWER_FIXED"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    forecast_df = X_test.copy()
    forecast_df["Actual"] = y_test.values
    forecast_df["Predicted"] = y_pred
    mse = mean_squared_error(y_test, y_pred)

    return forecast_df.sort_index(), mse

# ---------- AI Summary ----------
def generate_summary(df):
    total = df["Value"].sum()
    mean = df["Value"].mean()
    anomaly_count = df["anomaly"].sum() if "anomaly" in df else 0
    return (
        f"- Total Consumption: **{total:.2f} Units**  \n"
        f"- Monthly Average: **{mean:.2f} Units**  \n"
        f"- Anomalies Detected: **{anomaly_count} points**"
    )

# ---------- Export Alerts ----------
def export_alerts(df):
    today = pd.Timestamp(date.today())
    df['DateOnly'] = pd.to_datetime(df['TIME_STAMP']).dt.date
    alerts = df[(df['anomaly']) & (df['DateOnly'] == today)][["TIME_STAMP", "Value"]]
    if not alerts.empty:
        alerts_path = "alerts_today.csv"
        alerts.to_csv(alerts_path, index=False)
        st.success(f"✅ {len(alerts)} alert(s) exported to '{alerts_path}' for Zapier.")
    else:
        st.info("No new anomalies today.")

# ---------- Inverter Quartile Summary ----------
def grouping_data_with_summary(df):
    df_eff = df[df['EFFICIENCY_%'].between(0.01, 100)].copy()
    res2 = df_eff.groupby('SOURCE_ID_NUMBER')['AC_POWER_FIXED'].mean()
    res_df = pd.DataFrame(data=res2).reset_index()

    low_param = res2.quantile(0.25)
    middle_param = res2.quantile(0.5)
    high_param = res2.quantile(0.75)

    low = res_df[res_df['AC_POWER_FIXED'] <= low_param]
    medium_low = res_df[(res_df['AC_POWER_FIXED'] > low_param) & (res_df['AC_POWER_FIXED'] <= middle_param)]
    medium_high = res_df[(res_df['AC_POWER_FIXED'] > middle_param) & (res_df['AC_POWER_FIXED'] <= high_param)]
    high = res_df[res_df['AC_POWER_FIXED'] > high_param]

    summary = {
        'Inverter Counts': {
            'High (4th Quartile)': len(high),
            'Medium High (3rd Quartile)': len(medium_high),
            'Medium Low (2nd Quartile)': len(medium_low),
            'Low (1st Quartile)': len(low),
        }
    }

    if len(low) > 0:
        summary['Alert'] = f"{len(low)} inverter(s) are in the LOW performance group. REVIEW IS NECESSARY."

    max_row = res_df.loc[res_df['AC_POWER_FIXED'].idxmax()]
    min_row = res_df.loc[res_df['AC_POWER_FIXED'].idxmin()]
    max_val = res_df['AC_POWER_FIXED'].max()
    min_val = res_df['AC_POWER_FIXED'].min()

    summary['Highest_Output_Inverter'] = {
        'Inverter': f"S{int(max_row['SOURCE_ID_NUMBER'])}",
        'Average Output': f"{max_val:.2f} kW"
    }

    summary['Lowest_Output_Inverter'] = {
        'Inverter': f"S{int(min_row['SOURCE_ID_NUMBER'])}",
        'Average Output': f"{min_val:.2f} kW"
    }

    return high, medium_high, medium_low, low, summary

# ---------- Advanced Plotly Efficiency Anomaly ----------
def anomaly_detect(df):
    df_clean = df[df['EFFICIENCY_%'] > 0].copy()
    df_clean['TIME_STAMP'] = pd.to_datetime(df_clean['TIME_STAMP'])
    df_clean = df_clean.sort_values(['SOURCE_ID', 'TIME_STAMP'])

    full_range = pd.date_range(start=df_clean['TIME_STAMP'].min(), end=df_clean['TIME_STAMP'].max(), freq='15T')
    all_data = []
    inverter_list = sorted(df_clean['SOURCE_ID'].unique())

    for inv in inverter_list:
        inv_df = df_clean[df_clean['SOURCE_ID'] == inv].copy()
        inv_df = inv_df.set_index('TIME_STAMP')
        inv_df = inv_df.reindex(full_range)
        inv_df['SOURCE_ID'] = inv
        inv_df = inv_df.rename_axis('TIME_STAMP').reset_index()

        inv_df['anomaly'] = False
        mask = inv_df['EFFICIENCY_%'] > 0
        if mask.sum() > 10:
            model = IsolationForest(contamination=0.01, random_state=42)
            inv_df.loc[mask, 'anomaly'] = model.fit_predict(inv_df.loc[mask, ['EFFICIENCY_%']]) == -1

        all_data.append(inv_df)

    final_df = pd.concat(all_data)
    final_df['Status'] = final_df['anomaly'].map({True: 'Anomaly', False: 'Normal'})
    final_df.to_csv("Anomaly_Data.csv", index=False)

    fig = go.Figure()
    dropdowns = []

    for i, inv in enumerate(inverter_list):
        temp = final_df[final_df['SOURCE_ID'] == inv]
        fig.add_trace(go.Scatter(x=temp['TIME_STAMP'], y=temp['EFFICIENCY_%'], mode='lines', name=f'{inv} Efficiency', visible=(i == 0), line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=temp[temp['anomaly']]['TIME_STAMP'], y=temp[temp['anomaly']]['EFFICIENCY_%'], mode='markers', name=f'{inv} Anomaly', visible=(i == 0), marker=dict(color='red', size=6)))

        vis_flags = [False] * (2 * len(inverter_list))
        vis_flags[2 * i] = True
        vis_flags[2 * i + 1] = True

        dropdowns.append(dict(label=inv, method='update', args=[{'visible': vis_flags}, {'title': f'Efficiency for {inv}'}]))

    fig.update_layout(updatemenus=[dict(active=0, buttons=dropdowns, x=1.05, xanchor='left', y=1.15, yanchor='top')],
                      title=f'Efficiency for {inverter_list[0]}',
                      xaxis_title='Timestamp',
                      yaxis_title='Efficiency (%)',
                      height=600,
                      template='plotly_white')
    return fig

# ---------- Streamlit App ----------
st.set_page_config(page_title="EnergyAI", layout="wide")

st.sidebar.markdown("### ⚡ EnergyAI")
st.sidebar.info(
    """
    AI-driven energy monitoring designed for solar, wind, and other renewable systems.
    Detect anomalies, forecast output, and optimize efficiency — all in one place.
    Upload your CSV file to get started.
    """
)

uploaded_file = st.sidebar.file_uploader("Upload Your Energy Data (CSV)", type=["csv"])

tab = st.sidebar.radio("📌 Navigation", ["Home", "Energy & Summary", "Anomalies & Groupings", "Forecasting"])

if tab == "Home":
    st.title("⚡ EnergyAI — Smarter Energy Monitoring")
    st.markdown(
        """
        Welcome to **EnergyAI** — your AI-powered assistant for smarter energy management.
        
        We help energy developers and producers visualize, analyze, and forecast energy system performance
        with easy-to-use tools tailored for renewable and traditional energy sources.
        
        Upload your energy data from solar, wind, or other projects and explore:
        - Energy output over time  
        - AI-driven anomaly detection  
        - Efficiency groupings and inverter performance  
        - Forecasting future power output  
        
        Our goal is to help you make faster, data-driven decisions to optimize your energy assets.
        """
    )
    st.markdown("---")
    st.markdown(
        """
        **Features**  
        - Automated anomaly detection with clear alerts  
        - Visual and statistical summaries of energy output  
        - Interactive efficiency charts by inverter or source  
        - Forecasting models to predict future production  
        - Export alerts for integration with your workflows  
        """
    )
    st.markdown("---")
    st.markdown("### What Our Users Are Saying")
    st.markdown(
        """
        Power producers and energy analysts across the U.S. rely on EnergyAI to optimize operations:

        > “EnergyAI gave us back control of our workflows. What used to take days to sort through now gets done in one sitting.”
        >
        > — Project Manager, Regional Solar Operator

        > “We’ve been able to flag issues much earlier thanks to the anomaly detection tools. It's like having a second pair of eyes on everything.”
        >
        > — Reliability Engineer, Utility-Scale Wind Developer

        > “Our team was buried in data before. With EnergyAI, we get the insights we need instantly — it’s changed how we plan and scale projects.”
        >
        > — VP of Operations, Independent Energy Investment Group
        """
    )

elif tab == "Energy & Summary":
    st.title("Energy & Summary")
    if uploaded_file:
        df = load_clean_data(uploaded_file)
        if not df.empty:
            df = detect_anomalies(df)

            st.subheader("Energy Over Time")
            st.line_chart(df.set_index("TIME_STAMP")["Value"])

            st.subheader("AI Summary")
            st.markdown(generate_summary(df))
        else:
            st.warning("Uploaded file is empty or couldn't be processed.")
    else:
        st.info("Please upload a CSV file to continue.")

elif tab == "Anomalies & Groupings":
    st.title("Anomalies & Groupings")
    if uploaded_file:
        df = load_clean_data(uploaded_file)
        if not df.empty:
            df = detect_anomalies(df)

            if 'EFFICIENCY_%' in df.columns and 'TIME_STAMP' in df.columns:
                st.subheader("Inverter Efficiency Anomaly Detector")
                fig_plotly = anomaly_detect(df)
                st.plotly_chart(fig_plotly, use_container_width=True)
            else:
                st.info("Inverter efficiency data not detected. Skipping efficiency anomaly graph.")

            st.subheader("Detected Anomalies")
            anomaly_df = df[df["anomaly"]].copy()
            anomaly_df['TIME_STAMP'] = pd.to_datetime(anomaly_df['TIME_STAMP'], errors='coerce')
            anomaly_df['Formatted Time'] = anomaly_df['TIME_STAMP'].dt.strftime('%Y-%m-%d %H:%M')
            if not anomaly_df.empty:
                with st.expander(f"🔍 View {len(anomaly_df)} Anomalies"):
                    st.dataframe(anomaly_df[["Formatted Time", "Value"]].rename(columns={
                        "Formatted Time": "Timestamp",
                        "Value": "Anomalous Reading"
                    }))
            else:
                st.info("No anomalies detected.")

            if 'EFFICIENCY_%' in df.columns and 'SOURCE_ID' in df.columns:
                high, medium_high, medium_low, low, summary = grouping_data_with_summary(df)

                st.subheader("Inverter Average AC Power Output Groupings")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🔋 High Efficient Inverters (4th Quarter)")
                    st.dataframe(high)

                    st.markdown("#### 🔆 Low Medium Efficient Inverters (2nd Quarter)")
                    st.dataframe(medium_low)

                with col2:
                    st.markdown("#### ⚡ Medium High Efficient Inverters (3rd Quarter)")
                    st.dataframe(medium_high)

                    st.markdown("#### 🪫 Low Efficient Inverters (1st Quarter)")
                    st.dataframe(low)

                st.markdown("### 📊 Inverter Performance Summary")
                st.json(summary)

            st.subheader("Export Anomaly Alerts for Zapier")
            if st.button("Export Alerts to CSV"):
                export_alerts(df)
        else:
            st.warning("Uploaded file is empty or couldn't be processed.")
    else:
        st.info("Please upload a CSV file to continue.")

elif tab == "Forecasting":
    st.title("Forecasting")
    if uploaded_file:
        df = load_clean_data(uploaded_file)
        if not df.empty:
            df = detect_anomalies(df)

            st.subheader("Forecasting with Linear Regression")
            forecast_df, mse = run_forecast(df)
            st.write(f"Mean Squared Error: **{mse:.2f}**")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(forecast_df["time_index"], forecast_df["Actual"], label="Actual", alpha=0.6)
            ax2.plot(forecast_df["time_index"], forecast_df["Predicted"], label="Predicted", alpha=0.8)
            ax2.set_title("Forecasted vs Actual AC Power Output")
            ax2.set_xlabel("Time Index")
            ax2.set_ylabel("AC Power (kW)")
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.warning("Uploaded file is empty or couldn't be processed.")
    else:
        st.info("Please upload a CSV file to continue.")
