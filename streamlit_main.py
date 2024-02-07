import warnings
from seffaflik.elektrik import tuketim
import seffaflik
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

def visual_analysis(df,start_date,end_date):
    df_vis = df.loc[(df["date"] >= start_date) & (df["date"] <= end_date)]
    fig = px.line(df_vis, x = "datetime",y = "Consumption", title = "Türkiye Elektrik Tüketim Verisi (MW)")
    st.plotly_chart(fig)
    
def visual_analysis_2(df,feature_col):
    vis_df = df.copy()
    vis_df[feature_col] = vis_df[feature_col].astype("category")
    fig = px.box(df, x = feature_col,y = "Consumption", title = "Feature - Target İlişkisi")
    st.plotly_chart(fig)

def train_linear_regression(df,feature_cols,target_col,categoric_columns,split_date):
    features_df = df[feature_cols]
    
    if "trend" in categoric_columns:
        categoric_columns.remove("trend")
    
    
    features_df[categoric_columns] = features_df[categoric_columns].astype("category")
    
    X = pd.get_dummies(features_df,columns = categoric_columns)
    y = df[target_col]
    
    split_datetime = pd.to_datetime(split_date)
    
    train_idx = df[df["datetime"]<split_datetime].index
    test_idx = df[df["datetime"]>=split_datetime].index
    
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    linear_reg = LinearRegression().fit(X_train,y_train)
    y_preds = linear_reg.predict(X_test)
    y_fit = linear_reg.predict(X_train)
    
    r2_score = linear_reg.score(X_train,y_train)
    
    # Create subplots
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=False, subplot_titles=[
        f"Actual vs Fit in Train Period (Before {split_date})",
        f"Actual vs Predicted in Test Period (After {split_date})"
    ])
    
    # Plot actual vs fit in train period
    fig.add_trace(
        go.Scatter(x=df.loc[train_idx, "datetime"], y=df.loc[train_idx, "Consumption"], name="Actual"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.loc[train_idx, "datetime"], y=y_fit, name="Fit" ,marker=dict(color="#000005")),
        row=1, col=1
    )
    #fig.update_layout(title_text=f"Actual vs Fit in Train Period (Before {split_date})", row=1, col=1)
    
    # Plot actual vs predicted in test period
    fig.add_trace(
        go.Scatter(x=df.loc[test_idx, "datetime"], y=df.loc[test_idx, "Consumption"], name="Actual",marker=dict(color="#000001")),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.loc[test_idx, "datetime"], y=y_preds, name="Predicted",marker=dict(color="#000005")),
        row=2, col=1
    )
    #fig.update_layout(title_text=f"Actual vs Predicted in Test Period (After {split_date})", row=2, col=1)
    
    # Update layout and show figure
    fig.update_layout(height=600, width=800,xaxis=dict(title="Date"),yaxis=dict(title="Consumption"))
    st.plotly_chart(fig)
    
    mape = mean_absolute_percentage_error(y_test,y_preds)*100
    st.write(f"Average Percentage Error in Test Period: {mape:.2f}%")
    st.write(f"R^2 Score: {r2_score:.2f}")

st.set_page_config(
    page_title="OKR Time Series Forecast",
    layout="wide",
)

df = pd.read_csv("data_consumption.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df["date"] = df["datetime"].dt.date
st.title("OKR Time Series Forecast")

tab1,tab2 = st.tabs(["Visual Analysis","Model Performance"])

with tab1:
    
    with st.columns(2)[0]:
        date_range = st.slider("Select Time Range",min_value = df["date"].min(),max_value = df["date"].max(),value = (df["date"].min(),df["date"].max()))
    
    visual_analysis(df,date_range[0],date_range[1])
    
    with st.columns(2)[0]:
        feature_selector = st.selectbox("Select Feature",("year","quarter","month","week","day","hour"),index = 0)
    visual_analysis_2(df,feature_selector)
    
with tab2:
    with st.columns(2)[0]:
        multiple_features = st.multiselect("Select Features to Train the Model",["year","quarter","month","week","day","hour","trend"],default = "quarter")
    if st.button("Train the model"):
        if multiple_features:
            with st.spinner("Training the model"):
                train_linear_regression(df,multiple_features,"Consumption",multiple_features,"2023-12-01")
        
    
