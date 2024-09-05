
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
from scipy.stats.mstats import winsorize

# Title of the app
st.title('Student Performance Data Analysis')

# File uploader
uploaded_file = st.file_uploader("Upload CSV file here", type="csv")

if uploaded_file is not None:
    # 1. Load the data
    df = pd.read_csv(uploaded_file, delimiter=';')

    # Sidebar filters
    st.sidebar.header('Filter the Data')

    # Filter by gender (sex)
    gender = st.sidebar.multiselect('Select Gender', df['sex'].unique(), default=df['sex'].unique())

    # Filter by mother education level (Medu)
    medu = st.sidebar.multiselect('Select Mother Education Level (Medu)', df['Medu'].unique(), default=df['Medu'].unique())

    # Filter by father education level (Fedu)
    fedu = st.sidebar.multiselect('Select Father Education Level (Fedu)', df['Fedu'].unique(), default=df['Fedu'].unique())

    # Apply filters to dataset
    df_filtered = df[
        (df['sex'].isin(gender)) &
        (df['Medu'].isin(medu)) &
        (df['Fedu'].isin(fedu))
    ]

    # Layout with tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Summary Stats", "Visualizations", "Outlier Detection"])

    # Data Preview
    with tab1:
        st.subheader('Filtered Data Preview')
        st.write(df_filtered.head())

    # 2. Data Summary - Generate descriptive statistics
    with tab2:
        st.subheader('Summary Statistics')
        st.write(df_filtered.describe())

        # Data Information
        st.subheader('Data Info')
        buffer = io.StringIO()
        df_filtered.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    # 3. Data Visualization
    with tab3:
        st.subheader('Data Visualizations')

        # Columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Histograms')
            num_cols = df_filtered.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                fig, ax = plt.subplots()
                df_filtered[col].hist(ax=ax, bins=20)
                ax.set_title(f'Histogram of {col}')
                st.pyplot(fig)
                plt.close(fig)

        with col2:
            st.subheader('Density Plots')
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.kdeplot(df_filtered[col], ax=ax, fill=True)
                ax.set_title(f'Density Plot of {col}')
                st.pyplot(fig)
                plt.close(fig)

        # Box and Whisker Plots
        st.subheader('Box and Whisker Plots')
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=df_filtered[col], ax=ax)
            ax.set_title(f'Box and Whisker Plot of {col}')
            st.pyplot(fig)
            plt.close(fig)

        # Correlation Heatmap
        st.subheader('Correlation Heatmap')
        numeric_df = df_filtered.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig, ax = plt.subplots()
        num_cols_heatmap = len(corr.columns)
        font_size = min(10, 100 / num_cols_heatmap)  # Adjust dynamically
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, annot_kws={"size": font_size}, fmt='.2f', linewidths=0.5)
        st.pyplot(fig)
        plt.close(fig)

    # 4. Outlier Detection and Handling
    with tab4:
        st.subheader('Outlier Detection using Z-Score')

        # Identifying Outliers using Z-Score
        z_scores = np.abs(stats.zscore(df_filtered.select_dtypes(include=[np.number])))
        outliers_zscore = (z_scores > 3).sum(axis=0)
        st.write('Number of outliers detected using Z-Score:')
        st.write(outliers_zscore)

        # Handling Outliers - Removing outliers
        st.subheader('Handling Outliers')
        df_no_outliers = df_filtered[~((z_scores > 3).any(axis=1))]
        st.write('Data shape before removing outliers:', df_filtered.shape)
        st.write('Data shape after removing outliers:', df_no_outliers.shape)

        # Log Transformation (handling only positive values)
        st.write('Applying log transformation...')
        df_transformed = df_filtered.copy()
        for col in num_cols:
            if (df_transformed[col] > 0).all():  # Apply log1p only if values are positive
                df_transformed[col] = np.log1p(df_transformed[col])
        st.write('Data after log transformation:')
        st.write(df_transformed.head())

        # Winsorizing
        st.write('Winsorizing...')
        df_winsorized = df_filtered.copy()
        for col in num_cols:
            df_winsorized[col] = winsorize(df_winsorized[col], limits=[0.05, 0.05])
        st.write('Data after winsorizing:')
        st.write(df_winsorized.head())

        # Data Summary after winsorizing
        st.subheader('Summary Statistics (Winsorized Data)')
        st.write(df_winsorized.describe())

        st.subheader('Data Info (Winsorized Data)')
        buffer = io.StringIO()
        df_winsorized.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    # 5. Interactive Scatter Plot using Plotly
    st.subheader('Interactive Scatter Plot (Plotly)')
    col1 = st.selectbox('Select X-axis', num_cols)
    col2 = st.selectbox('Select Y-axis', num_cols, index=1)

    if col1 and col2:
        fig = px.scatter(df_filtered, x=col1, y=col2, title=f'Scatter Plot of {col1} vs {col2}')
        st.plotly_chart(fig)
 
