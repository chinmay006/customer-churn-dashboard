import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import time

# Page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

# Animate.css
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/>
""", unsafe_allow_html=True)

# CSS Styling
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #f7f8fc, #e3e8f0);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main-header {
    font-size: 2.8rem;
    color: #2c3e50;
    text-align: center;
    font-weight: bold;
    animation: fadeIn 2s ease-in-out;
}
.sub-header {
    font-size: 1.6rem;
    color: #34495e;
    font-weight: bold;
    border-bottom: 3px solid #2980b9;
    padding-bottom: 0.5rem;
    margin-top: 1.5rem;
}
.stButton>button {
    background: linear-gradient(90deg, #4CAF50, #45A049);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 25px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
    text-align: center;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 0.8rem;
    margin-top: 2rem;
}
.info-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px);}
    to { opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header animate__animated animate__fadeInDown">📊 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2345/2345563.png", width=100)
    st.markdown("## Navigation")
    st.markdown("""
    - Data Upload: Upload datasets and review structure
    - 📈 EDA: Explore data insights and trends
    - Model Training: Train and tune ML models
    - Predict & Visualize: Run predictions and analyze results
    """)
    st.markdown("## About")
    st.info("This app predicts customer churn with machine learning, enhanced with interactive visualizations and animations.")
    st.markdown("## Dataset Notes")
    st.warning("""
    - Target in last column
    - CSV format with headers
    - Preprocess categorical fields before upload
    """)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Upload", "📈 EDA", "🧠 Model Training", "📊 Predict & Visualize"])

# --- TAB 1: Data Upload ---
with tab1:
    st.markdown('<h2 class="sub-header animate__animated animate__fadeIn">Upload Your Dataset Files</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader("Upload Train.csv", type="csv")
    with col2:
        test_file = st.file_uploader("Upload Test.csv", type="csv")

    if train_file and test_file:
        with st.spinner("Reading datasets..."):
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            st.session_state['train_df'] = train_df
            st.session_state['test_df'] = test_df
            time.sleep(1)
        st.success("✅ Datasets uploaded successfully!")

        st.markdown("### Overview of Training Data")
        st.dataframe(train_df.head(10), use_container_width=True)
        st.markdown(f"Shape: {train_df.shape}")
        st.markdown("#### Target Distribution")
        target_col = train_df.columns[-1]
        if target_col in train_df.columns:
            dist = train_df[target_col].value_counts()
            fig = px.pie(values=dist.values, names=dist.index, title="Target Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Overview of Test Data")
        st.dataframe(test_df.head(10), use_container_width=True)
        st.markdown(f"Shape: {test_df.shape}")

        st.markdown("### Dataset Guidelines")
        st.info("""
        ✅ Make sure:
        - Target column is in the last column
        - Dataset is in CSV format with headers
        - Categorical columns are encoded before uploading (or handled later)
        - Missing values are minimal; median imputation will be applied where necessary
        """)

    elif train_file or test_file:
        st.warning("⚠ Please upload both training and test datasets.")

# --- TAB 2: EDA ---
with tab2:
    st.markdown('<h2 class="sub-header animate__animated animate__fadeIn">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    if 'train_df' not in st.session_state:
        st.warning("Please upload datasets first!")
    else:
        df = st.session_state['train_df']
        st.markdown("### Correlation Heatmap")
        corr = df.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("### Feature Distributions")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected = st.selectbox("Select Feature", numeric_cols)
        fig_dist = px.histogram(df, x=selected, nbins=30, title=f"Distribution of {selected}")
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("### Churn Trends")
        target_col = df.columns[-1]
        churn_counts = df[target_col].value_counts()
        fig_bar = px.bar(churn_counts, x=churn_counts.index, y=churn_counts.values, labels={'x':'Churn', 'y':'Count'},
                         title="Churn Frequency")
        st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 3: Model Training ---
with tab3:
    st.markdown('<h2 class="sub-header animate__animated animate__fadeIn">Train a Machine Learning Model</h2>', unsafe_allow_html=True)
    if 'train_df' not in st.session_state:
        st.warning("Please upload dataset first!")
    else:
        model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "Gradient Boosting",
                                                     "SVM", "KNN", "LightGBM", "Naive Bayes"])
        st.markdown("### Hyperparameters")
        if model_choice in ["Random Forest", "Gradient Boosting", "LightGBM"]:
            n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
        else:
            n_estimators = 100
        if model_choice in ["Random Forest", "Gradient Boosting"]:
            max_depth = st.slider("Max Depth", 3, 20, 5)
        else:
            max_depth = None
        if model_choice == "SVM":
            C_val = st.number_input("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
        else:
            C_val = 1.0

        if st.button("🚀 Train Model"):
            df = st.session_state['train_df']
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            imputer = SimpleImputer(strategy='median')
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif model_choice == "Gradient Boosting":
                model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif model_choice == "SVM":
                model = SVC(probability=True, C=C_val, random_state=42)
            elif model_choice == "KNN":
                model = KNeighborsClassifier()
            elif model_choice == "LightGBM":
                model = lgb.LGBMClassifier(n_estimators=n_estimators)
            else:
                model = GaussianNB()

            progress = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
            for pct in range(0, 101, 10):
                time.sleep(0.2)
                progress.progress(pct)
                status_text.text(f"Training... {pct}%")
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            progress.empty()
            status_text.empty()

            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)

            st.session_state.update({
                'model': model,
                'scaler': scaler,
                'imputer': imputer,
                'acc': acc,
                'f1': f1,
                'model_name': model_choice,
                'training_time': training_time
            })

            st.success(f"✅ Model trained in {training_time:.2f} seconds!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{acc:.4f}")
            with col2:
                st.metric("F1 Score", f"{f1:.4f}")
            with col3:
                st.metric("Training Time", f"{training_time:.2f}s")

# --- TAB 4: Predict & Visualize ---
with tab4:
    st.markdown('<h2 class="sub-header animate__animated animate__fadeIn">Predictions and Insights</h2>', unsafe_allow_html=True)
    if 'model' not in st.session_state or 'test_df' not in st.session_state:
        st.warning("Please upload and train the model first!")
    else:
        if st.button("📊 Generate Predictions"):
            with st.spinner("Making predictions..."):
                model = st.session_state['model']
                scaler = st.session_state['scaler']
                imputer = st.session_state['imputer']
                test_df = st.session_state['test_df']

                X_test = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)
                X_test_scaled = scaler.transform(X_test)

                preds = model.predict(X_test_scaled)
                probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else [0]*len(preds)

                results = pd.DataFrame({
                    "Index": test_df.index,
                    "Prediction": preds,
                    "Churn Probability": probs
                })

                churn_rate = results['Prediction'].mean() * 100

                st.success(f"✅ Predictions completed! Estimated churn rate: {churn_rate:.2f}%")

                st.markdown("### Prediction Results")
                st.dataframe(results.head(10), use_container_width=True)

                st.markdown("### Churn Distribution")
                pred_counts = results['Prediction'].value_counts().sort_index()
                fig_pie = px.pie(values=pred_counts.values,
                                 names=['Not Churn', 'Churn'],
                                 title="Churn Distribution",
                                 color_discrete_sequence=['lightgreen', 'tomato'])
                st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("### Churn Probability Distribution")
                fig_hist = px.histogram(results, x="Churn Probability", nbins=20,
                                        title="Churn Probability Histogram")
                fig_hist.update_traces(marker_color='lightcoral')
                st.plotly_chart(fig_hist, use_container_width=True)

                csv = results.to_csv(index=False)
                st.download_button("📥 Download CSV", data=csv, file_name="churn_predictions.csv")

# Footer once at the end
st.markdown("<hr>")
st.markdown('<p class="footer">Customer Churn Predictor • Enhanced with EDA, visualizations, and animations • Created for learning and exploration</p>', unsafe_allow_html=True)
