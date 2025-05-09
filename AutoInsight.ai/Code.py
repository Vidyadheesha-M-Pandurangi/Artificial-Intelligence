import streamlit as st
import pandas as pd
import os
import base64
from fpdf import FPDF
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model,predict_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,classification_report,recall_score, precision_score, cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize empty dataframe
df = None

# Load dataset if it exists
if os.path.exists("dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)


# Sidebar
with st.sidebar:
    st.image("Logo.jpg")
    st.title("AutoInsight.ai")
    st.info("AUTO TRAIN YOUR DATASET WITHIN A FEW CLICKS.")
    choice = st.radio("Select the task", ["Upload", "Profiling", "Modeling", "Download"])
    st.info("Analyze --> Train --> View Insights --> Download")

# Upload section
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=['csv'])
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.success("File uploaded and saved successfully!")
        st.dataframe(df)
        if df is not None:
            st.subheader("🔧 Data Preprocessing & Cleaning")
            original_shape = df.shape
            duplicates = df.duplicated().sum()
            null_columns = df.columns[df.isnull().all()]
            null_values = df.isnull().sum().sum()

            # Remove duplicates
            df.drop_duplicates(inplace=True)

            # Drop fully null columns
            df.dropna(axis=1, how='all', inplace=True)

            # Fill missing values (basic strategy)
            for col in df.select_dtypes(include='number').columns:
                df[col].fillna(df[col].median(), inplace=True)

            for col in df.select_dtypes(include='object').columns:
                df[col].fillna(df[col].mode()[0], inplace=True)

            cleaned_shape = df.shape

            # Save the cleaned data
            df.to_csv("dataset.csv", index=False)
            if 'Unnamed: 0' in df.columns:
                df.drop(columns=['Unnamed: 0'], inplace=True)

            # Summary of cleaning
            st.success("✅ Data cleaned successfully!")
            st.markdown(f"**Original Shape:** {original_shape} → **Cleaned Shape:** {cleaned_shape}")
            st.markdown(f"- Removed **{duplicates}** duplicate rows")
            st.markdown(f"- Dropped **{len(null_columns)}** columns with all nulls")
            st.markdown(f"- Filled **{null_values}** missing values")
# Profiling section
if choice == "Profiling":
    if df is not None:
        st.title("Exploratory Data Analysis")
        profile = ydata_profiling.ProfileReport(df, explorative=True)
        profile.to_file("eda_report.html")
        st_profile_report(profile)
    else:
        st.warning("Please upload a dataset first!")

# Modeling section
elif choice == "Modeling":
    if df is not None:
        st.title("🔧 Model Building")

        # Ensure session_state keys exist
        if "model_trained" not in st.session_state:
            st.session_state.model_trained = False
        if "model_metrics" not in st.session_state:
            st.session_state.model_metrics = None

        chosen_target = st.selectbox("🎯 Select the Target Column", df.columns)

        if st.button("🚀 Run Modeling"):
            # Step 1: Check for rare classes
            value_counts = df[chosen_target].value_counts()
            rare_classes = value_counts[value_counts < 2]
            if not rare_classes.empty:
                st.error("❗ The following classes have less than 2 samples, which will cause an error during model training:")
                st.write(rare_classes)
                st.warning("Please choose a different target column or ensure each class has at least 2 samples.")
            else:
                # Step 2: PyCaret setup
                setup(df, target=chosen_target, verbose=False, session_id=123)

                # Step 3: Show setup config
                setup_df = pull()
                st.subheader("⚙️ Setup Configurations")
                st.dataframe(setup_df)

                # Step 4: Compare models
                best_model = compare_models()
                compare_df = pull()
                st.subheader("📊 Model Comparison")
                st.dataframe(compare_df)

                # Step 5: Save best model
                save_model(best_model, "trained_model")

                # Step 6: Show best model
                st.subheader("🏆 Best Performing Model")
                st.write(type(best_model).__name__)
                params_df = pd.DataFrame(best_model.get_params().items(), columns=["Parameter", "Value"])
                st.markdown("### 🔧 Model Parameters")
                st.dataframe(params_df)

                # Step 7: Predict and Evaluate
                predicted_df = predict_model(best_model)
                y_true = predicted_df[chosen_target]
                y_pred = predicted_df["prediction_label"]

                # Step 8: Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average="weighted")
                recall = recall_score(y_true, y_pred, average="weighted")
                precision = precision_score(y_true, y_pred, average="weighted")
                kappa = cohen_kappa_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)
                try:
                    auc = roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), average="macro", multi_class="ovr")
                except:
                    auc = None

                # Save metrics to session_state
                metrics_df = pd.DataFrame({
                    "Metric": ["Accuracy", "F1 Score", "AUC Score", "Recall", "Precision", "Kappa", "MCC"],
                    "Score": [accuracy, f1, auc if auc is not None else "N/A", recall, precision, kappa, mcc]
                })

                st.session_state.model_trained = True
                st.session_state.model_metrics = metrics_df

                # Step 9: Classification report
                report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
                report_df.to_csv("model_metrics.csv")

                # Step 10: Graphical visualization
                st.subheader("📈 Model Performance Metrics")
                st.dataframe(metrics_df)

                graph_df = metrics_df[metrics_df["Metric"].isin(["Accuracy", "F1 Score"])]
                if auc is not None:
                    graph_df = pd.concat([
                        graph_df,
                        pd.DataFrame([{"Metric": "AUC Score", "Score": auc}])
                    ], ignore_index=True)

                st.bar_chart(graph_df.set_index("Metric"))

                st.success("✅ Model trained and metrics saved!")
    else:
        st.warning("Please upload and clean the dataset first.")

# Download section - Replaced functionality
if choice == "Download":
    st.title("Download Reports")

    # Download EDA Report as PDF
    if os.path.exists("eda_report.html"):
        st.subheader("EDA Report (HTML)")
        
        # Convert HTML report to PDF using external tools (example assumes it's saved externally)
        # For demo, show the HTML option with download button
        with open("eda_report.html", "r", encoding='utf-8') as f:
            html_content = f.read()
            b64 = base64.b64encode(html_content.encode()).decode()
            href = f'<a href="data:file/html;base64,{b64}" download="EDA_Report.html">Download EDA Report as HTML</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("EDA report not found. Please run profiling first.")

    # Download Model Metrics
    if os.path.exists("model_metrics.csv"):
        st.subheader("📥 Model Comparison Metrics")
        if os.path.exists("model_metrics.csv"):
            with open("model_metrics.csv", "rb") as f:
                st.download_button(
                label="📊 Download Model Metrics (CSV)",
                data=f,
                file_name="model_metrics.csv",
                mime="text/csv" )
        else:
            st.warning("No model metrics to download. Run modeling first.")

# 🔽 Trained model download
    st.subheader("📥 Download Trained Model")
    if os.path.exists("trained_model.pkl"):
        with open("trained_model.pkl", "rb") as f:
            st.download_button("📦 Download Trained Model (.pkl)", f, file_name="trained_model.pkl")
    else:
        st.warning("❌ No trained model found. Please run modeling first.")
    
