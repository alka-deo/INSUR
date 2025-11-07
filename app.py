
import streamlit as st
import pandas as pd
import numpy as np
import io
from helpers import smart_detect_columns, preprocess_for_modeling, train_models, evaluate_model, cross_val_scores
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import base64
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide", page_title="Insurance / Attrition Dashboard")

st.title("Insurance / Attrition Analytics Dashboard")

@st.cache_data
def load_default_data():
    return pd.read_csv("Insurance.csv")

df = load_default_data()

# Sidebar - global filters
st.sidebar.header("Global Filters")
job_col, sat_col = smart_detect_columns(df)
if job_col:
    job_options = st.sidebar.multiselect(f"Filter by {job_col}", options=sorted(df[job_col].dropna().unique()), default=sorted(df[job_col].dropna().unique())[:5])
else:
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    job_col = cat_cols[0] if cat_cols else None
    job_options = st.sidebar.multiselect(f"Filter by {job_col}", options=sorted(df[job_col].dropna().unique()) if job_col else [], default=sorted(df[job_col].dropna().unique())[:5] if job_col else [])

# satisfaction slider fallback
if sat_col:
    try:
        minv = float(df[sat_col].min())
        maxv = float(df[sat_col].max())
        sat_range = st.sidebar.slider(f"{sat_col} range", min_value=minv, max_value=maxv, value=(minv, maxv))
    except Exception:
        sat_range = None
else:
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    fallback = 'PI_AGE' if 'PI_AGE' in num_cols else (num_cols[0] if num_cols else None)
    sat_col = fallback
    if sat_col:
        minv = float(df[sat_col].min())
        maxv = float(df[sat_col].max())
        sat_range = st.sidebar.slider(f"{sat_col} range", min_value=minv, max_value=maxv, value=(minv, maxv))
    else:
        sat_range = None

# apply filters
df_filtered = df.copy()
if job_col and job_options:
    df_filtered = df_filtered[df_filtered[job_col].isin(job_options)]
if sat_col and sat_range is not None:
    df_filtered = df_filtered[df_filtered[sat_col].between(sat_range[0], sat_range[1])]

# Main tabs
tabs = st.tabs(["Dashboard", "Modeling", "Predict & Upload", "About / Help"])
with tabs[0]:
    st.header("Interactive Dashboard - Insights for HR / Policy Decisions")
    st.markdown("Use the filters (left) to refine the charts. Charts are interactive where possible.")
    # Chart 1: Policy Status distribution (donut) by gender or job role
    st.subheader("1) Policy Status Distribution")
    if 'POLICY_STATUS' in df_filtered.columns:
        status_counts = df_filtered['POLICY_STATUS'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')
        st.pyplot(fig1)
    else:
        st.info("POLICY_STATUS column missing.")

    # Chart 2: Sum assured vs Annual income scatter with trend and size by age
    st.subheader("2) Sum Assured vs Annual Income (risk/affordability view)")
    if {'SUM_ASSURED','PI_ANNUAL_INCOME','PI_AGE'}.issubset(df_filtered.columns):
        fig2, ax2 = plt.subplots(figsize=(8,4))
        sc = ax2.scatter(df_filtered['PI_ANNUAL_INCOME'], df_filtered['SUM_ASSURED'],
                         s=np.clip(df_filtered['PI_AGE'],10,60), alpha=0.6)
        ax2.set_xlabel("Annual Income")
        ax2.set_ylabel("Sum Assured")
        ax2.set_title("Scatter: Income vs Sum Assured (marker size ~ Age)")
        st.pyplot(fig2)
    else:
        st.info("Required columns for this chart are missing in dataset.")

    # Chart 3: Claim Reason vs Policy Status stacked bar (complex insight)
    st.subheader("3) Reason for Claim vs Policy Status (Stacked)")
    if 'REASON_FOR_CLAIM' in df_filtered.columns:
        ctab = pd.crosstab(df_filtered['REASON_FOR_CLAIM'], df_filtered['POLICY_STATUS'])
        fig3 = ctab.plot(kind='bar', stacked=True, figsize=(10,4)).get_figure()
        st.pyplot(fig3)
    else:
        st.info("REASON_FOR_CLAIM column missing.")

    # Chart 4: Age distribution by Policy Status (hist + KDE)
    st.subheader("4) Age Distribution by Policy Status")
    if 'PI_AGE' in df_filtered.columns:
        fig4, ax4 = plt.subplots()
        for status in df_filtered['POLICY_STATUS'].unique():
            subset = df_filtered[df_filtered['POLICY_STATUS']==status]
            ax4.hist(subset['PI_AGE'], bins=15, alpha=0.5, label=str(status), density=True)
        ax4.set_xlabel("Age")
        ax4.set_ylabel("Density")
        ax4.legend()
        st.pyplot(fig4)
    else:
        st.info("PI_AGE missing.")

    # Chart 5: Top 10 features correlated with rejected policies (complex insight)
    st.subheader("5) Top 10 Correlations with a target (numerical columns)")
    try:
        df_corr = df_filtered.copy()
        if 'Repudiate' in ' '.join(df_corr['POLICY_STATUS'].unique()):
            df_corr['target_bin'] = df_corr['POLICY_STATUS'].apply(lambda x: 1 if 'Repudiate' in x else 0)
        else:
            df_corr['target_bin'] = (df_corr['POLICY_STATUS'] != df_corr['POLICY_STATUS'].mode().iloc[0]).astype(int)
        num_cols = df_corr.select_dtypes(include=['number']).columns.tolist()
        if 'target_bin' in num_cols:
            num_cols.remove('target_bin')
        corrs = {}
        for c in num_cols:
            corrs[c] = df_corr[c].corr(df_corr['target_bin'])
        corr_series = pd.Series(corrs).abs().sort_values(ascending=False).head(10)
        fig5, ax5 = plt.subplots(figsize=(8,4))
        corr_series[::-1].plot(kind='barh', ax=ax5)
        ax5.set_title("Top 10 absolute correlations with repudiation/target")
        st.pyplot(fig5)
    except Exception as e:
        st.write("Could not compute correlations:", e)

with tabs[1]:
    st.header("Modeling: Train and compare DT / RF / GBRT")
    st.markdown("Select target column (default POLICY_STATUS) and click 'Run Models' to train with cv=5 and see metrics.")
    target_col = st.selectbox("Select target column", options=df.columns.tolist(), index=df.columns.get_loc("POLICY_STATUS") if "POLICY_STATUS" in df.columns else 0)
    test_size = st.slider("Test set fraction", 0.1, 0.4, 0.2)
    run_button = st.button("Run Models (cv=5)")
    if run_button:
        st.info("Preprocessing (imputation, encoding) and training started...")
        X, y, le = preprocess_for_modeling(df_filtered, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        models = train_models(X_train, y_train)
        results = {}
        fig_roc, axroc = plt.subplots(figsize=(7,6))
        for name, model in models.items():
            metrics = evaluate_model(model, X_test, y_test)
            cv_mean, cv_std = cross_val_scores(model, X_train, y_train, cv=5)
            results[name] = {
                "train_acc": accuracy_score(y_train, model.predict(X_train)),
                "test_acc": metrics['accuracy'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "f1": metrics['f1'],
                "auc": metrics['auc'],
                "cv_mean": cv_mean,
                "cv_std": cv_std
            }
            if metrics.get('roc_curve') is not None:
                fpr, tpr = metrics['roc_curve']
                axroc.plot(fpr, tpr, label=f"{name} (AUC={metrics['auc']:.3f})")
        axroc.plot([0,1],[0,1],'k--', alpha=0.5)
        axroc.set_xlabel("False Positive Rate"); axroc.set_ylabel("True Positive Rate"); axroc.set_title("ROC Comparison")
        axroc.legend()
        st.pyplot(fig_roc)

        for name, model in models.items():
            metrics = evaluate_model(model, X_test, y_test)
            st.subheader(name)
            st.write("Train acc:", results[name]['train_acc'], "Test acc:", results[name]['test_acc'])
            cm = metrics['confusion_matrix']
            figcm, axcm = plt.subplots()
            im = axcm.imshow(cm, interpolation='nearest', cmap='Blues')
            axcm.figure.colorbar(im, ax=axcm)
            axcm.set_title(f"{name} Confusion Matrix")
            axcm.set_xlabel("Predicted"); axcm.set_ylabel("True")
            st.pyplot(figcm)
            if metrics.get('feature_importances') is not None:
                fi = metrics['feature_importances']
                feat_series = pd.Series(fi, index=X.columns).sort_values(ascending=False).head(20)
                figfi, axfi = plt.subplots(figsize=(6, max(3,0.25*len(feat_series))))
                feat_series[::-1].plot(kind='barh', ax=axfi)
                axfi.set_title(f"{name} - Top feature importances")
                st.pyplot(figfi)

        resdf = pd.DataFrame(results).T
        st.write("Summary table (train/test acc, precision, recall, f1, auc, cv mean/std):")
        st.dataframe(resdf)
        st.session_state['models'] = models
        st.session_state['preprocess_target'] = target_col
        st.session_state['label_encoder'] = le
        st.success("Models trained and saved to session. You can now use the Predict & Upload tab.")

with tabs[2]:
    st.header("Predict on new data / Upload CSV")
    uploaded = st.file_uploader("Upload CSV for prediction (columns must match training features or at least contain the raw columns)", type=['csv'])
    if uploaded is not None:
        newdf = pd.read_csv(uploaded)
        st.write("Uploaded data (first 5 rows):")
        st.dataframe(newdf.head())
        if 'models' not in st.session_state:
            st.warning("No trained models in session. Please train models in the Modeling tab first (or reload app after training).")
        else:
            if st.button("Run prediction (using Random Forest)"):
                target = st.session_state.get('preprocess_target', 'POLICY_STATUS')
                try:
                    X_new, _, _ = preprocess_for_modeling(pd.concat([newdf, pd.DataFrame({target: [np.nan]*len(newdf)})], axis=1), target)
                    trained_model = st.session_state['models']['Random Forest']
                    if hasattr(trained_model, 'feature_names_in_'):
                        X_new = X_new.reindex(columns=trained_model.feature_names_in_, fill_value=0)
                    preds = trained_model.predict(X_new)
                    le = st.session_state.get('label_encoder', None)
                    if le is not None:
                        pred_labels = le.inverse_transform(preds)
                    else:
                        pred_labels = preds
                    newdf['Predicted_'+target] = pred_labels
                    st.write("Predictions (first 10 rows):")
                    st.dataframe(newdf.head(10))
                    csv = newdf.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download predictions.csv</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error("Prediction failed: " + str(e))

with tabs[3]:
    st.header("About & Help")
    st.markdown(\"\"\"
    **What this app does**
    - Interactive dashboard with 5 charts and filters.
    - Train three models (DT, RF, GBRT) with cv=5 and view confusion matrices, ROC, AUC, feature importances.
    - Upload new CSV and predict (Random Forest used by default for prediction).

    **Notes & tips**
    - The app applies median imputation for numeric columns and mode for categoricals.
    - If your dataset column names differ (e.g., Job Role or Satisfaction), the app attempts to auto-detect similar column names and falls back to sensible defaults.
    - For best results, upload a CSV with the same raw columns used in training (not only one-hot encoded).
    \"\"\")
