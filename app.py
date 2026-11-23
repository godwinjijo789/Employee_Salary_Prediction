import io
import warnings
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SalaryScope",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXPECTED = ["Age", "Experience", "Department", "Education", "Gender", "Salary"]
CATEGORICAL_FEATURES = ["Department", "Education", "Gender"]
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "data" / "employee_salary_cleaned.csv"
CHART_COLORS = ["#2563EB", "#06B6D4", "#F59E0B", "#8B5CF6", "#EC4899", "#10B981", "#EF4444", "#6366F1"]
px.defaults.color_discrete_sequence = CHART_COLORS

st.markdown("""
<style>
    .brand-masthead {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: 0 0 .7rem;
        padding: .9rem 1.25rem;
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 18px;
        background: linear-gradient(110deg, #0B1B3A 0%, #1D4ED8 68%, #0891B2 100%);
        box-shadow: 0 10px 24px rgba(37, 99, 235, .16);
        color: #fff;
    }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: .85rem;
    }
    .brand-mark {
        display: flex;
        align-items: end;
        gap: 3px;
        height: 30px;
        padding: 6px;
        border: 1px solid rgba(255,255,255,.28);
        border-radius: 10px;
        background: rgba(255,255,255,.12);
    }
    .brand-mark span {
        display: block;
        width: 5px;
        border-radius: 4px;
        background: #67E8F9;
    }
    .brand-mark span:nth-child(1) { height: 10px; }
    .brand-mark span:nth-child(2) { height: 17px; }
    .brand-mark span:nth-child(3) { height: 13px; }
    .brand-name {
        color: #fff;
        font-size: 1.65rem;
        font-weight: 750;
        letter-spacing: -.02em;
        line-height: 1.2;
    }
    .brand-subtitle {
        margin-top: .18rem;
        color: rgba(255,255,255,.76);
        font-size: .78rem;
        letter-spacing: .04em;
    }
    .brand-status {
        padding: .42rem .7rem;
        border: 1px solid rgba(255,255,255,.24);
        border-radius: 999px;
        color: rgba(255,255,255,.9);
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .1em;
        white-space: nowrap;
    }
    .page-hero {
        padding: 1.7rem 1.9rem;
        margin: 0 0 1.15rem;
        border: 1px solid rgba(255,255,255,.2);
        border-radius: 22px;
        background: linear-gradient(118deg, #0B1B3A 0%, #2563EB 56%, #06B6D4 100%);
        box-shadow: 0 14px 34px rgba(37, 99, 235, .17);
        color: #fff;
    }
    .page-hero__eyebrow {
        margin-bottom: .45rem;
        color: rgba(255,255,255,.78);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .page-hero h1 {
        margin: 0;
        color: #fff;
        font-size: clamp(1.7rem, 3vw, 2.45rem);
        line-height: 1.14;
    }
    .page-hero p {
        max-width: 760px;
        margin: .65rem 0 0;
        color: rgba(255,255,255,.88);
        font-size: 1rem;
    }
    .feature-card {
        min-height: 152px;
        padding: 1.1rem;
        border: 1px solid rgba(59,130,246,.2);
        border-radius: 16px;
        background: linear-gradient(145deg, rgba(6,182,212,.12), rgba(37,99,235,.08) 58%, rgba(139,92,246,.06));
    }
    .feature-card__number {
        color: #0284C7;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .1em;
    }
    .feature-card h3 {
        margin: .5rem 0 .35rem;
        font-size: 1.05rem;
    }
    .feature-card p {
        margin: 0;
        font-size: .88rem;
        opacity: .8;
    }
    .insight-banner {
        margin: .75rem 0;
        padding: .9rem 1rem;
        border-left: 4px solid #06B6D4;
        border-radius: 0 12px 12px 0;
        background: linear-gradient(90deg, rgba(6,182,212,.12), rgba(37,99,235,.05));
    }
    .stButton button {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-height: 42px;
        font-size: 0.78rem;
        line-height: 1.1;
    }
    .stButton button p {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    @media (max-width: 640px) {
        .brand-masthead { padding: .8rem .9rem; }
        .brand-status { display: none; }
    }
</style>
""", unsafe_allow_html=True)

def money(v):
    if pd.isna(v):
        return "—"
    return f"₹{v:,.0f}"

def r2_value(v):
    if pd.isna(v):
        return "—"
    return f"{v:.5f}"

def snapshot_percentage(v):
    if pd.isna(v):
        return "—"
    return f"{v:.2f}%"

def r2_snapshot_percentage(v):
    if pd.isna(v):
        return "—"
    return f"{v * 100:.2f}%"

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

@st.cache_data(show_spinner="Loading employee dataset...")
def load_default_dataset(path, modified_at_ns):
    return normalize_columns(pd.read_csv(path))

def validate_dataset(df):
    missing = [c for c in EXPECTED if c not in df.columns]
    return missing

def select_page(page_name):
    st.session_state["active_dashboard_page"] = page_name

def get_feature_types(df, features):
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical

def make_preprocessor(numeric, categorical, scale_numeric=True):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric),
        ("cat", Pipeline(cat_steps), categorical),
    ], remainder="drop")

def train_regressors(df, target, features, test_size, random_state):
    work = df[features + [target]].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])
    X, y = work[features], work[target]
    if len(work) < 10:
        raise ValueError("At least 10 rows with a valid salary target are needed for model training.")
    if y.nunique() < 2:
        raise ValueError("Salary must contain at least two different numeric values.")
    numeric, categorical = get_feature_types(X, features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state, min_samples_leaf=2),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, random_state=random_state, min_samples_leaf=2, n_jobs=-1
        ),
    }
    fitted, rows, predictions = {}, [], {}
    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", make_preprocessor(numeric, categorical, scale_numeric=True)),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        predictions[name] = pred
        mse = mean_squared_error(y_test, pred)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R² Score": r2_score(y_test, pred),
        })
    return fitted, pd.DataFrame(rows), X_test, y_test, predictions, work

def run_clustering(df, features, k, random_state):
    X = df[features].copy()
    numeric, categorical = get_feature_types(X, features)
    prep = make_preprocessor(numeric, categorical, scale_numeric=True)
    transformed = prep.fit_transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(transformed)
    result = df.copy()
    result["Cluster"] = labels
    sil = None
    if k > 1 and len(result) > k and len(np.unique(labels)) > 1:
        # Sampling keeps silhouette computation responsive on larger datasets.
        if len(result) > 4000:
            sil = silhouette_score(transformed, labels, sample_size=4000, random_state=random_state)
        else:
            sil = silhouette_score(transformed, labels)
    return result, model, prep, sil

def is_dark_mode():
    theme_context = getattr(getattr(st, "context", None), "theme", None)
    return getattr(theme_context, "type", "light") == "dark"

def fig_theme(fig, title=None):
    dark_mode = is_dark_mode()
    grid_color = "rgba(148, 163, 184, 0.18)" if dark_mode else "rgba(100, 116, 139, 0.16)"
    fig.update_layout(
        title=title,
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CHART_COLORS,
        margin=dict(l=15, r=15, t=55 if title else 20, b=15),
        font=dict(family="Arial", size=12, color="#E8EEF8" if dark_mode else "#26384A"),
        legend_title_text="",
        hoverlabel=dict(bgcolor="#16243A" if dark_mode else "#FFFFFF", font_color="#E8EEF8" if dark_mode else "#172033"),
    )
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color, automargin=True)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color, automargin=True)
    return fig

def page_header(title, description, eyebrow="EMPLOYEE ANALYTICS"):
    st.markdown(
        f'<section class="page-hero"><div class="page-hero__eyebrow">{escape(eyebrow)}</div>'
        f'<h1>{escape(title)}</h1><p>{escape(description)}</p></section>',
        unsafe_allow_html=True,
    )

def numeric_range_slider(series, label, key, step):
    values = series.dropna()
    if values.empty:
        st.caption(f"{label}: no numeric values available.")
        return None
    lower, upper = float(values.min()), float(values.max())
    if lower == upper:
        st.caption(f"{label}: only {lower:.1f} is present.")
        return (lower, upper)
    return st.slider(
        label,
        min_value=lower,
        max_value=upper,
        value=(lower, upper),
        step=min(step, upper - lower),
        format="%.1f",
        key=key,
    )

# ---------- Sidebar ----------
st.sidebar.caption("Appearance")
st.sidebar.caption("Choose Light or Dark from the Theme controls in the top-right menu.")
with st.sidebar.expander("Dataset requirements"):
    st.caption("Target: Salary")
    st.caption("Features: Age, Experience, Department, Education, Gender")

navigation_pages = [
    "Dashboard",
    "Data Explorer",
    "Preprocessing",
    "Salary Predictor",
    "Pattern",
    "Model Evaluation",
    "Smart Insights",
    "About Project",
]
navigation_labels = {
    "Dashboard": ":material/dashboard: Dashboard",
    "Data Explorer": ":material/query_stats: Data Explorer",
    "Preprocessing": ":material/rule: Preprocessing",
    "Salary Predictor": ":material/payments: Salary Predictor",
    "Pattern": ":material/hub: Pattern",
    "Model Evaluation": ":material/monitoring: Models",
    "Smart Insights": ":material/lightbulb: Smart Insights",
    "About Project": ":material/menu_book: About",
}
if "active_dashboard_page" not in st.session_state:
    st.session_state["active_dashboard_page"] = st.session_state.get(
        "dashboard_navigation", "Dashboard"
    )
if st.session_state["active_dashboard_page"] == "Overview":
    st.session_state["active_dashboard_page"] = "Dashboard"
if st.session_state["active_dashboard_page"] not in navigation_pages:
    st.session_state["active_dashboard_page"] = "Dashboard"

st.markdown(
    '<div class="brand-masthead">'
    '<div class="brand-lockup">'
    '<span class="brand-mark"><span></span><span></span><span></span></span>'
    '<div><div class="brand-name">SalaryScope</div>'
    '<div class="brand-subtitle">People analytics · Salary intelligence</div></div>'
    '</div><div class="brand-status">WORKFORCE INSIGHTS</div></div>',
    unsafe_allow_html=True,
)
with st.container(border=True):
    nav_grid = st.columns(8, gap="small")
    for index, option in enumerate(navigation_pages):
        with nav_grid[index]:
            st.button(
                navigation_labels[option],
                key=f"top_navigation_{index}",
                type="primary" if st.session_state["active_dashboard_page"] == option else "secondary",
                width="stretch",
                wrap=False,
                on_click=select_page,
                args=(option,),
            )
page = st.session_state["active_dashboard_page"]

if not DEFAULT_DATASET_PATH.is_file():
    st.error(f"Employee dataset not found: `{DEFAULT_DATASET_PATH}`.")
    st.stop()

try:
    df = load_default_dataset(
        str(DEFAULT_DATASET_PATH),
        DEFAULT_DATASET_PATH.stat().st_mtime_ns,
    )
except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
    st.error(f"Could not load employee dataset `{DEFAULT_DATASET_PATH}`: {e}")
    st.stop()

st.session_state["raw_df"] = df

missing = validate_dataset(df)
if missing:
    st.error("The employee dataset is missing required columns: " + ", ".join(missing))
    st.write("Available columns:", list(df.columns))
    st.warning(f"Check the column names in `{DEFAULT_DATASET_PATH}`.")
    st.stop()

# Numeric coercion for expected numeric fields.
for col in ["Age", "Experience", "Salary"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

data_fingerprint = str(int(pd.util.hash_pandas_object(df, index=True).sum()))

# Model controls and cached-in-session results.
with st.sidebar.expander("Model settings", expanded=False):
    test_fraction = st.slider("Test data (%)", 10, 40, 20, 5) / 100
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    include_salary_cluster = st.checkbox("Include Salary in clustering", value=True,
        help="When enabled, clustering uses salary as one of the input dimensions, as in your earlier experiment.")
    run_button = st.button("Train / refresh analysis", type="primary", use_container_width=True)
cluster_k = 2

if "analysis_signature" not in st.session_state:
    st.session_state["analysis_signature"] = None

signature = (
    data_fingerprint, tuple(df.columns), test_fraction, cluster_k,
    int(random_state), include_salary_cluster,
)
if run_button or st.session_state.get("analysis_signature") != signature:
    try:
        with st.spinner("Training salary models and discovering employee clusters..."):
            features = ["Age", "Experience", "Department", "Education", "Gender"]
            fitted, metrics, X_test, y_test, preds, modelling_df = train_regressors(
                df, "Salary", features, test_fraction, int(random_state)
            )
            cluster_features = features + (["Salary"] if include_salary_cluster else [])
            cluster_source = df.dropna(subset=["Age", "Experience"]).copy()
            cluster_source = cluster_source.reset_index(drop=True)
            clustered, km, cluster_prep, sil = run_clustering(
                cluster_source, cluster_features, cluster_k, int(random_state)
            )
            st.session_state.update({
                "models": fitted, "metrics": metrics, "X_test": X_test, "y_test": y_test,
                "preds": preds, "modelling_df": modelling_df, "clustered": clustered,
                "cluster_model": km, "cluster_preprocessor": cluster_prep, "silhouette": sil,
                "cluster_features": cluster_features, "analysis_signature": signature,
            })
    except Exception as e:
        st.error(f"Analysis could not be completed: {e}")
        st.stop()

models = st.session_state["models"]
metrics = st.session_state["metrics"]
clustered = st.session_state["clustered"]
sil = st.session_state["silhouette"]
best_model_name = metrics.sort_values("MAE").iloc[0]["Model"]
best_model = models[best_model_name]

# ---------- Pages ----------
if page == "Dashboard":
    page_header(
        "Workforce dashboard",
        "A live snapshot of salary, team composition, and model performance.",
        f"DASHBOARD  /  {DEFAULT_DATASET_PATH.name}",
    )
    department_options = sorted(df["Department"].fillna("Missing").astype(str).unique().tolist())
    filter_col, view_col = st.columns([1.5, 1])
    selected_departments = filter_col.multiselect(
        "Departments",
        department_options,
        default=department_options,
        key=f"overview_departments_{data_fingerprint}",
    )
    salary_view = view_col.selectbox(
        "Salary chart",
        ["Salary distribution", "By department", "By experience", "3D salary landscape"],
        key="overview_salary_view",
    )
    overview_df = df[df["Department"].fillna("Missing").astype(str).isin(selected_departments)]
    valid_salary = df["Salary"].dropna()
    overview_salary = overview_df["Salary"].dropna()
    st.caption("KPI cards summarize the full employee dataset; the department filter below applies to the charts.")
    kpi_row_one = st.columns(4)
    kpi_row_one[0].metric(":material/groups: Total employees", f"{len(df):,}")
    kpi_row_one[1].metric(":material/payments: Average salary", money(valid_salary.mean()) if len(valid_salary) else "—")
    kpi_row_one[2].metric(":material/vertical_align_center: Median salary", money(valid_salary.median()) if len(valid_salary) else "—")
    kpi_row_one[3].metric(":material/arrow_upward: Maximum salary", money(valid_salary.max()) if len(valid_salary) else "—")
    kpi_row_two = st.columns(4)
    kpi_row_two[0].metric(":material/arrow_downward: Minimum salary", money(valid_salary.min()) if len(valid_salary) else "—")
    experience = df["Experience"].dropna()
    kpi_row_two[1].metric(":material/work_history: Average experience", f"{experience.mean():.1f} years" if len(experience) else "—")
    age = df["Age"].dropna()
    kpi_row_two[2].metric(":material/person: Average age", f"{age.mean():.1f} years" if len(age) else "—")
    kpi_row_two[3].metric(":material/apartment: Departments", f"{df['Department'].nunique(dropna=True):,}")
    st.subheader("Explore salary patterns")
    if overview_df.empty:
        st.info("Choose at least one department to show overview charts.")
    else:
        left, right = st.columns([1.35, 1])
        with left.container(border=True):
            st.markdown(f"**{salary_view}**")
            if salary_view == "Salary distribution":
                fig = px.histogram(
                    overview_df.dropna(subset=["Salary"]),
                    x="Salary",
                    nbins=30,
                    marginal="violin",
                    color_discrete_sequence=[CHART_COLORS[0]],
                    labels={"Salary": "Monthly salary (₹)"},
                )
                if not overview_salary.empty:
                    fig.add_vline(
                        x=overview_salary.mean(),
                        line_dash="dash",
                        line_color=CHART_COLORS[1],
                        annotation_text="Mean",
                    )
                    fig.add_vline(
                        x=overview_salary.median(),
                        line_dash="dot",
                        line_color=CHART_COLORS[2],
                        annotation_text="Median",
                    )
                fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
            elif salary_view == "By department":
                salary_by_department = (
                    overview_df.groupby("Department", dropna=False)["Salary"]
                    .mean().reset_index()
                ).dropna(subset=["Salary"]).sort_values("Salary")
                fig = px.bar(
                    salary_by_department,
                    x="Salary",
                    y="Department",
                    orientation="h",
                    color="Salary",
                    color_continuous_scale=["#BAE6FD", "#2563EB", "#0B1B3A"],
                    text_auto=".2s",
                    labels={"Salary": "Average monthly salary (₹)"},
                )
                fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
            elif salary_view == "By experience":
                fig = px.scatter(
                    overview_df.dropna(subset=["Experience", "Salary"]),
                    x="Experience",
                    y="Salary",
                    color="Department",
                    hover_data=["Age", "Education"],
                    opacity=0.78,
                    render_mode="webgl",
                    labels={
                        "Experience": "Experience (years)",
                        "Salary": "Monthly salary (₹)",
                    },
                )
                fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
            else:
                plot_df = overview_df.dropna(
                    subset=["Age", "Experience", "Salary"]
                )
                if len(plot_df) > 5000:
                    plot_df = plot_df.sample(5000, random_state=42)
                    st.caption("Showing a deterministic 5,000-record sample for responsive 3D rendering.")
                fig = px.scatter_3d(
                    plot_df,
                    x="Age",
                    y="Experience",
                    z="Salary",
                    color="Department",
                    hover_data=["Education", "Gender"],
                    opacity=0.78,
                    labels={
                        "Age": "Age (years)",
                        "Experience": "Experience (years)",
                        "Salary": "Monthly salary (₹)",
                    },
                )
                fig.update_traces(marker=dict(size=4))
                fig.update_layout(
                    scene=dict(
                        xaxis_title="Age (years)",
                        yaxis_title="Experience (years)",
                        zaxis_title="Monthly salary (₹)",
                    )
                )
            st.plotly_chart(fig_theme(fig), use_container_width=True)
        with right.container(border=True):
            st.markdown("**Team composition**")
            counts = (
                overview_df["Department"].fillna("Missing")
                .value_counts().rename_axis("Department")
                .reset_index(name="Employees")
            )
            fig = px.pie(
                counts,
                names="Department",
                values="Employees",
                hole=0.66,
                color_discrete_sequence=CHART_COLORS,
            )
            fig.update_traces(textposition="inside", textinfo="percent")
            fig.add_annotation(
                text=f"<b>{len(overview_df):,}</b><br>employees",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#E8EEF8" if is_dark_mode() else "#26384A"),
            )
            st.plotly_chart(fig_theme(fig), use_container_width=True)
    st.subheader("Model snapshot")
    snapshot = metrics.sort_values("MAE").copy()
    snapshot["MAE"] = snapshot["MAE"].map(snapshot_percentage)
    snapshot["MSE"] = snapshot["MSE"].map(snapshot_percentage)
    snapshot["RMSE"] = snapshot["RMSE"].map(snapshot_percentage)
    snapshot["R² Score"] = snapshot["R² Score"].map(r2_snapshot_percentage)
    st.dataframe(snapshot,
                 use_container_width=True, hide_index=True)
    st.caption(f"Best model by MAE on this run: {best_model_name}. This is a relative comparison, not a guarantee of accurate salary prediction.")

elif page == "Data Explorer":
    page_header(
        "Exploratory data analysis",
        "Compare salary and workforce patterns with interactive filters and linked visual views.",
    )
    department_options = sorted(df["Department"].fillna("Missing").astype(str).unique().tolist())
    gender_options = sorted(df["Gender"].fillna("Missing").astype(str).unique().tolist())
    education_options = sorted(df["Education"].fillna("Missing").astype(str).unique().tolist())
    st.markdown("#### Focus the analysis")
    filter_row_one = st.columns(3)
    selected_departments = filter_row_one[0].multiselect(
        "Department", department_options, default=department_options,
        key=f"eda_departments_{data_fingerprint}",
    )
    selected_genders = filter_row_one[1].multiselect(
        "Gender", gender_options, default=gender_options,
        key=f"eda_genders_{data_fingerprint}",
    )
    selected_education = filter_row_one[2].multiselect(
        "Education level", education_options, default=education_options,
        key=f"eda_education_{data_fingerprint}",
    )
    filter_row_two = st.columns(2)
    with filter_row_two[0]:
        age_range = numeric_range_slider(
            df["Age"], "Age range (years)", f"eda_age_{data_fingerprint}", 1.0
        )
    with filter_row_two[1]:
        experience_range = numeric_range_slider(
            df["Experience"], "Experience range (years)",
            f"eda_experience_{data_fingerprint}", 0.5,
        )
    salary_range = numeric_range_slider(
        df["Salary"], "Salary range (₹)",
        f"eda_salary_{data_fingerprint}", 1000.0,
    )
    filtered = df.copy()
    for column, selected in (
        ("Department", selected_departments),
        ("Gender", selected_genders),
        ("Education", selected_education),
    ):
        filtered = filtered[
            filtered[column].fillna("Missing").astype(str).isin(selected)
        ]
    if age_range is not None:
        filtered = filtered[
            filtered["Age"].between(*age_range) | filtered["Age"].isna()
        ]
    if experience_range is not None:
        filtered = filtered[
            filtered["Experience"].between(*experience_range)
            | filtered["Experience"].isna()
        ]
    if salary_range is not None:
        filtered = filtered[
            filtered["Salary"].between(*salary_range)
            | filtered["Salary"].isna()
        ]
    st.caption(
        f"Showing **{len(filtered):,}** of {len(df):,} records. "
        "Rows with missing age or experience remain included when those ranges are used."
    )
    if filtered.empty:
        st.info("No records match these filters. Expand one or more selections to continue.")
    else:
        compensation_tab, workforce_tab, three_d_tab = st.tabs(
            ["Salary patterns", "Workforce patterns", "3D salary explorer"]
        )
        with compensation_tab:
            salary_records = filtered.dropna(subset=["Salary"]).copy()
            histogram_col, salary_box_col = st.columns(2)
            with histogram_col.container(border=True):
                st.markdown("**Salary distribution**")
                if salary_records.empty:
                    st.info("No numeric salary values match these filters.")
                else:
                    fig = px.histogram(
                        salary_records,
                        x="Salary",
                        nbins=30,
                        marginal="violin",
                        color_discrete_sequence=[CHART_COLORS[0]],
                        labels={"Salary": "Monthly salary (₹)"},
                    )
                    fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
                    st.plotly_chart(fig_theme(fig), use_container_width=True)
            with salary_box_col.container(border=True):
                st.markdown("**Salary spread and outliers**")
                if salary_records.empty:
                    st.info("No numeric salary values match these filters.")
                else:
                    fig = px.box(
                        salary_records,
                        y="Salary",
                        points="outliers",
                        color_discrete_sequence=[CHART_COLORS[1]],
                        labels={"Salary": "Monthly salary (₹)"},
                    )
                    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
                    st.plotly_chart(fig_theme(fig), use_container_width=True)

            department_salary = (
                salary_records.assign(
                    Department=salary_records["Department"].fillna("Missing").astype(str)
                )
                .groupby("Department", dropna=False)["Salary"]
                .agg(Mean="mean", Median="median")
                .reset_index()
                .melt(
                    id_vars="Department",
                    value_vars=["Mean", "Median"],
                    var_name="Measure",
                    value_name="Salary",
                )
            )
            dept_col, education_col = st.columns(2)
            with dept_col.container(border=True):
                st.markdown("**Salary by department**")
                if department_salary.empty:
                    st.info("No department salary values match these filters.")
                else:
                    fig = px.bar(
                        department_salary,
                        x="Salary",
                        y="Department",
                        color="Measure",
                        barmode="group",
                        orientation="h",
                        color_discrete_sequence=CHART_COLORS,
                        labels={"Salary": "Monthly salary (₹)"},
                    )
                    fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
                    st.plotly_chart(fig_theme(fig), use_container_width=True)
            with education_col.container(border=True):
                st.markdown("**Salary by education level**")
                education_salary = salary_records.assign(
                    Education=salary_records["Education"].fillna("Missing").astype(str)
                )
                if education_salary.empty:
                    st.info("No education-level salary values match these filters.")
                else:
                    fig = px.box(
                        education_salary,
                        x="Education",
                        y="Salary",
                        color="Education",
                        points="outliers",
                        color_discrete_sequence=CHART_COLORS,
                        labels={"Salary": "Monthly salary (₹)"},
                    )
                    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
                    st.plotly_chart(fig_theme(fig), use_container_width=True)
            with st.container(border=True):
                st.markdown("**Salary by gender**")
                gender_salary = salary_records.assign(
                    Gender=salary_records["Gender"].fillna("Missing").astype(str)
                )
                if gender_salary.empty:
                    st.info("No gender salary values match these filters.")
                else:
                    fig = px.box(
                        gender_salary,
                        x="Gender",
                        y="Salary",
                        color="Gender",
                        points="outliers",
                        color_discrete_sequence=CHART_COLORS,
                        labels={"Salary": "Monthly salary (₹)"},
                    )
                    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
                    st.plotly_chart(fig_theme(fig), use_container_width=True)
        with workforce_tab:
            age_salary_col, experience_salary_col = st.columns(2)
            with age_salary_col.container(border=True):
                st.markdown("**Age vs. salary**")
                age_salary = filtered.dropna(subset=["Age", "Salary"])
                if age_salary.empty:
                    st.info("No age and salary pairs match these filters.")
                else:
                    fig = px.scatter(
                        age_salary,
                        x="Age",
                        y="Salary",
                        color="Department",
                        hover_data=["Experience", "Education", "Gender"],
                        opacity=0.78,
                        render_mode="webgl",
                        labels={"Age": "Age (years)", "Salary": "Monthly salary (₹)"},
                    )
                    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
                    st.plotly_chart(fig_theme(fig), use_container_width=True)
            with experience_salary_col.container(border=True):
                st.markdown("**Experience vs. salary**")
                experience_salary = filtered.dropna(
                    subset=["Experience", "Salary"]
                )
                if experience_salary.empty:
                    st.info("No experience and salary pairs match these filters.")
                else:
                    fig = px.scatter(
                        experience_salary,
                        x="Experience",
                        y="Salary",
                        color="Department",
                        hover_data=["Age", "Education", "Gender"],
                        opacity=0.78,
                        render_mode="webgl",
                        labels={
                            "Experience": "Experience (years)",
                            "Salary": "Monthly salary (₹)",
                        },
                    )
                    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
                    st.plotly_chart(fig_theme(fig), use_container_width=True)

            density_col, department_count_col = st.columns(2)
            with density_col.container(border=True):
                st.markdown("**Age vs. experience distribution**")
                age_experience = filtered.dropna(subset=["Age", "Experience"])
                if age_experience.empty:
                    st.info("No age and experience pairs match these filters.")
                else:
                    fig = px.density_heatmap(
                        age_experience,
                        x="Age",
                        y="Experience",
                        nbinsx=20,
                        nbinsy=20,
                        color_continuous_scale=["#DBEAFE", "#2563EB", "#0B1B3A"],
                        labels={"Age": "Age (years)", "Experience": "Experience (years)"},
                    )
                    st.plotly_chart(fig_theme(fig), use_container_width=True)
            with department_count_col.container(border=True):
                st.markdown("**Employees by department**")
                department_counts = (
                    filtered["Department"].fillna("Missing").astype(str)
                    .value_counts().rename_axis("Department")
                    .reset_index(name="Employees")
                )
                fig = px.bar(
                    department_counts,
                    x="Department",
                    y="Employees",
                    color="Department",
                    text="Employees",
                    color_discrete_sequence=CHART_COLORS,
                    labels={"Employees": "Employee records"},
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(fig_theme(fig), use_container_width=True)

            with st.container(border=True):
                st.markdown("**Numeric feature correlation**")
                numeric_data = filtered[["Age", "Experience", "Salary"]]
                numeric_data = numeric_data.dropna(axis=1, how="all")
                numeric_data = numeric_data.loc[:, numeric_data.nunique(dropna=True) > 1]
                correlation_data = numeric_data.dropna()
                if len(correlation_data) < 2 or correlation_data.shape[1] < 2:
                    st.info("At least two varying numeric fields with paired values are needed for a correlation heatmap.")
                else:
                    correlation = correlation_data.corr()
                    fig = px.imshow(
                        correlation,
                        text_auto=".2f",
                        zmin=-1,
                        zmax=1,
                        color_continuous_scale=["#EF4444", "#F8FAFC", "#2563EB"],
                        labels={"color": "Correlation"},
                    )
                    st.plotly_chart(fig_theme(fig), use_container_width=True)
        with three_d_tab:
            st.markdown("**Rotate, zoom, and hover to inspect the salary landscape.**")
            plot_3d = filtered.dropna(
                subset=["Age", "Experience", "Salary"]
            )
            if len(plot_3d) > 5000:
                plot_3d = plot_3d.sample(5000, random_state=42)
                st.caption("Showing a deterministic 5,000-record sample for responsive 3D rendering.")
            if plot_3d.empty:
                st.info("At least one complete age, experience, and salary record is needed for this view.")
            else:
                fig = px.scatter_3d(
                    plot_3d,
                    x="Age",
                    y="Experience",
                    z="Salary",
                    color="Department",
                    hover_data=["Education", "Gender"],
                    opacity=0.8,
                    labels={
                        "Age": "Age (years)",
                        "Experience": "Experience (years)",
                        "Salary": "Monthly salary (₹)",
                    },
                )
                fig.update_traces(marker=dict(size=4))
                fig.update_layout(
                    scene=dict(
                        xaxis_title="Age (years)",
                        yaxis_title="Experience (years)",
                        zaxis_title="Monthly salary (₹)",
                    )
                )
                st.plotly_chart(fig_theme(fig), use_container_width=True)

    with st.expander("Filtered employee records"):
        table_search = st.text_input("Search employee records", key=f"eda_search_{data_fingerprint}")
        table_df = filtered
        if table_search.strip():
            search = table_search.strip().lower()
            table_df = filtered[
                filtered.astype(str).apply(
                    lambda column: column.str.lower().str.contains(search, regex=False, na=False)
                ).any(axis=1)
            ]
        st.caption(f"{len(table_df):,} matching records")
        st.dataframe(table_df.head(500), use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered data (CSV)",
            filtered.to_csv(index=False).encode("utf-8"),
            "filtered_employee_data.csv",
            "text/csv",
        )

elif page == "Preprocessing":
    page_header(
        "Data preparation",
        "Trace the actual input data through quality checks, feature processing, and the model split.",
    )
    modeling_rows = len(st.session_state["modelling_df"])
    test_rows = len(st.session_state["y_test"])
    train_rows = modeling_rows - test_rows
    fitted_preprocessor = best_model.named_steps["preprocess"]
    fitted_columns = {
        name: list(columns)
        for name, _, columns in fitted_preprocessor.transformers_
    }
    numeric_columns = fitted_columns["num"]
    categorical_columns = fitted_columns["cat"]
    feature_columns = [column for column in EXPECTED if column != "Salary"]
    transformed_shape = fitted_preprocessor.transform(
        st.session_state["modelling_df"][feature_columns]
    ).shape
    missing_cells = int(df.isna().sum().sum())
    quality_score = 100 * (1 - missing_cells / max(df.size, 1))

    metric_row = st.columns(4)
    metric_row[0].metric("Dataset rows", f"{len(df):,}")
    metric_row[1].metric("Rows with usable salary", f"{modeling_rows:,}")
    metric_row[2].metric("Training / test rows", f"{train_rows:,} / {test_rows:,}")
    metric_row[3].metric("Encoded feature columns", f"{transformed_shape[1]:,}")
    st.caption(
        f"{len(df) - modeling_rows:,} rows with missing/non-numeric salary were excluded "
        "from regression training. Feature missing values are imputed within each fitted model pipeline."
    )

    st.subheader("Preprocessing workflow")
    workflow = st.columns(4)
    workflow_cards = [
        ("01", "Inspect", "Original employee rows, missing fields, and exact duplicate count."),
        ("02", "Split", f"{train_rows:,} training rows and {test_rows:,} held-out test rows."),
        ("03", "Transform", "Numeric features: median imputation and standard scaling. Categorical features: most-frequent imputation and one-hot encoding."),
        ("04", "Fit", "Each model fits its own preprocessing pipeline on training data; unknown categories are ignored at prediction time."),
    ]
    for column, (number, title, description) in zip(workflow, workflow_cards):
        column.markdown(
            f'<div class="feature-card"><div class="feature-card__number">{number} / PIPELINE</div>'
            f'<h3>{title}</h3><p>{escape(description)}</p></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "This view reports the current employee dataset and fitted pipeline. Duplicate and IQR-flagged rows "
        "are measured for review; the existing training pipeline does not automatically delete them."
    )

    missing_by_column = pd.DataFrame({
        "Column": df.columns,
        "Missing values": [int(df[column].isna().sum()) for column in df.columns],
        "Missing (%)": [float(df[column].isna().mean() * 100) for column in df.columns],
    }).sort_values("Missing (%)", ascending=False)
    missing_col, duplicate_col = st.columns([1.3, 1])
    with missing_col.container(border=True):
        st.markdown("**Missing values before model preprocessing**")
        if missing_cells == 0:
            st.success("No missing cells found in the employee dataset.")
        else:
            fig = px.bar(
                missing_by_column[missing_by_column["Missing values"] > 0],
                x="Missing (%)",
                y="Column",
                orientation="h",
                color="Missing (%)",
                color_continuous_scale=["#BAE6FD", "#2563EB", "#0B1B3A"],
                text_auto=".1f",
                labels={"Missing (%)": "Missing values (%)"},
            )
            st.plotly_chart(fig_theme(fig), use_container_width=True)
        st.caption(f"Raw dataset completeness: {quality_score:.1f}%.")
    with duplicate_col.container(border=True):
        st.markdown("**Duplicate detection**")
        st.metric("Exact duplicate rows", f"{int(df.duplicated().sum()):,}")
        st.write(
            "Duplicates are reported, not removed, to preserve the existing input and modeling behavior."
        )
        st.dataframe(missing_by_column, use_container_width=True, hide_index=True)

    st.subheader("Numerical outlier review")
    outlier_rows = []
    for column in ["Age", "Experience", "Salary"]:
        values = df[column].dropna()
        if values.empty:
            outlier_rows.append({
                "Field": column, "Valid values": 0, "IQR lower bound": np.nan,
                "IQR upper bound": np.nan, "Flagged values": 0,
            })
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = int(((values < lower_bound) | (values > upper_bound)).sum())
        outlier_rows.append({
            "Field": column,
            "Valid values": int(values.size),
            "IQR lower bound": lower_bound,
            "IQR upper bound": upper_bound,
            "Flagged values": flagged,
        })
    outlier_summary = pd.DataFrame(outlier_rows)
    outlier_left, outlier_right = st.columns([1, 1.2])
    with outlier_left.container(border=True):
        st.dataframe(
            outlier_summary.style.format({
                "IQR lower bound": "{:,.2f}",
                "IQR upper bound": "{:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("IQR flags use Q1 − 1.5×IQR and Q3 + 1.5×IQR; flagged records remain in the analysis.")
    with outlier_right.container(border=True):
        plot_outliers = outlier_summary.copy()
        fig = px.bar(
            plot_outliers,
            x="Field",
            y="Flagged values",
            color="Field",
            text="Flagged values",
            color_discrete_sequence=CHART_COLORS,
            labels={"Flagged values": "Values outside IQR bounds"},
        )
        st.plotly_chart(fig_theme(fig), use_container_width=True)

    st.subheader("Fitted feature transformations")
    categorical_pipeline = fitted_preprocessor.named_transformers_["cat"]
    encoder = categorical_pipeline.named_steps["onehot"]
    category_summary = pd.DataFrame({
        "Feature": categorical_columns,
        "Observed training categories": [
            ", ".join(str(value) for value in categories)
            for categories in encoder.categories_
        ],
        "Encoded columns": [len(categories) for categories in encoder.categories_],
    })
    transform_col, category_col = st.columns([1, 1.3])
    with transform_col.container(border=True):
        st.markdown("**Numerical feature processing**")
        numeric_feature_text = ", ".join(numeric_columns) if numeric_columns else "No numeric features"
        st.write(f"{numeric_feature_text} → median imputation → StandardScaler.")
        st.write("Salary is the prediction target and is not included in the numeric feature scaling.")
        st.markdown("**Categorical feature processing**")
        categorical_feature_text = ", ".join(categorical_columns) if categorical_columns else "No categorical features"
        st.write(f"{categorical_feature_text} → most-frequent imputation → one-hot encoding.")
        st.metric("Transformed feature matrix", f"{transformed_shape[0]:,} rows × {transformed_shape[1]:,} columns")
    with category_col.container(border=True):
        st.markdown("**One-hot categories learned from training data**")
        st.dataframe(category_summary, use_container_width=True, hide_index=True)

    with st.expander("Preview original employee rows"):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

elif page == "Salary Predictor":
    page_header("Salary prediction", "Estimate monthly salary with a model trained on the employee dataset.")
    selected_model = st.selectbox("Prediction model", list(models.keys()), index=list(models.keys()).index(best_model_name))
    with st.container(border=True):
        st.markdown("**Build an employee profile**")
        st.caption("Choose the closest profile values, then estimate monthly pay.")
        with st.form("prediction_form"):
            x1,x2,x3 = st.columns(3)
            age = x1.number_input("Age", min_value=16, max_value=100, value=30)
            exp = x2.number_input("Experience (years)", min_value=0.0, max_value=80.0, value=5.0, step=0.5)
            dept_opts = sorted(df["Department"].dropna().unique().tolist(), key=str)
            edu_opts = sorted(df["Education"].dropna().unique().tolist(), key=str)
            gender_opts = sorted(df["Gender"].dropna().unique().tolist(), key=str)
            department = x3.selectbox("Department", dept_opts)
            x4,x5 = st.columns(2)
            education = x4.selectbox("Education level", edu_opts)
            gender = x5.selectbox("Gender", gender_opts)
            submitted = st.form_submit_button("Estimate salary", type="primary", use_container_width=True)
    if submitted:
        sample = pd.DataFrame([{
            "Age": age, "Experience": exp, "Department": department,
            "Education": education, "Gender": gender
        }])
        value = float(models[selected_model].predict(sample)[0])
        with st.container(border=True):
            st.markdown("#### Estimated monthly salary")
            st.metric("Model estimate", money(value))
            st.markdown("**Input employee profile**")
            st.dataframe(
                pd.DataFrame([{
                    "Age": age,
                    "Experience (years)": exp,
                    "Department": department,
                    "Education level": education,
                    "Gender": gender,
                    "Model": selected_model,
                }]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"{selected_model} estimate for the selected profile. This is exploratory, "
                "not an official salary recommendation."
            )
            salary_data = df["Salary"].dropna()
            fig = px.histogram(
                salary_data,
                x="Salary",
                nbins=28,
                color_discrete_sequence=[CHART_COLORS[0]],
                labels={"Salary": "Monthly salary (₹)"},
            )
            fig.add_vline(
                x=value,
                line_width=3,
                line_dash="dash",
                line_color=CHART_COLORS[4],
                annotation_text="Prediction",
                annotation_position="top",
            )
            fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
            st.plotly_chart(fig_theme(fig), use_container_width=True)
    st.subheader("Prediction reliability")
    chosen = metrics[metrics["Model"] == selected_model].iloc[0]
    r1,r2,r3 = st.columns(3)
    r1.metric("Test MAE", money(chosen["MAE"]))
    r2.metric("Test RMSE", money(chosen["RMSE"]))
    r3.metric("Test R²", r2_value(chosen["R² Score"]))
    if chosen["R² Score"] < 0:
        st.warning("This model has a negative R² score on the current test split. It performed worse than predicting the test-set mean. Treat individual predictions as experimental only.")
    with st.container(border=True):
        st.markdown("**Compare test-set error**")
        error_comparison = metrics.sort_values("MAE")
        fig = px.bar(
            error_comparison,
            x="Model",
            y="MAE",
            color="Model",
            text_auto=".2s",
            labels={"MAE": "Mean absolute error (₹)"},
        )
        fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(fig_theme(fig), use_container_width=True)

elif page == "Pattern":
    page_header(
        "Employee cluster analysis",
        "Explore K-Means segments using observed employee profiles, salary distributions, and interactive 3D views.",
    )
    cluster_count = clustered["Cluster"].nunique()
    st.caption(
        f"K = {cluster_count} · Features used: {', '.join(st.session_state['cluster_features'])}"
    )
    metric_row = st.columns(3)
    metric_row[0].metric("Clustered employees", f"{len(clustered):,}")
    metric_row[1].metric("Silhouette score", f"{sil:.3f}" if sil is not None else "Not available")
    metric_row[2].metric("Clusters", f"{cluster_count:,}")
    if sil is not None and sil < 0.25:
        st.warning(
            "The silhouette score suggests weak separation. Treat these groups as exploratory "
            "patterns rather than naturally distinct employee categories."
        )
    if "Salary" not in st.session_state["cluster_features"]:
        st.info("Monthly salary is not an input to this clustering run; it is shown only for profile comparison.")
    else:
        st.caption("Monthly salary is included as a clustering input for this run.")

    sizes = (
        clustered["Cluster"].value_counts().sort_index()
        .rename_axis("Cluster").reset_index(name="Employees")
    )
    sizes["Cluster label"] = sizes["Cluster"].map(lambda value: f"Cluster {value}")
    distribution_col, map_col = st.columns([0.85, 1.5])
    with distribution_col.container(border=True):
        st.markdown("**Cluster distribution**")
        fig = px.pie(
            sizes,
            names="Cluster label",
            values="Employees",
            hole=0.58,
            color="Cluster label",
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_traces(textposition="inside", textinfo="label+value+percent")
        st.plotly_chart(fig_theme(fig), use_container_width=True)
    with map_col.container(border=True):
        st.markdown("**Employee map**")
        map_view = st.radio(
            "Choose a view",
            ["3D salary landscape", "2D age vs. experience"],
            horizontal=True,
            key="cluster_map_view",
        )
        cluster_plot = clustered.copy()
        cluster_plot["Cluster label"] = cluster_plot["Cluster"].map(
            lambda value: f"Cluster {value}"
        )
        if map_view == "3D salary landscape":
            cluster_plot = cluster_plot.dropna(
                subset=["Age", "Experience", "Salary"]
            )
            if len(cluster_plot) > 5000:
                cluster_plot = cluster_plot.sample(5000, random_state=42)
                st.caption("Showing a deterministic 5,000-record sample for responsive 3D rendering.")
            if cluster_plot.empty:
                st.info("No complete age, experience, and salary records are available for the 3D map.")
            else:
                fig = px.scatter_3d(
                    cluster_plot,
                    x="Age",
                    y="Experience",
                    z="Salary",
                    color="Cluster label",
                    hover_data=["Department", "Education", "Gender"],
                    opacity=0.78,
                    color_discrete_sequence=CHART_COLORS,
                    labels={
                        "Age": "Age (years)",
                        "Experience": "Experience (years)",
                        "Salary": "Monthly salary (₹)",
                    },
                )
                fig.update_traces(marker=dict(size=4))
                fig.update_layout(scene=dict(
                    xaxis_title="Age (years)",
                    yaxis_title="Experience (years)",
                    zaxis=dict(title="Monthly salary (₹)", tickprefix="₹", tickformat=",.0f"),
                ))
                st.plotly_chart(fig_theme(fig), use_container_width=True)
        else:
            cluster_plot = cluster_plot.dropna(subset=["Age", "Experience"])
            if cluster_plot.empty:
                st.info("No complete age and experience records are available for the 2D map.")
            else:
                fig = px.scatter(
                    cluster_plot,
                    x="Age",
                    y="Experience",
                    color="Cluster label",
                    hover_data=["Salary", "Department", "Education", "Gender"],
                    opacity=0.78,
                    render_mode="webgl",
                    color_discrete_sequence=CHART_COLORS,
                    labels={
                        "color": "Cluster",
                        "Age": "Age (years)",
                        "Experience": "Experience (years)",
                    },
                )
                st.plotly_chart(fig_theme(fig), use_container_width=True)

    profile = clustered.groupby("Cluster", dropna=False).agg(
        Employees=("Cluster", "size"),
        Mean_age=("Age", "mean"),
        Median_age=("Age", "median"),
        Mean_experience=("Experience", "mean"),
        Median_experience=("Experience", "median"),
        Mean_salary=("Salary", "mean"),
        Median_salary=("Salary", "median"),
    ).reset_index()
    profile_display = profile.rename(columns={
        "Cluster": "Cluster label",
        "Mean_age": "Average age",
        "Median_age": "Median age",
        "Mean_experience": "Average experience",
        "Median_experience": "Median experience",
        "Mean_salary": "Average salary",
        "Median_salary": "Median salary",
    })
    st.subheader("Cluster profiles")
    profile_table_col, salary_plot_col = st.columns([1.2, 1])
    with profile_table_col.container(border=True):
        st.dataframe(
            profile_display.style.format({
                "Average age": "{:.1f}",
                "Median age": "{:.1f}",
                "Average experience": "{:.1f}",
                "Median experience": "{:.1f}",
                "Average salary": "₹{:,.0f}",
                "Median salary": "₹{:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    with salary_plot_col.container(border=True):
        st.markdown("**Salary distribution by cluster**")
        salary_profile = clustered.dropna(subset=["Salary"]).copy()
        salary_profile["Cluster label"] = salary_profile["Cluster"].map(
            lambda value: f"Cluster {value}"
        )
        if salary_profile.empty:
            st.info("No numeric salary values are available for cluster comparison.")
        else:
            fig = px.violin(
                salary_profile,
                x="Cluster label",
                y="Salary",
                color="Cluster label",
                box=True,
                points="outliers",
                color_discrete_sequence=CHART_COLORS,
                labels={"Salary": "Monthly salary (₹)"},
            )
            fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
            st.plotly_chart(fig_theme(fig), use_container_width=True)
    st.markdown("**Observed cluster characteristics**")
    category_descriptions = {
        "Department": "most common department",
        "Education": "most common education level",
        "Gender": "most common gender",
    }
    for _, row in profile.iterrows():
        members = clustered[clustered["Cluster"] == row["Cluster"]]
        characteristics = [f"{int(row['Employees']):,} employees"]
        if pd.notna(row["Mean_age"]):
            characteristics.append(f"average age {row['Mean_age']:.1f} years")
        if pd.notna(row["Mean_experience"]):
            characteristics.append(f"average experience {row['Mean_experience']:.1f} years")
        if pd.notna(row["Mean_salary"]):
            characteristics.append(f"average salary {money(row['Mean_salary'])}")
        for column, description in category_descriptions.items():
            common_values = members[column].dropna().astype(str).value_counts()
            if not common_values.empty:
                characteristics.append(f"{description}: {common_values.index[0]}")
        st.write(f"**Cluster {int(row['Cluster'])}:** " + "; ".join(characteristics) + ".")
    salary_means = profile.dropna(subset=["Mean_salary"])
    if len(salary_means) >= 2:
        high_cluster = int(salary_means.loc[salary_means["Mean_salary"].idxmax(), "Cluster"])
        low_cluster = int(salary_means.loc[salary_means["Mean_salary"].idxmin(), "Cluster"])
        salary_gap = float(
            salary_means["Mean_salary"].max() - salary_means["Mean_salary"].min()
        )
        st.info(
            f"In this run, Cluster {high_cluster} has the higher observed mean salary by "
            f"{money(salary_gap)}. Cluster labels are arbitrary and the groups should not "
            "be described as salary tiers without considering the full profile."
        )
    composition_col, profile_distribution_col = st.columns([1.1, 1])
    with composition_col.container(border=True):
        st.markdown("**Workforce mix by cluster**")
        category_choice = st.selectbox(
            "Compare category",
            CATEGORICAL_FEATURES,
            key="cluster_category_choice",
        )
        composition = (
            clustered.assign(
                Category=clustered[category_choice].fillna("Missing").astype(str),
                Cluster_label=clustered["Cluster"].map(lambda value: f"Cluster {value}"),
            )
            .groupby(["Cluster_label", "Category"], dropna=False)
            .size()
            .reset_index(name="Employees")
        )
        fig = px.bar(
            composition,
            x="Category",
            y="Employees",
            color="Cluster_label",
            barmode="relative",
            color_discrete_sequence=CHART_COLORS,
            labels={"Employees": "Share of cluster (%)", "Category": category_choice},
        )
        fig.update_layout(barnorm="percent")
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_theme(fig), use_container_width=True)
    with profile_distribution_col.container(border=True):
        st.markdown("**Choose a profile distribution**")
        profile_metric = st.selectbox(
            "Profile field",
            ["Age", "Experience", "Salary"],
            format_func=lambda value: {
                "Age": "Age",
                "Experience": "Experience",
                "Salary": "Monthly salary",
            }[value],
            key="cluster_profile_metric",
        )
        distribution_data = clustered.dropna(subset=[profile_metric]).copy()
        distribution_data["Cluster label"] = distribution_data["Cluster"].map(
            lambda value: f"Cluster {value}"
        )
        if distribution_data.empty:
            st.info("No values are available for this profile field.")
        else:
            fig = px.box(
                distribution_data,
                x="Cluster label",
                y=profile_metric,
                color="Cluster label",
                points="outliers",
                color_discrete_sequence=CHART_COLORS,
                labels={
                    profile_metric: "Monthly salary (₹)"
                    if profile_metric == "Salary"
                    else profile_metric.replace("_", " "),
                },
            )
            if profile_metric == "Salary":
                fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
            st.plotly_chart(fig_theme(fig), use_container_width=True)
    st.download_button(
        "Download clustered dataset (CSV)",
        clustered.to_csv(index=False).encode("utf-8"),
        "employee_clusters.csv",
        "text/csv",
    )

elif page == "Smart Insights":
    page_header(
        "Business insights",
        "Automatically surface workforce observations from the current employee dataset.",
    )
    department_data = df.assign(
        Department=df["Department"].fillna("Missing").astype(str)
    )
    department_summary = (
        department_data.groupby("Department", dropna=False)
        .agg(
            Employees=("Department", "size"),
            Salary_records=("Salary", "count"),
            Mean_salary=("Salary", "mean"),
            Median_salary=("Salary", "median"),
        )
        .reset_index()
    )
    valid_department_salary = department_summary.dropna(subset=["Mean_salary"])
    most_populous = department_summary.sort_values(
        ["Employees", "Department"], ascending=[False, True]
    ).iloc[0] if not department_summary.empty else None
    highest_department = (
        valid_department_salary.loc[valid_department_salary["Mean_salary"].idxmax()]
        if not valid_department_salary.empty else None
    )
    lowest_department = (
        valid_department_salary.loc[valid_department_salary["Mean_salary"].idxmin()]
        if not valid_department_salary.empty else None
    )

    experience_salary = df.dropna(subset=["Experience", "Salary"])
    experience_correlation = None
    if (
        len(experience_salary) > 1
        and experience_salary["Experience"].nunique() > 1
        and experience_salary["Salary"].nunique() > 1
    ):
        experience_correlation = float(
            experience_salary["Experience"].corr(experience_salary["Salary"])
        )
        if not np.isfinite(experience_correlation):
            experience_correlation = None

    insights_row = st.columns(4)
    insights_row[0].metric(
        "Highest mean salary",
        str(highest_department["Department"]) if highest_department is not None else "Not available",
        money(highest_department["Mean_salary"]) if highest_department is not None else None,
    )
    insights_row[1].metric(
        "Lowest mean salary",
        str(lowest_department["Department"]) if lowest_department is not None else "Not available",
        money(lowest_department["Mean_salary"]) if lowest_department is not None else None,
    )
    insights_row[2].metric(
        "Most employees",
        str(most_populous["Department"]) if most_populous is not None else "Not available",
        f"{int(most_populous['Employees']):,} records" if most_populous is not None else None,
    )
    insights_row[3].metric(
        "Experience / salary correlation",
        f"{experience_correlation:.2f}" if experience_correlation is not None else "Not available",
        "Pearson r; association only",
    )

    generated_insights = []
    if highest_department is not None and lowest_department is not None:
        salary_gap = float(
            highest_department["Mean_salary"] - lowest_department["Mean_salary"]
        )
        generated_insights.append(
            f"Mean salary is highest in {highest_department['Department']} "
            f"({money(highest_department['Mean_salary'])}) and lowest in "
            f"{lowest_department['Department']} ({money(lowest_department['Mean_salary'])}); "
            f"the difference between these means is {money(salary_gap)}."
        )
    if most_populous is not None:
        generated_insights.append(
            f"{most_populous['Department']} has the largest headcount "
            f"({int(most_populous['Employees']):,} records)."
        )
    if experience_correlation is not None:
        direction = "positive" if experience_correlation > 0 else "negative" if experience_correlation < 0 else "near-zero"
        generated_insights.append(
            f"Experience and salary have a {direction} linear association "
            f"(Pearson r = {experience_correlation:.2f}, based on {len(experience_salary):,} complete records)."
        )
    if generated_insights:
        st.subheader("Highlights from this dataset")
        for insight in generated_insights:
            st.markdown(f"- {insight}")
    st.caption("These are descriptive statistics for employee records; association does not establish causation.")

    department_chart_col, education_chart_col = st.columns(2)
    with department_chart_col.container(border=True):
        st.markdown("**Department salary comparison**")
        if valid_department_salary.empty:
            st.info("No numeric salary values are available by department.")
        else:
            department_chart = valid_department_salary.melt(
                id_vars="Department",
                value_vars=["Mean_salary", "Median_salary"],
                var_name="Measure",
                value_name="Salary",
            )
            department_chart["Measure"] = department_chart["Measure"].map({
                "Mean_salary": "Mean",
                "Median_salary": "Median",
            })
            fig = px.bar(
                department_chart,
                x="Salary",
                y="Department",
                color="Measure",
                barmode="group",
                orientation="h",
                color_discrete_sequence=CHART_COLORS,
                labels={"Salary": "Monthly salary (₹)"},
            )
            fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
            st.plotly_chart(fig_theme(fig), use_container_width=True)
    with education_chart_col.container(border=True):
        st.markdown("**Salary differences by education level**")
        education_summary = (
            df.assign(Education=df["Education"].fillna("Missing").astype(str))
            .groupby("Education", dropna=False)["Salary"]
            .agg(Employees="count", Mean_salary="mean", Median_salary="median")
            .reset_index()
            .dropna(subset=["Mean_salary"])
            .sort_values("Mean_salary")
        )
        if education_summary.empty:
            st.info("No numeric salary values are available by education level.")
        else:
            fig = px.bar(
                education_summary,
                x="Mean_salary",
                y="Education",
                orientation="h",
                color="Mean_salary",
                color_continuous_scale=["#BAE6FD", "#2563EB", "#0B1B3A"],
                text_auto=".2s",
                hover_data=["Employees", "Median_salary"],
                labels={"Mean_salary": "Average monthly salary (₹)"},
            )
            fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
            st.plotly_chart(fig_theme(fig), use_container_width=True)
            high_education = education_summary.iloc[-1]
            low_education = education_summary.iloc[0]
            st.caption(
                f"Observed means range from {low_education['Education']} "
                f"({money(low_education['Mean_salary'])}) to {high_education['Education']} "
                f"({money(high_education['Mean_salary'])})."
            )

    st.subheader("Experience and salary relationship")
    if experience_salary.empty:
        st.info("No complete experience and salary records are available.")
    else:
        plot_experience = experience_salary
        if len(plot_experience) > 5000:
            plot_experience = plot_experience.sample(5000, random_state=42)
            st.caption("Showing a deterministic 5,000-record sample in the scatter plot; correlation uses all complete records.")
        fig = px.scatter(
            plot_experience,
            x="Experience",
            y="Salary",
            color="Department",
            hover_data=["Age", "Education", "Gender"],
            opacity=0.78,
            render_mode="webgl",
            labels={
                "Experience": "Experience (years)",
                "Salary": "Monthly salary (₹)",
            },
        )
        if (
            experience_salary["Experience"].nunique() > 1
            and experience_salary["Salary"].nunique() > 1
        ):
            slope, intercept = np.polyfit(
                experience_salary["Experience"],
                experience_salary["Salary"],
                1,
            )
            x_values = np.linspace(
                experience_salary["Experience"].min(),
                experience_salary["Experience"].max(),
                100,
            )
            fig.add_trace(go.Scatter(
                x=x_values,
                y=slope * x_values + intercept,
                mode="lines",
                name="Linear trend",
                line=dict(color="#F97316", width=3, dash="dash"),
            ))
        fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(fig_theme(fig), use_container_width=True)

    st.subheader("Discovered cluster differences")
    cluster_summary = clustered.groupby("Cluster").agg(
        Employees=("Cluster", "size"),
        Mean_age=("Age", "mean"),
        Mean_experience=("Experience", "mean"),
        Mean_salary=("Salary", "mean"),
        Median_salary=("Salary", "median"),
    ).reset_index()
    if cluster_summary.empty:
        st.info("No cluster profiles are available.")
    else:
        st.dataframe(
            cluster_summary.style.format({
                "Mean_age": "{:.1f}",
                "Mean_experience": "{:.1f}",
                "Mean_salary": "₹{:,.0f}",
                "Median_salary": "₹{:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        salary_cluster_summary = cluster_summary.dropna(subset=["Mean_salary"])
        if len(salary_cluster_summary) >= 2:
            cluster_gap = (
                salary_cluster_summary["Mean_salary"].max()
                - salary_cluster_summary["Mean_salary"].min()
            )
            st.caption(
                f"Observed mean salary gap between clusters: {money(cluster_gap)}. "
                "Cluster numbers are arbitrary; interpret this alongside age, experience, and category mix."
            )

elif page == "Model Evaluation":
    page_header("Model evaluation", "Evaluate the current train/test results before interpreting salary estimates.")
    st.caption("Lower MAE / RMSE is better. R² closer to 1 is better; a negative score means the model underperformed a mean-only baseline.")
    metrics_col, error_col = st.columns([1, 1.3])
    with metrics_col.container(border=True):
        st.markdown("**Test-set scorecard**")
        st.dataframe(
            metrics.sort_values("MAE").style.format(
                {"MAE": "₹{:,.2f}", "MSE": "{:,.2f}", "RMSE": "₹{:,.2f}", "R² Score": "{:.5f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    with error_col.container(border=True):
        st.markdown("**Absolute and squared error**")
        error_metrics = metrics.melt(
            id_vars="Model",
            value_vars=["MAE", "RMSE"],
            var_name="Measure",
            value_name="Error",
        )
        fig = px.bar(
            error_metrics,
            x="Model",
            y="Error",
            color="Measure",
            barmode="group",
            text_auto=".2s",
            color_discrete_map={"MAE": CHART_COLORS[0], "RMSE": CHART_COLORS[1]},
            labels={"Error": "Error (₹)"},
        )
        fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(fig_theme(fig), use_container_width=True)
    r2_col, diagnostics_col = st.columns([1, 1.35])
    with r2_col.container(border=True):
        st.markdown("**R² compared with baseline**")
        r2_chart = metrics.sort_values("R² Score").copy()
        fig = px.bar(
            r2_chart,
            x="R² Score",
            y="Model",
            orientation="h",
            color="R² Score",
            color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
            text_auto=".5f",
            labels={"R² Score": "R² score"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color=CHART_COLORS[2], annotation_text="Mean-only baseline")
        st.plotly_chart(fig_theme(fig), use_container_width=True)
    with diagnostics_col.container(border=True):
        st.markdown("**Prediction diagnostics**")
        chosen_name = st.selectbox("Model to inspect", list(models.keys()))
        y = st.session_state["y_test"]
        p = st.session_state["preds"][chosen_name]
        diag = pd.DataFrame({"Actual salary": y.values, "Predicted salary": p})
        diag["Absolute error"] = (diag["Actual salary"] - diag["Predicted salary"]).abs()
        fig = px.scatter(
            diag,
            x="Actual salary",
            y="Predicted salary",
            color="Absolute error",
            opacity=0.8,
            color_continuous_scale=["#14B8A6", "#F59E0B", "#EF4444"],
            labels={"Actual salary": "Actual salary (₹)", "Predicted salary": "Predicted salary (₹)"},
        )
        lo = float(min(diag["Actual salary"].min(), diag["Predicted salary"].min()))
        hi = float(max(diag["Actual salary"].max(), diag["Predicted salary"].max()))
        fig.add_trace(go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Perfect estimate",
            line=dict(color="#94A3B8", dash="dash"),
        ))
        fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
        fig.update_yaxes(tickprefix="₹", tickformat=",.0f", scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig_theme(fig), use_container_width=True)
    st.subheader("Residual diagnostics")
    diag["Residual"] = diag["Actual salary"] - diag["Predicted salary"]
    residual_col, spread_col = st.columns(2)
    with residual_col.container(border=True):
        st.markdown("**Error distribution**")
        fig = px.histogram(
            diag,
            x="Residual",
            nbins=35,
            color_discrete_sequence=[CHART_COLORS[3]],
            labels={"Residual": "Residual (₹)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#EF4444")
        fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(fig_theme(fig), use_container_width=True)
    with spread_col.container(border=True):
        st.markdown("**Residuals vs. predicted salary**")
        fig = px.scatter(
            diag,
            x="Predicted salary",
            y="Residual",
            color="Absolute error",
            color_continuous_scale=["#14B8A6", "#F59E0B", "#EF4444"],
            opacity=0.78,
            labels={"Predicted salary": "Predicted salary (₹)", "Residual": "Residual (₹)"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#EF4444")
        fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
        fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(fig_theme(fig), use_container_width=True)
    st.download_button("Download model metrics (CSV)", metrics.to_csv(index=False).encode("utf-8"),
                       "model_metrics.csv", "text/csv")

elif page == "About Project":
    page_header(
        "About this project",
        "A practical employee salary analytics workflow with transparent, data-derived results.",
    )
    overview_tab, runtime_tab, limitations_tab = st.tabs(
        ["Project overview", "Current run", "Limitations"]
    )
    with overview_tab:
        st.markdown("**Purpose**")
        st.write(
            "Explore salary distributions and workforce composition, compare salary "
            "regressors, and investigate K-Means employee segments using the configured employee CSV."
        )
        st.code(
            "Load CSV → Validate fields → Inspect data quality\n"
            "  ├─ Regression: Linear Regression / Decision Tree / Random Forest\n"
            "  ├─ K-Means: default K=2, with actual cluster profiles and silhouette score\n"
            "  └─ Insights: department, education, experience, and salary comparisons",
            language="text",
        )
        st.write(
            f"The regression and clustering pipelines are fitted from `{DEFAULT_DATASET_PATH}` "
            "and held in the Streamlit session. The employee CSV is included with the project; "
            "trained-model artifacts are not serialized."
        )
        st.markdown(
            '<div class="insight-banner"><b>Explore:</b> Use the EDA filters to compare '
            'groups, then inspect the model diagnostics and cluster profiles before drawing conclusions.</div>',
            unsafe_allow_html=True,
        )
    with runtime_tab:
        st.markdown("**Results computed for the employee dataset**")
        st.caption(
            f"Training rows: {len(st.session_state['modelling_df']):,} before the train/test split · "
            f"Cluster count: {clustered['Cluster'].nunique()} · "
            f"Silhouette: {sil:.3f}" if sil is not None
            else f"Training rows: {len(st.session_state['modelling_df']):,} before the train/test split · "
            f"Cluster count: {clustered['Cluster'].nunique()} · Silhouette unavailable"
        )
        st.dataframe(
            metrics.sort_values("MAE").style.format({
                "MAE": "₹{:,.2f}",
                "MSE": "{:,.2f}",
                "RMSE": "₹{:,.2f}",
                "R² Score": "{:.5f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        negative_scores = metrics[metrics["R² Score"] < 0]
        if not negative_scores.empty:
            negative_labels = ", ".join(
                f"{row['Model']} ({r2_value(row['R² Score'])})"
                for _, row in negative_scores.iterrows()
            )
            st.warning(
                "Negative R² on this held-out split means the listed model(s) performed "
                f"worse than a mean-only baseline: {negative_labels}."
            )
        else:
            st.info("No negative R² values were found in this run's model comparison.")
    with limitations_tab:
        st.warning(
            "Salary predictions are exploratory estimates, not official compensation recommendations. "
            "Check the held-out metrics and residual diagnostics before considering any use."
        )
        st.write("- A random train/test split provides one estimate of generalization performance.")
        st.write("- Correlation and group differences are descriptive and do not establish causality.")
        st.write("- K-Means segments depend on the selected features, scaling, and chosen K; cluster numbers are arbitrary.")
        st.write("- IQR-flagged outliers and duplicate rows are reported but not removed from model training.")
        st.write("- Models are not serialized; they are retrained when the employee dataset or settings change.")
    with st.container(border=True):
        st.markdown("**Run locally**")
        st.code(
            ".venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
            ".venv\\Scripts\\python.exe -m streamlit run app.py",
            language="powershell",
        )
