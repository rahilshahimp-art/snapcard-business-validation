"""
Snap Card Contract Packaging — Business Validation Dashboard
Run:  streamlit run app.py
The survey data is generated in-app by snapcard_data.py (seed 42), identical to the submitted CSV files.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from snapcard_data import generate, clean

st.set_page_config(page_title="Snap Card Business Validation", page_icon="📦", layout="wide")
PALETTE = ["#2a6fdb", "#e0912f", "#3aa57a", "#c0504d", "#8064a2", "#4bacc6", "#9b9b9b", "#f2c14e", "#6c8ebf"]
LVL = {"Definitely not": 0, "Probably not": 1, "Unsure": 2, "Probably": 3, "Definitely": 4}
LVL_ORDER = list(LVL)
REV_ORDER = ["Under 1 Cr", "1-10 Cr", "10-50 Cr", "50-250 Cr", "Above 250 Cr"]
LIKERT = {"Q11_LeakPain": "Q11 Leak / spill pain", "Q12_PremiumImportance": "Q12 Premium look matters",
          "Q13_Sustainability": "Q13 Sustainability priority", "Q14_Hygiene": "Q14 Hygiene matters",
          "Q15_PriceSensitivity": "Q15 Price sensitivity", "Q16_Openness": "Q16 Openness to new formats",
          "Q19_Appeal": "Q19 Appeal of Snap Card"}
SERV = ["Liquid filling", "Custom printing & branding", "Sample & trial packs", "Bulk contract manufacturing",
        "White-label ready products", "Artwork & design", "Formulation compatibility testing",
        "Kit / subscription-box assembly", "Warehousing & dispatch", "Recyclable material option"]


def explain(what: str, insight: str):
    st.markdown(f"**What this shows:** {what}  \n**Insight:** {insight}")


@st.cache_data
def load():
    raw, log = generate(seed=42)
    df, outliers = clean(raw)
    # ---- transformations (feature engineering) ----
    df["PilotLevel"] = df["Q30_PilotLikelihood"].map(LVL)                       # ordinal 0-4
    df["Interested"] = (df["PilotLevel"] >= 3).astype(int)                       # binary target
    df["Qualified"] = ((df["Interested"] == 1) & (df["Q29_FreeTest"] == "Yes")).astype(int)
    df["RevenueIdx"] = df["Q4_Revenue"].map({r: i for i, r in enumerate(REV_ORDER)})
    df["log_units"] = np.log1p(df["Q8_MonthlyUnits"])
    df["log_budget"] = np.log1p(df["Q27_MonthlyBudget_Rs"])
    df["nserv"] = df[[f"Q22_{s}" for s in SERV]].sum(axis=1)
    df["PriceMultiple"] = np.where(df["Q9_CostPerUnit_Rs"] > 0, df["Q25_WTP_PerCard_Rs"] / df["Q9_CostPerUnit_Rs"], np.nan)
    df["AttitudeScore"] = (df["Q11_LeakPain"] + df["Q12_PremiumImportance"] + df["Q16_Openness"]
                           + (6 - df["Q15_PriceSensitivity"])) / 4               # composite, higher = warmer
    outliers["planted"] = [log[f"outliers_{c}"] for c in outliers["column"]]
    return raw, df, outliers[["column", "planted", "flagged"]]


raw, df_all, outlier_tbl = load()

# ---------------- sidebar filters ----------------
st.sidebar.title("Filters")
types = st.sidebar.multiselect("Company type (Q3)", sorted(df_all["Q3_CompanyType"].unique()),
                               default=sorted(df_all["Q3_CompanyType"].unique()))
inds = st.sidebar.multiselect("Industry (Q1)", sorted(df_all["Q1_Industry"].unique()),
                              default=sorted(df_all["Q1_Industry"].unique()))
revs = st.sidebar.multiselect("Revenue band (Q4)", REV_ORDER, default=REV_ORDER)
df = df_all[df_all["Q3_CompanyType"].isin(types) & df_all["Q1_Industry"].isin(inds) & df_all["Q4_Revenue"].isin(revs)]
st.sidebar.caption(f"{len(df):,} of {len(df_all):,} brands selected")
st.sidebar.markdown("---")
st.sidebar.caption("Data is synthetic: 3,000 simulated B2B brand decision-makers answering a 30-question survey.")
if len(df) < 60:
    st.warning("Too few brands selected for reliable charts. Widen the filters.")
    st.stop()

st.title("Snap Card Contract Packaging — Business Validation")
st.caption("Synthetic survey of 3,000 Indian brands · descriptive, diagnostic and predictive analytics")

tabs = st.tabs(["Overview", "Data & cleaning", "Descriptive EDA", "Diagnostic & correlation",
                "Classification", "Clustering", "Association rules", "Regression", "Business case"])

# =============== OVERVIEW ===============
with tabs[0]:
    n = len(df); it = df["Interested"].mean(); q = df["Qualified"].mean()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Brands", f"{n:,}")
    c2.metric("Would pilot in 6 months", f"{it:.0%}")
    c3.metric("Qualified leads", f"{q:.0%}", help="Interested AND will send a free test sample (Q29)")
    c4.metric("Median price / card", f"₹{df['Q25_WTP_PerCard_Rs'].median():.2f}")
    c5.metric("Median budget (interested)", f"₹{df.loc[df.Interested == 1, 'Q27_MonthlyBudget_Rs'].median():,.0f}")
    st.markdown("""
### The business idea
A **B2B contract-packaging service** that fills, prints and brands *Snap Cards* — thin cards holding a single dose of
liquid that dispenses cleanly when bent — for cosmetics, personal care, food, pharma and hotel brands.

### Validation questions → methods
| Question | Method |
|---|---|
| Is the data clean and usable? | Missing-value imputation, IQR outlier treatment, noise check |
| Who are the respondents and what do they say? | **Descriptive** EDA: distributions, shares, summaries |
| What drives interest and spend? | **Diagnostic**: Spearman correlation, interest-rate breakdowns, chi-square / Cramér's V |
| Will a brand be interested? | Classification (logistic regression, decision tree, random forest) |
| Which segments exist? | K-means clustering |
| Which services are bought together? | Apriori association rules |
| How much will a brand spend? | Linear and random-forest regression |
| Can a plant make money? | Break-even and payback model |
""")
    top = df.groupby("Q3_CompanyType")["Interested"].mean().sort_values(ascending=False)
    st.info(f"**Headline:** {it:.0%} of selected brands would pilot. The warmest group is **{top.index[0]}** "
            f"({top.iloc[0]:.0%}); the coldest is **{top.index[-1]}** ({top.iloc[-1]:.0%}). "
            "See the *Business case* tab for the go / no-go verdict.")

# =============== DATA & CLEANING ===============
with tabs[1]:
    st.subheader("1 · The synthetic dataset")
    st.markdown(f"""
- **{len(raw):,} rows × {raw.shape[1]} columns**: one row per brand, one column per survey answer (Q22 is split into 10 yes/no service columns).
- Generated from **5 company-type personas** (D2C startup, SME brand, large FMCG, hotel supplier, pharma), each with its own typical answers.
- **Cause and effect was built in**: leak pain, openness and appeal raise interest; price sensitivity lowers it; volume drives spend.
- **Messiness was planted on purpose** so cleaning could be demonstrated: ±1 noise on 5% of rating answers, 2.5% outliers in four money/volume columns, 4% blanks in five columns.
""")
    with st.expander("Preview raw data (first 50 rows)"):
        st.dataframe(raw.head(50), width="stretch")

    st.subheader("2 · Missing values")
    miss = raw.isna().sum(); miss = miss[miss > 0].sort_values(ascending=True)
    fig = px.bar(x=miss.values, y=miss.index, orientation="h", labels={"x": "Blank answers", "y": ""},
                 color_discrete_sequence=[PALETTE[3]], text=miss.values)
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    explain("Number of blank answers per column in the raw data.",
            f"{int(miss.sum()):,} cells were blank, all in money or count questions. Q9 has extra blanks because brands "
            "with no current single-use pack ('None' in Q7) cannot give a cost — those are filled with 0, not the median. "
            "All other blanks were filled with the **median of the same company type and format**, which keeps a D2C "
            "startup from being given an FMCG-sized value.")

    st.subheader("3 · Outliers: before vs after cleaning")
    col = st.selectbox("Column", ["Q25_WTP_PerCard_Rs", "Q9_CostPerUnit_Rs", "Q8_MonthlyUnits", "Q27_MonthlyBudget_Rs"])
    bb = pd.concat([pd.DataFrame({"value": raw[col], "stage": "Raw", "type": raw["Q3_CompanyType"]}),
                    pd.DataFrame({"value": df_all[col], "stage": "Cleaned", "type": df_all["Q3_CompanyType"]})])
    bb = bb[bb["value"] > 0]
    fig = px.box(bb, x="type", y="value", color="stage", log_y=True, color_discrete_sequence=[PALETTE[3], PALETTE[0]],
                 labels={"value": col + " (log scale)", "type": ""})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend_title="")
    st.plotly_chart(fig, width="stretch")
    rmax, cmax = raw[col].max(), df_all[col].max()
    explain("Box plots of the chosen column by company type, raw (red) vs cleaned (blue), on a log scale.",
            f"Raw maximum {rmax:,.1f} vs cleaned maximum {cmax:,.1f}. Points far above each red box are planted errors "
            "(decimal typos ×100 or ×1,000, or inflated volumes). They were found with the **IQR rule on the log scale "
            "within each company type** (value > Q3 + 1.5 × IQR), set to blank and re-filled with the segment median. "
            "Working within company type matters: a 1-million-unit order is normal for FMCG but an error for a D2C startup.")
    out_tbl = outlier_tbl.rename(columns={"column": "Column", "planted": "Planted", "flagged": "Flagged by IQR"})
    st.dataframe(out_tbl, hide_index=True)
    st.caption("Q9 and Q25 flag slightly more than planted (some genuine extreme answers). Q27 flags fewer because budgets "
               "already span 1,000× across types, so moderate inflation hides inside the spread — an honest limit of IQR.")

    st.subheader("4 · Noise and inconsistent answers")
    incon = int(((df_all["Q30_PilotLikelihood"] == "Definitely") & (df_all["Q29_FreeTest"] == "No")).sum())
    st.markdown(f"""
- **Rating noise:** about 5% of answers to Q11–Q16 and Q19 were shifted by one point. Noise is not removed (you cannot tell which answers moved); it simply weakens correlations slightly, which keeps the analysis realistic.
- **Inconsistent answers:** {incon} brands said they would *definitely* pilot but refused a free test (Q29). They are kept but are **not** counted as qualified leads.
""")

    st.subheader("5 · Transformations applied")
    st.markdown("""
| New variable | How it is built | Why |
|---|---|---|
| Interested | 1 if Q30 is *Probably* or *Definitely* | Binary target for classification |
| PilotLevel | Q30 coded 0–4 | Ordinal version for correlation |
| Qualified | Interested **and** Q29 = Yes | Realistic sales lead |
| RevenueIdx | Q4 band coded 0–4 | Ordinal encoding |
| log_units, log_budget | log(1 + value) | Volumes and budgets are heavily right-skewed |
| nserv | Count of services ticked in Q22 | Engagement depth |
| PriceMultiple | Q25 ÷ Q9 | How much dearer a Snap Card is than today's pack |
| AttitudeScore | Mean of Q11, Q12, Q16 and (6 − Q15) | One composite 'warmth' score |
| One-hot columns | Q1, Q2, Q3, Q5, Q6, Q7, Q10, Q18, Q21 | Categorical inputs for the models |
| Standard scaling | z-scores | Needed for K-means and logistic regression |
""")
    skew_b, skew_a = df_all["Q27_MonthlyBudget_Rs"].skew(), df_all["log_budget"].skew()
    fig = go.Figure()
    fig.add_histogram(x=df_all["Q27_MonthlyBudget_Rs"], nbinsx=60, name="Raw budget", marker_color=PALETTE[3])
    fig2 = go.Figure()
    fig2.add_histogram(x=df_all.loc[df_all.Q27_MonthlyBudget_Rs > 0, "log_budget"], nbinsx=40, name="log budget", marker_color=PALETTE[0])
    a, b = st.columns(2)
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="Q27 budget (₹)")
    fig2.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="log(1 + budget), budgets above 0")
    a.plotly_chart(fig, width="stretch"); b.plotly_chart(fig2, width="stretch")
    explain("The monthly budget before and after the log transform.",
            f"Skewness falls from {skew_b:.1f} to {skew_a:.1f}. Raw budgets pile up near zero with a long tail of large "
            "FMCG brands; after the log they form a near-normal hump, which suits linear regression and K-means.")
    st.download_button("Download cleaned + transformed dataset (CSV)", df_all.to_csv(index=False).encode(),
                       "snapcard_survey_transformed.csv", "text/csv")

# =============== DESCRIPTIVE ===============
with tabs[2]:
    st.subheader("Who answered")
    a, b = st.columns(2)
    ct = df["Q3_CompanyType"].value_counts()
    fig = px.bar(x=ct.values, y=ct.index, orientation="h", text=ct.values, color_discrete_sequence=[PALETTE[0]],
                 labels={"x": "Brands", "y": ""}, title="Company type (Q3)")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), yaxis={"categoryorder": "total ascending"})
    a.plotly_chart(fig, width="stretch")
    ic = df["Q1_Industry"].value_counts()
    fig = px.bar(x=ic.values, y=ic.index, orientation="h", text=ic.values, color_discrete_sequence=[PALETTE[1]],
                 labels={"x": "Brands", "y": ""}, title="Industry (Q1)")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), yaxis={"categoryorder": "total ascending"})
    b.plotly_chart(fig, width="stretch")
    explain("How many brands of each type and industry are in the (filtered) sample.",
            f"The sample leans to **{ct.index[0]}** ({ct.iloc[0] / len(df):.0%}) and **{ic.index[0]}** "
            f"({ic.iloc[0] / len(df):.0%}), mirroring India's small-brand-heavy market. Large FMCG is a minority "
            "but carries the largest volumes, so averages must be read by segment.")

    st.subheader("Answer to the key question: will you pilot Snap Cards? (Q30)")
    q30 = df["Q30_PilotLikelihood"].value_counts().reindex(LVL_ORDER).fillna(0)
    fig = px.bar(x=q30.index, y=q30.values, text=[f"{v / len(df):.0%}" for v in q30.values],
                 color=q30.index, color_discrete_sequence=["#c0504d", "#e8a09a", "#b0b0b0", "#8fc1a9", "#3aa57a"],
                 labels={"x": "", "y": "Brands"})
    fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    explain("Distribution of pilot likelihood across all five answer options.",
            f"{df['Interested'].mean():.0%} say *Probably* or *Definitely*, {(df['PilotLevel'] <= 1).mean():.0%} lean no, "
            f"and {(df['PilotLevel'] == 2).mean():.0%} are unsure. A sizeable undecided middle means a free trial or sample "
            "could move the needle.")

    st.subheader("Attitudes (Likert 1–5)")
    lk = df[list(LIKERT)].rename(columns=LIKERT).melt(var_name="Statement", value_name="Score")
    tab = lk.groupby(["Statement", "Score"]).size().reset_index(name="n")
    tab["share"] = tab["n"] / tab.groupby("Statement")["n"].transform("sum")
    fig = px.bar(tab, y="Statement", x="share", color=tab["Score"].astype(str), orientation="h",
                 color_discrete_sequence=["#c0504d", "#e8a09a", "#d0d0d0", "#8fc1a9", "#3aa57a"],
                 labels={"share": "Share of brands", "color": "Score"})
    fig.update_layout(height=380, barmode="stack", xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10), legend_title="Score")
    st.plotly_chart(fig, width="stretch")
    means = df[list(LIKERT)].mean().rename(LIKERT).sort_values(ascending=False)
    explain("100% stacked bars of every attitude question, from 1 (red, disagree) to 5 (green, agree).",
            f"Highest average: **{means.index[0]}** ({means.iloc[0]:.2f}); lowest: **{means.index[-1]}** ({means.iloc[-1]:.2f}). "
            "Brands care about hygiene and a premium look, but openness to new formats is the weakest — adoption needs "
            "proof (samples, compatibility tests) rather than just a pitch.")

    st.subheader("Money and volume")
    a, b = st.columns(2)
    fig = px.histogram(df, x="Q25_WTP_PerCard_Rs", nbins=50, color_discrete_sequence=[PALETTE[0]],
                       labels={"Q25_WTP_PerCard_Rs": "Price willing to pay per Snap Card (₹)"})
    fig.add_vline(df["Q25_WTP_PerCard_Rs"].median(), line_dash="dash", annotation_text="median")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    a.plotly_chart(fig, width="stretch")
    fig = px.box(df[df.Q8_MonthlyUnits > 0], x="Q3_CompanyType", y="Q8_MonthlyUnits", log_y=True, color="Q3_CompanyType",
                 color_discrete_sequence=PALETTE, labels={"Q8_MonthlyUnits": "Single-use units packed / month (log)", "Q3_CompanyType": ""})
    fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    b.plotly_chart(fig, width="stretch")
    p25, p75 = df["Q25_WTP_PerCard_Rs"].quantile([.25, .75])
    explain("Left: spread of the price brands would pay per filled card. Right: current monthly single-use volume by company type (log scale).",
            f"Half of brands would pay between ₹{p25:.2f} and ₹{p75:.2f} per card (median ₹{df['Q25_WTP_PerCard_Rs'].median():.2f}). "
            "Volumes differ by about 100× between a D2C startup and a large FMCG, which is why every later analysis uses logs "
            "or works within company type.")

    st.subheader("Summary statistics")
    num = ["Q8_MonthlyUnits", "Q9_CostPerUnit_Rs", "Q17_SamplingCampaigns", "Q25_WTP_PerCard_Rs", "Q26_MonthlyCards", "Q27_MonthlyBudget_Rs"]
    desc = df[num].describe(percentiles=[.25, .5, .75]).T
    desc["skew"] = df[num].skew()
    st.dataframe(desc.style.format("{:,.2f}"), width="stretch")
    explain("Count, mean, spread and percentiles of every numeric answer, after cleaning.",
            "Every money and volume column has mean far above median and high skew — a few big brands pull the mean up. "
            "Medians are the honest 'typical brand' figure and are used throughout.")

# =============== DIAGNOSTIC ===============
with tabs[3]:
    st.subheader("Correlation matrix (Spearman)")
    cmap = {**LIKERT, "Q17_SamplingCampaigns": "Q17 Sampling campaigns", "Q25_WTP_PerCard_Rs": "Q25 Price / card",
            "log_units": "Q8 Monthly units (log)", "RevenueIdx": "Q4 Revenue", "nserv": "Q22 No. of services",
            "log_budget": "Q27 Budget (log)", "PilotLevel": "Q30 Pilot likelihood"}
    corr = df[list(cmap)].rename(columns=cmap).corr(method="spearman")
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    # rank only pre-pitch drivers; services, price and budget are other answers given at the same time as Q30
    pl = corr["Q30 Pilot likelihood"].drop(["Q30 Pilot likelihood", "Q22 No. of services", "Q27 Budget (log)", "Q25 Price / card"]).sort_values()
    explain("Spearman rank correlation between every pair of numeric and rating variables (blue = move together, red = move opposite). "
            "Spearman is used because the ratings are ordinal and money columns are skewed.",
            f"Pilot likelihood rises most with **{pl.index[-1]}** (ρ = {pl.iloc[-1]:.2f}) and **{pl.index[-2]}** "
            f"(ρ = {pl.iloc[-2]:.2f}), and falls most with **{pl.index[0]}** (ρ = {pl.iloc[0]:.2f}). Price sensitivity "
            "is the main brake. Services wanted, price and budget also move with interest (ρ ≈ 0.5–0.75), as expected: brands "
            "that want to pilot naturally tick more services and set budgets. Monthly volume is negatively linked to interest: the biggest packers are the "
            "least keen, because they already buy sachets very cheaply.")

    st.subheader("Interest rate by key drivers")
    drivers = {"Q19_Appeal": "Q19 Appeal (1–5)", "Q15_PriceSensitivity": "Q15 Price sensitivity (1–5)",
               "Q11_LeakPain": "Q11 Leak pain (1–5)", "Q16_Openness": "Q16 Openness (1–5)"}
    cols = st.columns(2)
    notes = {}
    for i, (c, lab) in enumerate(drivers.items()):
        g = df.groupby(c)["Interested"].agg(["mean", "size"]).reset_index()
        fig = px.bar(g, x=c, y="mean", text=g["mean"].map("{:.0%}".format), color_discrete_sequence=[PALETTE[i]],
                     labels={c: lab, "mean": "Would pilot"}, hover_data={"size": True})
        fig.update_layout(height=280, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
        cols[i % 2].plotly_chart(fig, width="stretch")
        notes[c] = (g["mean"].iloc[0], g["mean"].iloc[-1])
    explain("Share of brands that would pilot, at each score of four attitude questions.",
            f"Appeal is a near-perfect staircase: {notes['Q19_Appeal'][0]:.0%} interest at score 1 vs "
            f"{notes['Q19_Appeal'][1]:.0%} at 5. Price sensitivity runs the other way "
            f"({notes['Q15_PriceSensitivity'][0]:.0%} → {notes['Q15_PriceSensitivity'][1]:.0%}). Leak pain and openness "
            "both lift interest steadily — the product solves a real pain for brands willing to try new things.")

    st.subheader("Interest by business profile")
    a, b = st.columns(2)
    g = df.groupby("Q4_Revenue")["Interested"].mean().reindex(REV_ORDER).dropna()
    fig = px.bar(x=g.index, y=g.values, text=[f"{v:.0%}" for v in g.values], color_discrete_sequence=[PALETTE[4]],
                 labels={"x": "Revenue band (Q4)", "y": "Would pilot"})
    fig.update_layout(height=300, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
    a.plotly_chart(fig, width="stretch")
    g2 = df.groupby("Q10_PackagingMode")["Interested"].mean().sort_values()
    fig = px.bar(x=g2.values, y=g2.index, orientation="h", text=[f"{v:.0%}" for v in g2.values],
                 color_discrete_sequence=[PALETTE[5]], labels={"x": "Would pilot", "y": "Packaging today (Q10)"})
    fig.update_layout(height=300, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
    b.plotly_chart(fig, width="stretch")
    explain("Left: interest by company revenue. Right: interest by how brands package today.",
            f"Interest falls as revenue grows ({g.iloc[0]:.0%} for the smallest band vs {g.iloc[-1]:.0%} for the largest). "
            f"Brands that already **fully outsource** packaging are the most open ({g2.get('Fully outsourced', np.nan):.0%}) — "
            "they are used to buying packaging as a service, so they are the natural first customers for a contract packer.")

    st.subheader("Which categorical answers matter? (chi-square test)")
    rows = []
    for c, lab in [("Q3_CompanyType", "Q3 Company type"), ("Q1_Industry", "Q1 Industry"), ("Q2_Role", "Q2 Role"),
                   ("Q10_PackagingMode", "Q10 Packaging mode"), ("Q18_Awareness", "Q18 Prior awareness"),
                   ("Q7_CurrentFormat", "Q7 Current format"), ("Q6_Channel", "Q6 Sales channel"), ("Q5_Location", "Q5 Location")]:
        t = pd.crosstab(df[c], df["Interested"])
        if t.shape[0] < 2 or t.shape[1] < 2: continue
        chi, p, _, _ = stats.chi2_contingency(t)
        v = np.sqrt(chi / (t.values.sum() * (min(t.shape) - 1)))
        rows.append({"Variable": lab, "Chi-square": chi, "p-value": p, "Cramér's V": v,
                     "Significant (p<0.05)": "Yes" if p < .05 else "No"})
    chi_df = pd.DataFrame(rows).sort_values("Cramér's V", ascending=False)
    st.dataframe(chi_df.style.format({"Chi-square": "{:.1f}", "p-value": "{:.2e}", "Cramér's V": "{:.3f}"}), hide_index=True, width="stretch")
    ns = chi_df[chi_df["Significant (p<0.05)"] == "No"]["Variable"].tolist()
    explain("A chi-square test of independence between each categorical answer and being interested. Cramér's V measures strength "
            "(0 = none, 0.1 weak, 0.3 moderate, 0.5 strong).",
            f"**{chi_df.iloc[0]['Variable']}** is by far the strongest (V = {chi_df.iloc[0]['Cramér' + chr(39) + 's V']:.2f}): who the brand is "
            f"matters more than where it is. {('Not significant: ' + ', '.join(ns) + ' — so a single city launch is not a handicap.') if ns else ''}")

    st.subheader("Price vs current cost: the sachet problem")
    seg = df.groupby("Q3_CompanyType").agg(current=("Q9_CostPerUnit_Rs", lambda s: s[s > 0].median()), wtp=("Q25_WTP_PerCard_Rs", "median"),
                                           interest=("Interested", "mean")).reset_index()
    fig = go.Figure()
    for _, r in seg.iterrows():
        fig.add_trace(go.Scatter(x=[r.current, r.wtp], y=[r.Q3_CompanyType] * 2, mode="lines", line=dict(color="#bbb", width=3), showlegend=False))
    fig.add_trace(go.Scatter(x=seg.current, y=seg.Q3_CompanyType, mode="markers", marker=dict(size=13, color=PALETTE[6]), name="Pays today per unit (Q9)"))
    fig.add_trace(go.Scatter(x=seg.wtp, y=seg.Q3_CompanyType, mode="markers", marker=dict(size=13, color=PALETTE[0]), name="Would pay per Snap Card (Q25)"))
    fig.update_layout(height=320, xaxis_title="₹ per unit", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    explain("For each company type, the median cost of today's single-use pack (grey) against the median price they'd pay for a Snap Card (blue).",
            "For FMCG and SME brands using ~₹1 sachets, a Snap Card is several times dearer — this explains their low interest. "
            "For hotels, a Snap Card is *cheaper* than the mini bottles they buy today, and for D2C and pharma it is roughly at par. "
            "**Diagnosis: Snap Cards replace mini bottles, vials and tubes, not sachets.**")

    st.subheader("Price vs budget by company type")
    sc = df[df.Q27_MonthlyBudget_Rs > 0].sample(min(1200, (df.Q27_MonthlyBudget_Rs > 0).sum()), random_state=1)
    fig = px.scatter(sc, x="Q25_WTP_PerCard_Rs", y="Q27_MonthlyBudget_Rs", color="Q3_CompanyType", log_y=True, opacity=.6,
                     color_discrete_sequence=PALETTE, labels={"Q25_WTP_PerCard_Rs": "Price / card (₹)", "Q27_MonthlyBudget_Rs": "Monthly budget (₹, log)", "Q3_CompanyType": ""})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    rho = stats.spearmanr(sc["Q25_WTP_PerCard_Rs"], sc["Q27_MonthlyBudget_Rs"]).correlation
    explain("Each dot is a brand with a budget above zero: price it would pay per card vs its monthly budget.",
            f"Overall correlation is weak (ρ = {rho:.2f}), but colours form separate bands: budget is set mostly by "
            "**company type and volume**, not by price per card. D2C startups pay the most per card but have the smallest "
            "budgets; pharma and hotels pay less per card but spend more in total.")

    st.subheader("Biggest concerns among interested brands (Q21)")
    cc = df[df.Interested == 1]["Q21_TopConcern"].value_counts(normalize=True).sort_values()
    fig = px.bar(x=cc.values, y=cc.index, orientation="h", text=[f"{v:.0%}" for v in cc.values], color_discrete_sequence=[PALETTE[3]],
                 labels={"x": "Share of interested brands", "y": ""})
    fig.update_layout(height=280, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    explain("What would stop interested brands from switching.",
            f"**{cc.index[-1]}** ({cc.iloc[-1]:.0%}) and **{cc.index[-2]}** ({cc.iloc[-2]:.0%}) lead. Answer: transparent "
            "per-card pricing and a free formulation compatibility test as the standard first step of every sale.")

# =============== CLASSIFICATION ===============
CAT = ["Q1_Industry", "Q2_Role", "Q3_CompanyType", "Q5_Location", "Q6_Channel", "Q7_CurrentFormat", "Q10_PackagingMode", "Q18_Awareness", "Q21_TopConcern"]
NUM = ["RevenueIdx", "log_units", "Q9_CostPerUnit_Rs", *[c for c in LIKERT if c != "Q19_Appeal"], "Q17_SamplingCampaigns", "Q19_Appeal"]


@st.cache_data
def classify(data: pd.DataFrame):
    X = pd.concat([data[NUM], pd.get_dummies(data[CAT]).astype(int)], axis=1)
    y = data["Interested"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=7)
    sc = StandardScaler().fit(Xtr)
    models = {"Logistic regression": (LogisticRegression(max_iter=2000, C=.5), True),
              "Decision tree": (DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=7), False),
              "Random forest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=7, n_jobs=-1), False)}
    res, rocs, fitted = [], {}, {}
    for name, (m, s) in models.items():
        a, b = (sc.transform(Xtr), sc.transform(Xte)) if s else (Xtr, Xte)
        m.fit(a, ytr); p = m.predict_proba(b)[:, 1]; pr = (p >= .5).astype(int)
        res.append({"Model": name, "Accuracy": accuracy_score(yte, pr), "Precision": precision_score(yte, pr),
                    "Recall": recall_score(yte, pr), "F1": f1_score(yte, pr), "ROC-AUC": roc_auc_score(yte, p)})
        rocs[name] = roc_curve(yte, p); fitted[name] = (m, p)
    imp = pd.Series(fitted["Random forest"][0].feature_importances_, index=X.columns)
    key = lambda c: next((k for k in NUM + CAT if c == k or c.startswith(k + "_")), c)
    imp = imp.groupby(imp.index.map(key)).sum().sort_values(ascending=False)
    best = max(res, key=lambda r: r["ROC-AUC"])["Model"]
    cm = confusion_matrix(yte, (fitted[best][1] >= .5).astype(int))
    return pd.DataFrame(res), rocs, imp, best, cm, len(ytr), len(yte)


with tabs[4]:
    st.subheader("Will a brand be interested? — classification")
    st.markdown("**Target:** Interested (Q30 = Probably/Definitely). **Inputs:** Q1–Q21. 70 / 30 stratified train–test split.")
    res, rocs, imp, best, cm, ntr, nte = classify(df)
    st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns if c != "Model"}).highlight_max(axis=0, subset=res.columns[1:], color="#d7ecd9"),
                 hide_index=True, width="stretch")
    a, b = st.columns(2)
    fig = go.Figure()
    for i, (n_, (f_, t_, _)) in enumerate(rocs.items()):
        fig.add_trace(go.Scatter(x=f_, y=t_, mode="lines", name=n_, line=dict(color=PALETTE[i], width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#aaa"), showlegend=False))
    fig.update_layout(height=380, xaxis_title="False-positive rate", yaxis_title="True-positive rate", margin=dict(l=10, r=10, t=10, b=10), legend=dict(x=.45, y=.1))
    a.plotly_chart(fig, width="stretch")
    ti = imp.head(10).sort_values()
    fig = px.bar(x=ti.values, y=ti.index, orientation="h", color_discrete_sequence=[PALETTE[0]], labels={"x": "Importance", "y": ""})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    b.plotly_chart(fig, width="stretch")
    bestrow = res.set_index("Model").loc[best]
    explain("Left: ROC curves — the further a curve bows to the top-left, the better the model separates interested from uninterested brands. "
            "Right: random-forest importance of each question.",
            f"Best model: **{best}** with ROC-AUC {bestrow['ROC-AUC']:.3f} and accuracy {bestrow['Accuracy']:.0%} on {nte:,} unseen brands "
            f"(training on {ntr:,}). The top predictor is **{imp.index[0]}**, then **{imp.index[1]}** — consistent with the correlation tab.")
    st.markdown(f"**Confusion matrix ({best})**")
    cmdf = pd.DataFrame(cm, index=["Actually not", "Actually interested"], columns=["Predicted not", "Predicted interested"])
    st.dataframe(cmdf, width="content")
    st.caption(f"Missed leads (false negatives): {cm[1, 0]} · wasted pitches (false positives): {cm[0, 1]}. "
               "Use the model to rank a prospect list before sales calls.")

# =============== CLUSTERING ===============
@st.cache_data
def cluster(data: pd.DataFrame, k: int):
    f = pd.DataFrame({"Revenue": data["RevenueIdx"], "log_units": data["log_units"], "log_cost": np.log1p(data["Q9_CostPerUnit_Rs"]),
                      **{c: data[c] for c in LIKERT}, "log_sampling": np.log1p(data["Q17_SamplingCampaigns"]), "log_budget": data["log_budget"]})
    Z = StandardScaler().fit_transform(f)
    elbow = [(kk, KMeans(kk, n_init=10, random_state=7).fit(Z)) for kk in range(2, 9)]
    el = pd.DataFrame({"k": [e[0] for e in elbow], "Inertia": [e[1].inertia_ for e in elbow],
                       "Silhouette": [silhouette_score(Z, e[1].labels_, sample_size=min(2000, len(Z)), random_state=7) for e in elbow]})
    lab = KMeans(k, n_init=20, random_state=7).fit(Z).labels_
    P = PCA(2, random_state=7).fit_transform(Z)
    return el, lab, P


with tabs[5]:
    st.subheader("Customer segments — K-means clustering")
    k = st.slider("Number of segments (k)", 2, 6, 3)
    el, lab, P = cluster(df, k)
    d = df.copy(); d["cluster"] = lab
    rank = d.groupby("cluster")["Interested"].mean().sort_values(ascending=False)
    names = {c: (["High", "Medium", "Low"][i] + " probability") if k == 3 else f"Segment {i + 1} ({rank[c]:.0%} interested)"
             for i, c in enumerate(rank.index)}
    d["Segment"] = d["cluster"].map(names)
    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=el.k, y=el.Inertia, name="Inertia", mode="lines+markers", line=dict(color=PALETTE[1], dash="dash")))
    fig.add_trace(go.Scatter(x=el.k, y=el.Silhouette, name="Silhouette", mode="lines+markers", yaxis="y2", line=dict(color=PALETTE[0])))
    fig.update_layout(height=340, xaxis_title="k", yaxis=dict(title="Inertia"), yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12))
    a.plotly_chart(fig, width="stretch")
    samp = np.random.default_rng(1).choice(len(d), min(1200, len(d)), replace=False)
    fig = px.scatter(x=P[samp, 0], y=P[samp, 1], color=d["Segment"].values[samp], opacity=.65, color_discrete_sequence=PALETTE,
                     labels={"x": "Principal component 1", "y": "Principal component 2", "color": ""})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12))
    b.plotly_chart(fig, width="stretch")
    explain("Left: the elbow (inertia keeps falling, with a bend) and silhouette (higher = better separated) for k = 2–8. "
            "Right: brands projected onto two principal components and coloured by segment.",
            f"Silhouette at k = {k} is {el.set_index('k').loc[k, 'Silhouette']:.2f}: segments overlap, as is normal for attitude data. "
            "k = 3 is used because it matches the high / medium / low brief and gives clearly different interest rates.")
    prof = d.groupby("Segment").agg(Brands=("Interested", "size"), Interested=("Interested", "mean"), Qualified=("Qualified", "mean"),
                                    Appeal=("Q19_Appeal", "mean"), PriceSensitivity=("Q15_PriceSensitivity", "mean"),
                                    Openness=("Q16_Openness", "mean"), MedianBudget=("Q27_MonthlyBudget_Rs", "median"),
                                    MedianPrice=("Q25_WTP_PerCard_Rs", "median"),
                                    TopType=("Q3_CompanyType", lambda s: s.value_counts().index[0])).sort_values("Interested", ascending=False)
    st.dataframe(prof.style.format({"Interested": "{:.0%}", "Qualified": "{:.0%}", "Appeal": "{:.2f}", "PriceSensitivity": "{:.2f}",
                                    "Openness": "{:.2f}", "MedianBudget": "₹{:,.0f}", "MedianPrice": "₹{:.2f}"}), width="stretch")
    explain("Profile of each segment: size, interest, attitudes and spend.",
            f"The top segment is mostly **{prof.iloc[0]['TopType']}** with {prof.iloc[0]['Interested']:.0%} interest — chase them now. "
            f"The bottom segment is mostly **{prof.iloc[-1]['TopType']}** with {prof.iloc[-1]['Interested']:.0%} interest and the highest "
            "price sensitivity — skip for now.")

# =============== ASSOCIATION ===============
@st.cache_data
def assoc(data: pd.DataFrame, sup: float, conf: float):
    from mlxtend.frequent_patterns import apriori, association_rules
    B = data[[f"Q22_{s}" for s in SERV]].astype(bool); B.columns = SERV
    B = B[B.sum(1) > 0]
    fi = apriori(B, min_support=sup, use_colnames=True)
    if fi.empty: return B, pd.DataFrame()
    r = association_rules(fi, metric="confidence", min_threshold=conf)
    r = r[(r["lift"] > 1) & (r["consequents"].apply(len) == 1)].sort_values("lift", ascending=False)
    r["If"] = r["antecedents"].apply(lambda s: " + ".join(sorted(s))); r["Then"] = r["consequents"].apply(lambda s: next(iter(s)))
    return B, r[["If", "Then", "support", "confidence", "lift"]].drop_duplicates()


with tabs[6]:
    st.subheader("Which services are bought together? — Apriori")
    a, b = st.columns(2)
    sup = a.slider("Minimum support", .02, .20, .05, .01)
    conf = b.slider("Minimum confidence", .3, .9, .5, .05)
    B, rules = assoc(df, sup, conf)
    pop = B.mean().sort_values()
    fig = px.bar(x=pop.values, y=pop.index, orientation="h", text=[f"{v:.0%}" for v in pop.values], color_discrete_sequence=[PALETTE[2]],
                 labels={"x": "Share of brands wanting it", "y": ""})
    fig.update_layout(height=380, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    explain("How often each of the 10 services in Q22 is wanted, among brands that ticked at least one.",
            f"**{pop.index[-1]}** is the entry service ({pop.iloc[-1]:.0%}); **{pop.index[0]}** is the rarest ({pop.iloc[0]:.0%}).")
    if rules.empty:
        st.warning("No rules at these thresholds. Lower support or confidence.")
    else:
        st.dataframe(rules.head(15).style.format({"support": "{:.1%}", "confidence": "{:.0%}", "lift": "{:.2f}"}), hide_index=True, width="stretch")
        top = rules.iloc[0]
        explain("Association rules: *support* = share of all brands with the whole combination; *confidence* = share of brands wanting the "
                "'If' services that also want the 'Then' service; *lift* > 1 = more often than chance (above 2 is strong).",
                f"Strongest rule: **{top['If']} → {top['Then']}** (confidence {top['confidence']:.0%}, lift {top['lift']:.2f}). "
                "Package services as bundles: a D2C *launch kit* (sample packs + printing + artwork), a *volume programme* "
                "(filling + bulk + warehousing) and *private label* (white-label + compatibility testing).")

# =============== REGRESSION ===============
@st.cache_data
def regress(data: pd.DataFrame):
    d = data[data.Q27_MonthlyBudget_Rs > 0]
    X = pd.concat([d[["RevenueIdx", "log_units", "Q12_PremiumImportance", "Q15_PriceSensitivity", "Q16_Openness", "Q19_Appeal", "nserv", "PilotLevel"]],
                   np.log1p(d[["Q17_SamplingCampaigns"]]), pd.get_dummies(d[["Q1_Industry", "Q3_CompanyType", "Q24_EngagementModel"]]).astype(int)], axis=1)
    y = np.log(d["Q27_MonthlyBudget_Rs"])
    a, b, ya, yb = train_test_split(X, y, test_size=.3, random_state=7)
    out, preds = [], {}
    for name, m in [("Linear regression", LinearRegression()), ("Random forest", RandomForestRegressor(300, min_samples_leaf=5, random_state=7, n_jobs=-1))]:
        m.fit(a, ya); p = m.predict(b); preds[name] = (m, p)
        out.append({"Model": name, "R²": r2_score(yb, p), "MAE (₹)": mean_absolute_error(np.exp(yb), np.exp(p)),
                    "Median % error": float(np.median(np.abs(np.exp(p) - np.exp(yb)) / np.exp(yb)))})
    rf = preds["Random forest"][0]
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    key = lambda c: next((k for k in ["Q1_Industry", "Q3_CompanyType", "Q24_EngagementModel"] if c.startswith(k)), c)
    imp = imp.groupby(imp.index.map(key)).sum().sort_values(ascending=False)
    best = max(out, key=lambda r: r["R²"])["Model"]
    return pd.DataFrame(out), imp, np.exp(yb), np.exp(preds[best][1]), best, len(d)


with tabs[7]:
    st.subheader("How much will a brand spend per month? — regression")
    st.markdown("**Target:** Q27 monthly budget (log ₹), brands with budget > 0. Price (Q25) and quantity (Q26) are *excluded* as inputs "
                "because budget ≈ price × quantity — including them would inflate R² without telling us anything new.")
    rr, rimp, ya, yp, rbest, rn = regress(df)
    st.dataframe(rr.style.format({"R²": "{:.3f}", "MAE (₹)": "₹{:,.0f}", "Median % error": "{:.0%}"}), hide_index=True, width="stretch")
    a, b = st.columns(2)
    fig = px.scatter(x=ya, y=yp, log_x=True, log_y=True, opacity=.5, color_discrete_sequence=[PALETTE[0]],
                     labels={"x": "Actual monthly budget (₹)", "y": "Predicted (₹)"})
    lo, hi = min(ya.min(), yp.min()), max(ya.max(), yp.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="#999"), showlegend=False))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    a.plotly_chart(fig, width="stretch")
    ri = rimp.head(8).sort_values()
    fig = px.bar(x=ri.values, y=ri.index, orientation="h", color_discrete_sequence=[PALETTE[1]], labels={"x": "Importance", "y": ""})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    b.plotly_chart(fig, width="stretch")
    br = rr.set_index("Model").loc[rbest]
    explain("Left: predicted vs actual budget for test brands (dots on the dashed line are perfect). Right: what drives the prediction.",
            f"**{rbest}** explains {br['R²']:.0%} of the variation in (log) budget across {rn:,} brands; a typical prediction is off by "
            f"{br['Median % error']:.0%}. **{rimp.index[0]}** dominates — a brand's current volume is the best guide to its Snap Card spend.")

# =============== BUSINESS CASE ===============
with tabs[8]:
    st.subheader("Can one Snap Card machine make money?")
    st.caption("Machine price and speed from a supplier listing (ZONESUN ZS-FS120C, US$58,999, ~60 packs/min). Other inputs are assumptions — move the sliders.")
    c = st.columns(4)
    fx = c[0].number_input("₹ per US$", 70.0, 110.0, 88.0)
    landed = c[1].slider("Landed cost factor", 1.0, 1.8, 1.35, .05)
    var = c[2].slider("Variable cost / card (₹)", .8, 4.0, 1.8, .1)
    fixed_op = c[3].slider("Fixed overheads / month (₹ lakh)", 2.0, 15.0, 6.5, .5) * 1e5
    c = st.columns(4)
    per_m = c[0].slider("New clients signed / month", .5, 5.0, 2.0, .5)
    churn = c[1].slider("Monthly churn", 0.0, .10, .03, .01)
    conv = c[2].slider("Qualified leads that sign", .05, .6, .25, .05)
    incl_fmcg = c[3].checkbox("Include large FMCG clients", False)
    capex = 58999 * fx * landed
    r_, n_ = .11 / 12, 60
    emi = capex * r_ / (1 - (1 + r_) ** -n_)
    fixed = fixed_op + emi
    cap = 60 * 60 * 8 * 25 * .8 * 2
    pool = df_all[(df_all.Qualified == 1) & (incl_fmcg | (df_all.Q3_CompanyType != "Large FMCG"))]
    price = np.average(pool["Q25_WTP_PerCard_Rs"], weights=pool["Q26_MonthlyCards"])
    contrib = price - var
    mean_cards = pool["Q26_MonthlyCards"].mean()
    if contrib <= 0:
        st.error(f"At ₹{var:.2f} cost per card the average price of ₹{price:.2f} leaves no margin — the business cannot work.")
    else:
        be_cards = fixed / contrib; be_clients = be_cards / mean_cards
        clients, cum, payback, series = 0.0, -capex, None, []
        for m in range(1, 61):
            clients = clients * (1 - churn) + per_m
            cum += min(cap, clients * mean_cards) * contrib - fixed
            series.append(cum)
            if payback is None and cum >= 0: payback = m
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Machine cost (landed)", f"₹{capex / 1e5:,.1f} lakh")
        k2.metric("Margin per card", f"₹{contrib:.2f}", help=f"Volume-weighted price ₹{price:.2f} minus variable cost")
        k3.metric("Break-even clients", f"{be_clients:.1f}")
        k4.metric("Payback", f"{payback} months" if payback else "Not within 5 years")
        base = df_all[incl_fmcg | (df_all.Q3_CompanyType != "Large FMCG")]
        k5.metric("Brands to approach", f"{be_clients / (len(pool) / len(base) * conv):,.0f}",
                  help="Brands to pitch so that enough become qualified leads and sign to reach break-even")
        fig = px.line(x=list(range(1, 61)), y=np.array(series) / 1e5, labels={"x": "Month", "y": "Cumulative cash (₹ lakh)"})
        fig.add_hline(0, line_color="#888")
        fig.update_traces(line_color=PALETTE[0], line_width=3)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        explain("Cumulative cash after buying one machine, adding each month's margin and subtracting fixed costs and loan repayment.",
                f"With these inputs the plant breaks even at about {be_clients:.0f} average clients ({be_cards:,.0f} cards a month, "
                f"{be_cards / (cap / 2):.0%} of one shift) and {'pays back in ' + str(payback) + ' months' if payback else 'does not pay back within 5 years'}.")
    st.markdown("""
### Verdict: **conditional GO**
- **Viable** as a premium contract packager for **D2C cosmetics/personal care, hotel toiletry suppliers and pharma sampling** — the segments where a Snap Card is at par with or cheaper than today's mini bottles, vials and tubes.
- **Not viable** as a mass **sachet replacement**: large FMCG and SME brands pay ~₹1 per sachet and show the lowest interest.
- Sell **bundles**, lead with a **free compatibility test**, and target brands that already **outsource packaging**.
- Before buying a machine: validate with 100–150 real brands, get 3 machine quotes and a costed bill of materials, and sign 3–5 paid pilots.

*All survey data here is synthetic; it demonstrates the method and the logic, and real responses must confirm the numbers.*
""")
