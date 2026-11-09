"""Synthetic survey generator and cleaner for the Snap Card validation app.

generate() rebuilds the 3,000-brand survey (seed 42), including planted noise,
outliers and blanks. clean() applies the cleaning steps described in the app.
"""
import numpy as np
import pandas as pd

LVL = {"Definitely not": 0, "Probably not": 1, "Unsure": 2, "Probably": 3, "Definitely": 4}
MONEY = ["Q8_MonthlyUnits", "Q9_CostPerUnit_Rs", "Q25_WTP_PerCard_Rs", "Q27_MonthlyBudget_Rs"]


def generate(seed=42, N=3000):
    """Return (raw survey DataFrame, log of injected problems)."""
    rng = np.random.default_rng(seed)

    # ---------- personas (Q3 company type) ----------
    personas = ["D2C startup", "SME brand", "Large FMCG", "Hotel/hospitality supplier", "Pharma company"]
    p_share = [0.32, 0.28, 0.10, 0.18, 0.12]
    Q3 = rng.choice(personas, N, p=p_share)

    ind_by_persona = {
        "D2C startup": (["Cosmetics", "Personal care", "Coffee & beverages", "Food & condiments", "Ayurvedic", "Perfume"],
                        [0.32, 0.22, 0.16, 0.12, 0.10, 0.08]),
        "SME brand": (["Food & condiments", "Ayurvedic", "Personal care", "Cosmetics", "Coffee & beverages", "Other"],
                      [0.28, 0.22, 0.20, 0.14, 0.10, 0.06]),
        "Large FMCG": (["Personal care", "Food & condiments", "Cosmetics", "Coffee & beverages"],
                       [0.40, 0.30, 0.18, 0.12]),
        "Hotel/hospitality supplier": (["Hotel toiletries supply", "Personal care", "Food & condiments"], [0.70, 0.18, 0.12]),
        "Pharma company": (["Pharma", "Ayurvedic"], [0.75, 0.25]),
    }
    Q1 = np.array([rng.choice(ind_by_persona[p][0], p=ind_by_persona[p][1]) for p in Q3])

    Q2 = rng.choice(["Founder", "Procurement", "Marketing", "Product & R&D", "Operations"], N)
    Q2 = np.where(Q3 == "D2C startup", rng.choice(["Founder", "Marketing", "Product & R&D"], N, p=[.6, .25, .15]), Q2)

    rev_levels = ["Under 1 Cr", "1-10 Cr", "10-50 Cr", "50-250 Cr", "Above 250 Cr"]
    rev_p = {
        "D2C startup": [.35, .45, .15, .05, 0],
        "SME brand": [.10, .40, .35, .13, .02],
        "Large FMCG": [0, 0, .05, .30, .65],
        "Hotel/hospitality supplier": [.10, .40, .35, .13, .02],
        "Pharma company": [0, .10, .30, .35, .25],
    }
    Q4_idx = np.array([rng.choice(5, p=rev_p[p]) for p in Q3])
    Q4 = np.array(rev_levels)[Q4_idx]

    Q5 = rng.choice(["Metro", "Tier 1", "Tier 2-3"], N, p=[.50, .30, .20])
    chan_p = {
        "D2C startup": (["Own website", "Marketplaces", "Modern retail"], [.50, .40, .10]),
        "SME brand": (["General trade", "Modern retail", "Marketplaces", "Own website"], [.40, .25, .25, .10]),
        "Large FMCG": (["General trade", "Modern retail", "Marketplaces"], [.55, .30, .15]),
        "Hotel/hospitality supplier": (["Institutional (B2B)"], [1.0]),
        "Pharma company": (["Institutional (B2B)", "General trade", "Modern retail"], [.45, .40, .15]),
    }
    Q6 = np.array([rng.choice(chan_p[p][0], p=chan_p[p][1]) for p in Q3])

    fmt_p = {
        "D2C startup": (["Sachets", "Mini bottles", "Tubes", "Vials", "None"], [.30, .30, .15, .10, .15]),
        "SME brand": (["Sachets", "Mini bottles", "Tubes", "None"], [.55, .20, .10, .15]),
        "Large FMCG": (["Sachets", "Mini bottles", "Tubes"], [.80, .12, .08]),
        "Hotel/hospitality supplier": (["Mini bottles", "Sachets", "Tubes"], [.60, .30, .10]),
        "Pharma company": (["Sachets", "Vials", "Tubes", "Mini bottles"], [.35, .30, .20, .15]),
    }
    Q7 = np.array([rng.choice(fmt_p[p][0], p=fmt_p[p][1]) for p in Q3])

    # Q8 monthly single-use units (log-normal by persona and revenue)
    base_units = {"D2C startup": 8_000, "SME brand": 40_000, "Large FMCG": 1_500_000,
                  "Hotel/hospitality supplier": 60_000, "Pharma company": 90_000}
    Q8 = np.array([base_units[p] for p in Q3]) * np.exp(0.45 * (Q4_idx - 1.5)) * rng.lognormal(0, 0.6, N)
    Q8 = np.where(Q7 == "None", 0, Q8).round(-2)

    # Q9 current cost per unit (Rs) by format
    cost_mu = {"Sachets": 0.9, "Mini bottles": 7.0, "Tubes": 6.0, "Vials": 4.5, "None": np.nan}
    Q9 = np.array([cost_mu[f] for f in Q7]) * rng.lognormal(0, 0.30, N)
    Q9 = np.where(Q3 == "Large FMCG", Q9 * 0.75, Q9).round(2)

    out_p = {"D2C startup": [.65, .25, .10], "SME brand": [.35, .40, .25], "Large FMCG": [.15, .45, .40],
             "Hotel/hospitality supplier": [.55, .30, .15], "Pharma company": [.40, .40, .20]}
    Q10 = np.array([rng.choice(["Fully outsourced", "Partly outsourced", "Fully in-house"], p=out_p[p]) for p in Q3])

    # ---------- attitudes (Likert 1-5) driven by persona ----------
    def likert(mu, sd=0.95):
        return np.clip(np.rint(rng.normal(mu, sd)), 1, 5).astype(int)

    pm = lambda d: np.array([d[p] for p in Q3])
    Q11 = likert(pm({"D2C startup": 3.3, "SME brand": 3.4, "Large FMCG": 3.0, "Hotel/hospitality supplier": 3.9, "Pharma company": 3.2})
                 + np.where(Q7 == "Sachets", 0.4, 0))
    Q12 = likert(pm({"D2C startup": 4.2, "SME brand": 3.3, "Large FMCG": 3.2, "Hotel/hospitality supplier": 3.8, "Pharma company": 2.8})
                 + np.where(np.isin(Q1, ["Cosmetics", "Perfume"]), 0.4, 0))
    Q13 = likert(pm({"D2C startup": 3.9, "SME brand": 3.0, "Large FMCG": 3.6, "Hotel/hospitality supplier": 3.7, "Pharma company": 2.9}))
    Q14 = likert(pm({"D2C startup": 3.6, "SME brand": 3.4, "Large FMCG": 3.5, "Hotel/hospitality supplier": 4.1, "Pharma company": 4.5}))
    Q15 = likert(pm({"D2C startup": 3.0, "SME brand": 3.9, "Large FMCG": 4.4, "Hotel/hospitality supplier": 3.8, "Pharma company": 3.2}))
    Q16 = likert(pm({"D2C startup": 4.0, "SME brand": 2.9, "Large FMCG": 2.7, "Hotel/hospitality supplier": 3.1, "Pharma company": 2.9}))
    Q17 = np.rint(pm({"D2C startup": 5, "SME brand": 3, "Large FMCG": 8, "Hotel/hospitality supplier": 1.5, "Pharma company": 7})
                  * rng.lognormal(0, 0.5, N)).astype(float)

    Q18 = rng.choice(["Never", "Heard of it", "Seen or used it"], N, p=[.55, .33, .12])
    Q18 = np.where((Q3 == "D2C startup") & (rng.random(N) < .25), "Heard of it", Q18)

    # Q19 appeal: driven by pain, premium, openness, awareness, cost-sensitivity
    aw = pd.Series(Q18).map({"Never": 0, "Heard of it": 0.25, "Seen or used it": 0.5}).values
    Q19 = likert(3.0 + 0.30 * (Q11 - 3) + 0.30 * (Q12 - 3) + 0.30 * (Q16 - 3) - 0.25 * (Q15 - 3)
                 + 0.15 * (Q14 - 3) + aw, 0.8)

    # ---------- latent interest -> Q30 ----------
    pers_eff = pm({"D2C startup": 0.45, "SME brand": -0.35, "Large FMCG": -0.65,
                   "Hotel/hospitality supplier": 0.35, "Pharma company": 0.0})
    z = (-0.75 + 0.85 * (Q19 - 3) + 0.30 * (Q11 - 3) + 0.25 * (Q16 - 3) - 0.30 * (Q15 - 3)
         + 0.12 * (Q12 - 3) + pers_eff + 0.15 * np.log1p(Q17) + 0.25 * (Q10 == "Fully outsourced")
         + 0.20 * aw + rng.normal(0, 0.9, N))
    prob = 1 / (1 + np.exp(-z))
    cuts = np.quantile(z, [0.25, 0.48, 0.60, 0.80])  # roughly 20/20/12/23/25
    lvl = np.digitize(z, cuts)  # 0..4
    Q30 = np.array(["Definitely not", "Probably not", "Unsure", "Probably", "Definitely"])[lvl]

    Q21 = np.where(Q15 >= 4, rng.choice(["Cost", "Minimum order size", "Product compatibility & shelf life",
                                          "Customers unfamiliar with format", "Lead time"], N, p=[.50, .15, .15, .12, .08]),
                   rng.choice(["Cost", "Minimum order size", "Product compatibility & shelf life",
                               "Customers unfamiliar with format", "Lead time"], N, p=[.20, .22, .28, .20, .10]))

    prod_by_ind = {
        "Cosmetics": ["Serums & oils", "Lotions & gels"], "Personal care": ["Shampoo & body wash", "Lotions & gels"],
        "Food & condiments": ["Sauces & honey"], "Coffee & beverages": ["Coffee & drink concentrates"],
        "Ayurvedic": ["Ayurvedic oils", "Lotions & gels"], "Perfume": ["Perfume testers"],
        "Hotel toiletries supply": ["Shampoo & body wash", "Lotions & gels"], "Pharma": ["Medicine samples", "Lotions & gels"],
        "Other": ["Lotions & gels", "Sauces & honey"],
    }
    Q20 = []
    for i in range(N):
        if lvl[i] == 0 and rng.random() < .6:
            Q20.append("None"); continue
        opts = prod_by_ind[Q1[i]]
        pick = [opts[0]] + ([opts[1]] if len(opts) > 1 and rng.random() < .45 else [])
        Q20.append(";".join(pick))
    Q20 = np.array(Q20)

    # ---------- Q22 services with planted associations ----------
    services = ["Liquid filling", "Custom printing & branding", "Sample & trial packs", "Bulk contract manufacturing",
                "White-label ready products", "Artwork & design", "Formulation compatibility testing",
                "Kit / subscription-box assembly", "Warehousing & dispatch", "Recyclable material option"]
    S = np.zeros((N, 10), dtype=int)
    engage = np.clip(0.15 + 0.17 * lvl, 0.10, 0.85)
    for i in range(N):
        p = Q3[i]; e = engage[i]
        b = np.array([.85, .55, .30, .15, .12, .20, .25, .10, .18, .20]) * e * 1.2
        if p == "D2C startup": b[[2, 1, 7, 4]] += [.35 * e, .15 * e, .15 * e, .15 * e]
        if p == "Large FMCG": b[[3, 8]] += [.45 * e, .25 * e]
        if p == "Hotel/hospitality supplier": b[[3, 1, 9]] += [.30 * e, .20 * e, .25 * e]
        if p == "Pharma company": b[[2, 6]] += [.40 * e, .30 * e]
        if p == "SME brand": b[[4, 5]] += [.15 * e, .10 * e]
        b[9] += 0.12 * (Q13[i] - 3) * e
        b = np.clip(b, 0, .97)
        s = (rng.random(10) < b).astype(int)
        # planted rules
        if s[2] and s[1] and rng.random() < .70: s[5] = 1            # samples + printing -> artwork
        if s[0] and s[3] and rng.random() < .65: s[8] = 1            # filling + bulk -> warehousing
        if s[4] and rng.random() < .60: s[6] = 1                     # white-label -> compatibility testing
        if s[2] and p in ("D2C startup", "Hotel/hospitality supplier") and rng.random() < .40: s[7] = 1  # samples -> kits
        if lvl[i] == 0 and rng.random() < .55: s[:] = 0
        S[i] = s

    Q23 = np.array([services[np.argmax(S[i] * rng.random(10) + S[i] * np.array([.3, .1, .1, .1, 0, 0, 0, 0, 0, 0]))]
                    if S[i].sum() else "None" for i in range(N)])
    Q24 = np.array([rng.choice(["Per order", "Monthly contract", "Annual contract"],
                               p=[.55, .30, .15] if lvl[i] < 3 else [.30, .42, .28]) for i in range(N)])

    # ---------- spend ----------
    wtp_base = {"Cosmetics": 6.5, "Perfume": 8.0, "Personal care": 3.8, "Food & condiments": 2.6,
                "Coffee & beverages": 3.6, "Ayurvedic": 4.2, "Hotel toiletries supply": 3.4, "Pharma": 5.5, "Other": 3.5}
    Q25 = (np.array([wtp_base[x] for x in Q1]) * np.exp(0.10 * (Q12 - 3) - 0.12 * (Q15 - 3) + 0.08 * (lvl - 2))
           * rng.lognormal(0, 0.22, N)).round(2)

    trial_share = np.array([.02, .05, .12, .25, .40])[lvl]
    size_units = np.where(Q8 > 0, Q8, pm(base_units) * 0.3)
    Q26 = np.where(lvl == 0, 0, size_units * trial_share * rng.lognormal(0, 0.45, N)
                   * (1 + 0.6 * (Q3 == "D2C startup"))).round(-2)
    Q26 = np.maximum(Q26, np.where(lvl >= 3, 2000, 0))
    nserv = S.sum(1)
    Q27 = (Q25 * Q26 * rng.uniform(0.7, 1.05, N) * (1 + 0.04 * nserv)).round(-2)
    Q27 = np.where(lvl == 0, 0, Q27)

    Q28 = np.where(np.isin(Q3, ["Large FMCG"]), rng.choice(["10,000", "50,000+"], N, p=[.3, .7]),
                   np.where(Q3 == "D2C startup", rng.choice(["1,000", "5,000", "10,000"], N, p=[.55, .35, .10]),
                            rng.choice(["1,000", "5,000", "10,000", "50,000+"], N, p=[.15, .40, .35, .10])))
    Q29 = np.where(rng.random(N) < np.array([.05, .18, .40, .68, .85])[lvl], "Yes", "No")

    df = pd.DataFrame({
        "RespondentID": [f"R{i+1:04d}" for i in range(N)],
        "Q1_Industry": Q1, "Q2_Role": Q2, "Q3_CompanyType": Q3, "Q4_Revenue": Q4, "Q5_Location": Q5,
        "Q6_Channel": Q6, "Q7_CurrentFormat": Q7, "Q8_MonthlyUnits": Q8, "Q9_CostPerUnit_Rs": Q9,
        "Q10_PackagingMode": Q10, "Q11_LeakPain": Q11, "Q12_PremiumImportance": Q12, "Q13_Sustainability": Q13,
        "Q14_Hygiene": Q14, "Q15_PriceSensitivity": Q15, "Q16_Openness": Q16, "Q17_SamplingCampaigns": Q17,
        "Q18_Awareness": Q18, "Q19_Appeal": Q19, "Q20_Products": Q20, "Q21_TopConcern": Q21,
    })
    for j, s in enumerate(services):
        df[f"Q22_{s}"] = S[:, j]
    df["Q23_TopService"] = Q23
    df["Q24_EngagementModel"] = Q24
    df["Q25_WTP_PerCard_Rs"] = Q25
    df["Q26_MonthlyCards"] = Q26
    df["Q27_MonthlyBudget_Rs"] = Q27
    df["Q28_MaxMOQ"] = Q28
    df["Q29_FreeTest"] = Q29
    df["Q30_PilotLikelihood"] = Q30

    # ---------- inject messiness ----------
    log = {}
    lik = ["Q11_LeakPain", "Q12_PremiumImportance", "Q13_Sustainability", "Q14_Hygiene",
           "Q15_PriceSensitivity", "Q16_Openness", "Q19_Appeal"]
    m = rng.random((N, len(lik))) < 0.05
    for k, c in enumerate(lik):
        df.loc[m[:, k], c] = np.clip(df.loc[m[:, k], c] + rng.choice([-1, 1], m[:, k].sum()), 1, 5)
    log["likert_noise_cells"] = int(m.sum())

    for c, how in [("Q8_MonthlyUnits", "x"), ("Q9_CostPerUnit_Rs", "typo"), ("Q25_WTP_PerCard_Rs", "typo"),
                   ("Q27_MonthlyBudget_Rs", "x")]:
        idx = rng.choice(N, int(0.025 * N), replace=False)
        if how == "x":
            df.loc[idx, c] = df.loc[idx, c] * rng.uniform(15, 60, len(idx))
        else:
            df.loc[idx, c] = df.loc[idx, c] * rng.choice([100, 1000], len(idx))  # decimal-point typos
        log[f"outliers_{c}"] = len(idx)

    for c in ["Q9_CostPerUnit_Rs", "Q17_SamplingCampaigns", "Q25_WTP_PerCard_Rs", "Q26_MonthlyCards", "Q27_MonthlyBudget_Rs"]:
        idx = rng.choice(N, int(0.04 * N), replace=False)
        df.loc[idx, c] = np.nan
        log[f"missing_{c}"] = len(idx)
    return df, log


def clean(raw):
    """Return (cleaned DataFrame, outlier report)."""
    df = raw.copy()
    lvl = df["Q30_PilotLikelihood"].map(LVL)
    report = []
    # outliers: IQR fence on the log scale, within each company type
    for c in MONEY:
        flagged = pd.Series(False, index=df.index)
        for g, sub in df.groupby("Q3_CompanyType"):
            v = np.log1p(sub[c].where(sub[c] > 0))
            q1, q3 = v.quantile([.25, .75])
            flagged.loc[sub.index] = v > q3 + 1.5 * (q3 - q1)
        flagged = flagged.fillna(False).astype(bool)
        df.loc[flagged, c] = np.nan
        report.append({"column": c, "flagged": int(flagged.sum())})
    # missing values: median of the same company type and pack format
    for c in ["Q8_MonthlyUnits", "Q9_CostPerUnit_Rs", "Q17_SamplingCampaigns", "Q25_WTP_PerCard_Rs"]:
        df[c] = df[c].fillna(df.groupby(["Q3_CompanyType", "Q7_CurrentFormat"])[c].transform("median"))
        df[c] = df[c].fillna(df.groupby("Q3_CompanyType")[c].transform("median"))
    df["Q9_CostPerUnit_Rs"] = df["Q9_CostPerUnit_Rs"].where(df["Q7_CurrentFormat"] != "None", 0)
    # spend: 0 for "Definitely not", else median of the same type and pilot level
    for c in ["Q26_MonthlyCards", "Q27_MonthlyBudget_Rs"]:
        med = df.assign(lvl=lvl).groupby(["Q3_CompanyType", "lvl"])[c].transform("median")
        df[c] = df[c].fillna(pd.Series(np.where(lvl == 0, 0, med), index=df.index))
        df[c] = df[c].fillna(df.groupby("Q3_CompanyType")[c].transform("median"))
    return df, pd.DataFrame(report)
