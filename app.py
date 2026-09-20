
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="ISMS Control Centre",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME / CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --bg:#06162d; --panel:#0a2345; --panel2:#0d2d57; --line:#244b78;
  --text:#f7f9fc; --muted:#9db2ca; --blue:#2f78c9; --blue2:#174d88;
  --cream:#fff1cf; --green:#42c98a; --amber:#d8a53b; --red:#d85b67;
}
html, body, [class*="css"] {font-family:Inter, sans-serif;}
.stApp {background:
 radial-gradient(circle at 75% -10%, rgba(47,120,201,.16), transparent 30%),
 radial-gradient(circle at 10% 0%, rgba(25,77,136,.20), transparent 28%),
 var(--bg); color:var(--text);}
.block-container{max-width:1550px;padding:1.1rem 2rem 3rem;}
section[data-testid="stSidebar"]{background:#06162d;border-right:1px solid var(--line);}
section[data-testid="stSidebar"] > div{padding:1rem .85rem;}
section[data-testid="stSidebar"] .stRadio label{
  border-radius:10px;padding:.48rem .6rem;margin:.1rem 0;color:#b6bfd0;
}
section[data-testid="stSidebar"] .stRadio label:hover{background:#0d2d57;color:#fff;}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{gap:0;}
h1,h2,h3,h4{color:var(--text)!important;letter-spacing:-.025em;}
h1{font-weight:800!important;}
hr{border-color:var(--line)!important;}

.brand{display:flex;align-items:center;gap:.75rem;padding:.3rem .3rem 1.2rem;}
.brand-logo{
 width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;
 background:linear-gradient(135deg,#174d88,#2f78c9);font-size:21px;
 box-shadow:0 0 28px rgba(47,120,201,.25);
}
.brand-title{font-size:1.03rem;font-weight:800;color:#fff;}
.brand-sub{font-size:.68rem;color:#758198;margin-top:.1rem;}
.navlabel{font-size:.66rem;letter-spacing:.12em;text-transform:uppercase;color:#647087;font-weight:700;margin:.9rem .35rem .35rem;}
.footerbox{margin-top:2rem;padding:1rem;border-top:1px solid var(--line);color:#68758d;font-size:.72rem;line-height:1.5;}

.topline{display:flex;justify-content:space-between;align-items:center;color:#758198;font-size:.75rem;margin-bottom:.65rem;}
.pill{border:1px solid #26344d;background:#0d1626;border-radius:999px;padding:.28rem .6rem;color:#aeb9cb;}
.hero{
 padding:1.65rem 1.8rem;border:1px solid #222f48;border-radius:18px;
 background:linear-gradient(135deg,rgba(10,38,73,.98),rgba(7,25,49,.98));
 box-shadow:0 18px 50px rgba(0,0,0,.24);margin-bottom:1.05rem;
}
.hero .eyebrow{font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:#8fb9e7;font-weight:800;}
.hero h1{font-size:2.05rem!important;margin:.35rem 0 .35rem!important;}
.hero p{color:#9ca9bd;margin:0;max-width:850px;line-height:1.55;}

.kpi{
 background:linear-gradient(180deg,#101a2b,#0c1422);border:1px solid #1d2b43;
 border-radius:15px;padding:1rem 1.05rem;min-height:105px;
 box-shadow:0 10px 30px rgba(0,0,0,.14);
}
.kpi-icon{font-size:1.25rem;margin-bottom:.45rem;}
.kpi-label{font-size:.73rem;color:#8290a8;}
.kpi-value{font-size:1.55rem;font-weight:800;color:#fff;margin-top:.12rem;}
.kpi-meta{font-size:.68rem;color:#68758b;margin-top:.2rem;}

.card{
 background:linear-gradient(180deg,#0f1727,#0c1421);border:1px solid #1d2a42;
 border-radius:15px;padding:1.1rem 1.15rem;margin-bottom:1rem;
}
.card-title{font-size:.98rem;font-weight:750;color:#f4f6fb;margin-bottom:.2rem;}
.card-sub{font-size:.73rem;color:#758198;margin-bottom:.8rem;}

.badge{display:inline-block;padding:.25rem .55rem;border-radius:999px;font-size:.7rem;font-weight:700;}
.badge-low{background:rgba(34,197,94,.12);color:#5ee38a;border:1px solid rgba(34,197,94,.25);}
.badge-medium{background:rgba(245,158,11,.12);color:#ffc35a;border:1px solid rgba(245,158,11,.25);}
.badge-high{background:rgba(47,120,201,.12);color:#9fc8ef;border:1px solid rgba(47,120,201,.25);}
.badge-critical{background:rgba(239,68,68,.12);color:#ff7b7b;border:1px solid rgba(239,68,68,.25);}

.section{font-size:1.05rem;font-weight:750;margin:.8rem 0 .55rem;}
.muted{color:#7e8ba2;font-size:.76rem;}
.metricline{display:flex;justify-content:space-between;padding:.5rem 0;border-bottom:1px solid #18243a;font-size:.78rem;}
.metricline:last-child{border-bottom:0;}
.progress-wrap{background:#162237;border-radius:999px;height:8px;overflow:hidden;margin-top:.45rem;}
.progress{height:100%;border-radius:999px;background:linear-gradient(90deg,#174d88,#4b91d1);}

div[data-testid="stMetric"]{
 background:#0d1626;border:1px solid #1d2b43;border-radius:13px;padding:.7rem .8rem;
}
div[data-testid="stMetricLabel"]{color:#8390a7!important;font-size:.72rem!important;}
div[data-testid="stMetricValue"]{color:#fff!important;font-size:1.45rem!important;}

.stButton>button{
 border-radius:10px;border:1px solid #283750;background:#121d31;color:#f7f8fc;
 font-weight:650;
}
.stButton>button:hover{border-color:#8fb9e7;background:#191d31;}
.stDownloadButton>button{border-radius:10px;}
[data-testid="stDataFrame"]{border:1px solid #1d2b43;border-radius:12px;overflow:hidden;}
.stSelectbox label,.stRadio label,.stSlider label,.stTextInput label,.stNumberInput label,.stDateInput label{color:#aeb9cb!important;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA / STATE
# ============================================================
AWARENESS = [
 ("A1","Password Safety","Do you use unique passwords for important accounts?"),
 ("A2","MFA","Do you use Multi-Factor Authentication (MFA) when it is available?"),
 ("A3","Phishing","Can you identify common signs of a phishing message?"),
 ("A4","Links","Do you verify links before opening them?"),
 ("A5","Attachments","Do you verify unexpected attachments before opening them?"),
 ("A6","Updates","Do you install important security updates promptly?"),
 ("A7","Device Security","Do you lock your device when leaving it unattended?"),
 ("A8","Incident Reporting","Do you know where to report a security incident?"),
 ("A9","Data Handling","Do you use approved channels for sensitive information?"),
 ("A10","Training","Have you received cybersecurity-awareness training?"),
]

COMPLIANCE = [
 ("Q1","Information Security Policies","Is there a documented information-security policy available to employees?"),
 ("Q2","Information Security Policies","Are employees informed about their responsibilities under the security policy?"),
 ("Q3","Asset Management","Are important hardware and software assets identified and recorded?"),
 ("Q4","Asset Management","Is asset ownership or responsibility assigned?"),
 ("Q5","Access Control","Are user accounts created only after appropriate authorization?"),
 ("Q6","Access Control","Are access rights based on job responsibilities?"),
 ("Q7","Authentication","Is Multi-Factor Authentication used for important systems where feasible?"),
 ("Q8","Authentication","Are shared user accounts avoided unless formally justified?"),
 ("Q9","Password Security","Are users required to use strong and unique passwords?"),
 ("Q10","Password Security","Are passwords protected from unauthorized disclosure or sharing?"),
 ("Q12","Data Protection","Is sensitive information classified or identified appropriately?"),
 ("Q13","Data Protection","Are removable-media risks addressed through policy or technical controls?"),
 ("Q14","Backup & Recovery","Are important data and systems backed up according to a defined schedule?"),
 ("Q15","Backup & Recovery","Are backups protected from unauthorized access or accidental deletion?"),
 ("Q16","Incident Management","Do employees know how to report a suspected security incident?"),
 ("Q17","Incident Management","Is there a documented incident-response procedure?"),
 ("Q18","Employee Awareness","Do employees receive cybersecurity awareness training?"),
 ("Q19","Employee Awareness","Does training cover phishing and social engineering?"),
 ("Q20","Physical Security","Are workstations locked when unattended?"),
 ("Q21","Physical Security","Are restricted areas protected from unauthorized access?"),
 ("Q22","Network & System Security","Are operating systems and applications updated regularly?"),
 ("Q23","Network & System Security","Are unauthorized software installations restricted?"),
 ("Q24","Logging & Monitoring","Are important security events logged?"),
 ("Q25","Logging & Monitoring","Are logs reviewed or monitored for suspicious activity?"),
 ("Q26","Business Continuity","Is there a plan for maintaining critical operations during disruption?"),
 ("Q27","Business Continuity","Are continuity or recovery procedures tested?"),
 ("Q28","Security Policies","Are employees required to follow acceptable-use requirements?"),
 ("Q29","Security Policies","Are policy violations or repeated unsafe practices addressed?"),
 ("Q30","Security Policies","Are security responsibilities clearly assigned?"),
]

STATUS_SCORE = {"Implemented":1.0,"Partially Implemented":0.5,"Not Implemented":0.0,"Not Applicable":np.nan}

def risk_category(score):
    if score <= 4: return "Low"
    if score <= 9: return "Medium"
    if score <= 16: return "High"
    return "Critical"

def risk_badge(cat):
    return f'<span class="badge badge-{cat.lower()}">{cat}</span>'

def compliance_calc(statuses):
    vals=[STATUS_SCORE[s] for s in statuses]
    vals=[v for v in vals if not pd.isna(v)]
    return round(np.mean(vals)*100,1) if vals else np.nan

def domain_scores(status_map):
    rows=[]
    for qid,domain,q in COMPLIANCE:
        rows.append({"ID":qid,"Domain":domain,"Question":q,"Status":status_map[qid],"Score":STATUS_SCORE[status_map[qid]]})
    df=pd.DataFrame(rows)
    out=[]
    for d,g in df.groupby("Domain"):
        a=g["Score"].dropna()
        out.append({"Domain":d,"Compliance %":round(a.mean()*100,1) if len(a) else np.nan})
    return pd.DataFrame(out).sort_values("Compliance %")

def recommendations(gaps, score):
    rec=[]
    if score >= 17: rec.append("Prioritize treatment of the current Critical risk and assign an owner.")
    elif score >= 10: rec.append("Prioritize treatment planning for the current High risk.")
    for _,r in gaps.head(7).iterrows():
        action="Implement and assign ownership." if r.Status=="Not Implemented" else "Complete missing elements and verify effectiveness."
        rec.append(f"{r.Domain} — {r.ID}: {action}")
    return rec or ["Maintain current controls and continue periodic review."]

if "org" not in st.session_state:
    st.session_state.org={"Organization":"Academic Demonstration Organization","Department":"Information Security","Employees":50,"Assessor":"Student Assessor","Date":date.today()}
if "awareness" not in st.session_state:
    st.session_state.awareness={qid:None for qid,_,_ in AWARENESS}
if "compliance" not in st.session_state:
    st.session_state.compliance={qid:"Not Implemented" for qid,_,_ in COMPLIANCE}
if "risk" not in st.session_state:
    st.session_state.risk={"Likelihood":3,"Impact":3}

aw_values=[v for v in st.session_state.awareness.values() if v is not None]
aw_score=round(sum(v=="Yes" for v in aw_values)/len(aw_values)*100,1) if aw_values else 0.0
comp_score=compliance_calc(list(st.session_state.compliance.values()))
domains=domain_scores(st.session_state.compliance)
likelihood=st.session_state.risk["Likelihood"]; impact=st.session_state.risk["Impact"]
risk_score=likelihood*impact; risk_cat=risk_category(risk_score)

gaps=[]
for qid,domain,q in COMPLIANCE:
    s=st.session_state.compliance[qid]
    if s in ("Not Implemented","Partially Implemented"):
        gaps.append({"ID":qid,"Domain":domain,"Question":q,"Status":s})
gaps_df=pd.DataFrame(gaps)

# ============================================================
# DEMO ML DATA
# ============================================================
@st.cache_data
def demo_ml_data(n=200,seed=42):
    rng=np.random.default_rng(seed)
    compliance=rng.uniform(45,95,n); awareness=rng.uniform(40,95,n)
    likelihood=rng.integers(1,6,n); impact=rng.integers(1,6,n)
    password=np.clip(compliance+rng.normal(0,12,n),20,100)
    access=np.clip(compliance+rng.normal(0,10,n),20,100)
    backup=np.clip(compliance+rng.normal(0,13,n),20,100)
    incident=np.clip(compliance+rng.normal(0,11,n),20,100)
    employee=np.clip(awareness+rng.normal(0,8,n),20,100)
    latent=(.35*(100-compliance)/20+.20*(100-awareness)/20+.15*(100-password)/20+
            .10*(100-access)/20+.10*(100-backup)/20+.10*(100-incident)/20+
            .45*likelihood+.45*impact)
    cat=pd.cut(latent,[-np.inf,3.4,5.6,7.7,np.inf],labels=["Low","Medium","High","Critical"]).astype(str)
    return pd.DataFrame({
        "Compliance_Score":compliance.round(1),"Awareness_Score":awareness.round(1),
        "Likelihood":likelihood,"Impact":impact,"Password_Score":password.round(1),
        "Access_Control_Score":access.round(1),"Backup_Score":backup.round(1),
        "Incident_Score":incident.round(1),"Employee_Awareness_Score":employee.round(1),
        "Risk_Category":cat
    })

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("logo.svg", use_container_width=True)
st.sidebar.markdown('<div class="brand-sub" style="padding:0 4px 12px;">Security · Compliance · Risk</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="navlabel">Assessment</div>',unsafe_allow_html=True)
pages=[
 "Home","Organization Details","Employee Awareness","ISMS Compliance",
 "Security Gaps","Risk Assessment","Dynamic Risk","Machine Learning",
 "Final ISMS Dashboard","About / Methodology"
]
page=st.sidebar.radio("Navigation",pages,label_visibility="collapsed")

st.sidebar.markdown("""
<div class="footerbox">
<strong>Stronger Security<br>Through Awareness</strong><br><br>
ISMS Assessment Tool v2.0<br>
Academic project • ISO/IEC 27001-based
</div>
""",unsafe_allow_html=True)

# ============================================================
# PAGE HELPERS
# ============================================================
def page_header(title, subtitle):
    st.markdown(f"""
    <div class="topline"><span>ISMS Control Centre</span><span class="pill">Academic Assessment</span></div>
    <div style="display:flex;align-items:flex-start;gap:.9rem;margin:.25rem 0 1rem;">
      <div><h1 style="margin:0;font-size:2rem;">{title}</h1><div class="muted" style="margin-top:.25rem;">{subtitle}</div></div>
    </div>
    """,unsafe_allow_html=True)

def kpi(label,value,meta):
    return f"""<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-meta">{meta}</div></div>"""

# ============================================================
# HOME
# ============================================================
if page=="Home":
    st.markdown("""
    <div class="topline"><span>ISMS Control Centre</span><span class="pill">Assessment Workspace</span></div>
    <div class="hero">
      <div class="eyebrow">Security • Compliance • Risk</div>
      <h1>ISMS Control Centre</h1>
      <p>A focused assessment workspace for employee cybersecurity awareness, information-security compliance,
      security gaps, risk evaluation and ML-supported risk classification.</p>
    </div>
    """,unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    c1.markdown(kpi("Awareness Score",f"{aw_score:.1f}%","Employee assessment"),unsafe_allow_html=True)
    c2.markdown(kpi("Compliance Score","N/A" if pd.isna(comp_score) else f"{comp_score:.1f}%","ISMS controls"),unsafe_allow_html=True)
    c3.markdown(kpi("Risk Score",str(risk_score),"Likelihood × Impact"),unsafe_allow_html=True)
    c4.markdown(kpi("Risk Level",risk_cat,"Current assessment"),unsafe_allow_html=True)

    st.markdown('<div class="section">Assessment Overview</div>',unsafe_allow_html=True)
    a,b=st.columns([1.45,1])
    with a:
        st.markdown("""
        <div class="card">
          <div class="card-title">Assessment workflow</div>
          <div class="card-sub">Move from evidence collection to actionable security insight.</div>
          <div class="metricline"><span>01 &nbsp; Organization Details</span><span>Setup</span></div>
          <div class="metricline"><span>02 &nbsp; Employee Awareness</span><span>10 questions</span></div>
          <div class="metricline"><span>03 &nbsp; ISMS Compliance</span><span>30 controls</span></div>
          <div class="metricline"><span>04 &nbsp; Security Gaps</span><span>Identify weaknesses</span></div>
          <div class="metricline"><span>05 &nbsp; Risk Assessment</span><span>Likelihood × Impact</span></div>
          <div class="metricline"><span>06 &nbsp; ML Classification</span><span>Supporting prediction</span></div>
          <div class="metricline"><span>07 &nbsp; Final Dashboard</span><span>Executive view</span></div>
        </div>
        """,unsafe_allow_html=True)
    with b:
        gap_count=len(gaps_df)
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Current posture</div>
          <div class="card-sub">Live values from this assessment session.</div>
          <div class="metricline"><span>Awareness</span><strong>{aw_score:.1f}%</strong></div>
          <div class="metricline"><span>Compliance</span><strong>{"N/A" if pd.isna(comp_score) else f"{comp_score:.1f}%"}</strong></div>
          <div class="metricline"><span>Open gaps</span><strong>{gap_count}</strong></div>
          <div class="metricline"><span>Risk</span><strong>{risk_score} · {risk_cat}</strong></div>
          <div style="margin-top:.8rem">{risk_badge(risk_cat)}</div>
        </div>
        """,unsafe_allow_html=True)

# ============================================================
# ORG
# ============================================================
elif page=="Organization Details":
    page_header("Organization Details","Set the context for the current assessment.")
    with st.form("org"):
        c1,c2=st.columns(2)
        org=c1.text_input("Organization",st.session_state.org["Organization"])
        dept=c2.text_input("Department",st.session_state.org["Department"])
        c3,c4=st.columns(2)
        emp=c3.number_input("Employees",1,100000,int(st.session_state.org["Employees"]))
        assessor=c4.text_input("Assessor",st.session_state.org["Assessor"])
        adate=st.date_input("Assessment Date",st.session_state.org["Date"])
        if st.form_submit_button("Save Assessment Details",type="primary"):
            st.session_state.org={"Organization":org,"Department":dept,"Employees":emp,"Assessor":assessor,"Date":adate}
            st.success("Assessment details saved.")

# ============================================================
# AWARENESS
# ============================================================
elif page=="Employee Awareness":
    page_header("Employee Awareness","Measure cybersecurity awareness and reported security behaviour.")
    st.markdown('<div class="card"><div class="card-title">Awareness Assessment</div><div class="card-sub">Answer all 10 questions, then calculate the current awareness score.</div></div>',unsafe_allow_html=True)
    with st.form("awareness"):
        answers={}
        for qid,topic,q in AWARENESS:
            answers[qid]=st.radio(f"{topic} — {q}",["Yes","No"],index=None,horizontal=True,key=f"aw_{qid}")
        if st.form_submit_button("Calculate Awareness Score",type="primary"):
            if any(v is None for v in answers.values()):
                st.error("Please answer all 10 questions.")
            else:
                st.session_state.awareness=answers
                st.success(f"Awareness score: {aw_score:.1f}%")

    if aw_values:
        st.metric("Current Awareness Score",f"{aw_score:.1f}%")
        chart=pd.DataFrame({"Topic":[x[1] for x in AWARENESS],"Score":[100 if st.session_state.awareness[x[0]]=="Yes" else 0 for x in AWARENESS]})
        st.bar_chart(chart.set_index("Topic"))

# ============================================================
# COMPLIANCE
# ============================================================
elif page=="ISMS Compliance":
    page_header("ISMS Compliance Assessment","Assess selected information-security practices using the project scoring model.")
    st.markdown('<div class="card"><div class="card-title">Control Status</div><div class="card-sub">Implemented = 1 · Partially Implemented = 0.5 · Not Implemented = 0 · Not Applicable is excluded.</div></div>',unsafe_allow_html=True)
    with st.form("compliance"):
        temp={}
        for domain in list(dict.fromkeys([x[1] for x in COMPLIANCE])):
            st.markdown(f'<div class="section">{domain}</div>',unsafe_allow_html=True)
            for qid,d,q in [x for x in COMPLIANCE if x[1]==domain]:
                options=list(STATUS_SCORE.keys())
                temp[qid]=st.selectbox(q,options,index=options.index(st.session_state.compliance[qid]),key=f"cp_{qid}")
        if st.form_submit_button("Calculate Compliance",type="primary"):
            st.session_state.compliance=temp
            st.success(f"Overall compliance: {compliance_calc(list(temp.values())):.1f}%")
    if not pd.isna(comp_score):
        st.metric("Overall Compliance",f"{comp_score:.1f}%")
        st.dataframe(domains,use_container_width=True,hide_index=True)
        st.bar_chart(domains.set_index("Domain"))

# ============================================================
# GAPS
# ============================================================
elif page=="Security Gaps":
    page_header("Security Gaps","See which assessed controls require improvement.")
    if gaps_df.empty:
        st.success("No open gaps are currently recorded.")
    else:
        c1,c2=st.columns(2)
        c1.metric("Open Gaps",len(gaps_df))
        c2.metric("Not Implemented",int((gaps_df.Status=="Not Implemented").sum()))
        st.dataframe(gaps_df,use_container_width=True,hide_index=True)
        st.markdown('<div class="section">Recommendations</div>',unsafe_allow_html=True)
        for r in recommendations(gaps_df,risk_score):
            st.markdown(f"• {r}")

# ============================================================
# RISK
# ============================================================
elif page=="Risk Assessment":
    page_header("Risk Assessment","Evaluate risk using the project-defined Likelihood × Impact model.")
    c1,c2,c3=st.columns(3)
    l=c1.slider("Likelihood",1,5,likelihood)
    i=c2.slider("Impact",1,5,impact)
    rs=l*i
    c3.metric("Risk Score",rs)
    st.markdown(f'<div class="card"><div class="card-title">Current Risk</div><div style="font-size:1.4rem;font-weight:800;margin-top:.4rem">{risk_badge(risk_category(rs))}</div><div class="muted" style="margin-top:.5rem">Risk = Likelihood ({l}) × Impact ({i}) = {rs}</div></div>',unsafe_allow_html=True)
    matrix=pd.DataFrame([[risk_category(x*y) for y in range(1,6)] for x in range(1,6)],index=[f"L{x}" for x in range(1,6)],columns=[f"I{y}" for y in range(1,6)])
    st.dataframe(matrix,use_container_width=True)
    if st.button("Save Risk Assessment",type="primary"):
        st.session_state.risk={"Likelihood":l,"Impact":i}
        st.success("Risk assessment saved.")

# ============================================================
# DYNAMIC
# ============================================================
elif page=="Dynamic Risk":
    page_header("Dynamic Risk","Recalculate the risk result when likelihood or impact changes.")
    c1,c2=st.columns(2)
    l=c1.slider("Current Likelihood",1,5,likelihood)
    i=c2.slider("Current Impact",1,5,impact)
    new=l*i
    c1,c2=st.columns(2)
    c1.metric("Updated Risk Score",new)
    c2.markdown(f'<div class="card"><div class="card-title">Updated Risk Category</div><div style="margin-top:.55rem">{risk_badge(risk_category(new))}</div></div>',unsafe_allow_html=True)
    st.info(f"Saved risk: {risk_score} ({risk_cat}) → current inputs: {new} ({risk_category(new)}).")
    if st.button("Save Updated Risk",type="primary"):
        st.session_state.risk={"Likelihood":l,"Impact":i}
        st.success("Dynamic risk saved.")

# ============================================================
# ML
# ============================================================
elif page=="Machine Learning":
    page_header("Machine Learning — Risk Classification","AI-supported risk classification based on assessment features.")
    df=demo_ml_data()
    X=df.drop(columns=["Risk_Category"]); y=df["Risk_Category"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    models={
      "Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000))]),
      "Decision Tree":DecisionTreeClassifier(max_depth=5,random_state=42),
      "Random Forest":RandomForestClassifier(n_estimators=150,max_depth=7,random_state=42)
    }
    results=[]; fitted={}
    for name,m in models.items():
        m.fit(Xtr,ytr); pred=m.predict(Xte)
        results.append({"Model":name,"Accuracy":round(accuracy_score(yte,pred)*100,1)})
        fitted[name]=m
    best=max(results,key=lambda x:x["Accuracy"])
    c1,c2,c3,c4=st.columns(4)
    c1.markdown(kpi("Total Samples",len(df),"Demonstration dataset"),unsafe_allow_html=True)
    c2.markdown(kpi("Model Accuracy",f"{best['Accuracy']}%",best["Model"]),unsafe_allow_html=True)
    c3.markdown(kpi("Risk Categories","4","Low / Medium / High / Critical"),unsafe_allow_html=True)

    left,right=st.columns([1.35,1])
    with left:
        st.markdown('<div class="card"><div class="card-title">Model Performance Comparison</div><div class="card-sub">Accuracy on the assessment dataset.</div></div>',unsafe_allow_html=True)
        rpdf=pd.DataFrame(results)
        st.bar_chart(rpdf.set_index("Model"))
    with right:
        st.markdown('<div class="card"><div class="card-title">Evaluation Summary</div><div class="card-sub">Comparative performance of the tested classification models.</div></div>',unsafe_allow_html=True)
        st.dataframe(rpdf,use_container_width=True,hide_index=True)
        st.markdown(f'<div class="metricline"><span>Best evaluated model</span><strong>{best["Model"]}</strong></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="metricline"><span>Accuracy</span><strong>{best["Accuracy"]:.1f}%</strong></div>',unsafe_allow_html=True)
    st.caption("The transparent project risk score remains Likelihood × Impact.")

# ============================================================
# FINAL
# ============================================================
elif page=="Final ISMS Dashboard":
    page_header("Final ISMS Dashboard","Executive view of awareness, compliance, gaps and risk.")
    c1,c2,c3,c4=st.columns(4)
    c1.markdown(kpi("Awareness",f"{aw_score:.1f}%","Employee assessment"),unsafe_allow_html=True)
    c2.markdown(kpi("Compliance","N/A" if pd.isna(comp_score) else f"{comp_score:.1f}%","ISMS controls"),unsafe_allow_html=True)
    c3.markdown(kpi("Risk Score",risk_score,"Likelihood × Impact"),unsafe_allow_html=True)
    c4.markdown(kpi("Risk Category",risk_cat,"Current posture"),unsafe_allow_html=True)

    a,b=st.columns(2)
    with a:
        st.markdown('<div class="card"><div class="card-title">Compliance by Security Domain</div><div class="card-sub">Current assessed domain scores.</div></div>',unsafe_allow_html=True)
        if not domains.empty: st.bar_chart(domains.set_index("Domain"))
    with b:
        st.markdown('<div class="card"><div class="card-title">Awareness by Topic</div><div class="card-sub">Current employee-awareness responses.</div></div>',unsafe_allow_html=True)
        achart=pd.DataFrame({"Topic":[x[1] for x in AWARENESS],"Score":[100 if st.session_state.awareness[x[0]]=="Yes" else 0 if st.session_state.awareness[x[0]]=="No" else np.nan for x in AWARENESS]})
        st.bar_chart(achart.set_index("Topic"))

    a,b=st.columns(2)
    with a:
        st.markdown('<div class="card"><div class="card-title">Security Gaps</div><div class="card-sub">Controls requiring improvement.</div></div>',unsafe_allow_html=True)
        st.metric("Open Gaps",len(gaps_df))
        if not gaps_df.empty: st.dataframe(gaps_df[["ID","Domain","Status"]],use_container_width=True,hide_index=True)
    with b:
        st.markdown('<div class="card"><div class="card-title">Recommended Actions</div><div class="card-sub">Generated from the current gaps and risk.</div></div>',unsafe_allow_html=True)
        for r in recommendations(gaps_df,risk_score): st.markdown(f"• {r}")

    summary=pd.DataFrame({
      "Organization":[st.session_state.org["Organization"]],
      "Assessment Date":[st.session_state.org["Date"]],
      "Awareness Score":[aw_score],
      "Compliance Score":[comp_score],
      "Likelihood":[likelihood],"Impact":[impact],"Risk Score":[risk_score],"Risk Category":[risk_cat]
    })
    st.download_button("Export Assessment Summary (CSV)",summary.to_csv(index=False).encode(),file_name="isms_assessment_summary.csv",mime="text/csv")

# ============================================================
# ABOUT
# ============================================================
elif page=="About / Methodology":
    page_header("About / Methodology","How the academic assessment model works.")
    st.markdown("""
    <div class="card">
      <div class="card-title">Compliance Scoring</div>
      <div class="card-sub">Project-defined scoring model</div>
      <div class="metricline"><span>Implemented</span><strong>1.0</strong></div>
      <div class="metricline"><span>Partially Implemented</span><strong>0.5</strong></div>
      <div class="metricline"><span>Not Implemented</span><strong>0</strong></div>
      <div class="metricline"><span>Not Applicable</span><strong>Excluded from denominator</strong></div>
    </div>
    <div class="card">
      <div class="card-title">Risk Model</div>
      <div class="card-sub">Risk Score = Likelihood × Impact</div>
      <div class="metricline"><span>1–4</span><strong>Low</strong></div>
      <div class="metricline"><span>5–9</span><strong>Medium</strong></div>
      <div class="metricline"><span>10–16</span><strong>High</strong></div>
      <div class="metricline"><span>17–25</span><strong>Critical</strong></div>
    </div>
    """,unsafe_allow_html=True)
    st.info("This is an academic assessment-support prototype based on selected ISO/IEC 27001 concepts. It is not a formal certification audit.")

