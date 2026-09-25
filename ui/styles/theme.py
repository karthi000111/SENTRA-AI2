"""Central visual language for the SENTRA AI Streamlit workspace."""


def inject_theme(st: object) -> None:
    st.markdown("""
    <style>
      .stApp { background: #08111f; color: #e8eef7; }
      [data-testid="stSidebar"] { background: #0c192b; border-right: 1px solid #203554; }
      h1, h2, h3 { color: #f4f8ff; letter-spacing: -0.02em; }
      .sentra-header { padding: 1.2rem 0 1.4rem; border-bottom: 1px solid #203554; margin-bottom: 1.5rem; }
      .brand { font-size: 2rem; font-weight: 750; letter-spacing: .08em; color: #77d7ff; }
      .subtitle, .muted { color: #a9b9ce; }
      .status { color: #73e6ad; font-weight: 650; font-size: .86rem; padding-top: .55rem; text-align: right; }
      .card { background: #101f34; border: 1px solid #243d60; border-radius: 12px; padding: 1.15rem 1.25rem; margin: .7rem 0; }
      .answer-card { border-left: 4px solid #77d7ff; }
      .label { color: #86a1c3; font-size: .74rem; font-weight: 700; letter-spacing: .11em; }
      .metric-value { color: #f4f8ff; font-size: 1.35rem; font-weight: 700; }
      .ready { color: #73e6ad; font-weight: 700; } .empty { color: #f6c76d; font-weight: 700; }
      .session-id { font-family: monospace; color: #9ecfff; overflow-wrap: anywhere; }
      .evidence-text { color: #c9d5e5; line-height: 1.55; }
      div.stButton > button { border-radius: 8px; font-weight: 650; }
      [data-testid="stMetricValue"] { overflow-wrap: break-word; white-space: normal; line-height: 1.2; font-size: 1.2rem; }
      [data-testid="stMetricLabel"] { overflow-wrap: break-word; white-space: normal; }
    </style>
    """, unsafe_allow_html=True)
