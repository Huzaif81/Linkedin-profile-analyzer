import streamlit as st
import pdfplumber
import json
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# 1. Global Page Config & Branding Setup
st.set_page_config(
    page_title="ProProfile AI | Portfolio Optimization Suite",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Premium Professional UI Injector Layer (CSS)
st.markdown("""
    <style>
    /* Global App Container Adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1250px;
    }
    
    /* Modern Glassmorphism Hero Frame */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(45deg, #3B82F6, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 400;
    }

    /* Functional Metric Display Dashboard Panel */
    .metric-panel {
        background: #1E293B;
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.75rem;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-panel:hover {
        transform: translateY(-2px);
        border-color: #3B82F6;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: #3B82F6;
        margin-top: 0.5rem;
    }

    /* Structural Section Formatting overrides */
    h3 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px 6px 0px 0px;
        padding: 0.75rem 1.5rem;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border-color: #3B82F6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Permanent Key Initialization (Pulls from Background Secrets Vault)
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel("gemini-2.5-flash")
except Exception:
    st.error("🔑 Environment Configuration Missing: Please declare 'GEMINI_API_KEY' within your cloud dashboard secrets framework panel.")
    st.stop()

# 4. Load Text Vectorizer Model into Memory Cache
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

v_model = load_embedding_model()

# --- HERO PRESENTATION LAYER ---
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">💎 ProProfile AI Studio</div>
        <div class="hero-subtitle">Deploy semantic vector models and generative deep-scans to engineer high-conversion professional portfolios.</div>
    </div>
""", unsafe_allow_html=True)

# --- WORKSPACE GRID LAYOUT ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📂 Profile Ingestion")
    uploaded_file = st.file_uploader(
        "Upload LinkedIn Profile PDF or Resume", 
        type=["pdf"],
        label_visibility="collapsed"
    )
    
    # Practical Placeholder Data for a Data Scientist role
    default_jd_placeholder = (
        "We are seeking a Data Scientist to build predictive models, design SQL schemas, "
        "and deploy analytical business intelligence dashboards using Python and Power BI..."
    )

with col_right:
    st.markdown("### 🎯 Target Industry Benchmark")
    job_description = st.text_area(
        "Target Job Description",
        placeholder=default_jd_placeholder,
        height=125,
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- CORE ANALYTICAL PROCESSING PIPELINE ---
if uploaded_file and job_description.strip():
    
    # Primary Action Button Center Alignment Grid
    btn_col_1, btn_col_2, btn_col_3 = st.columns([1, 2, 1])
    with btn_col_2:
        execute_click = st.button("🚀 Run Comprehensive Dual-Engine Analysis", type="primary", use_container_width=True)
        
    if execute_click:
        # Ingestion parsing
        with pdfplumber.open(uploaded_file) as pdf:
            profile_text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        st.session_state['profile_text'] = profile_text
        
        # Engine A: Compute Vector Mathematics
        with st.spinner("Processing deep text transformations..."):
            profile_emb = v_model.encode(profile_text, convert_to_tensor=True)
            jd_emb = v_model.encode(job_description, convert_to_tensor=True)
            match_score = round(util.cos_sim(profile_emb, jd_emb).item() * 100, 1)
        
        # Engine B: Direct Generative Evaluation Pipeline
        try:
            advanced_prompt = f"""
            You are an elite corporate technical recruiter. Analyze the candidate's Profile Text against the Target Job Description.
            
            Profile Text: {profile_text}
            Target Job Description: {job_description}
            
            Respond strictly with a valid JSON object matching this structural layout:
            {{
                "headline_suggestions": ["Suggested Headline 1", "Suggested Headline 2"],
                "about_section_critique": "Detailed optimization feedback for their LinkedIn Summary section",
                "optimized_about_text": "A fully rewritten, keyword-rich LinkedIn Summary paragraph emphasizing data skills",
                "work_experience_corrections": [
                    {{"original_bullet": "handled analytics tasks", "corrected_bullet": "Engineered automated data pipelines using Python and SQL, increasing team diagnostic reporting efficiency by 20%", "reasoning": "Inject action metrics"}}
                ],
                "related_trending_jobs": ["Job Title 1", "Job Title 2"],
                "recommended_next_projects": ["Detailed project concept 1 focused on filling their skill gaps", "Detailed project concept 2"]
            }}
            """
            
            with st.spinner("Orchestrating model evaluation streams..."):
                response = llm_model.generate_content(
                    advanced_prompt, 
                    generation_config={"response_mime_type": "application/json"}
                )
                ai_data = json.loads(response.text)
            
            # --- RENDER RESULTS REGION ---
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Render custom CSS glassmorphism metric widget card
            st.markdown(f"""
                <div class="metric-panel">
                    <div class="metric-label">Overall Semantic Alignment Coefficient</div>
                    <div class="metric-value">{match_score}% Fit Factor</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Segment outputs across Tab interfaces
            tab_branding, tab_experience, tab_market = st.tabs([
                "✨ Executive Profile Branding", 
                "🛠️ Experience Refinement", 
                "📈 Strategic Market Roadmap"
            ])
            
            with tab_branding:
                st.markdown("### 💡 High-Conversion Headlines")
                st.caption("Swap your current headline for one of these keyword-optimized structures to improve search discoverability:")
                for h in ai_data.get("headline_suggestions", []):
                    st.code(h, language="text")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📝 Profile Summary Critique")
                st.info(ai_data.get("about_section_critique"))
                
                st.markdown("### 🛠️ Generated Profile Summary Copy")
                st.caption("Ready to copy and paste directly into your LinkedIn 'About' panel:")
                st.markdown(f"> *{ai_data.get('optimized_about_text')}*")
                
            with tab_experience:
                st.markdown("### 🔄 Resume Bullet-Point Optimizations")
                st.caption("Transform passive task statements into performance-driven, data-backed metric lines:")
                st.table(ai_data.get("work_experience_corrections", []))
                
            with tab_market:
                st.markdown("### 🎯 Parallel Career Trajectories")
                st.caption("Based on the mathematical density of your skill vectors, you have an advantage in these adjacent sectors:")
                for job in ai_data.get("related_trending_jobs", []):
                    st.markdown(f"- **{job}**")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 🚀 Strategic Portfolio Project Blueprints")
                st.caption("Build these futuristic micro-projects to automatically patch the structural knowledge gaps discovered during the scan:")
                for proj in ai_data.get("recommended_next_projects", []):
                    st.markdown(f"👉 {proj}")
                    
        except Exception as err:
            st.error(f"Core Interface Engine Error: {err}")
else:
    st.info("💡 Application Dashboard Status: Provide an initial profile asset and target parameter set to deploy algorithms.")