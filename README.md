# Linkedin-profile-analyzer

import streamlit as st
import pdfplumber
import json
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# 1. Professional Page Configurations
st.set_page_config(
    page_title="ProProfile AI | LinkedIn Optimization Suite",
    page_icon="💼",
    layout="wide"
)

# Custom CSS styling to make the interface look crisp and modern
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 1rem; }
    .metric-card { background-color: #F3F4F6; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar Configurations
st.sidebar.title("🛠️ Control Center")
st.sidebar.markdown("Configure your AI keys and environment settings below:")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

v_model = load_embedding_model()

# --- MAIN INTERFACE ---
st.markdown('<div class="main-header">💼 ProProfile AI Studio</div>', unsafe_allow_html=True)
st.write("Transform your raw resume or LinkedIn profile data into an optimized, high-conversion career portfolio.")
st.markdown("---")

# Setup layout blocks for inputs
col_inputs, col_empty = st.columns([3, 1])
with col_inputs:
    uploaded_file = st.file_uploader("📂 Drop your LinkedIn Export or CV PDF here", type=["pdf"])
    job_description = st.text_area("🎯 Target Job Description / Industry Role", placeholder="Paste the text requirements here...", height=150)

if uploaded_file and job_description.strip():
    if st.button("🚀 Execute Comprehensive Deep-Scan Analysis", type="primary"):
        
        # Ingestion layer
        with pdfplumber.open(uploaded_file) as pdf:
            profile_text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        st.session_state['profile_text'] = profile_text
        
        # Calculate Vector Match Score
        profile_emb = v_model.encode(profile_text, convert_to_tensor=True)
        jd_emb = v_model.encode(job_description, convert_to_tensor=True)
        match_score = round(util.cos_sim(profile_emb, jd_emb).item() * 100, 1)
        
        if not gemini_key:
            st.error("Please insert your Gemini API Key in the left sidebar control center to display full feature tabs.")
        else:
            try:
                genai.configure(api_key=gemini_key)
                llm_model = genai.GenerativeModel("gemini-2.5-flash")
                
                # Highly structured comprehensive prompt covering our new core features
                advanced_prompt = f"""
                You are an elite executive career strategist. Break down your analysis of the candidate's Profile Text against the Target Job Description into specific segments.
                
                Profile Text: {profile_text}
                Target Job Description: {job_description}
                
                Respond ONLY with a valid JSON object matching this exact structural schema layout:
                {{
                    "headline_suggestions": ["Suggested Headline 1", "Suggested Headline 2"],
                    "about_section_critique": "Detailed optimization feedback for their LinkedIn 'About' or summary section",
                    "optimized_about_text": "A fully rewritten, high-impact, keyword-rich 'About' summary paragraph",
                    "work_experience_corrections": [
                        {{"original_bullet": "handled database management", "corrected_bullet": "Optimized relational database schemas using SQL, improving operational query efficiency by 25%", "reasoning": "Incorporate strong action verbs, quantifiable performance metrics, and target technical skills."}}
                    ],
                    "related_trending_jobs": ["Job Title 1", "Job Title 2", "Job Title 3"],
                    "recommended_next_projects": ["Project Idea 1 Description", "Project Idea 2 Description"]
                }}
                """
                
                with st.spinner("Analyzing profile structures and mapping industry metrics..."):
                    response = llm_model.generate_content(advanced_prompt, generation_config={"response_mime_type": "application/json"})
                    ai_data = json.loads(response.text)
                
                # --- PROFESSIONAL RE-DESIGN LAYER (TABS) ---
                st.markdown("### 📊 Diagnostic Dashboard")
                
                # Beautiful core metric summary header card
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Overall Semantic Job-Fit Alignment Score</h4>
                        <h2 style="color: #2563EB;">{match_score}% Match Factor</h2>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Initialize analytical display tabs
                tab_branding, tab_experience, tab_market = st.tabs([
                    "✨ Executive Branding (Headline & About)", 
                    "🛠️ Experience & Bullet Corrections", 
                    "📈 Market Fit & Next Projects"
                ])
                
                with tab_branding:
                    st.subheader("💡 Tailored Branding Headlines")
                    for h in ai_data.get("headline_suggestions", []):
                        st.code(h, language="text")
                        
                    st.markdown("---")
                    st.subheader("📝 Summary ('About' Section) Optimization Strategy")
                    st.info(ai_data.get("about_section_critique"))
                    st.markdown("**Optimized Profile Summary Copy:**")
                    st.write(ai_data.get("optimized_about_text"))
                    
                with tab_experience:
                    st.subheader("🔄 Bullet-Point Impact Improvements")
                    st.caption("Review these specific structural transformations designed to pass modern applicant tracking filters:")
                    st.table(ai_data.get("work_experience_corrections", []))
                    
                with tab_market:
                    st.subheader("🎯 Complementary Job Markets to Target")
                    st.write("Based on your structural skillset, you are also highly competitive for these alternative job paths:")
                    for job in ai_data.get("related_trending_jobs", []):
                        st.markdown(f"- **{job}**")
                        
                    st.markdown("---")
                    st.subheader("💡 High-Ticket Portfolio Projects to Build Next")
                    st.caption("Building these targeted projects will automatically patch the structural knowledge gaps found in your profile embedding context:")
                    for proj in ai_data.get("recommended_next_projects", []):
                        st.markdown(f"👉 {proj}")
                        
            except Exception as e:
                st.error(f"Execution Error: {e}")
else:
    st.info("💡 Drop a candidate profile PDF and input a target industry bench post to launch the analytical pipeline.")

streamlit
pdfplumber
sentence-transformers
torch
google-generativeai
