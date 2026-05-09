import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import textwrap
from datetime import datetime

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
# This section configures the Streamlit application’s layout, title, icon, and sidebar behavior.

st.set_page_config(
    page_title="DS Salary Predictor",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# LOAD MODEL AND COMPONENTS
# ==============================================================================
# This section loads the pre-trained salary prediction model, feature scaler, and metadata required for consistent inference.

@st.cache_resource
def load_model_components():
    """Load the trained model, scaler, and metadata"""
    try:
        with open('salary_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, metadata
    
    except FileNotFoundError as e: #This code block handles missing model files by displaying an error message and stopping the application.
        st.error(f"Error loading model files: {e}")
        st.error("Please ensure all model files are in the same directory as this script.")
        st.stop()

model, scaler, metadata = load_model_components() # This function retrieves the serialized model, scaler, and metadata from disk and caches them to avoid repeated loading.

# ==============================================================================
# FEATURE ENGINEERING FUNCTION
# ==============================================================================

def engineer_features(raw_input):  # Transforms raw user inputs into the 31 engineered features expected by the trained salary prediction model.
    """
    Transform raw user inputs into 31 engineered features that match
    the training data format.
    
    Features created (in exact training order):
    1. remote_ratio (numeric: 0, 50, or 100)
    2-5. Experience level binary indicators
    6. experience_level_ordinal
    7-10. Employment type binary indicators
    11-14. Remote work binary indicators
    15. remote_ratio_normalized
    16-18. Company size binary indicators
    19. company_size_ordinal
    20-21. Location binary indicators
    22-26. Job role indicators
    27-29. Title seniority indicators
    30-31. Interaction features
    
    Total: 31 features
    """
    
    features = {}
    
    # Feature 1: remote_ratio (keep as numeric)
    features['remote_ratio'] = raw_input['remote_ratio']
    
    # Features 2-5: Experience level binary indicators
    features['is_entry_level'] = 1 if raw_input['experience_level'] == 'EN' else 0
    features['is_mid_level'] = 1 if raw_input['experience_level'] == 'MI' else 0
    features['is_senior_level'] = 1 if raw_input['experience_level'] == 'SE' else 0
    features['is_executive_level'] = 1 if raw_input['experience_level'] == 'EX' else 0
    
    # Feature 6: Experience level ordinal
    experience_map = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
    features['experience_level_ordinal'] = experience_map[raw_input['experience_level']]
    
    # Features 7-10: Employment type binary indicators
    features['is_full_time'] = 1 if raw_input['employment_type'] == 'FT' else 0
    features['is_part_time'] = 1 if raw_input['employment_type'] == 'PT' else 0
    features['is_contract'] = 1 if raw_input['employment_type'] == 'CT' else 0
    features['is_freelance'] = 1 if raw_input['employment_type'] == 'FL' else 0
    
    # Features 11-15: Remote work features
    features['is_fully_remote'] = 1 if raw_input['remote_ratio'] == 100 else 0
    features['is_hybrid'] = 1 if raw_input['remote_ratio'] == 50 else 0
    features['is_onsite'] = 1 if raw_input['remote_ratio'] == 0 else 0
    features['remote_ratio_normalized'] = raw_input['remote_ratio'] / 100
    
    # Features 16-19: Company size features
    features['is_small_company'] = 1 if raw_input['company_size'] == 'S' else 0
    features['is_medium_company'] = 1 if raw_input['company_size'] == 'M' else 0
    features['is_large_company'] = 1 if raw_input['company_size'] == 'L' else 0
    
    company_size_map = {'S': 1, 'M': 2, 'L': 3}
    features['company_size_ordinal'] = company_size_map[raw_input['company_size']]
    
    # Features 20-21: Location binary indicators
    features['is_us_based'] = 1 if raw_input['employee_residence'] == 'US' else 0
    features['is_canada_based'] = 1 if raw_input['employee_residence'] == 'CA' else 0
    
    # Features 22-28: Job title text extraction
    job_title_lower = raw_input['job_title'].lower()
    features['is_data_scientist'] = 1 if 'data scientist' in job_title_lower else 0
    features['is_data_engineer'] = 1 if 'data engineer' in job_title_lower else 0
    features['is_data_analyst'] = 1 if 'data analyst' in job_title_lower or 'analyst' in job_title_lower else 0
    features['is_ml_engineer'] = 1 if 'machine learning' in job_title_lower or 'ml engineer' in job_title_lower else 0
    features['is_manager'] = 1 if any(word in job_title_lower for word in ['manager', 'lead', 'head', 'director']) else 0
    features['title_has_senior'] = 1 if 'senior' in job_title_lower or 'sr' in job_title_lower else 0
    features['title_has_junior'] = 1 if 'junior' in job_title_lower or 'jr' in job_title_lower else 0
    features['title_has_lead'] = 1 if any(word in job_title_lower for word in ['lead', 'principal', 'staff']) else 0
    
    # Features 29-31: Interaction features
    features['exp_company_interaction'] = features['experience_level_ordinal'] * features['company_size_ordinal']
    features['exp_remote_interaction'] = features['experience_level_ordinal'] * features['remote_ratio_normalized']
    features['senior_large_company'] = features['is_senior_level'] * features['is_large_company']
    
    # Create DataFrame with features in exact training order (31 features total)
    feature_list = [
        'remote_ratio',
        'is_entry_level', 'is_mid_level', 'is_senior_level', 'is_executive_level',
        'experience_level_ordinal',
        'is_full_time', 'is_part_time', 'is_contract', 'is_freelance',
        'is_fully_remote', 'is_hybrid', 'is_onsite', 'remote_ratio_normalized',
        'is_small_company', 'is_medium_company', 'is_large_company', 'company_size_ordinal',
        'is_us_based', 'is_canada_based',
        'is_data_scientist', 'is_data_engineer', 'is_data_analyst', 'is_ml_engineer', 'is_manager',
        'title_has_senior', 'title_has_junior', 'title_has_lead',
        'exp_company_interaction', 'exp_remote_interaction', 'senior_large_company'
    ]
    
    feature_df = pd.DataFrame([features])[feature_list]
    
    return feature_df

# ==============================================================================
# MAIN APP INTERFACE
# ==============================================================================

def main():
    # Set the main title and description of the application
    st.title("Data Science Salary Predictor")
    st.markdown("### Predict salaries for data science roles in the US/Canada market")
    
    # Display model info in sidebar
    with st.sidebar:
        st.header("Model Information")
        st.markdown(f"""
        **Model Type:** {metadata['model_type']}
        
        **Performance (Test Set):**
        - R-squared Score: {metadata['model_performance']['test_r2']:.4f}
        - RMSE: ${metadata['model_performance']['test_rmse']:,.0f}
        - MAE: ${metadata['model_performance']['test_mae']:,.0f}
        
        **Training Info:**
        - Training Samples: {metadata['training_samples']}
        - Features: {metadata['n_features']}
        - Market: {metadata['market']}
        - Date: {metadata['training_date']}
        """)
        
        st.markdown("---")
        st.markdown("**About This Model:**")
        st.markdown("""
        Predicts salaries based on:
        - Experience level
        - Job role and title
        - Company size
        - Location (US/Canada)
        - Remote work arrangement
        - Employment type
        """)
    
    # Divider separating sidebar from main content
    st.markdown("---")
    st.subheader("Enter Job Details")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Role Information")
        
        experience_level = st.selectbox(
            "Experience Level",
            options=['EN', 'MI', 'SE', 'EX'],
            index=1,
            format_func=lambda x: {
                'EN': 'Entry Level',
                'MI': 'Mid Level',
                'SE': 'Senior Level',
                'EX': 'Executive Level'
            }[x],
            help="Your level of experience in data science"
        )
        
        job_title = st.text_input(
            "Job Title",
            value="Data Scientist",
            help="e.g., Senior Data Engineer, ML Engineer, Data Analyst"
        )
        
        employment_type = st.selectbox(
            "Employment Type",
            options=['FT', 'PT', 'CT', 'FL'],
            format_func=lambda x: {
                'FT': 'Full-Time',
                'PT': 'Part-Time',
                'CT': 'Contract',
                'FL': 'Freelance'
            }[x],
            help="Type of employment contract"
        )
    
    with col2:
        st.markdown("#### Company Information")
        
        company_size = st.selectbox(
            "Company Size",
            options=['S', 'M', 'L'],
            index=1,
            format_func=lambda x: {
                'S': 'Small (Less than 50 employees)',
                'M': 'Medium (50-250 employees)',
                'L': 'Large (More than 250 employees)'
            }[x],
            help="Size of the company"
        )
        
        employee_residence = st.selectbox(
            "Location",
            options=['US', 'CA'],
            format_func=lambda x: {
                'US': 'United States',
                'CA': 'Canada'
            }[x],
            help="Where you are based"
        )
        
        remote_ratio = st.select_slider(
            "Remote Work Arrangement",
            options=[0, 50, 100],
            value=100,
            format_func=lambda x: {
                0: 'Onsite (0% remote)',
                50: 'Hybrid (50% remote)',
                100: 'Fully Remote (100%)'
            }[x],
            help="Percentage of remote work"
        )
    
    # Create input dictionary
    user_input = {
        'experience_level': experience_level,
        'employment_type': employment_type,
        'job_title': job_title,
        'company_size': company_size,
        'employee_residence': employee_residence,
        'remote_ratio': remote_ratio
    }
    
    # Divider before prediction section
    st.markdown("---")
    # Run prediction pipeline when user clicks the button
    if st.button("Predict Salary", type="primary", use_container_width=True):
        with st.spinner("Analyzing job details and predicting salary..."):
            try:
                # Engineer features
                features_df = engineer_features(user_input)
                
                # Verify feature count
                if features_df.shape[1] != 31:
                    st.error(f"Feature count mismatch: Expected 31, got {features_df.shape[1]}")
                    st.stop()
                
                # Scale features
                features_scaled = scaler.transform(features_df)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                
                # Calculate confidence range using MAE
                mae = metadata['model_performance']['test_mae']
                lower_bound = max(0, prediction - mae)
                upper_bound = prediction + mae
                
                # Display results
                st.success("Prediction Complete!")
                
                # Show prediction in three columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Lower Estimate",
                        value=f"${lower_bound:,.0f}",
                        help=f"Predicted salary minus average error (MAE)"
                    )
                
                with col2:
                    st.metric(
                        label="Predicted Salary",
                        value=f"${prediction:,.0f}",
                        help="Model's best estimate"
                    )
                
                with col3:
                    st.metric(
                        label="Upper Estimate",
                        value=f"${upper_bound:,.0f}",
                        help=f"Predicted salary plus average error (MAE)"
                    )
                
                # Show confidence interval
                st.info(textwrap.dedent(f"""
                **Interpretation:**
                
                Based on your inputs, the predicted salary is **${prediction:,.0f}**.
                
                The model has an average error of **${mae:,.0f}**, so the actual salary is likely between **${lower_bound:,.0f}** and **${upper_bound:,.0f}**.
                
                This prediction is based on {metadata['training_samples']} salary data points from the US/Canada data science market.
                """))
                
                #  Basic sanity checks to flag unusually low or high predictions 
                if prediction < 40000:
                    st.warning("Note: This prediction seems low. Consider if the inputs are typical for the market.")
                elif prediction > 300000:
                    st.warning("Note: This prediction seems high. Consider if the inputs are typical for the market.")
                
                # Show input summary
                with st.expander("Input Summary"):
                    st.json(user_input)
                
                # Show engineered features
                with st.expander("Engineered Features"):
                    st.dataframe(features_df)
                    
                    st.markdown("**Non-zero features:**")
                    non_zero = []
                    for col in features_df.columns:
                        val = features_df[col].values[0]
                        if val != 0:
                            non_zero.append(f"{col}: {val}")
                    st.text("\n".join(non_zero))
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.error("Please check your inputs and try again.")
                import traceback
                st.code(traceback.format_exc())
    
    # Provide example job profiles to help users understand typical inputs
    st.markdown("---")
    with st.expander("Example Job Profiles"):
        st.markdown("""
        **Entry-Level Data Analyst**
        - Experience: Entry Level (EN)
        - Title: Data Analyst
        - Company: Small (S)
        - Location: US
        - Remote: Onsite (0)
        - Expected Range: \$60,000 - \$80,000
        
        **Mid-Level Data Scientist**
        - Experience: Mid Level (MI)
        - Title: Data Scientist
        - Company: Medium (M)
        - Location: Canada (CA)
        - Remote: Fully Remote (100)
        - Expected Range: \$90,000 - \$120,000
        
        **Senior Data Engineer**
        - Experience: Senior Level (SE)
        - Title: Senior Data Engineer
        - Company: Large (L)
        - Location: US
        - Remote: Hybrid (50)
        - Expected Range: \$130,000 - \$170,000
        """)

# ==============================================================================
# RUN APP
# ==============================================================================
# Ensure the app runs only when executed directly
if __name__ == "__main__":
    main()