# Student Dropout Prediction System
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Import mappings from external file
from mappings import MAPPINGS

# Helper functions for encoding/decoding


def get_label_to_code_map(category):
    """Create reverse mapping from label to code"""
    return {v: k for k, v in MAPPINGS[category].items()}


def get_selectbox_options(category):
    """Get sorted list of labels for a category"""
    return sorted(MAPPINGS[category].values())

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================


st.set_page_config(
    page_title="Student Dropout Prediction System",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fee 0%, #fcc 100%);
        border-left-color: #e74c3c;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe5a1 100%);
        border-left-color: #f39c12;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left-color: #2ecc71;
    }
    
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 15px 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #f39c12;
        margin: 15px 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2ecc71;
        margin: 15px 0;
    }
    
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        margin: 20px 0 10px 0;
        font-weight: bold;
    }
    
    .insight-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND DATA FUNCTIONS
# ============================================================================


@st.cache_resource
def load_model():
    """Load the trained dropout prediction model"""
    try:
        model = joblib.load('model_results/final_dropout_prediction_model.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found!")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None


@st.cache_resource
def load_scaler():
    """Load the data scaler for preprocessing"""
    try:
        scaler = joblib.load('model_results/scaler.pkl')
        return scaler
    except FileNotFoundError:
        # If no scaler exists, return None (for models that don't need scaling)
        return None
    except Exception as e:
        st.warning(f"Warning: Could not load scaler: {str(e)}")
        return None


@st.cache_data
def load_json_file(filename):
    """Load JSON configuration files"""
    try:
        with open(f'model_results/{filename}', 'r') as f:
            return json.load(f)
    except:
        return None


@st.cache_data
def load_feature_importance():
    """Load feature importance data"""
    try:
        importance_df = pd.read_csv(
            'model_results/feature_importance_full.csv')
        return importance_df
    except:
        return None


@st.cache_data
def load_feature_coefficients():
    """Load feature coefficients for Logistic Regression"""
    try:
        coefficients_df = pd.read_csv(
            'model_results/feature_coefficients.csv')
        return coefficients_df
    except:
        return None


@st.cache_data
def load_learning_curves():
    """Load learning curve data"""
    try:
        with open('model_results/learning_curve_summary.json', 'r') as f:
            return json.load(f)
    except:
        return None


# Load all necessary data
model = load_model()
scaler = load_scaler()
model_info = load_json_file('model_info.json')
performance_metrics = load_json_file('performance_metrics.json')
confusion_matrix_data = load_json_file('confusion_matrix.json')
feature_names = load_json_file('feature_names.json')
top_features = load_json_file('top_features.json')
risk_thresholds = load_json_file('risk_thresholds.json')
intervention_strategies = load_json_file('intervention_strategies.json')
importance_df = load_feature_importance()
coefficients_df = load_feature_coefficients()
learning_curves_data = load_learning_curves()

# Determine which feature analysis to use based on model type
model_name = model_info.get('model_name', '') if model_info else ''
is_linear_model = 'Logistic Regression' in model_name or 'SVM' in model_name or 'Support Vector' in model_name
feature_analysis_df = coefficients_df if is_linear_model else importance_df


# ============================================================================
# HEADER
# ============================================================================

st.markdown('''
    <div class="main-header">
        Student Dropout Prediction System
    </div>
''', unsafe_allow_html=True)

st.markdown('''
    <p class="subtitle">
        <b>Predicting University Student Dropout Risk Using Socioeconomic and Academic Performance Indicators</b>
    </p>
''', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a page:",
    [
        "Dashboard & Insights",
        "Student Risk Prediction"
    ],
    label_visibility="collapsed"
)

# Display model info in sidebar
if model_info:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    st.sidebar.info(f"""
    **Model:** {model_info.get('model_name', 'N/A')}
    **Training Date:** {model_info.get('training_date', 'N/A')[:10]}
    """)

if performance_metrics:
    test_perf = performance_metrics.get('test_performance', {})
    st.sidebar.markdown("### Performance Metrics")
    st.sidebar.success(f"""
    **F1-Score:** {test_perf.get('f1_score', 0):.3f}
    **Recall:** {test_perf.get('recall', 0):.3f}
    **Precision:** {test_perf.get('precision', 0):.3f}
    """)

# ============================================================================
# PAGE 1: COMPREHENSIVE DASHBOARD & INSIGHTS
# ============================================================================

if page == "Dashboard & Insights":

    # ==========================
    # SECTION 1: EXECUTIVE SUMMARY
    # ==========================
    st.markdown("## Executive Summary")
    st.info("**Quick Overview:** Key performance metrics and model snapshot")

    if performance_metrics and confusion_matrix_data and risk_thresholds:
        test_perf = performance_metrics['test_performance']
        cm_data = confusion_matrix_data
        dist = risk_thresholds.get('distribution', {})

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Model Accuracy", f"{test_perf['accuracy']*100:.1f}%")
        col2.metric("Recall (Key Metric)", f"{test_perf['recall']*100:.1f}%",
                    help="Percentage of at-risk students correctly identified")
        col3.metric("F1-Score", f"{test_perf['f1_score']:.3f}")
        col4.metric("Total Students", "3,137",
                    help="Total students in dataset")
        col5.metric("High Risk Students", f"{dist.get('high_risk_count', 'N/A')}",
                    delta=f"{dist.get('high_risk_percentage', 0):.1f}%",
                    delta_color="inverse")

    st.markdown("---")

    # ==========================
    # SECTION 2: MODEL PERFORMANCE ANALYSIS
    # ==========================
    st.markdown("## Model Performance Analysis")
    st.info("**Core Metrics:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrix")

    if performance_metrics and confusion_matrix_data:
        test_perf = performance_metrics['test_performance']
        cm_data = confusion_matrix_data

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            # Performance Radar Chart
            categories = ['Accuracy', 'Precision',
                          'Recall', 'F1-Score', 'ROC-AUC']
            values = [
                test_perf['accuracy'],
                test_perf['precision'],
                test_perf['recall'],
                test_perf['f1_score'],
                test_perf['roc_auc']
            ]

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Model Performance',
                line=dict(color='#667eea', width=2),
                fillcolor='rgba(102, 126, 234, 0.3)',
                hovertemplate='<b>%{theta}</b><br>Score: %{r:.4f}<extra></extra>'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickformat='.2f'
                    )
                ),
                showlegend=False,
                title="Performance Metrics",
                height=350
            )

            st.plotly_chart(fig, use_container_width=True, key="perf_radar")

        with col2:
            # Confusion Matrix
            cm = np.array([[cm_data['true_negatives'], cm_data['false_positives']],
                           [cm_data['false_negatives'], cm_data['true_positives']]])

            cm_percent = cm / cm.sum() * 100
            annotations = []
            for i in range(2):
                for j in range(2):
                    annotations.append(
                        f"<b>{cm[i][j]}</b><br>({cm_percent[i][j]:.1f}%)"
                    )

            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Graduate', 'Predicted Dropout'],
                y=['Actual Graduate', 'Actual Dropout'],
                colorscale='Blues',
                text=np.array(annotations).reshape(2, 2),
                texttemplate='%{text}',
                textfont={"size": 14},
                showscale=True,
                hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>',
                colorbar=dict(title="Count")
            ))

            fig.update_layout(
                title="Confusion Matrix",
                height=350
            )

            st.plotly_chart(fig, use_container_width=True, key="conf_matrix")

        with col3:
            # Performance Comparison Bar Chart
            metrics_names = ['Accuracy', 'Precision',
                             'Recall', 'F1-Score', 'ROC-AUC']
            metrics_values = [
                test_perf['accuracy'],
                test_perf['precision'],
                test_perf['recall'],
                test_perf['f1_score'],
                test_perf['roc_auc']
            ]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=metrics_values,
                y=metrics_names,
                orientation='h',
                marker=dict(
                    color=metrics_values,
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f"{v:.3f}" for v in metrics_values],
                textposition='outside'
            ))

            fig.update_layout(
                title="Metrics Comparison",
                xaxis_title="Score",
                xaxis_range=[0, 1.1],
                height=350
            )

            st.plotly_chart(fig, use_container_width=True, key="metrics_bar")

    st.markdown("---")
    
    # ==========================
    # SECTION 4: LEARNING CURVES ANALYSIS
    # ==========================
    if learning_curves_data and model_name:
        st.markdown("## Learning Curves Analysis")
        st.info("**Convergence Check:** Verifying model stability and optimal training data size")
        
        model_key = model_name  # Try exact match first
        if model_key not in learning_curves_data:
            # Try to find a matching key
            for key in learning_curves_data.keys():
                if model_name.lower() in key.lower():
                    model_key = key
                    break
        
        if model_key in learning_curves_data:
            curve_data = learning_curves_data[model_key]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Model Convergence Summary")
                
                # Display metrics from the summary
                final_train = curve_data.get('final_train_score', 0)
                final_cv = curve_data.get('final_cv_score', 0)
                gap = curve_data.get('train_cv_gap', 0)
                improvement = curve_data.get('score_improvement', 0)
                converged = curve_data.get('converged', False)
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Final Training Score", f"{final_train:.4f}")
                    st.metric("Score Improvement", f"{improvement:.4f}")
                
                with col_b:
                    st.metric("Final CV Score", f"{final_cv:.4f}")
                    st.metric("Train-CV Gap", f"{gap:.4f}",
                             delta="Good" if gap < 0.05 else "Moderate" if gap < 0.10 else "High",
                             delta_color="normal" if gap < 0.05 else "off" if gap < 0.10 else "inverse")
                
                # Visual representation
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Training Score',
                    x=['Final Performance'],
                    y=[final_train],
                    marker_color='#e67e22',
                    text=[f'{final_train:.4f}'],
                    textposition='outside'
                ))
                
                fig.add_trace(go.Bar(
                    name='Cross-Validation Score',
                    x=['Final Performance'],
                    y=[final_cv],
                    marker_color='#667eea',
                    text=[f'{final_cv:.4f}'],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title=f"Final Performance Comparison - {model_name}",
                    yaxis_title="F1-Score",
                    yaxis_range=[0, 1.1],
                    height=350,
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True, key="learning_curve_summary")
            
            with col2:
                st.markdown("### Convergence Analysis")
                
                if converged:
                    st.success("Model Converged")
                    st.markdown("""
                    <div class="success-box">
                        <p style="font-size: 12px;"><b>Excellent!</b></p>
                        <p style="font-size: 11px;">The model has reached optimal performance with the current training data.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Not Yet Converged")
                    st.markdown("""
                    <div class="warning-box">
                        <p style="font-size: 12px;"><b>Note:</b></p>
                        <p style="font-size: 11px;">Model could potentially benefit from more training data or additional epochs.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### Gap Interpretation:")
                if gap < 0.05:
                    st.success(f"**Excellent** (Gap: {gap:.4f})")
                    st.write("No overfitting detected")
                elif gap < 0.10:
                    st.warning(f"**Acceptable** (Gap: {gap:.4f})")
                    st.write("Slight overfitting, but acceptable")
                else:
                    st.error(f"**High** (Gap: {gap:.4f})")
                    st.write("Significant overfitting present")
                
                st.markdown("""
                <div class="info-box">
                    <p style="font-size: 11px;"><b>Key Metrics:</b></p>
                    <ul style="font-size: 10px;">
                        <li><b>Gap < 0.05:</b> Excellent generalization</li>
                        <li><b>Gap 0.05-0.10:</b> Acceptable performance</li>
                        <li><b>Gap > 0.10:</b> Overfitting concerns</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"Learning curve data not found for {model_name}")
        
        st.markdown("---")
    
    # ==========================
    # SECTION 5: ERROR ANALYSIS & MODEL BEHAVIOR
    # ==========================
    st.markdown("## Error Analysis & Model Behavior")
    st.info("**Deep Dive:** Understanding where the model makes mistakes and why")

    if confusion_matrix_data:
        col1, col2 = st.columns(2)

        with col1:
            # Error Types Breakdown
            error_data = {
                'Type': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives'],
                'Count': [cm_data['true_positives'], cm_data['true_negatives'],
                          cm_data['false_positives'], cm_data['false_negatives']],
                'Category': ['Correct', 'Correct', 'Error', 'Error']
            }
            error_df = pd.DataFrame(error_data)

            fig = px.sunburst(
                error_df,
                path=['Category', 'Type'],
                values='Count',
                color='Category',
                color_discrete_map={'Correct': '#2ecc71', 'Error': '#e74c3c'},
                title="Prediction Breakdown: Correct vs Errors"
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True,
                            key="error_sunburst")

        with col2:
            # Prediction Accuracy by Class
            total_actual_dropout = cm_data['true_positives'] + \
                cm_data['false_negatives']
            total_actual_graduate = cm_data['true_negatives'] + \
                cm_data['false_positives']

            dropout_recall = (
                cm_data['true_positives'] / total_actual_dropout) * 100
            graduate_recall = (
                cm_data['true_negatives'] / total_actual_graduate) * 100

            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='Correctly Identified',
                x=['At-Risk Students', 'Graduate Students'],
                y=[dropout_recall, graduate_recall],
                marker_color='#2ecc71',
                text=[f"{dropout_recall:.1f}%", f"{graduate_recall:.1f}%"],
                textposition='inside'
            ))

            fig.add_trace(go.Bar(
                name='Missed',
                x=['At-Risk Students', 'Graduate Students'],
                y=[100-dropout_recall, 100-graduate_recall],
                marker_color='#e74c3c',
                text=[f"{100-dropout_recall:.1f}%",
                      f"{100-graduate_recall:.1f}%"],
                textposition='inside'
            ))

            fig.update_layout(
                barmode='stack',
                title="Model Accuracy by Student Category",
                yaxis_title="Percentage (%)",
                height=400,
                yaxis_range=[0, 100]
            )

            st.plotly_chart(fig, use_container_width=True,
                            key="class_accuracy")

    st.markdown("---")

    # ==========================
    # SECTION 7: RISK DISTRIBUTION & PATTERNS  
    # ==========================
    st.markdown("## Student Risk Distribution & Patterns")
    st.info("**Real-World Impact:** How students are distributed across risk categories")

    if risk_thresholds and 'distribution' in risk_thresholds:
        dist = risk_thresholds['distribution']

        col1, col2 = st.columns(2)

        with col1:
            # Risk Distribution Donut Chart
            labels = ['Low Risk', 'Medium Risk', 'High Risk']
            values = [
                dist['low_risk_count'],
                dist['medium_risk_count'],
                dist['high_risk_count']
            ]
            colors = ['#2ecc71', '#f39c12', '#e74c3c']

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color='white', width=2)),
                hole=0.4,
                textinfo='label+percent',
                textfont_size=14,
                hovertemplate='<b>%{label}</b><br>%{value} students<br>%{percent}<extra></extra>'
            )])

            fig.update_layout(
                title="Risk Category Distribution",
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True, key="risk_dist_pie")

        with col2:
            # Risk Level Bar Chart with targets
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=values,
                textposition='outside',
                texttemplate='<b>%{text}</b><br>students',
                hovertemplate='<b>%{x}</b><br>%{y} students<extra></extra>'
            ))

            # Add target/benchmark line
            avg_line = sum(values) / len(values)
            fig.add_hline(y=avg_line, line_dash="dash", line_color="gray",
                          annotation_text="Average", annotation_position="right")

            fig.update_layout(
                title="Student Count by Risk Level",
                xaxis_title="Risk Category",
                yaxis_title="Number of Students",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True, key="risk_dist_bar")

        # Risk Summary Cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card risk-low">
                <h3 style="color: #2ecc71;">Low Risk</h3>
                <h2 style="color: #27ae60;">{dist['low_risk_count']} students</h2>
                <p><b>{dist['low_risk_percentage']:.1f}%</b> of total</p>
                <hr>
                <p style="font-size: 13px;">Dropout probability &lt; 30%</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card risk-medium">
                <h3 style="color: #f39c12;">Medium Risk</h3>
                <h2 style="color: #e67e22;">{dist['medium_risk_count']} students</h2>
                <p><b>{dist['medium_risk_percentage']:.1f}%</b> of total</p>
                <hr>
                <p style="font-size: 13px;">Dropout probability 30-60%</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card risk-high">
                <h3 style="color: #e74c3c;">High Risk</h3>
                <h2 style="color: #c0392b;">{dist['high_risk_count']} students</h2>
                <p><b>{dist['high_risk_percentage']:.1f}%</b> of total</p>
                <hr>
                <p style="font-size: 13px;">Dropout probability &gt; 60%</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ==========================
    # SECTION 6: FEATURE IMPORTANCE/COEFFICIENTS ANALYSIS
    # ==========================
    if is_linear_model:
        st.markdown("## Critical Predictive Factors (Feature Coefficients)")
        st.info("**Key Drivers:** Which factors most strongly influence dropout risk (Logistic Regression Coefficients)")
    else:
        st.markdown("## Critical Predictive Factors (Feature Importance)")
        st.info("**Key Drivers:** Which factors most strongly influence dropout risk (Feature Importance)")
    
    if is_linear_model:
        st.info("**For Logistic Regression:** Positive coefficients increase dropout probability, negative coefficients decrease it")
    else:
        st.info("**For tree-based models:** Importance indicates how much each feature contributes to predictions")

    if feature_analysis_df is not None and len(feature_analysis_df) > 0:
        
        # Determine column names based on model type
        feature_col = 'Feature'
        value_col = 'Coefficient' if is_linear_model else 'Importance'
        
        # Check actual column names
        if value_col not in feature_analysis_df.columns:
            # Try alternative naming
            if 'Importance' in feature_analysis_df.columns:
                value_col = 'Importance'
            elif 'Coefficient' in feature_analysis_df.columns:
                value_col = 'Coefficient'
        
        # Create absolute values for sorting (for coefficients)
        if is_linear_model and value_col in feature_analysis_df.columns:
            feature_analysis_df['AbsValue'] = feature_analysis_df[value_col].abs()
            feature_analysis_df = feature_analysis_df.sort_values('AbsValue', ascending=False)
        
        # Top 15 Features
        if is_linear_model:
            st.markdown("### Top 15 Most Important Features (by Absolute Coefficient)")
        else:
            st.markdown("### Top 15 Most Important Features")

        top_15 = feature_analysis_df.head(15)

        fig = go.Figure()
        
        # For coefficients, use diverging color scale (red for positive, blue for negative)
        if is_linear_model:
            colors_list = []
            for val in top_15[value_col]:
                if val > 0:
                    colors_list.append('#e74c3c')  # Red for positive (increases dropout)
                else:
                    colors_list.append('#3498db')  # Blue for negative (decreases dropout)
            
            fig.add_trace(go.Bar(
                x=top_15[value_col],
                y=top_15[feature_col],
                orientation='h',
                marker=dict(color=colors_list),
                text=[f"{val:+.4f}" for val in top_15[value_col]],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Coefficient: %{x:+.4f}<extra></extra>'
            ))
        else:
            fig.add_trace(go.Bar(
                x=top_15[value_col],
                y=top_15[feature_col],
                orientation='h',
                marker=dict(
                    color=top_15[value_col],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f"{val:.4f}" for val in top_15[value_col]],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ))

        fig.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Coefficient Value" if is_linear_model else "Importance Score",
            yaxis_title="",
            hovermode='closest'
        )
        
        if is_linear_model:
            # Add vertical line at x=0 for coefficient plots
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

        st.plotly_chart(fig, use_container_width=True, key="top15_importance")

        st.markdown("---")

        # Top 10 with percentages
        st.markdown("### Top 10 Features - Relative Importance")

        col1, col2 = st.columns([2, 1])

        with col1:
            top_10 = importance_df.head(10)
            total_importance = top_10['Importance'].sum()
            top_10['Percentage'] = (
                top_10['Importance'] / importance_df['Importance'].sum()) * 100

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(value_col + ' Score', 'Percentage of Total'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}]]
            )

            fig.add_trace(
                go.Bar(
                    y=top_10[feature_col],
                    x=top_10[value_col],
                    orientation='h',
                    marker_color='#667eea',
                    text=[f"{v:+.4f}" if is_linear_model else f"{v:.4f}" for v in top_10[value_col]],
                    textposition='outside',
                    name=value_col
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Bar(
                    y=top_10[feature_col],
                    x=top_10['Percentage'],
                    orientation='h',
                    marker_color='#e67e22',
                    text=[f"{v:.2f}%" for v in top_10['Percentage']],
                    textposition='outside',
                    name='Percentage'
                ),
                row=1, col=2
            )

            fig.update_layout(
                height=500,
                showlegend=False
            )

            fig.update_yaxes(categoryorder='total ascending', row=1, col=1)
            fig.update_yaxes(categoryorder='total ascending', row=1, col=2)

            st.plotly_chart(fig, use_container_width=True,
                            key="top10_comparison")

        with col2:
            st.markdown("#### Statistics")

            st.metric("Top Feature", top_10.iloc[0][feature_col][:30] + "...")
            st.metric("Top Feature Impact",
                      f"{top_10.iloc[0]['Percentage']:.2f}%")
            st.metric("Top 10 Combined", f"{top_10['Percentage'].sum():.1f}%")
            st.metric("Remaining Features",
                      f"{100 - top_10['Percentage'].sum():.1f}%")

            st.markdown(f"""
            <div class="info-box">
                <p style="font-size: 12px;"><b>Analysis:</b> The top 10 features account for 
                <b>{top_10['Percentage'].sum():.1f}%</b> of the model's predictive power.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Cumulative Importance/Impact
        if is_linear_model:
            st.markdown("### Feature Impact Accumulation")
            st.info("Shows how feature impact accumulates (using absolute coefficient values)")
        else:
            st.markdown("### Feature Efficiency Analysis")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Calculate cumulative based on absolute values for coefficients
            if is_linear_model:
                feature_analysis_df['Cumulative'] = (feature_analysis_df['AbsValue'] / feature_analysis_df['AbsValue'].sum()).cumsum()
            else:
                feature_analysis_df['Cumulative'] = (feature_analysis_df[value_col] / feature_analysis_df[value_col].sum()).cumsum()

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=list(range(1, len(feature_analysis_df) + 1)),
                y=feature_analysis_df['Cumulative'],
                mode='lines',
                name='Cumulative Importance',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)',
                hovertemplate='<b>Features: %{x}</b><br>Cumulative: %{y:.3f}<extra></extra>'
            ))

            features_80 = (feature_analysis_df['Cumulative'] <= 0.8).sum() + 1
            features_90 = (feature_analysis_df['Cumulative'] <= 0.9).sum() + 1

            fig.add_trace(go.Scatter(
                x=[features_80, features_90],
                y=[0.8, 0.9],
                mode='markers+text',
                marker=dict(size=15, color=['red', 'orange'], symbol='star'),
                text=[f"  {features_80} features",
                      f"  {features_90} features"],
                textposition='middle right',
                textfont=dict(size=12),
                name='Key Thresholds',
                hovertemplate='<b>%{text}</b><br>Cumulative: %{y:.1%}<extra></extra>'
            ))

            fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                          annotation_text="80% Power", annotation_position="right")
            fig.add_hline(y=0.9, line_dash="dash", line_color="orange",
                          annotation_text="90% Power", annotation_position="right")

            fig.update_layout(
                title="Cumulative Feature Importance - Efficiency Frontier",
                xaxis_title="Number of Features",
                yaxis_title="Cumulative Importance",
                height=400,
                yaxis_tickformat='.0%'
            )

            st.plotly_chart(fig, use_container_width=True, key="cumulative")

        with col2:
            st.markdown("#### Efficiency Metrics")

            st.metric("Total Features", len(importance_df))
            st.metric("80% Power", features_80,
                      delta=f"{features_80/len(importance_df)*100:.0f}% of features")
            st.metric("90% Power", features_90,
                      delta=f"{features_90/len(importance_df)*100:.0f}% of features")

        st.markdown("---")

        
        # Category Analysis
        st.markdown("### Factor Categories Deep Dive")

        # Categorize features
        category_data = {
            'Academic': [],
            'Socioeconomic': [],
            'Demographic': []
        }

        for _, row in importance_df.iterrows():
            feat = row['Feature']
            imp = row['Importance']

            if any(term in feat.lower() for term in ['curricular', 'units', 'grade', 'approved', 'enrolled', 'evaluations', 'credited']):
                category_data['Academic'].append(imp)
            elif any(term in feat.lower() for term in ['mother', 'father', 'qualification', 'occupation', 'scholarship', 'tuition', 'debtor']):
                category_data['Socioeconomic'].append(imp)
            elif any(term in feat.lower() for term in ['age', 'gender', 'marital', 'displaced', 'international', 'nationality']):
                category_data['Demographic'].append(imp)

        category_totals = {k: sum(v) for k, v in category_data.items()}
        category_counts = {k: len(v) for k, v in category_data.items()}
        category_means = {k: np.mean(
            v) if v else 0 for k, v in category_data.items()}

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            colors_cat = ['#3498db', '#f39c12', '#2ecc71']

            fig = go.Figure(data=[go.Pie(
                labels=list(category_totals.keys()),
                values=list(category_totals.values()),
                marker=dict(colors=colors_cat, line=dict(
                    color='white', width=2)),
                hole=0.4,
                textinfo='label+percent',
                textfont_size=14,
                hovertemplate='<b>%{label}</b><br>Total Importance: %{value:.4f}<br>%{percent}<extra></extra>'
            )])

            fig.update_layout(
                title="Category Distribution by Total Importance",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True, key="category_pie")

        with col2:
            # Grouped bar chart
            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='Total Importance',
                x=list(category_totals.keys()),
                y=list(category_totals.values()),
                marker_color=colors_cat,
                text=[f"{v:.3f}" for v in category_totals.values()],
                textposition='outside',
                yaxis='y',
                hovertemplate='<b>%{x}</b><br>Total Importance: %{y:.4f}<extra></extra>'
            ))

            fig.add_trace(go.Bar(
                name='Feature Count',
                x=list(category_counts.keys()),
                y=list(category_counts.values()),
                marker_color=[f'rgba{tuple(list(int(c.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.5])}'
                              for c in colors_cat],
                text=list(category_counts.values()),
                textposition='outside',
                yaxis='y2',
                hovertemplate='<b>%{x}</b><br>Features: %{y}<extra></extra>'
            ))

            fig.update_layout(
                title="Category Comparison: Importance vs Count",
                yaxis=dict(title='Total Importance'),
                yaxis2=dict(title='Feature Count',
                            overlaying='y', side='right'),
                barmode='group',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True,
                            key="category_comparison")

        # Category statistics
        st.markdown("#### Category Statistics Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Academic Features**")
            st.metric("Count", category_counts['Academic'])
            st.metric("Total Importance", f"{category_totals['Academic']:.4f}")
            st.metric("Average Importance",
                      f"{category_means['Academic']:.4f}")

        with col2:
            st.markdown("**Socioeconomic Features**")
            st.metric("Count", category_counts['Socioeconomic'])
            st.metric("Total Importance",
                      f"{category_totals['Socioeconomic']:.4f}")
            st.metric("Average Importance",
                      f"{category_means['Socioeconomic']:.4f}")

        with col3:
            st.markdown("**Demographic Features**")
            st.metric("Count", category_counts['Demographic'])
            st.metric("Total Importance",
                      f"{category_totals['Demographic']:.4f}")
            st.metric("Average Importance",
                      f"{category_means['Demographic']:.4f}")

        st.markdown("---")

        # Category tabs
        st.markdown("### Detailed Category Breakdown")

        tab1, tab2, tab3 = st.tabs(
            ["Academic", "Socioeconomic", "Demographic"])

        with tab1:
            if top_features and top_features.get('top_academic'):
                academic_df = pd.DataFrame(top_features['top_academic'][:8])

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    y=academic_df['feature'],
                    x=academic_df['importance'],
                    orientation='h',
                    marker=dict(
                        color=academic_df['importance'],
                        colorscale='Blues',
                        showscale=True
                    ),
                    text=[f"{v:.4f}" for v in academic_df['importance']],
                    textposition='outside'))

                fig.update_layout(
                    title="Top 8 Academic Factors",
                    xaxis_title="Importance",
                    height=400,
                    yaxis={'categoryorder': 'total ascending'}
                )

                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if top_features and top_features.get('top_socioeconomic'):
                socio_df = pd.DataFrame(top_features['top_socioeconomic'][:8])

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    y=socio_df['feature'],
                    x=socio_df['importance'],
                    orientation='h',
                    marker=dict(
                        color=socio_df['importance'],
                        colorscale='YlOrRd',
                        showscale=True
                    ),
                    text=[f"{v:.4f}" for v in socio_df['importance']],
                    textposition='outside'
                ))

                fig.update_layout(
                    title="Top 8 Socioeconomic Factors",
                    xaxis_title="Importance",
                    height=400,
                    yaxis={'categoryorder': 'total ascending'}
                )

                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if top_features and top_features.get('top_demographic'):
                demo_df = pd.DataFrame(top_features['top_demographic'][:8])

                fig = go.Figure()

                fig.add_trace(go.Bar(
                        y=demo_df['feature'],
                        x=demo_df['importance'],
                        orientation='h',
                        marker=dict(
                            color=demo_df['importance'],
                            colorscale='Greens',
                            showscale=True
                        ),
                        text=[f"{v:.4f}" for v in demo_df['importance']],
                        textposition='outside'
                    ))

                fig.update_layout(
                        title="Top Demographic Factors",
                        xaxis_title="Importance",
                        height=400,
                        yaxis={'categoryorder': 'total ascending'}
                    )

                st.plotly_chart(fig, use_container_width=True)

        # Additional insights for demographic factors
        col1, col2 = st.columns(2)

        with col1:
            if top_features and top_features.get('top_demographic'):
                demo_df = pd.DataFrame(top_features['top_demographic'][:5])
                st.dataframe(demo_df, use_container_width=True, hide_index=True)

        with col2:
            if category_counts['Demographic'] > 0:
                st.metric("Demographic Features", category_counts['Demographic'])
                st.metric("Total Importance", f"{category_totals['Demographic']:.3f}")

    st.markdown("---")

# ============================================================================
# PAGE 2: STUDENT RISK PREDICTION
# ============================================================================

elif page == "Student Risk Prediction":
    st.markdown("### Individual Student Dropout Risk Assessment")

    st.markdown("""
    <div class="info-box">
        <b>Step-by-Step Assessment:</b> Complete each section to predict dropout risk. 
        All fields must be filled before proceeding to the next step!
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Model not loaded. Please ensure model file exists.")
        st.stop()

    # Initialize session state for multi-step form
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1

    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}

    # Progress bar
    progress = (st.session_state.current_step - 1) / 4
    st.progress(progress)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status1 = "[Done]" if st.session_state.current_step > 1 else "[Current]" if st.session_state.current_step == 1 else "[Pending]"
        st.markdown(f"**{status1} Step 1: Demographics**")
    with col2:
        status2 = "[Done]" if st.session_state.current_step > 2 else "[Current]" if st.session_state.current_step == 2 else "[Pending]"
        st.markdown(f"**{status2} Step 2: Academic**")
    with col3:
        status3 = "[Done]" if st.session_state.current_step > 3 else "[Current]" if st.session_state.current_step == 3 else "[Pending]"
        st.markdown(f"**{status3} Step 3: Financial**")
    with col4:
        status4 = "[Done]" if st.session_state.current_step > 4 else "[Current]" if st.session_state.current_step == 4 else "[Pending]"
        st.markdown(f"**{status4} Step 4: Economic**")

    st.markdown("---")

    # ============================================================================
    # STEP 1: DEMOGRAPHICS & BACKGROUND
    # ============================================================================

    if st.session_state.current_step == 1:
        st.markdown('<div class="section-header">Step 1: Demographics & Background</div>',
                    unsafe_allow_html=True)

        with st.form("step1_form"):
            st.markdown("#### Personal Information")
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age at Enrollment *", min_value=17, max_value=70, value=None,
                                      placeholder="Enter age")

                gender_options = ["-"] + get_selectbox_options('Gender')
                gender_label = st.selectbox(
                    "Gender *", options=gender_options, index=0)

                marital_status_options = ["-"] + \
                    get_selectbox_options('Marital Status')
                marital_status_label = st.selectbox(
                    "Marital Status *", options=marital_status_options, index=0)

            with col2:
                nationality_options = ["-"] + \
                    get_selectbox_options('Nationality')
                nationality_label = st.selectbox(
                    "Nationality *", options=nationality_options, index=0)

                displaced_options = ["-"] + get_selectbox_options('Attribute')
                displaced_label = st.selectbox(
                    "Displaced from Home *", options=displaced_options, index=0)

                international_options = ["-"] + \
                    get_selectbox_options('Attribute')
                international_label = st.selectbox("International Student *",
                                                   options=international_options, index=0, key="international")

            with col3:
                special_needs_options = ["-"] + \
                    get_selectbox_options('Attribute')
                special_needs_label = st.selectbox("Educational Special Needs *",
                                                   options=special_needs_options, index=0, key="special_needs")

                daytime_options = ["-"] + \
                    get_selectbox_options('Attendance Regime')
                daytime_label = st.selectbox(
                    "Attendance Type *", options=daytime_options, index=0)

            st.markdown("---")
            st.markdown("#### Academic Background")

            col1, col2 = st.columns(2)

            with col1:
                course_options = ["-"] + get_selectbox_options('Course Name')
                course_label = st.selectbox(
                    "Course *", options=course_options, index=0)

                previous_qual_options = [
                    "-"] + get_selectbox_options('Previous Qualification')
                previous_qual_label = st.selectbox("Previous Qualification *",
                                                   options=previous_qual_options, index=0)

            with col2:
                application_mode_options = [
                    "-"] + get_selectbox_options('Application Mode')
                application_mode_label = st.selectbox("Application Mode *",
                                                      options=application_mode_options, index=0)

                application_order = st.number_input("Application Order *",
                                                    min_value=0, max_value=9, value=None,
                                                    help="0 = First choice, 9 = Ninth choice",
                                                    placeholder="Enter 0-9")

            st.markdown("---")
            st.markdown("#### Family Background")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Mother's Information**")
                mother_qual_options = [
                    "-"] + get_selectbox_options('Parent Qualification')
                mother_qual_label = st.selectbox("Mother's Qualification *",
                                                 options=mother_qual_options, index=0, key="mother_qual")

                mother_occ_options = ["-"] + \
                    get_selectbox_options('Occupation')
                mother_occ_label = st.selectbox("Mother's Occupation *",
                                                options=mother_occ_options, index=0, key="mother_occ")

            with col2:
                st.markdown("**Father's Information**")
                father_qual_options = [
                    "-"] + get_selectbox_options('Parent Qualification')
                father_qual_label = st.selectbox("Father's Qualification *",
                                                 options=father_qual_options, index=0, key="father_qual")

                father_occ_options = ["-"] + \
                    get_selectbox_options('Occupation')
                father_occ_label = st.selectbox("Father's Occupation *",
                                                options=father_occ_options, index=0, key="father_occ")

            st.markdown("---")
            st.info("⚠️ All fields marked with * are required")

            submitted = st.form_submit_button(
                "Next: Academic Performance", use_container_width=True, type="primary")

            if submitted:
                # Validate all fields are filled
                errors = []

                if age is None:
                    errors.append("Age at Enrollment")
                if gender_label == "-":
                    errors.append("Gender")
                if marital_status_label == "-":
                    errors.append("Marital Status")
                if nationality_label == "-":
                    errors.append("Nationality")
                if displaced_label == "-":
                    errors.append("Displaced from Home")
                if international_label == "-":
                    errors.append("International Student")
                if special_needs_label == "-":
                    errors.append("Educational Special Needs")
                if daytime_label == "-":
                    errors.append("Attendance Type")
                if course_label == "-":
                    errors.append("Course")
                if previous_qual_label == "-":
                    errors.append("Previous Qualification")
                if application_mode_label == "-":
                    errors.append("Application Mode")
                if application_order is None:
                    errors.append("Application Order")
                if mother_qual_label == "-":
                    errors.append("Mother's Qualification")
                if mother_occ_label == "-":
                    errors.append("Mother's Occupation")
                if father_qual_label == "-":
                    errors.append("Father's Qualification")
                if father_occ_label == "-":
                    errors.append("Father's Occupation")

                if errors:
                    st.error(
                        f"⚠️ Please fill in the following required fields: {', '.join(errors)}")
                else:
                    # Save data to session state
                    st.session_state.form_data['age'] = age
                    st.session_state.form_data['gender_label'] = gender_label
                    st.session_state.form_data['marital_status_label'] = marital_status_label
                    st.session_state.form_data['nationality_label'] = nationality_label
                    st.session_state.form_data['displaced_label'] = displaced_label
                    st.session_state.form_data['international_label'] = international_label
                    st.session_state.form_data['special_needs_label'] = special_needs_label
                    st.session_state.form_data['daytime_label'] = daytime_label
                    st.session_state.form_data['course_label'] = course_label
                    st.session_state.form_data['previous_qual_label'] = previous_qual_label
                    st.session_state.form_data['application_mode_label'] = application_mode_label
                    st.session_state.form_data['application_order'] = application_order
                    st.session_state.form_data['mother_qual_label'] = mother_qual_label
                    st.session_state.form_data['mother_occ_label'] = mother_occ_label
                    st.session_state.form_data['father_qual_label'] = father_qual_label
                    st.session_state.form_data['father_occ_label'] = father_occ_label

                    st.session_state.current_step = 2
                    st.rerun()

    # ============================================================================
    # STEP 2: ACADEMIC PERFORMANCE
    # ============================================================================

    elif st.session_state.current_step == 2:
        st.markdown('<div class="section-header">Step 2: Academic Performance</div>',
                    unsafe_allow_html=True)

        with st.form("step2_form"):
            st.info(
                "First semester performance is the strongest predictor of dropout risk")

            st.markdown("#### First Semester Performance")
            col1, col2, col3 = st.columns(3)

            with col1:
                units_1_credited = st.number_input("Units Credited *", min_value=0, max_value=20,
                                                   value=None, key="s1_credit", placeholder="Enter 0-20")
                units_1_enrolled = st.number_input("Units Enrolled *", min_value=0, max_value=26,
                                                   value=None, key="s1_enroll", placeholder="Enter 0-26")

            with col2:
                units_1_evaluations = st.number_input("Evaluations *", min_value=0, max_value=45,
                                                      value=None, key="s1_eval", placeholder="Enter 0-45")
                units_1_approved = st.number_input("Units Approved *", min_value=0, max_value=26,
                                                   value=None, key="s1_approve", placeholder="Enter 0-26")

            with col3:
                units_1_grade = st.number_input("Grade Average *", min_value=0.0, max_value=20.0,
                                                value=None, step=0.1, key="s1_grade", placeholder="Enter 0-20")
                units_1_without_eval = st.number_input("Without Evaluations *", min_value=0, max_value=12,
                                                       value=None, key="s1_no_eval", placeholder="Enter 0-12")

            st.markdown("---")
            st.markdown("#### Second Semester Performance")
            st.info("Note: Enter 0 for all fields if second semester hasn't started yet")

            col1, col2, col3 = st.columns(3)

            with col1:
                units_2_credited = st.number_input("Units Credited *", min_value=0, max_value=20,
                                                   value=None, key="s2_credit", placeholder="Enter 0-20")
                units_2_enrolled = st.number_input("Units Enrolled *", min_value=0, max_value=26,
                                                   value=None, key="s2_enroll", placeholder="Enter 0-26")

            with col2:
                units_2_evaluations = st.number_input("Evaluations *", min_value=0, max_value=45,
                                                      value=None, key="s2_eval", placeholder="Enter 0-45")
                units_2_approved = st.number_input("Units Approved *", min_value=0, max_value=26,
                                                   value=None, key="s2_approve", placeholder="Enter 0-26")

            with col3:
                units_2_grade = st.number_input("Grade Average *", min_value=0.0, max_value=20.0,
                                                value=None, step=0.1, key="s2_grade", placeholder="Enter 0-20")
                units_2_without_eval = st.number_input("Without Evaluations *", min_value=0, max_value=12,
                                                       value=None, key="s2_no_eval", placeholder="Enter 0-12")

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                back = st.form_submit_button(
                    "Back to Demographics", use_container_width=True)
            with col2:
                submitted = st.form_submit_button(
                    "Next: Financial Status", use_container_width=True, type="primary")

            if back:
                st.session_state.current_step = 1
                st.rerun()

            if submitted:
                # Validate all fields are filled
                errors = []

                if units_1_credited is None:
                    errors.append("1st Sem Units Credited")
                if units_1_enrolled is None:
                    errors.append("1st Sem Units Enrolled")
                if units_1_evaluations is None:
                    errors.append("1st Sem Evaluations")
                if units_1_approved is None:
                    errors.append("1st Sem Units Approved")
                if units_1_grade is None:
                    errors.append("1st Sem Grade Average")
                if units_1_without_eval is None:
                    errors.append("1st Sem Without Evaluations")
                if units_2_credited is None:
                    errors.append("2nd Sem Units Credited")
                if units_2_enrolled is None:
                    errors.append("2nd Sem Units Enrolled")
                if units_2_evaluations is None:
                    errors.append("2nd Sem Evaluations")
                if units_2_approved is None:
                    errors.append("2nd Sem Units Approved")
                if units_2_grade is None:
                    errors.append("2nd Sem Grade Average")
                if units_2_without_eval is None:
                    errors.append("2nd Sem Without Evaluations")

                if errors:
                    st.error(
                        f"⚠️ Please fill in the following required fields: {', '.join(errors)}")
                else:
                    # Save data to session state
                    st.session_state.form_data['units_1_credited'] = units_1_credited
                    st.session_state.form_data['units_1_enrolled'] = units_1_enrolled
                    st.session_state.form_data['units_1_evaluations'] = units_1_evaluations
                    st.session_state.form_data['units_1_approved'] = units_1_approved
                    st.session_state.form_data['units_1_grade'] = units_1_grade
                    st.session_state.form_data['units_1_without_eval'] = units_1_without_eval
                    st.session_state.form_data['units_2_credited'] = units_2_credited
                    st.session_state.form_data['units_2_enrolled'] = units_2_enrolled
                    st.session_state.form_data['units_2_evaluations'] = units_2_evaluations
                    st.session_state.form_data['units_2_approved'] = units_2_approved
                    st.session_state.form_data['units_2_grade'] = units_2_grade
                    st.session_state.form_data['units_2_without_eval'] = units_2_without_eval

                    st.session_state.current_step = 3
                    st.rerun()

    # ============================================================================
    # STEP 3: FINANCIAL STATUS
    # ============================================================================

    elif st.session_state.current_step == 3:
        st.markdown('<div class="section-header">Step 3: Financial Status</div>',
                    unsafe_allow_html=True)

        with st.form("step3_form"):
            st.markdown("#### Financial Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                scholarship_options = ["-"] + \
                    get_selectbox_options('Attribute')
                scholarship_label = st.selectbox("Scholarship Holder *",
                                                 options=scholarship_options, index=0, key="scholarship")

            with col2:
                tuition_options = ["-"] + get_selectbox_options('Attribute')
                tuition_updated_label = st.selectbox("Tuition Fees Up to Date *",
                                                     options=tuition_options, index=0, key="tuition")

            with col3:
                debtor_options = ["-"] + get_selectbox_options('Attribute')
                debtor_label = st.selectbox("Debtor Status *",
                                            options=debtor_options, index=0, key="debtor")

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                back = st.form_submit_button(
                    "Back to Academic", use_container_width=True)
            with col2:
                submitted = st.form_submit_button(
                    "Next: Economic Factors", use_container_width=True, type="primary")

            if back:
                st.session_state.current_step = 2
                st.rerun()

            if submitted:
                # Validate all fields are filled
                errors = []

                if scholarship_label == "-":
                    errors.append("Scholarship Holder")
                if tuition_updated_label == "-":
                    errors.append("Tuition Fees Up to Date")
                if debtor_label == "-":
                    errors.append("Debtor Status")

                if errors:
                    st.error(
                        f"⚠️ Please fill in the following required fields: {', '.join(errors)}")
                else:
                    # Save data to session state
                    st.session_state.form_data['scholarship_label'] = scholarship_label
                    st.session_state.form_data['tuition_updated_label'] = tuition_updated_label
                    st.session_state.form_data['debtor_label'] = debtor_label

                    st.session_state.current_step = 4
                    st.rerun()

    # ============================================================================
    # STEP 4: ECONOMIC FACTORS & PREDICTION
    # ============================================================================
    elif st.session_state.current_step == 4:
        st.markdown('<div class="section-header">Step 4: Economic Indicators</div>',
                    unsafe_allow_html=True)

        with st.form("step4_form"):
            st.info("Note: These are macroeconomic indicators for the enrollment period")

            col1, col2, col3 = st.columns(3)

            with col1:
                unemployment_rate = st.number_input("Unemployment Rate (%) *",
                                                    min_value=0.0, max_value=20.0, value=None, step=0.1, placeholder="Enter unemployment rate")

            with col2:
                inflation_rate = st.number_input("Inflation Rate (%) *",
                                                 min_value=-5.0, max_value=10.0, value=None, step=0.1, placeholder="Enter inflation rate")

            with col3:
                gdp = st.number_input("GDP Growth Rate *",
                                      min_value=-5.0, max_value=5.0, value=None, step=0.01, placeholder="Enter GDP growth")

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                back = st.form_submit_button(
                    "Back to Financial", use_container_width=True)
            with col2:
                submitted = st.form_submit_button(
                    "Generate Prediction", use_container_width=True, type="primary")

            if back:
                st.session_state.current_step = 3
                st.rerun()

            if submitted:
                # Validate all fields are filled
                errors = []

                if unemployment_rate is None:
                    errors.append("Unemployment Rate")
                if inflation_rate is None:
                    errors.append("Inflation Rate")
                if gdp is None:
                    errors.append("GDP Growth Rate")

                if errors:
                    st.error(
                        f"⚠️ Please fill in the following required fields: {', '.join(errors)}")
                else:
                    # Save data to session state
                    st.session_state.form_data['unemployment_rate'] = unemployment_rate
                    st.session_state.form_data['inflation_rate'] = inflation_rate
                    st.session_state.form_data['gdp'] = gdp

                    # NOW GENERATE PREDICTION
                    with st.spinner("Analyzing student data..."):

                        # Retrieve all data from session state
                        data = st.session_state.form_data

                        # Convert labels back to codes
                        label_to_code = get_label_to_code_map

                        gender_code = label_to_code(
                            'Gender')[data['gender_label']]
                        marital_status_code = label_to_code('Marital Status')[
                            data['marital_status_label']]
                        nationality_code = label_to_code('Nationality')[
                            data['nationality_label']]
                        displaced_code = label_to_code(
                            'Attribute')[data['displaced_label']]
                        international_code = label_to_code(
                            'Attribute')[data['international_label']]
                        special_needs_code = label_to_code(
                            'Attribute')[data['special_needs_label']]
                        daytime_code = label_to_code('Attendance Regime')[
                            data['daytime_label']]
                        course_code = label_to_code('Course Name')[
                            data['course_label']]
                        previous_qual_code = label_to_code('Previous Qualification')[
                            data['previous_qual_label']]
                        application_mode_code = label_to_code('Application Mode')[
                            data['application_mode_label']]
                        mother_qual_code = label_to_code('Parent Qualification')[
                            data['mother_qual_label']]
                        mother_occ_code = label_to_code(
                            'Occupation')[data['mother_occ_label']]
                        father_qual_code = label_to_code('Parent Qualification')[
                            data['father_qual_label']]
                        father_occ_code = label_to_code(
                            'Occupation')[data['father_occ_label']]
                        scholarship_code = label_to_code(
                            'Attribute')[data['scholarship_label']]
                        tuition_updated_code = label_to_code(
                            'Attribute')[data['tuition_updated_label']]
                        debtor_code = label_to_code('Attribute')[
                            data['debtor_label']]

                        # Create input dataframe
                        input_data = pd.DataFrame({
                            'Marital status': [marital_status_code],
                            'Application mode': [application_mode_code],
                            'Application order': [data['application_order']],
                            'Course': [course_code],
                            'Daytime/evening attendance': [daytime_code],
                            'Previous qualification': [previous_qual_code],
                            'Nationality': [nationality_code],
                            "Mother's qualification": [mother_qual_code],
                            "Father's qualification": [father_qual_code],
                            "Mother's occupation": [mother_occ_code],
                            "Father's occupation": [father_occ_code],
                            'Displaced': [displaced_code],
                            'Educational special needs': [special_needs_code],
                            'Debtor': [debtor_code],
                            'Tuition fees up to date': [tuition_updated_code],
                            'Gender': [gender_code],
                            'Scholarship holder': [scholarship_code],
                            'Age at enrollment': [data['age']],
                            'International': [international_code],
                            'Curricular units 1st sem (credited)': [data['units_1_credited']],
                            'Curricular units 1st sem (enrolled)': [data['units_1_enrolled']],
                            'Curricular units 1st sem (evaluations)': [data['units_1_evaluations']],
                            'Curricular units 1st sem (approved)': [data['units_1_approved']],
                            'Curricular units 1st sem (grade)': [data['units_1_grade']],
                            'Curricular units 1st sem (without evaluations)': [data['units_1_without_eval']],
                            'Curricular units 2nd sem (credited)': [data['units_2_credited']],
                            'Curricular units 2nd sem (enrolled)': [data['units_2_enrolled']],
                            'Curricular units 2nd sem (evaluations)': [data['units_2_evaluations']],
                            'Curricular units 2nd sem (approved)': [data['units_2_approved']],
                            'Curricular units 2nd sem (grade)': [data['units_2_grade']],
                            'Curricular units 2nd sem (without evaluations)': [data['units_2_without_eval']],
                            'Unemployment rate': [unemployment_rate],
                            'Inflation rate': [inflation_rate],
                            'GDP': [gdp]
                        })

                        # Ensure correct column order
                        if feature_names:
                            try:
                                input_data = input_data[feature_names]
                            except KeyError:
                                pass

                        # Apply scaling if scaler exists (for Logistic Regression/SVM)
                        if scaler is not None:
                            numeric_cols = input_data.select_dtypes(include=[np.number]).columns.tolist()
                            input_data_scaled = input_data.copy()
                            input_data_scaled[numeric_cols] = scaler.transform(input_data[numeric_cols])
                            input_data = input_data_scaled

                        try:
                            # Make prediction
                            prediction = model.predict(input_data)[0]
                            prediction_proba = model.predict_proba(input_data)[
                                0]

                            dropout_risk = prediction_proba[1] * 100

                            # Determine risk level
                            if dropout_risk >= 60:
                                risk_class = "risk-high"
                                risk_label_display = "HIGH RISK"
                                risk_color = "#e74c3c"
                                risk_icon = "[!]"
                                recommendations = [
                                    "**URGENT**: Schedule immediate meeting with academic advisor within 48 hours",
                                    "Assess emergency financial aid eligibility and fast-track application",
                                    "Connect with academic tutoring services and study skills workshops",
                                    "Consider temporary course load reduction to improve focus",
                                    "Provide access to counseling services for stress management",
                                    "Assign peer mentor from same program/background",
                                    "Weekly check-ins for remainder of semester"
                                ]
                            elif dropout_risk >= 30:
                                risk_class = "risk-medium"
                                risk_label_display = "MEDIUM RISK"
                                risk_color = "#f39c12"
                                risk_icon = "[!]"
                                recommendations = [
                                    "Schedule proactive meeting with academic advisor within 2 weeks",
                                    "Monitor academic progress bi-weekly through semester",
                                    "Provide information on tutoring, study groups, and office hours",
                                    "Address any emerging financial or personal issues early",
                                    "Connect with student success coaching services",
                                    "Consider joining peer study groups or academic clubs"
                                ]
                            else:
                                risk_class = "risk-low"
                                risk_label_display = "LOW RISK"
                                risk_color = "#2ecc71"
                                risk_icon = "[✓]"
                                recommendations = [
                                    "Student showing positive indicators across multiple factors",
                                    "Maintain regular semester check-ins as standard practice",
                                    "Encourage continued academic engagement and participation",
                                    "Celebrate achievements and maintain motivation",
                                    "Connect with leadership and advanced opportunities",
                                    "Consider as potential peer mentor for at-risk students"
                                ]

                            # Store prediction in session state
                            st.session_state.prediction_complete = True
                            st.session_state.dropout_risk = dropout_risk
                            st.session_state.prediction_proba = prediction_proba
                            st.session_state.risk_class = risk_class
                            st.session_state.risk_label_display = risk_label_display
                            st.session_state.risk_color = risk_color
                            st.session_state.risk_icon = risk_icon
                            st.session_state.recommendations = recommendations

                            st.rerun()

                        except Exception as e:
                            st.error(f"⚠️ Prediction Error: {str(e)}")
                            with st.expander("Debug Information"):
                                st.write("**Error Details:**", str(e))
                                st.write("**Input Shape:**", input_data.shape)
                                st.dataframe(input_data)

    # ============================================================================
    # DISPLAY PREDICTION RESULTS
    # ============================================================================

    if 'prediction_complete' in st.session_state and st.session_state.prediction_complete:
        st.markdown("---")
        st.balloons()
        st.success("Risk Assessment Complete!")

        # Show student profile summary
        with st.expander("Complete Student Profile", expanded=False):
            data = st.session_state.form_data

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                **Demographics:**
                - Age: {data['age']}
                - Gender: {data['gender_label']}
                - Marital: {data['marital_status_label']}
                - Nationality: {data['nationality_label']}
                - Displaced: {data['displaced_label']}
                - International: {data['international_label']}
                - Special Needs: {data['special_needs_label']}
                - Attendance: {data['daytime_label']}
                """)

            with col2:
                st.markdown(f"""
                **Academic Background:**
                - Course: {data['course_label'][:30]}...
                - Previous Qual: {data['previous_qual_label'][:30]}...
                - Application Mode: {data['application_mode_label'][:30]}...
                - Application Order: {data['application_order']}
                - Mother's Qual: {data['mother_qual_label'][:30]}...
                - Father's Qual: {data['father_qual_label'][:30]}...
                """)

            with col3:
                st.markdown(f"""
                **Academic Performance:**
                - 1st Sem Enrolled: {data['units_1_enrolled']}
                - 1st Sem Approved: {data['units_1_approved']}
                - 1st Sem Grade: {data['units_1_grade']}/20
                - 2nd Sem Enrolled: {data['units_2_enrolled']}
                - 2nd Sem Approved: {data['units_2_approved']}
                - 2nd Sem Grade: {data['units_2_grade']}/20
                """)

            with col4:
                st.markdown(f"""
                **Financial & Economic:**
                - Scholarship: {data['scholarship_label']}
                - Tuition Updated: {data['tuition_updated_label']}
                - Debtor: {data['debtor_label']}
                - Unemployment: {data['unemployment_rate']}%
                - Inflation: {data['inflation_rate']}%
                - GDP Growth: {data['gdp']}
                """)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"""
            <div class="metric-card {st.session_state.risk_class}">
                <h2 style="text-align: center;">{st.session_state.risk_icon} {st.session_state.risk_label_display}</h2>
                <h1 style="text-align: center; color: {st.session_state.risk_color}; font-size: 72px; margin: 20px 0;">
                    {st.session_state.dropout_risk:.1f}%
                </h1>
                <p style="text-align: center; font-size: 18px;">
                    <b>Dropout Probability</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.dropout_risk,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Level"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': st.session_state.risk_color},
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 60], 'color': '#fff3cd'},
                        {'range': [60, 100], 'color': '#fee'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))

            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Recommended Interventions")

            for i, rec in enumerate(st.session_state.recommendations, 1):
                st.markdown(f"{i}. {rec}")

            st.markdown("---")

            # Confidence breakdown
            st.markdown("### Prediction Confidence")

            fig = px.bar(
                x=['Graduate', 'Dropout'],
                y=[st.session_state.prediction_proba[0]*100,
                    st.session_state.prediction_proba[1]*100],
                color=['Graduate', 'Dropout'],
                color_discrete_map={
                    'Graduate': '#2ecc71', 'Dropout': '#e74c3c'},
                text=[f"{st.session_state.prediction_proba[0]*100:.1f}%",
                      f"{st.session_state.prediction_proba[1]*100:.1f}%"]
            )

            fig.update_traces(textposition='outside')
            fig.update_layout(
                showlegend=False,
                height=300,
                yaxis_range=[0, 110],
                xaxis_title="",
                yaxis_title="Probability (%)"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Reset button
        if st.button("Assess Another Student", use_container_width=True, type="primary"):
            # Clear all session state
            st.session_state.current_step = 1
            st.session_state.form_data = {}
            st.session_state.prediction_complete = False
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 30px;'>
        <p style='font-size: 18px; font-weight: bold;'>Student Dropout Prediction System</p>
        <p>Empowering Institutions to Support Student Success Through Data-Driven Insights</p>
        <p style='margin-top: 15px;'>Built with Streamlit • Powered by Logistic Regression • © 2024-2025</p>
    </div>
""", unsafe_allow_html=True)
