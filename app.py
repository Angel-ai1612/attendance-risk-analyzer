"""
Smart Attendance Risk Analyzer - Dashboard
Interactive web interface for attendance risk predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Attendance Risk Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
    }
    .risk-warning {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
    }
    .risk-safe {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_config():
    """Load trained model and config."""
    try:
        model = joblib.load('models/xgboost_baseline.pkl')
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return model, config
    except FileNotFoundError as e:
        st.error(f"❌ Error loading files: {e}")
        st.stop()


@st.cache_data
def load_data():
    """Load feature data and predictions."""
    try:
        features_df = pd.read_csv('data/features.csv')
        return features_df
    except FileNotFoundError:
        st.error("❌ Features not found. Run feature_engineering.py first!")
        st.stop()


def get_risk_label(risk_prob):
    """Convert risk probability to label."""
    if risk_prob >= 0.70:
        return "High Risk", "#EF4444"
    elif risk_prob >= 0.50:
        return "Warning", "#F59E0B"
    else:
        return "Safe", "#10B981"


def make_predictions(model, features_df):
    """Generate predictions for all students."""
    feature_cols = [
        'consistency_score', 'absence_streak_max', 'absence_streak_current',
        'trend_slope', 'weekend_vs_weekday_ratio', 'early_vs_late_month_ratio',
        'days_since_last_absence', 'total_sessions_missed', 'has_multiple_risk_factors'
    ]
    
    X = features_df[feature_cols]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    features_df['prediction'] = predictions
    features_df['risk_probability'] = probabilities
    features_df['risk_label'] = features_df['risk_probability'].apply(
        lambda x: get_risk_label(x)[0]
    )
    
    return features_df


def overview_tab(df, config):
    """Overview dashboard with key metrics and charts."""
    st.markdown('<p class="main-header">🎓 Attendance Risk Dashboard</p>', unsafe_allow_html=True)
    st.markdown("**Predictive analytics for proactive student support**")
    st.markdown("---")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Students",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        high_risk = len(df[df['risk_label'] == 'High Risk'])
        st.metric(
            label="High Risk",
            value=f"{high_risk:,}",
            delta=f"{high_risk/len(df)*100:.1f}%"
        )
    
    with col3:
        warning = len(df[df['risk_label'] == 'Warning'])
        st.metric(
            label="Warning",
            value=f"{warning:,}",
            delta=f"{warning/len(df)*100:.1f}%"
        )
    
    with col4:
        safe = len(df[df['risk_label'] == 'Safe'])
        st.metric(
            label="Safe",
            value=f"{safe:,}",
            delta=f"{safe/len(df)*100:.1f}%"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk Distribution")
        risk_counts = df['risk_label'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker=dict(colors=['#EF4444', '#10B981', '#F59E0B']),
            textinfo='label+percent',
            textfont=dict(size=14)
        )])
        fig.update_layout(
            showlegend=True,
            height=350,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏫 Risk by Department")
        dept_risk = df.groupby('department')['risk_probability'].mean().sort_values(ascending=True)
        
        fig = go.Figure(data=[go.Bar(
            x=dept_risk.values,
            y=dept_risk.index,
            orientation='h',
            marker=dict(
                color=dept_risk.values,
                colorscale='RdYlGn_r',
                showscale=False
            )
        )])
        fig.update_layout(
            xaxis_title="Average Risk Probability",
            yaxis_title="",
            height=350,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk by Year
    st.subheader("📈 Risk Distribution by Academic Year")
    year_risk = df.groupby(['year', 'risk_label']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    colors = {'High Risk': '#EF4444', 'Warning': '#F59E0B', 'Safe': '#10B981'}
    
    for risk in ['High Risk', 'Warning', 'Safe']:
        if risk in year_risk.columns:
            fig.add_trace(go.Bar(
                name=risk,
                x=year_risk.index,
                y=year_risk[risk],
                marker_color=colors[risk]
            ))
    
    fig.update_layout(
        barmode='stack',
        xaxis_title="Academic Year",
        yaxis_title="Number of Students",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top At-Risk Students
    st.subheader("⚠️ Top 10 Highest Risk Students")
    top_risk = df.nlargest(10, 'risk_probability')[
        ['student_id', 'department', 'year', 'risk_probability', 'risk_label',
         'total_sessions_missed', 'absence_streak_max']
    ].copy()
    top_risk['risk_probability'] = top_risk['risk_probability'].apply(lambda x: f"{x*100:.1f}%")
    
    st.dataframe(
        top_risk,
        use_container_width=True,
        hide_index=True,
        column_config={
            "student_id": "Student ID",
            "department": "Department",
            "year": "Year",
            "risk_probability": "Risk Score",
            "risk_label": "Status",
            "total_sessions_missed": "Sessions Missed",
            "absence_streak_max": "Max Absence Streak"
        }
    )


def student_lookup_tab(df, model, config):
    """Individual student lookup and analysis."""
    st.markdown('<p class="main-header">🔍 Student Lookup</p>', unsafe_allow_html=True)
    st.markdown("Search for individual student predictions and insights")
    st.markdown("---")
    
    # Search
    col1, col2 = st.columns([2, 1])
    
    with col1:
        student_id = st.selectbox(
            "Select Student ID",
            options=df['student_id'].unique(),
            index=0
        )
    
    with col2:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=['High Risk', 'Warning', 'Safe'],
            default=['High Risk', 'Warning', 'Safe']
        )
    
    # Get student data
    student = df[df['student_id'] == student_id].iloc[0]
    
    # Student Overview
    st.markdown("### 👤 Student Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Student ID", student['student_id'])
    
    with col2:
        st.metric("Department", student['department'])
    
    with col3:
        st.metric("Year", student['year'])
    
    with col4:
        risk_label, color = get_risk_label(student['risk_probability'])
        st.markdown(f"**Risk Status:** <span style='color:{color}; font-weight:700; font-size:1.2rem;'>{risk_label}</span>", 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk Assessment
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📊 Risk Assessment")
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=student['risk_probability'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 20}},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#d1fae5'},
                    {'range': [50, 70], 'color': '#fef3c7'},
                    {'range': [70, 100], 'color': '#fee2e2'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(t=40, b=0, l=40, r=40))
        st.plotly_chart(fig, use_container_width=True)
        
        # Intervention
        st.markdown("### 💡 Recommended Action")
        risk_config = config['risk_levels']
        if student['risk_probability'] >= 0.70:
            intervention = risk_config['high_risk']['intervention']
        elif student['risk_probability'] >= 0.50:
            intervention = risk_config['warning']['intervention']
        else:
            intervention = risk_config['safe']['intervention']
        
        st.info(intervention)
    
    with col2:
        st.markdown("### 📈 Feature Breakdown")
        
        # Feature values
        features = {
            'Consistency Score': student['consistency_score'],
            'Max Absence Streak': student['absence_streak_max'],
            'Current Absence Streak': student['absence_streak_current'],
            'Trend Slope': student['trend_slope'],
            'Total Sessions Missed': student['total_sessions_missed'],
            'Days Since Last Absence': student['days_since_last_absence'],
            'Weekend/Weekday Ratio': student['weekend_vs_weekday_ratio'],
            'Multiple Risk Factors': 'Yes' if student['has_multiple_risk_factors'] else 'No'
        }
        
        # Create feature comparison
        feature_df = pd.DataFrame({
            'Feature': list(features.keys()),
            'Value': [str(v) if isinstance(v, str) else f"{v:.2f}" for v in features.values()]
        })
        
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
        
        # Key insights
        st.markdown("### 🔑 Key Insights")
        
        insights = []
        if student['absence_streak_max'] > 5:
            insights.append(f"⚠️ Extended absence streak detected ({student['absence_streak_max']} days)")
        if student['consistency_score'] < 0.5:
            insights.append("⚠️ Erratic attendance pattern (low consistency)")
        if student['total_sessions_missed'] > 20:
            insights.append(f"⚠️ High number of sessions missed ({student['total_sessions_missed']})")
        if student['trend_slope'] < -0.02:
            insights.append("📉 Declining attendance trend")
        if student['has_multiple_risk_factors']:
            insights.append("🚨 Multiple risk factors present")
        
        if not insights:
            insights.append("✅ No major risk factors detected")
        
        for insight in insights:
            st.markdown(f"- {insight}")


def insights_tab(df, model, config):
    """Model insights and feature importance."""
    st.markdown('<p class="main-header">💡 Model Insights</p>', unsafe_allow_html=True)
    st.markdown("Understanding what drives attendance risk predictions")
    st.markdown("---")
    
    # Load model metadata
    try:
        with open('models/model_metadata.yaml', 'r') as f:
            metadata = yaml.safe_load(f)
    except:
        metadata = None
    
    # Model Performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Performance")
        if metadata and 'metrics' in metadata:
            metrics = metadata['metrics']
            
            metric_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Score': [
                    f"{metrics.get('accuracy', 0)*100:.2f}%",
                    f"{metrics.get('precision', 0)*100:.2f}%",
                    f"{metrics.get('recall', 0)*100:.2f}%",
                    f"{metrics.get('f1_score', 0)*100:.2f}%",
                    f"{metrics.get('roc_auc', 0):.4f}"
                ]
            })
            
            st.dataframe(metric_df, use_container_width=True, hide_index=True)
            
            st.info("📌 **High Recall (81%)** means the model catches most at-risk students, though some safe students may be flagged.")
        else:
            st.warning("Model metadata not found")
    
    with col2:
        st.subheader("🎯 Feature Importance")
        if metadata and 'feature_importance' in metadata:
            feat_imp = pd.DataFrame(metadata['feature_importance']).head(8)
            
            fig = go.Figure(go.Bar(
                x=feat_imp['importance'],
                y=feat_imp['feature'],
                orientation='h',
                marker=dict(
                    color=feat_imp['importance'],
                    colorscale='Viridis',
                    showscale=False
                )
            ))
            fig.update_layout(
                xaxis_title="Importance",
                yaxis_title="",
                height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance data not found")
    
    # Distribution Analysis
    st.subheader("📈 Feature Distributions by Risk Level")
    
    feature_to_plot = st.selectbox(
        "Select Feature to Analyze",
        options=[
            'consistency_score', 'absence_streak_max', 'total_sessions_missed',
            'trend_slope', 'days_since_last_absence'
        ]
    )
    
    fig = px.box(
        df,
        x='risk_label',
        y=feature_to_plot,
        color='risk_label',
        color_discrete_map={'High Risk': '#EF4444', 'Warning': '#F59E0B', 'Safe': '#10B981'},
        category_orders={'risk_label': ['High Risk', 'Warning', 'Safe']}
    )
    fig.update_layout(
        xaxis_title="Risk Level",
        yaxis_title=feature_to_plot.replace('_', ' ').title(),
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Info
    st.markdown("---")
    st.subheader("ℹ️ About This Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Type:** XGBoost Classifier
        
        **Prediction Task:** Forecasts attendance risk 14 days ahead using only behavioral patterns (no current attendance data)
        
        **Training Data:** 15,000 students with 9 pattern-based features
        
        **Key Innovation:** Uses consistency, trends, and streaks instead of raw attendance rates to prevent data leakage
        """)
    
    with col2:
        if metadata:
            st.markdown(f"""
            **Model Version:** 2.0
            
            **Trained:** {metadata.get('trained_at', 'Unknown')[:10]}
            
            **Cross-Validation:** 5-fold
            
            **Primary Metric:** ROC-AUC = {metadata.get('metrics', {}).get('roc_auc', 0):.4f}
            """)


def main():
    """Main dashboard application."""
    # Load resources
    model, config = load_model_and_config()
    df = load_data()
    
    # Generate predictions
    df = make_predictions(model, df)
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f393.png", width=80)
        st.title("Navigation")
        
        tab_selection = st.radio(
            "Go to",
            ["📊 Overview", "🔍 Student Lookup", "💡 Insights"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📌 Quick Stats")
        st.metric("Students Analyzed", f"{len(df):,}")
        st.metric("Model Accuracy", "73.1%")
        st.metric("Prediction Horizon", "14 days")
        
        st.markdown("---")
        st.markdown("**🎓 Smart Attendance Risk Analyzer v2.0**")
        st.markdown("Built with Streamlit + XGBoost")
    
    # Main content
    if tab_selection == "📊 Overview":
        overview_tab(df, config)
    elif tab_selection == "🔍 Student Lookup":
        student_lookup_tab(df, model, config)
    else:
        insights_tab(df, model, config)


if __name__ == "__main__":
    main()