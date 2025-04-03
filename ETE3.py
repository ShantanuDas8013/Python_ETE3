import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import altair as alt
import re
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
# Import the image processing module
from image_processing import (
    get_placeholder_sports_images, load_image, apply_filter, 
    adjust_brightness, adjust_contrast, adjust_color, 
    add_sport_overlay, create_image_gallery, create_image_uploader_section
)

# Check for statsmodels availability
has_statsmodels = False
try:
    import statsmodels.api as sm
    has_statsmodels = True
except ImportError:
    pass

# Download nltk data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Inter-College Tournament Analysis",
    page_icon="🏆",
    layout="wide"
)

# Function to generate synthetic dataset
def generate_tournament_data():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define constants
    num_participants = 300
    num_days = 5
    sports = [
        "Basketball", "Cricket", "Football"  # Only sports with available images
    ]
    colleges = [
        "Engineering College", "Arts College", "Medical College", 
        "Science College", "Business College", "Law College",
        "Technology Institute", "Sports Academy", "Design School"
    ]
    
    # Add states data
    states = [
        "Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Uttar Pradesh",
        "West Bengal", "Gujarat", "Rajasthan", "Telangana", "Punjab"
    ]
    
    # Generate start date (e.g., a week ago)
    start_date = datetime.now() - timedelta(days=num_days+2)
    
    # Create lists for each column
    participant_ids = list(range(1, num_participants + 1))
    names = [f"Participant_{i}" for i in participant_ids]
    colleges_list = [random.choice(colleges) for _ in range(num_participants)]
    genders = np.random.choice(['Male', 'Female'], size=num_participants)
    ages = np.random.randint(18, 25, size=num_participants)
    sports_list = [random.choice(sports) for _ in range(num_participants)]
    
    # Add states for each participant - assign colleges to specific states for realism
    college_state_map = {college: random.choice(states) for college in colleges}
    states_list = [college_state_map[college] for college in colleges_list]
    
    # Assign dates ensuring distribution across all days
    dates = []
    for i in range(num_participants):
        day_offset = i % num_days
        dates.append((start_date + timedelta(days=day_offset)).strftime('%Y-%m-%d'))
    
    # Performance metrics (e.g. score out of 100)
    performances = np.random.randint(0, 101, size=num_participants)
    
    # Satisfaction ratings (1-5 scale)
    satisfaction = np.random.randint(1, 6, size=num_participants)
    
    # Generate synthetic feedback
    positive_comments = [
        "Great event, thoroughly enjoyed it!",
        "Well organized tournament, will participate again next year.",
        "Excellent facilities and friendly staff.",
        "Had a wonderful time competing with other colleges.",
        "Very professional organization and fair judging.",
        "The tournament schedule was perfectly planned.",
        "Loved the competitive spirit and sportsmanship."
    ]
    
    neutral_comments = [
        "The event was okay, some improvements needed.",
        "Average organization, could be better next time.",
        "Decent facilities but limited refreshments.",
        "It was a standard tournament experience.",
        "Some delays in the schedule but otherwise fine."
    ]
    
    negative_comments = [
        "Poor organization, too many delays.",
        "Facilities were inadequate for the number of participants.",
        "Judging seemed biased in some events.",
        "Not enough water stations available.",
        "The scheduling was chaotic and confusing."
    ]
    
    feedbacks = []
    for rating in satisfaction:
        if rating >= 4:
            feedbacks.append(random.choice(positive_comments))
        elif rating <= 2:
            feedbacks.append(random.choice(negative_comments))
        else:
            feedbacks.append(random.choice(neutral_comments))
    
    # Create DataFrame
    data = {
        'ParticipantID': participant_ids,
        'Name': names,
        'College': colleges_list,
        'State': states_list,
        'Gender': genders,
        'Age': ages,
        'Sport': sports_list,
        'Date': dates,
        'Performance': performances,
        'Satisfaction': satisfaction,
        'Feedback': feedbacks
    }
    
    return pd.DataFrame(data)

# Function to perform sentiment analysis on feedback
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Return polarity score: -1 (negative) to 1 (positive)
    return analysis.sentiment.polarity

# Add function to preprocess text
def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to extract top words from text
def get_top_words(text_list, n=10, custom_stopwords=None):
    """Extract top n words from a list of texts, excluding stopwords"""
    nltk_stop_words = set(stopwords.words('english'))
    
    # Add custom stopwords if provided
    if custom_stopwords:
        nltk_stop_words.update(custom_stopwords)
    
    # Process all texts
    all_words = []
    for text in text_list:
        # Preprocess text
        clean_text = preprocess_text(text)
        # Tokenize
        words = word_tokenize(clean_text)
        # Remove stopwords and short words
        words = [word for word in words if word not in nltk_stop_words and len(word) > 2]
        all_words.extend(words)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Get top n words
    top_words = word_counts.most_common(n)
    
    return top_words

# Generate dataset (only once)
if 'data' not in st.session_state:
    st.session_state.data = generate_tournament_data()

# App title and introduction
st.title("🏆 Inter-College Tournament Analysis")
st.markdown("""
This application analyzes participation data from an inter-college tournament
featuring 300 participants across 5 days and 10 different sports events across different states.
""")

# Sidebar for filters
st.sidebar.header("Filters")

# Sport filter
selected_sports = st.sidebar.multiselect(
    "Select Sports",
    options=st.session_state.data['Sport'].unique(),
    default=st.session_state.data['Sport'].unique()[:3]
)

# State filter
selected_states = st.sidebar.multiselect(
    "Select States",
    options=st.session_state.data['State'].unique(),
    default=st.session_state.data['State'].unique()[:3]
)

# College filter
selected_colleges = st.sidebar.multiselect(
    "Select Colleges",
    options=st.session_state.data['College'].unique(),
    default=st.session_state.data['College'].unique()[:3]
)

# Date filter
min_date = pd.to_datetime(st.session_state.data['Date']).min().date()
max_date = pd.to_datetime(st.session_state.data['Date']).max().date()
selected_date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply filters
filtered_df = st.session_state.data.copy()

if selected_sports:
    filtered_df = filtered_df[filtered_df['Sport'].isin(selected_sports)]
if selected_colleges:
    filtered_df = filtered_df[filtered_df['College'].isin(selected_colleges)]
if selected_states:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
if len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_df = filtered_df[
        (pd.to_datetime(filtered_df['Date']).dt.date >= start_date) & 
        (pd.to_datetime(filtered_df['Date']).dt.date <= end_date)
    ]

# Main content
st.header("Tournament Dashboard")

# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Participants", len(filtered_df))
with col2:
    st.metric("Avg. Satisfaction", round(filtered_df['Satisfaction'].mean(), 2))
with col3:
    st.metric("Avg. Performance", round(filtered_df['Performance'].mean(), 2))
with col4:
    st.metric("Sports Events", filtered_df['Sport'].nunique())
with col5:
    st.metric("States Represented", filtered_df['State'].nunique())

# Show raw data if requested
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.dataframe(filtered_df)

# Charts section
st.header("Participation Analysis")

# Create tabs - Add a new tab for Day-wise Image Gallery
tabs = st.tabs([
    "Sports Analysis", 
    "College Analysis",
    "State Analysis",
    "Time Analysis",
    "Participant Demographics",
    "Feedback Analysis",
    "Sports Feedback Analysis",
    "Event Gallery"  # New tab
])

# Use the tabs
with tabs[0]:
    # Sports Analysis tab content
    st.subheader("Participation by Sport")
    
    # Bar chart for sports participation
    sport_counts = filtered_df['Sport'].value_counts().reset_index()
    sport_counts.columns = ['Sport', 'Count']
    
    sport_chart = alt.Chart(sport_counts).mark_bar().encode(
        x=alt.X('Sport', sort='-y'),
        y='Count',
        color=alt.Color('Sport', legend=None),
        tooltip=['Sport', 'Count']
    ).properties(
        height=400
    ).interactive()
    
    st.altair_chart(sport_chart, use_container_width=True)
    
    # Performance by sport
    sport_performance = filtered_df.groupby('Sport')['Performance'].mean().sort_values(ascending=False).reset_index()
    
    performance_chart = alt.Chart(sport_performance).mark_bar().encode(
        x=alt.X('Sport', sort='-y'),
        y='Performance',
        color=alt.Color('Sport', legend=None),
        tooltip=['Sport', 'Performance']
    ).properties(
        height=400,
        title="Average Performance by Sport"
    ).interactive()
    
    st.altair_chart(performance_chart, use_container_width=True)
    
    # NEW: Add bubble chart for sport popularity and performance
    st.subheader("Sport Popularity vs Performance")
    
    sport_bubble_data = filtered_df.groupby('Sport').agg(
        Count=('ParticipantID', 'count'),
        Avg_Performance=('Performance', 'mean'),
        Avg_Satisfaction=('Satisfaction', 'mean')
    ).reset_index()
    
    fig_bubble = px.scatter(
        sport_bubble_data, 
        x="Count", 
        y="Avg_Performance",
        size="Avg_Satisfaction",
        color="Sport",
        hover_name="Sport",
        hover_data={
            "Count": True,
            "Avg_Performance": ":.1f",
            "Avg_Satisfaction": ":.1f",
            "Sport": False
        },
        labels={
            "Count": "Number of Participants",
            "Avg_Performance": "Average Performance",
            "Avg_Satisfaction": "Average Satisfaction"
        },
        size_max=60,
        title="Sport Popularity vs Performance (Size indicates satisfaction)"
    )

    fig_bubble.update_traces(
        marker=dict(line=dict(width=2, color='DarkSlateGray')),
        selector=dict(mode='markers')
    )

    fig_bubble.update_layout(
        xaxis_title="Number of Participants (Popularity)",
        yaxis_title="Average Performance Score",
        showlegend=True,
        legend_title="Sports",
        plot_bgcolor='white',
        hoverlabel=dict(bgcolor="white"),
        xaxis=dict(
            gridcolor='LightGray',
            zerolinecolor='LightGray',
        ),
        yaxis=dict(
            gridcolor='LightGray',
            zerolinecolor='LightGray',
        )
    )
    
    st.plotly_chart(fig_bubble, use_container_width=True)

    # NEW: Add radar chart for sports comparison
    st.subheader("Sports Performance Radar Chart")
    
    # Get top 5 sports by participation for better visualization
    top_sports = sport_bubble_data.nlargest(5, 'Count')
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=top_sports['Avg_Performance'],
        theta=top_sports['Sport'],
        fill='toself',
        name='Avg Performance'
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=top_sports['Avg_Satisfaction'] * 20,  # Scale up to match performance scale
        theta=top_sports['Sport'],
        fill='toself',
        name='Avg Satisfaction (scaled)'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

with tabs[1]:
    # College Analysis tab content
    st.subheader("Participation by College")
    
    # Bar chart for college participation
    college_counts = filtered_df['College'].value_counts().reset_index()
    college_counts.columns = ['College', 'Count']
    
    college_chart = alt.Chart(college_counts).mark_bar().encode(
        x=alt.X('College', sort='-y'),
        y='Count',
        color=alt.Color('College', legend=None),
        tooltip=['College', 'Count']
    ).properties(
        height=400
    ).interactive()
    
    st.altair_chart(college_chart, use_container_width=True)
    
    # College performance
    cols1, cols2 = st.columns(2)
    
    with cols1:
        college_performance = filtered_df.groupby('College')['Performance'].mean().sort_values(ascending=False)
        st.subheader("Average Performance by College")
        st.bar_chart(college_performance)
    
    with cols2:
        college_satisfaction = filtered_df.groupby('College')['Satisfaction'].mean().sort_values(ascending=False)
        st.subheader("Average Satisfaction by College")
        st.bar_chart(college_satisfaction)
    
    # NEW: Add a treemap for colleges grouped by state
    st.subheader("College Representation by State (Treemap)")
    
    college_state_counts = filtered_df.groupby(['State', 'College']).size().reset_index()
    college_state_counts.columns = ['State', 'College', 'Count']
    
    fig_treemap = px.treemap(
        college_state_counts, 
        path=[px.Constant("All States"), 'State', 'College'], 
        values='Count',
        color='Count',
        color_continuous_scale='RdBu',
        title='Participation by State and College'
    )
    
    st.plotly_chart(fig_treemap, use_container_width=True)

with tabs[2]:
    # State Analysis tab content
    st.subheader("Participation by State")
    
    # Bar chart for state participation
    state_counts = filtered_df['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']
    
    state_chart = alt.Chart(state_counts).mark_bar().encode(
        x=alt.X('State', sort='-y'),
        y='Count',
        color=alt.Color('State', legend=None),
        tooltip=['State', 'Count']
    ).properties(
        height=400
    ).interactive()
    
    st.altair_chart(state_chart, use_container_width=True)
    
    # Choropleth map (simulated with heatmap since we don't have real geo data)
    st.subheader("State Participation Distribution")
    
    # Performance metrics by state
    col1, col2 = st.columns(2)
    
    with col1:
        state_performance = filtered_df.groupby('State')['Performance'].mean().sort_values(ascending=False)
        
        fig_perf = px.bar(
            x=state_performance.index,
            y=state_performance.values,
            labels={'x': 'State', 'y': 'Average Performance'},
            title="Average Performance by State",
            color=state_performance.values,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with col2:
        sports_by_state = filtered_df.groupby(['State', 'Sport']).size().reset_index()
        sports_by_state.columns = ['State', 'Sport', 'Count']
        
        # Heatmap of sports by state
        fig_heatmap = px.density_heatmap(
            sports_by_state,
            x='State',
            y='Sport',
            z='Count',
            color_continuous_scale="Viridis",
            title="Sport Participation by State"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # NEW: Sunburst chart for state -> sport -> gender hierarchy
    st.subheader("State, Sport, and Gender Distribution")
    
    state_sport_gender = filtered_df.groupby(['State', 'Sport', 'Gender']).size().reset_index()
    state_sport_gender.columns = ['State', 'Sport', 'Gender', 'Count']
    
    fig_sunburst = px.sunburst(
        state_sport_gender,
        path=['State', 'Sport', 'Gender'],
        values='Count',
        color='Count',
        title="Hierarchical View of Participation"
    )
    
    st.plotly_chart(fig_sunburst, use_container_width=True)

with tabs[3]:
    # Time Analysis tab content
    st.subheader("Daily Participation")
    
    # Line chart for daily participation
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
    daily_counts = filtered_df.groupby(filtered_df['Date'].dt.date).size().reset_index()
    daily_counts.columns = ['Date', 'Count']
    
    daily_chart = alt.Chart(daily_counts).mark_line(point=True).encode(
        x='Date',
        y='Count',
        tooltip=['Date', 'Count']
    ).properties(
        height=400,
        title="Daily Participation"
    ).interactive()
    
    st.altair_chart(daily_chart, use_container_width=True)
    
    # Heatmap of sports across days
    st.subheader("Sports Participation by Day")
    
    # Prepare data for heatmap
    filtered_df['DateFormatted'] = filtered_df['Date'].dt.date
    heatmap_data = filtered_df.groupby(['DateFormatted', 'Sport']).size().reset_index()
    heatmap_data.columns = ['Date', 'Sport', 'Count']
    
    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x='Date:O',
        y='Sport:O',
        color='Count:Q',
        tooltip=['Date', 'Sport', 'Count']
    ).properties(
        height=400
    )
    
    st.altair_chart(heatmap, use_container_width=True)
    
    # NEW: Add state participation trend over days
    st.subheader("State Participation Trend")
    
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
    state_daily = filtered_df.groupby([filtered_df['Date'].dt.date, 'State']).size().reset_index()
    state_daily.columns = ['Date', 'State', 'Count']
    
    fig_state_trend = px.line(
        state_daily,
        x='Date',
        y='Count',
        color='State',
        title="State Participation Over Time",
        markers=True
    )
    
    st.plotly_chart(fig_state_trend, use_container_width=True)
    
    # NEW: Add stacked area chart for sports over time
    st.subheader("Sports Participation Trend")
    
    sports_daily = filtered_df.groupby([filtered_df['Date'].dt.date, 'Sport']).size().reset_index()
    sports_daily.columns = ['Date', 'Sport', 'Count']
    
    fig_area = px.area(
        sports_daily,
        x='Date',
        y='Count',
        color='Sport',
        title="Sports Participation Over Time",
        groupnorm='percent'  # Shows relative percentage
    )
    
    st.plotly_chart(fig_area, use_container_width=True)

with tabs[4]:
    # Participant Demographics tab content
    st.subheader("Participant Demographics")
    
    # Display summary statistics at the top
    st.info(f"""
    **Demographic Summary**: 
    - Total Participants: {len(filtered_df)}
    - Age Range: {filtered_df['Age'].min()} to {filtered_df['Age'].max()} years
    - Average Age: {filtered_df['Age'].mean():.1f} years
    - Gender Ratio: {filtered_df['Gender'].value_counts(normalize=True)['Male']:.1%} Male / {filtered_df['Gender'].value_counts(normalize=True)['Female']:.1%} Female
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution pie chart - improved with percentages and formatting
        gender_counts = filtered_df['Gender'].value_counts()
        gender_pcts = filtered_df['Gender'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
        
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Gender Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        # Add percentage labels to the pie chart
        fig_gender.update_traces(
            textposition='inside',
            textinfo='percent+label',
            insidetextfont=dict(color='white'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
        )
        
        fig_gender.update_layout(
            title_x=0.5,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # Age distribution histogram - improved with additional statistics
        fig_age = px.histogram(
            filtered_df,
            x='Age',
            color='Gender',
            marginal='box',
            title="Age Distribution by Gender",
            opacity=0.8,
            barmode='overlay',  # Overlay for better comparison
            histnorm='percent',  # Show as percentage for better interpretation
            color_discrete_sequence=['#636EFA', '#EF553B']
        )
        
        # Add mean age lines
        for gender in filtered_df['Gender'].unique():
            mean_age = filtered_df[filtered_df['Gender'] == gender]['Age'].mean()
            fig_age.add_vline(
                x=mean_age, 
                line_dash="dash", 
                line_color="black",
                annotation_text=f"{gender} Avg: {mean_age:.1f}",
                annotation_position="top"
            )
        
        fig_age.update_layout(
            title_x=0.5,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            bargap=0.1
        )
        
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Scatter plot for age vs performance - improved with trend annotations
    st.subheader("Age vs. Performance Analysis")
    
    # Create tabs for different views of the age-performance relationship
    age_perf_tab1, age_perf_tab2 = st.tabs(["By Sport", "By Gender"])
    
    with age_perf_tab1:
        # Check if statsmodels is available for trendline
        if has_statsmodels:
            fig_scatter = px.scatter(
                filtered_df,
                x='Age',
                y='Performance',
                color='Sport',
                hover_name='Sport',
                opacity=0.7,
                title="Age vs Performance by Sport",
                trendline="ols",
                labels={"Performance": "Performance Score (0-100)", "Age": "Age (years)"}
            )
            
            # Add overall trendline annotation
            age_perf_corr = filtered_df['Age'].corr(filtered_df['Performance']).round(2)
            correlation_note = f"Overall Correlation: {age_perf_corr}"
            fig_scatter.add_annotation(
                x=0.5, y=0.05,
                xref="paper", yref="paper",
                text=correlation_note,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
            
        else:
            # Create scatter plot without trendline if statsmodels is not available
            fig_scatter = px.scatter(
                filtered_df,
                x='Age',
                y='Performance',
                color='Sport',
                hover_name='Sport',
                opacity=0.7,
                title="Age vs Performance by Sport",
                labels={"Performance": "Performance Score (0-100)", "Age": "Age (years)"}
            )
            st.info("📊 Install statsmodels package for trendline visualization: `pip install statsmodels`")
        
        fig_scatter.update_layout(
            title_x=0.5,
            legend_title="Sport"
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with age_perf_tab2:
        # Create a scatter plot by gender
        fig_gender_scatter = px.scatter(
            filtered_df,
            x='Age',
            y='Performance',
            color='Gender',
            symbol='Gender',  # Add different symbols for genders
            opacity=0.7,
            title="Age vs Performance by Gender",
            trendline="ols" if has_statsmodels else None,
            labels={"Performance": "Performance Score (0-100)", "Age": "Age (years)"},
            color_discrete_sequence=['#636EFA', '#EF553B']
        )
        
        # Add annotations for gender-specific analysis
        for gender in filtered_df['Gender'].unique():
            gender_df = filtered_df[filtered_df['Gender'] == gender]
            avg_perf = gender_df['Performance'].mean()
            fig_gender_scatter.add_annotation(
                x=gender_df['Age'].max(),
                y=avg_perf,
                text=f"{gender} Avg: {avg_perf:.1f}",
                showarrow=True,
                arrowhead=1
            )
        
        fig_gender_scatter.update_layout(title_x=0.5)
        
        st.plotly_chart(fig_gender_scatter, use_container_width=True)
    
    # Box plot for performance by gender across sports - improved with better visualization
    st.subheader("Performance Distribution by Gender across Sports")
    
    fig_box = px.box(
        filtered_df,
        x='Sport',
        y='Performance',
        color='Gender',
        title="Performance Distribution by Gender and Sport",
        points="all",  # Show all points for smaller datasets
        notched=True,  # Add notches to show confidence interval around median
        color_discrete_sequence=['#636EFA', '#EF553B'],
        labels={"Performance": "Performance Score", "Sport": "Sport", "Gender": "Gender"}
    )
    
    # Add median lines for easier comparison
    fig_box.update_layout(
        boxmode='group',  # Group by Sport and Gender
        title_x=0.5,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    # Add overall average line
    overall_avg = filtered_df['Performance'].mean()
    fig_box.add_hline(
        y=overall_avg, 
        line_dash="dash", 
        line_color="black",
        annotation_text=f"Overall Average: {overall_avg:.1f}",
        annotation_position="top right"
    )
    
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Enhanced: Age Distribution across Sports with improved accuracy
    st.subheader("Age Distribution across Sports")
    
    # Create tabs for different age distribution views
    age_dist_tab1, age_dist_tab2 = st.tabs(["Heatmap View", "Distribution View"])
    
    with age_dist_tab1:
        # Calculate average age by sport and gender with sample size info
        age_heatmap_data = filtered_df.groupby(['Sport', 'Gender']).agg(
            avg_age=('Age', 'mean'),
            median_age=('Age', 'median'),
            min_age=('Age', 'min'),
            max_age=('Age', 'max'),
            std_age=('Age', 'std'),
            count=('Age', 'count')
        ).reset_index()
        
        # Check if we have enough data
        if len(age_heatmap_data) > 0:
            # Create pivot table for heatmap
            age_pivot = age_heatmap_data.pivot(
                index='Sport', 
                columns='Gender', 
                values='avg_age'
            )
            
            # Calculate overall average for centering the color scale
            overall_avg_age = filtered_df['Age'].mean()
            
            # Define color scale limits based on deviation from mean
            max_deviation = max(
                abs(filtered_df['Age'].max() - overall_avg_age),
                abs(filtered_df['Age'].min() - overall_avg_age)
            )
            min_color = overall_avg_age - max_deviation
            max_color = overall_avg_age + max_deviation
            
            # Calculate counts and other stats for annotation
            count_pivot = age_heatmap_data.pivot(index='Sport', columns='Gender', values='count')
            median_pivot = age_heatmap_data.pivot(index='Sport', columns='Gender', values='median_age')
            range_data = {}
            
            for sport in age_heatmap_data['Sport'].unique():
                for gender in age_heatmap_data['Gender'].unique():
                    subset = age_heatmap_data[(age_heatmap_data['Sport'] == sport) & (age_heatmap_data['Gender'] == gender)]
                    if not subset.empty:
                        key = (sport, gender)
                        range_data[key] = f"{subset['min_age'].iloc[0]}-{subset['max_age'].iloc[0]}"
            
            # Create annotation text with rich statistics
            annotations = []
            for i, sport in enumerate(age_pivot.index):
                for j, gender in enumerate(age_pivot.columns):
                    if pd.notna(age_pivot.iloc[i, j]) and pd.notna(count_pivot.iloc[i, j]):
                        # Calculate if this cell is significantly different from overall average
                        is_significant = abs(age_pivot.iloc[i, j] - overall_avg_age) > age_heatmap_data['std_age'].mean()
                        
                        # Format annotation text
                        text = (
                            f"<b>{age_pivot.iloc[i, j]:.1f}</b> yrs<br>"
                            f"Med: {median_pivot.iloc[i, j]:.1f}<br>"
                            f"n={int(count_pivot.iloc[i, j])}"
                        )
                        
                        # Add significance marker if applicable
                        if is_significant:
                            text = "★ " + text
                        
                        # Determine text color based on background color brightness
                        font_color = "white" if age_pivot.iloc[i, j] > (min_color + max_color) / 2 else "black"
                        
                        annotations.append(dict(
                            x=j,
                            y=i,
                            text=text,
                            showarrow=False,
                            font=dict(color=font_color, size=10),
                            align="center"
                        ))
            
            # Fill NaN values for visualization (doesn't affect actual data)
            age_pivot_filled = age_pivot.fillna(filtered_df['Age'].mean())
            
            # Create heatmap with annotations
            fig_heatmap = px.imshow(
                age_pivot_filled,
                aspect="auto",
                title="Age Distribution by Sport and Gender",
                color_continuous_scale="RdBu_r",  # Better for age contrasts
                labels=dict(x="Gender", y="Sport", color="Average Age"),
                text_auto=False,  # We'll use custom annotations
                zmin=min_color,   # Set color scale limits
                zmax=max_color,
            )
            
            # Add annotations
            for annotation in annotations:
                fig_heatmap.add_annotation(annotation)
            
            # Add gridlines for better readability
            fig_heatmap.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            fig_heatmap.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            
            # Update layout
            fig_heatmap.update_layout(
                title={
                    'text': f"Age Distribution by Sport and Gender<br><sup>Overall Average: {overall_avg_age:.1f} years (★ indicates significant deviation)</sup>",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                coloraxis_colorbar=dict(
                    title="Average Age",
                    ticksuffix=" yrs",
                    lenmode="fraction",
                    len=0.75,
                ),
                margin=dict(t=80, b=50)
            )
            
            # Add reference annotation for overall average
            fig_heatmap.add_annotation(
                xref="paper",
                yref="paper",
                x=1.02,
                y=0.5,
                text=f"<b>Age Legend</b><br>Avg: {overall_avg_age:.1f} yrs<br>Min: {filtered_df['Age'].min()} yrs<br>Max: {filtered_df['Age'].max()} yrs",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10)
            )
            
            # Display the enhanced heatmap
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Add helpful explanation
            st.caption(
                "**Reading the heatmap:** Each cell shows the average age, median age, and sample size (n). "
                "★ indicates a statistically significant deviation from the overall average. "
                "Colors represent average ages - darker blue for younger, darker red for older participants. "
                "Gray cells indicate no data available for that combination."
            )
            
            # Add age range summary table for reference
            with st.expander("View Age Ranges by Sport & Gender"):
                # Create a table showing age ranges
                range_table = []
                for sport in age_pivot.index:
                    for gender in age_pivot.columns:
                        key = (sport, gender)
                        if key in range_data:
                            subset = age_heatmap_data[(age_heatmap_data['Sport'] == sport) & (age_heatmap_data['Gender'] == gender)]
                            range_table.append({
                                'Sport': sport,
                                'Gender': gender,
                                'Age Range': range_data[key],
                                'Std Dev': subset['std_age'].iloc[0].round(2),
                                'Sample Size': int(subset['count'].iloc[0])
                            })
                
                if range_table:
                    st.table(pd.DataFrame(range_table))
        else:
            st.warning("Not enough data to create the age distribution heatmap.")
    
    with age_dist_tab2:
        # Create a violin plot to show full age distribution by sport
        fig_violin = px.violin(
            filtered_df,
            x='Sport',
            y='Age',
            color='Gender',
            box=True,  # Add box plot inside violin
            points='all',  # Show all data points
            title="Age Distribution by Sport and Gender",
            labels=dict(Sport="Sport", Age="Age (years)", Gender="Gender"),
            color_discrete_sequence=['#636EFA', '#EF553B']
        )
        
        # Add mean age reference line
        overall_avg_age = filtered_df['Age'].mean()
        fig_violin.add_hline(
            y=overall_avg_age,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Overall Average: {overall_avg_age:.1f}",
            annotation_position="top left"
        )
        
        fig_violin.update_layout(
            title_x=0.5,
            violinmode='group',
            violingap=0.1,
            xaxis_title="Sport",
            yaxis_title="Age (years)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add age range annotation
        fig_violin.add_annotation(
            x=0.5, 
            y=-0.15,
            xref="paper", 
            yref="paper",
            text=f"Age Range: {filtered_df['Age'].min()} - {filtered_df['Age'].max()} years",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        st.plotly_chart(fig_violin, use_container_width=True)
        
        # Add statistical summary table
        st.subheader("Age Statistics by Sport")
        
        # Create detailed statistics table
        age_stats = filtered_df.groupby('Sport')['Age'].agg([
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Std Dev', 'std'),
            ('Count', 'count')
        ]).reset_index()
        
        # Format numerical columns
        for col in ['Mean', 'Median', 'Std Dev']:
            age_stats[col] = age_stats[col].round(2)
        
        # Display table
        st.dataframe(
            age_stats,
            column_config={
                "Mean": st.column_config.NumberColumn("Mean Age", format="%.1f"),
                "Median": st.column_config.NumberColumn("Median Age", format="%.1f"),
                "Std Dev": st.column_config.NumberColumn("Std Dev", format="%.2f"),
                "Count": st.column_config.NumberColumn("Sample Size", format="%d")
            },
            hide_index=True,
            use_container_width=True
        )

with tabs[5]:
    # Feedback Analysis tab content
    st.subheader("Feedback Analysis")
    
    # Add sentiment scores to the dataframe
    filtered_df['Sentiment'] = filtered_df['Feedback'].apply(analyze_sentiment)
    
    # Display average sentiment by sport
    sentiment_by_sport = filtered_df.groupby('Sport')['Sentiment'].mean().sort_values(ascending=False).reset_index()
    
    # NEW: Add sentiment analysis by state
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_by_sport_chart = alt.Chart(sentiment_by_sport).mark_bar().encode(
            x='Sport',
            y='Sentiment',
            color=alt.condition(
                alt.datum.Sentiment > 0,
                alt.value('green'),
                alt.value('red')
            ),
            tooltip=['Sport', 'Sentiment']
        ).properties(
            height=300,
            title="Average Sentiment by Sport"
        ).interactive()
        
        st.altair_chart(sentiment_by_sport_chart, use_container_width=True)
    
    with col2:
        sentiment_by_state = filtered_df.groupby('State')['Sentiment'].mean().sort_values(ascending=False).reset_index()
        
        sentiment_by_state_chart = alt.Chart(sentiment_by_state).mark_bar().encode(
            x='State',
            y='Sentiment',
            color=alt.condition(
                alt.datum.Sentiment > 0,
                alt.value('green'),
                alt.value('red')
            ),
            tooltip=['State', 'Sentiment']
        ).properties(
            height=300,
            title="Average Sentiment by State"
        ).interactive()
        
        st.altair_chart(sentiment_by_state_chart, use_container_width=True)
    
    # Word cloud from feedback
    st.subheader("Feedback Word Cloud")
    
    all_feedback = " ".join(filtered_df['Feedback'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_feedback)
    
    # Display word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    # Show feedback samples
    st.subheader("Sample Feedback")
    
    # Group by sentiment category
    positive_feedback = filtered_df[filtered_df['Sentiment'] > 0.2].sample(min(5, len(filtered_df[filtered_df['Sentiment'] > 0.2])))
    neutral_feedback = filtered_df[(filtered_df['Sentiment'] >= -0.2) & (filtered_df['Sentiment'] <= 0.2)].sample(min(5, len(filtered_df[(filtered_df['Sentiment'] >= -0.2) & (filtered_df['Sentiment'] <= 0.2)])))
    negative_feedback = filtered_df[filtered_df['Sentiment'] < -0.2].sample(min(5, len(filtered_df[filtered_df['Sentiment'] < -0.2])))
    
    st.write("**Positive Feedback:**")
    for idx, row in positive_feedback.iterrows():
        st.write(f"- {row['Feedback']} (Sport: {row['Sport']}, Satisfaction: {row['Satisfaction']})")
    
    st.write("**Neutral Feedback:**")
    for idx, row in neutral_feedback.iterrows():
        st.write(f"- {row['Feedback']} (Sport: {row['Sport']}, Satisfaction: {row['Satisfaction']})")
    
    st.write("**Negative Feedback:**")
    for idx, row in negative_feedback.iterrows():
        st.write(f"- {row['Feedback']} (Sport: {row['Sport']}, Satisfaction: {row['Satisfaction']})")

with tabs[6]:
    # Sports Feedback Analysis tab content
    st.subheader("Sports-wise Feedback Analysis")
    
    # Add a sport selector for individual analysis
    all_sports = filtered_df['Sport'].unique()
    selected_sport_for_analysis = st.selectbox(
        "Select a Sport to Analyze Feedback", 
        options=all_sports
    )
    
    # Filter data for the selected sport
    sport_df = filtered_df[filtered_df['Sport'] == selected_sport_for_analysis]
    
    # Check if we have data
    if len(sport_df) == 0:
        st.warning(f"No data available for {selected_sport_for_analysis}")
    else:
        # Display basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Participants", len(sport_df))
        with col2:
            avg_sentiment = sport_df['Feedback'].apply(analyze_sentiment).mean()
            # Fix: Use delta instead of delta_color for sentiment indication
            sentiment_direction = "↑" if avg_sentiment > 0 else "↓"
            st.metric(
                "Average Sentiment", 
                f"{avg_sentiment:.2f}", 
                delta=sentiment_direction if avg_sentiment != 0 else None
            )
        with col3:
            positive_count = len(sport_df[sport_df['Feedback'].apply(analyze_sentiment) > 0])
            positive_pct = (positive_count / len(sport_df)) * 100
            st.metric("Positive Feedback %", f"{positive_pct:.1f}%")
        
        # Generate word cloud for the selected sport
        st.subheader(f"Word Cloud for {selected_sport_for_analysis}")
        
        # Join all feedback for this sport
        sport_feedback = " ".join(sport_df['Feedback'].tolist())
        
        # Define stopwords - renamed to avoid conflict with NLTK stopwords
        wordcloud_stopwords = set(STOPWORDS)
        custom_stops = {"sports", "tournament", "event", "college", "game", "games", "play", "played"}
        wordcloud_stopwords.update(custom_stops)
        
        # Create and display word cloud
        sport_wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            stopwords=wordcloud_stopwords,
            collocations=False
        ).generate(sport_feedback)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(sport_wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"Word Cloud for {selected_sport_for_analysis}")
        st.pyplot(fig)
        
        # Extract top words
        st.subheader(f"Most Common Words in {selected_sport_for_analysis} Feedback")
        top_words = get_top_words(sport_df['Feedback'].tolist(), n=15, custom_stopwords=custom_stops)
        
        # Create bar chart for top words
        top_words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
        
        fig_words = px.bar(
            top_words_df,
            x='Count',
            y='Word',
            orientation='h',
            title=f"Top 15 Words in {selected_sport_for_analysis} Feedback",
            color='Count',
            labels={'Count': 'Frequency', 'Word': 'Word'}
        )
        st.plotly_chart(fig_words, use_container_width=True)
        
        # Feedback sentiment distribution
        st.subheader("Feedback Sentiment Distribution")
        
        # Add sentiment scores if not already present
        if 'Sentiment' not in sport_df.columns:
            sport_df['Sentiment'] = sport_df['Feedback'].apply(analyze_sentiment)
        
        # Create histogram of sentiment scores
        fig_sentiment = px.histogram(
            sport_df,
            x='Sentiment',
            nbins=20,
            color_discrete_sequence=['#3366CC'],
            title=f"Sentiment Distribution for {selected_sport_for_analysis}",
            labels={'Sentiment': 'Sentiment Score (-1 to 1)', 'count': 'Number of Responses'}
        )
        
        fig_sentiment.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_sentiment.add_annotation(x=0.5, y=1, text="Positive", showarrow=False, yref="paper")
        fig_sentiment.add_annotation(x=-0.5, y=1, text="Negative", showarrow=False, yref="paper")
        
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Compare feedback between sports
    st.subheader("Compare Feedback Between Sports")
    
    # Select sports to compare
    compare_sports = st.multiselect(
        "Select Sports to Compare",
        options=all_sports,
        default=all_sports[:3] if len(all_sports) >= 3 else all_sports
    )
    
    if len(compare_sports) > 1:
        # Create dataframe with sentiment by sport
        compare_data = []
        word_counts_by_sport = {}
        
        for sport in compare_sports:
            sport_feedback = filtered_df[filtered_df['Sport'] == sport]['Feedback'].tolist()
            
            if not sport_feedback:  # Skip if no feedback available
                continue
                
            # Get sentiment stats
            sentiments = [analyze_sentiment(text) for text in sport_feedback]
            avg_sentiment = sum(sentiments) / len(sentiments)
            positive_pct = sum(1 for s in sentiments if s > 0) / len(sentiments) * 100
            neutral_pct = sum(1 for s in sentiments if -0.1 <= s <= 0.1) / len(sentiments) * 100
            negative_pct = sum(1 for s in sentiments if s < 0) / len(sentiments) * 100
            
            compare_data.append({
                'Sport': sport,
                'Average Sentiment': avg_sentiment,
                'Positive %': positive_pct,
                'Neutral %': neutral_pct,
                'Negative %': negative_pct,
                'Sample Size': len(sport_feedback)
            })
            
            # Get word frequency for this sport
            word_counts_by_sport[sport] = dict(get_top_words(sport_feedback, n=30, custom_stopwords=custom_stops))
        
        # Create comparison dataframe
        compare_df = pd.DataFrame(compare_data)
        
        # Display comparison metrics
        st.subheader("Sentiment Comparison")
        
        # Create a comparison bar chart
        fig_compare = px.bar(
            compare_df,
            x='Sport',
            y=['Positive %', 'Neutral %', 'Negative %'],
            title="Sentiment Breakdown by Sport",
            barmode='stack',
            labels={'value': 'Percentage', 'variable': 'Sentiment Type'}
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Create radar chart for sentiment comparison
        fig_radar = go.Figure()
        
        for sport in compare_df['Sport']:
            sport_data = compare_df[compare_df['Sport'] == sport].iloc[0]
            fig_radar.add_trace(go.Scatterpolar(
                r=[sport_data['Average Sentiment']*5 + 5, sport_data['Positive %']/10, sport_data['Negative %']/10],
                theta=['Sentiment Score', 'Positive %', 'Negative %'],
                fill='toself',
                name=sport
            ))
            
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )
            ),
            title="Sentiment Metrics Comparison"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Create word cloud grid for comparison
        st.subheader("Word Cloud Comparison")
        
        # Set number of columns for the grid
        num_sports = len(compare_sports)
        cols_per_row = min(3, num_sports)  # Max 3 columns per row
        
        # Create rows as needed
        rows = []
        for i in range(0, num_sports, cols_per_row):
            row = st.columns(cols_per_row)
            rows.append(row)
        
        # Generate word cloud for each sport
        for idx, sport in enumerate(compare_sports):
            row_idx = idx // cols_per_row
            col_idx = idx % cols_per_row
            
            with rows[row_idx][col_idx]:
                st.write(f"**{sport}**")
                sport_feedback = " ".join(filtered_df[filtered_df['Sport'] == sport]['Feedback'].tolist())
                
                if sport_feedback:
                    sport_wordcloud = WordCloud(
                        width=300, 
                        height=200, 
                        background_color='white',
                        max_words=50,
                        stopwords=wordcloud_stopwords,  # Use renamed variable
                        collocations=False
                    ).generate(sport_feedback)
                    
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.imshow(sport_wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.write("No feedback available")
        
        # Compare common words across sports
        st.subheader("Top Words Comparison")
        
        # Get unique words across all selected sports
        all_words = set()
        for word_dict in word_counts_by_sport.values():
            all_words.update(word_dict.keys())
        
        # Create a dataframe with word frequencies by sport
        word_freq_data = []
        for word in all_words:
            row = {'Word': word}
            for sport in word_counts_by_sport:
                row[sport] = word_counts_by_sport[sport].get(word, 0)
            word_freq_data.append(row)
        
        word_freq_df = pd.DataFrame(word_freq_data)
        
        # Filter to most common words across all sports
        total_freq = word_freq_df.drop('Word', axis=1).sum(axis=1)
        top_common_words = word_freq_df.loc[total_freq.nlargest(15).index]
        
        # Melt dataframe for visualization
        plot_data = pd.melt(
            top_common_words, 
            id_vars=['Word'], 
            value_vars=[s for s in compare_sports if s in word_counts_by_sport],
            var_name='Sport', 
            value_name='Frequency'
        )
        
        # Create grouped bar chart
        fig_words_compare = px.bar(
            plot_data,
            x='Word',
            y='Frequency',
            color='Sport',
            title="Common Words Across Sports",
            barmode='group'
        )
        
        st.plotly_chart(fig_words_compare, use_container_width=True)
    
    else:
        st.info("Please select at least two sports to compare.")

with tabs[7]:
    # Event Gallery tab content
    st.subheader("Tournament Event Gallery")
    
    # Gallery options
    st.markdown("""
    This gallery showcases images from tournament events organized by date.
    You can upload your own images or view placeholder images for each sport.
    """)
    
    # Add tabs for Gallery and Upload functions
    gallery_tab, upload_tab = st.tabs(["View Gallery", "Upload Images"])
    
    with gallery_tab:
        # Image processing controls
        with st.expander("Gallery Display Options"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gallery_filter = st.selectbox(
                    "Image Filter",
                    ["none", "grayscale", "sepia", "edge_enhance", "sharpen"],
                    index=0
                )
            
            with col2:
                brightness_factor = st.slider(
                    "Brightness",
                    min_value=0.5,
                    max_value=1.5,
                    value=1.0,
                    step=0.1
                )
            
            with col3:
                contrast_factor = st.slider(
                    "Contrast",
                    min_value=0.5,
                    max_value=1.5,
                    value=1.0,
                    step=0.1
                )
                
            col4, col5 = st.columns(2)
            
            with col4:
                saturation_factor = st.slider(
                    "Saturation",
                    min_value=0.5,
                    max_value=1.5,
                    value=1.0,
                    step=0.1
                )
            
            with col5:
                show_sport_labels = st.checkbox("Show Sport Labels", True)
                label_position = st.radio(
                    "Label Position", 
                    ["bottom", "top", "center"],
                    horizontal=True
                )
        
        # Create gallery filters dict
        gallery_filters = {
            "brightness": brightness_factor,
            "contrast": contrast_factor,
            "saturation": saturation_factor,
            "filter": gallery_filter,
            "sport_overlay": show_sport_labels,
            "overlay_position": label_position
        }
        
        # Display the image gallery based on filtered data
        create_image_gallery(
            filtered_df, 
            date_column='Date', 
            sport_column='Sport', 
            image_dict=st.session_state.get('uploaded_images', {}),
            filters=gallery_filters
        )
    
    with upload_tab:
        # Image upload interface
        create_image_uploader_section()

# Footer
st.markdown("---")
st.markdown("© 2023 Inter-College Tournament Analysis App | Created with Streamlit")
