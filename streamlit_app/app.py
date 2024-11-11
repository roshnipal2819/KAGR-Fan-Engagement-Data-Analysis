import streamlit as st
import pandas as pd
import plotly.express as px
import openai
import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Fan Engagement Analysis", layout="wide")

# Set up OpenAI API key (Make sure to set your API key in your environment)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
else:
    openai.api_key = OPENAI_API_KEY

# Load the processed data for visualization
@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned_fan_data.csv')

@st.cache_data
def load_external_data():
    attendance_df = pd.read_csv('data/external_year_wise_attendance.csv')
    rank_df = pd.read_csv('data/Rank.csv')
    engagement_df = pd.read_csv('data/external_fan_engagement.csv')
    return attendance_df, rank_df, engagement_df

def load_hotels_data():
        with open('data/hotel.json') as hotels_file:
            return json.load(hotels_file)
def load_restaurants_data():
        with open('data/restaurants.json') as restaurants_file:
            return json.load(restaurants_file)

# Load the cleaned fan data
data = load_data()
attendance_data, rank_data, engagement_data = load_external_data()
hotels_data = load_hotels_data()
restaurants_data = load_restaurants_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["Fan Engagement Analysis", "Market Analysis","Logistical Considerations","LLM Chatbot Q&A"])

# Scenario Settings - Add More Filters
st.sidebar.header("Scenario Settings")
discount_percentage = st.sidebar.slider("Ticket Discount Percentage (%)", min_value=0, max_value=50, value=10)
distance_filter = st.sidebar.slider("Maximum Distance to Arena (miles)", min_value=0, max_value=100, value=50)
income_level_filter = st.sidebar.selectbox("Filter by Income Level", options=['All', 'Low', 'Medium', 'High'], index=0)
fan_type_filter = st.sidebar.multiselect("Select Fan Type", options=data['Fan_Type'].unique(), default=data['Fan_Type'].unique())
age_filter = st.sidebar.slider("Filter by Age Range", min_value=int(data['Age'].min()), max_value=int(data['Age'].max()), value=(18, 65))
attendance_trend_filter = st.sidebar.selectbox("Filter by Attendance Trend", options=['All', 'Increasing', 'Stable', 'Decreasing'], index=0)

# Apply scenario logic to the data
filtered_data = data[
    (data['Distance_to_Arena_Miles'] <= distance_filter) & 
    (data['Fan_Type'].isin(fan_type_filter)) & 
    (data['Age'] >= age_filter[0]) & 
    (data['Age'] <= age_filter[1])
]
if income_level_filter != 'All':
    filtered_data = filtered_data[filtered_data['Income_Level'] == {'Low': 1, 'Medium': 2, 'High': 3}[income_level_filter]]
if attendance_trend_filter != 'All':
    filtered_data = filtered_data[filtered_data['Attendance_Trend'] == attendance_trend_filter]
distance_filtered_data = filtered_data[filtered_data['Distance_to_Arena_Miles'] <= distance_filter]

filtered_data['Original_Ticket_Units'] = filtered_data['Lifetime_Ticket_Units']
filtered_data['Discounted_Ticket_Units'] = filtered_data['Lifetime_Ticket_Units'] * (1 + discount_percentage / 100)
plot_data = filtered_data.groupby('Fan_Type').agg({
    'Original_Ticket_Units': 'sum',
    'Discounted_Ticket_Units': 'sum'}).reset_index()
plot_data['Percentage_Increase'] = (plot_data['Discounted_Ticket_Units'] - plot_data['Original_Ticket_Units']) / plot_data['Original_Ticket_Units'] * 100

# Page 1: Charts
if page_selection == "Fan Engagement Analysis":
    st.title("Fan Engagement Analysis")
    st.markdown("### Fan Demographics and Engagement Overview")
    fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],subplot_titles=['Fan Type Distribution', 'STM Holder Status', 'Attendance Trend'])
    fan_type_counts = data['Fan_Type'].value_counts()
    fig.add_trace(go.Pie(labels=fan_type_counts.index, values=fan_type_counts.values, name="Fan Type",textinfo='percent+label', hole=.3), 1, 1)
    stm_counts = data['STM_Holder'].value_counts()
    fig.add_trace(go.Pie(labels=stm_counts.index, values=stm_counts.values, name="STM Holder",textinfo='percent+label', hole=.3), 1, 2)
    attendance_counts = data['Attendance_Trend'].value_counts()
    fig.add_trace(go.Pie(labels=attendance_counts.index, values=attendance_counts.values, name="Attendance Trend",textinfo='percent+label', hole=.3), 1, 3)
    fig.update_layout(height=500, width=1000)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Key Insights:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Dominant Fan Type:** {fan_type_counts.index[0]} ({fan_type_counts.values[0]/len(data)*100:.1f}%)")
        st.write(f"**Least Common Fan Type:** {fan_type_counts.index[-1]} ({fan_type_counts.values[-1]/len(data)*100:.1f}%)")
    with col2:
        st.write(f"**STM Holders:** {stm_counts['Yes']/len(data)*100:.1f}%")
        st.write(f"**Non-STM Holders:** {stm_counts['No']/len(data)*100:.1f}%")
    with col3:
        st.write(f"**Most Common Trend:** {attendance_counts.index[0]} ({attendance_counts.values[0]/len(data)*100:.1f}%)")
        st.write(f"**Least Common Trend:** {attendance_counts.index[-1]} ({attendance_counts.values[-1]/len(data)*100:.1f}%)")
    avg_engagement = data['Total_Engagement_Score'].mean()
    st.markdown(f"**Average Engagement Score:** {avg_engagement:.2f}")
    stm_engagement = data.groupby('STM_Holder')['Total_Engagement_Score'].mean()
    st.markdown(f"**STM Holder Avg. Engagement:** {stm_engagement['Yes']:.2f}")
    st.markdown(f"**Non-STM Holder Avg. Engagement:** {stm_engagement['No']:.2f}")

    # 2. Gender vs. Fan Type
    st.markdown("### Gender Distribution by Fan Type")
    fan_gender_counts = data.groupby(['Fan_Type', 'Gender']).size().reset_index(name='Count')
    fig2 = px.bar(
    fan_gender_counts, 
    x='Fan_Type',
    y='Count',
    color='Gender',
    labels={'Count': 'Number of Fans', 'Fan_Type': 'Fan Type', 'Gender': 'Gender'},
    color_discrete_map={'Male': '#d35400', 'Female': '#1a5276', 'Non-binary': '#F08080'},
    barmode='group',
    text='Count')
    fig2.update_layout(
    xaxis_title="Fan Type",
    yaxis_title="Number of Fans",
    legend_title="Gender",
    height=500)
    fig2.update_traces(hovertemplate='%{y}')
    st.plotly_chart(fig2, use_container_width=True)

    # 4. Age Distribution
    st.markdown("### Age Distribution of Fans")
    fig4 = px.histogram(
    data,
    x='Age',
    nbins=20,
    marginal='box',  
    hover_data=data.columns,  
    labels={'Age': 'Age (years)', 'count': 'Number of Fans'},
    color_discrete_sequence=['#3366cc'])
    fig4.update_layout(
    xaxis_title_font=dict(size=14),
    yaxis_title_font=dict(size=14),
    bargap=0.1)
    mean_age = data['Age'].mean()
    fig4.add_vline(x=mean_age, line_dash="dash", line_color="red", annotation_text=f"Mean Age: {mean_age:.1f}", annotation_position="top right")
    st.plotly_chart(fig4, use_container_width=True)

    #5. Lifetime Games Attended
    st.markdown("### Lifetime Games Attended Distribution")
    mean_games = data['Lifetime_Games_Attended'].mean()
    median_games = data['Lifetime_Games_Attended'].median()
    fig5 = px.histogram(
    data,
    x='Lifetime_Games_Attended',
    nbins=30,
    labels={'Lifetime_Games_Attended': 'Number of Games Attended'},
    color_discrete_sequence=['#1f77b4'])
    fig5.update_layout(
    xaxis_title="Number of Games Attended",
    yaxis_title="Count of Fans",
    bargap=0.1,  # Add some gap between bars
    showlegend=False)
    fig5.add_vline(x=mean_games, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean_games:.1f}")
    fig5.add_vline(x=median_games, line_dash="dash", line_color="green", annotation_text=f"Median: {median_games:.1f}")
    fig5.add_annotation(
    text=f"Mean: {mean_games:.1f}<br>Median: {median_games:.1f}",
    xref="paper", yref="paper",
    x=0.90, y=0.90,
    showarrow=False,
    bgcolor="black",
    bordercolor="white",
    borderwidth=1)
    st.plotly_chart(fig5, use_container_width=True)
    st.write(f"""
This histogram shows the distribution of lifetime games attended by fans. 
Key observations:
- The average (mean) number of games attended is {mean_games:.1f}.
- The median number of games attended is {median_games:.1f}.
- The distribution appears to be right-skewed, indicating that while most fans attend a moderate number of games, 
  there are some highly dedicated fans who have attended a large number of games over their lifetime.
""")

    # 7. Distance to Arena vs. Engagement Score
    st.markdown("### How Does Distance Affect Fan Engagement?")
    data['Distance_Category'] = pd.cut(data['Distance_to_Arena_Miles'], bins=[0, 5, 10, 20, 50, 100, float('inf')],labels=['0-5', '6-10', '11-20', '21-50', '51-100', '100+'])
    fig7 = px.box(data, x='Distance_Category', y='Total_Engagement_Score', 
              title="Fan Engagement Score by Distance to Arena",
              labels={'Distance_Category': 'Distance to Arena (miles)', 
                      'Total_Engagement_Score': 'Engagement Score'},
              color='Distance_Category',
              color_discrete_sequence=px.colors.qualitative.Set3)
    fig7.update_layout(
    xaxis_title="Distance to Arena (miles)",
    yaxis_title="Engagement Score",
    showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown("""This box plot shows how fan engagement varies based on the distance from the arena:
- The boxes represent the middle 50% of engagement scores for each distance category.
- The line in each box is the median engagement score.
- The whiskers extend to the minimum and maximum scores (excluding outliers).
- Any points beyond the whiskers are considered outliers.
                
**Key Observations:**
1. Fans living closer to the arena (0-5 miles) tend to have higher engagement scores.
2. There's a general trend of decreasing engagement as distance increases.
3. However, there's significant variation within each distance category, suggesting other factors also influence engagement.
""")

    # 8. Age vs. Total Engagement Score
    st.markdown("### Age vs. Total Engagement Score")
    fig8 = px.scatter(data, x='Age', y='Total_Engagement_Score')
    st.plotly_chart(fig8, use_container_width=True)

    # 9. Distance to Arena vs. Lifetime Ticket Spend
    st.markdown("### Average Lifetime Ticket Spend by Distance to Arena")
    distance_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    distance_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    distance_filtered_data['Distance_Category'] = pd.cut(distance_filtered_data['Distance_to_Arena_Miles'], bins=distance_bins, labels=distance_labels, include_lowest=True)    
    avg_spend_by_distance = distance_filtered_data.groupby('Distance_Category')['Lifetime_Ticket_Spend'].mean().reset_index()
    fig_line = px.line(
    avg_spend_by_distance,
    x='Distance_Category',
    y='Lifetime_Ticket_Spend',
    markers=True,
    labels={
        "Distance_Category": "Distance to Arena (miles)",
        "Lifetime_Ticket_Spend": "Average Lifetime Ticket Spend ($)"
    },
    title=f"Average Lifetime Ticket Spend by Distance to Arena (Up to {distance_filter} miles)")
    st.plotly_chart(fig_line, use_container_width=True)

    # Chart 1: Impact of Ticket Discount on Attendance by Fan Type
    st.markdown("### Impact of Ticket Discount on Attendance by Fan Type")
    fig_discount = go.Figure()
    fig_discount.add_trace(go.Bar(
    x=plot_data['Fan_Type'],
    y=plot_data['Original_Ticket_Units'],
    name='Original Ticket Units',
    marker_color='royalblue'))
    fig_discount.add_trace(go.Bar(
    x=plot_data['Fan_Type'],
    y=plot_data['Discounted_Ticket_Units'],
    name=f'Discounted Ticket Units ({discount_percentage}% Off)',
    marker_color='lightgreen',
    text=plot_data['Percentage_Increase'].apply(lambda x: f'+{x:.1f}%'),
    textposition='outside'))
    fig_discount.update_layout(
    title=f"Impact of {discount_percentage}% Ticket Discount on Attendance by Fan Type",
    xaxis_title="Fan Type",
    yaxis_title="Total Ticket Units",
    barmode='group',
    legend_title="Ticket Type",
    height=500)
    st.plotly_chart(fig_discount, use_container_width=True)

    #6. Season Ticket Member (STM) Impact
    st.markdown("### Season Ticket Member (STM) Impact")
    filtered_data_stm = filtered_data[filtered_data['Fan_Type'].isin(fan_type_filter)]
    stm_impact = filtered_data_stm.groupby('STM_Holder')[['Lifetime_Ticket_Spend', 'Lifetime_Concessions_Spend', 'Total_Engagement_Score']].mean().reset_index()
    stm_impact_melted = pd.melt(stm_impact, id_vars=['STM_Holder'], value_vars=['Lifetime_Ticket_Spend', 'Lifetime_Concessions_Spend', 'Total_Engagement_Score'],var_name='Metric', value_name='Value')
    fig6 = px.bar(stm_impact_melted, x='STM_Holder', y='Value', color='Metric', barmode='group',
              title="STM Holder Impact on Spend and Engagement",
              labels={'STM_Holder': 'Season Ticket Member', 'Value': 'Average Value'},
              color_discrete_map={'Lifetime_Ticket_Spend': '#1f77b4', 
                                  'Lifetime_Concessions_Spend': '#ff7f0e',
                                  'Total_Engagement_Score': '#2ca02c'})
    fig6.update_layout(
    xaxis_title="Season Ticket Member Status",
    yaxis_title="Average Value",
    legend_title="Metrics",
    font=dict(size=12),
    bargap=0.2,
    bargroupgap=0.1)
    fig6.update_yaxes(tickprefix="$", showgrid=True)
    fig6.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    st.plotly_chart(fig6, use_container_width=True)

    # 10. Lifetime Ticket Units vs. Total Engagement Score
    st.markdown("### Lifetime Ticket Units vs. Total Engagement Score")
    fig10 = px.scatter(data, x='Lifetime_Ticket_Units', y='Total_Engagement_Score')
    st.plotly_chart(fig10, use_container_width=True)

    # # Chart 4: Engagement Score Distribution by Fan Type
    st.markdown("### Engagement Score Distribution by Fan Type")
    fig_engagement_score = px.box(
        filtered_data, x='Fan_Type', y='Total_Engagement_Score', color='Income_Level',
        labels={"Total_Engagement_Score": "Total Engagement Score", "Fan_Type": "Fan Type"})
    st.plotly_chart(fig_engagement_score, use_container_width=True)
    st.write(f"**Summary**: Engagement scores vary significantly among the {', '.join(fan_type_filter)} fan types, with {income_level_filter if income_level_filter != 'All' else 'all'} income levels contributing to the observed distribution.")

     # 11. Correlation Matrix
    st.markdown("### Correlation Matrix")
    corr_cols = ['Age', 'Distance_to_Arena_Miles', 'Lifetime_Ticket_Spend', 'Lifetime_Concessions_Spend', 'Total_Engagement_Score']
    corr_matrix = data[corr_cols].corr()
    fig4 = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig4, use_container_width=True)

    # 12. Engagement Score by Demographics
    st.markdown("### Engagement Score by Demographics")
    fig3 = px.box(data, x='Age', y='Total_Engagement_Score', color='Gender', facet_col='Income_Level')
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 5: Concessions Spend by Distance to Arena
    st.markdown("### Concessions Spend by Distance to Arena")
    fig_distance_concessions = px.scatter(
    distance_filtered_data, 
    x='Distance_to_Arena_Miles', 
    y='Lifetime_Concessions_Spend', 
    color='Fan_Type',
    size='Lifetime_Games_Attended',  # Add size variation based on games attended
    hover_data=['Age', 'Income_Level'],  # Add more info to hover
    labels={
        "Distance_to_Arena_Miles": "Distance to Arena (miles)", 
        "Lifetime_Concessions_Spend": "Lifetime Concessions Spend ($)",
        "Lifetime_Games_Attended": "Games Attended"
    },
    title=f"Concessions Spend by Distance to Arena (Up to {distance_filter} miles)")
    fig_distance_concessions.update_layout(
    xaxis_title="Distance to Arena (miles)",
    yaxis_title="Lifetime Concessions Spend ($)",
    legend_title="Fan Type",
    height=600,)
    fig_distance_concessions.add_traces(px.scatter(
    distance_filtered_data, 
    x='Distance_to_Arena_Miles', 
    y='Lifetime_Concessions_Spend', 
    trendline="lowess").data)
    st.plotly_chart(fig_distance_concessions, use_container_width=True)
    near_fans = distance_filtered_data[distance_filtered_data['Distance_to_Arena_Miles'] <= distance_filter/2]
    far_fans = distance_filtered_data[distance_filtered_data['Distance_to_Arena_Miles'] > distance_filter/2]
    near_avg_spend = near_fans['Lifetime_Concessions_Spend'].mean()
    far_avg_spend = far_fans['Lifetime_Concessions_Spend'].mean()
    st.write(f"""
**Summary**: 
- Concessions spending tends to decrease as fans live further from the arena. 
- Fans within {distance_filter} miles of the arena have been included in this analysis.
- The average lifetime concessions spend for fans living within {distance_filter/2} miles is ${near_avg_spend:.2f}.
- For fans living between {distance_filter/2} and {distance_filter} miles, the average spend is ${far_avg_spend:.2f}.
- {', '.join(fan_type_filter)} fan types are represented in different colors.
- The size of each point represents the number of games attended, showing how frequency of attendance relates to spending.
- The trend line shows the overall relationship between distance and spending.""")

    # 8. Spend Analysis
    st.markdown("### Spend Analysis by Fan Type")
    spend_analysis = data.groupby('Fan_Type')[['Lifetime_Concessions_Spend', 'Lifetime_Retail_Spend']].mean().reset_index()
    fig8 = px.bar(spend_analysis, x='Fan_Type', y=['Lifetime_Concessions_Spend', 'Lifetime_Retail_Spend'],title="Average Lifetime Spend by Fan Type")
    st.plotly_chart(fig8, use_container_width=True)
    
    # 5. Lifetime Value Analysis
    st.markdown("### Lifetime Value Analysis")
    fig5 = px.scatter(data, x='Lifetime_Games_Attended', y='Total_Engagement_Score', color='Fan_Type', title="Lifetime Games Attended vs Total Engagement Score")
    st.plotly_chart(fig5, use_container_width=True)

# Page 2: Market Analysis Dashboard
elif page_selection == "Market Analysis":
    st.title("Market Analysis")

    # Summary Metrics Section
    st.markdown("### All-Time Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_fans = len(data)
        st.metric(label="Total Fans", value=f"{total_fans:,}")
    with col2:
        total_ticket_units = data['Lifetime_Ticket_Units'].sum()
        st.metric(label="Total Ticket Units", value=f"{total_ticket_units:,}")
    with col3:
        total_concessions_spend = data['Lifetime_Concessions_Spend'].sum()
        st.metric(label="Total Concessions Spend ($)", value=f"${total_concessions_spend:,.2f}")
    with col4:
        avg_engagement_score = data['Total_Engagement_Score'].mean()
        st.metric(label="Average Engagement Score", value=f"{avg_engagement_score:.2f}")

    # Selected Duration Metrics
    st.markdown("### Selected Duration Metrics")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        selected_total_ticket_units = filtered_data['Discounted_Ticket_Units'].sum()
        st.metric(label="Total Ticket Units", value=f"{selected_total_ticket_units:,.2f}")
    with col6:
        selected_concessions_spend = filtered_data['Lifetime_Concessions_Spend'].sum()
        st.metric(label="Total Concessions Spend ($)", value=f"${selected_concessions_spend:,.2f}")
    with col7:
        selected_avg_engagement_score = filtered_data['Total_Engagement_Score'].mean()
        st.metric(label="Average Engagement Score", value=f"{selected_avg_engagement_score:.2f}")
    with col8:
        selected_total_games_attended = filtered_data['Lifetime_Games_Attended'].sum()
        st.metric(label="Total Games Attended", value=f"{selected_total_games_attended:,}")

    # Top 5 Large Cities
    st.markdown("### Top 5 Large Cities by Score")
    top_5_large = rank_data.nlargest(5, 'Score_Large_City')
    fig15 = px.bar(top_5_large, x='Large_City', y='Score_Large_City', color='Score_Large_City', title="Top 5 Large Cities by Score")
    st.plotly_chart(fig15, use_container_width=True)

    # Top 5 States by Large City Score
    st.markdown("### Top 5 States by Large City Score")
    top_5_states = rank_data.groupby('Large_State')['Score_Large_City'].mean().nlargest(5).reset_index()
    fig21 = px.bar(top_5_states, x='Large_State', y='Score_Large_City')
    summary_text = "Massachusetts (MA) has the highest average score among large cities, significantly outperforming the other top states."
    st.markdown(f"**Summary:** {summary_text}")
    st.plotly_chart(fig21, use_container_width=True)

    # # 4. City Size Comparison
    st.subheader("City Size Comparison")
    city_size_data = pd.melt(rank_data, 
                             value_vars=['Score_Large_City', 'Score_Midsize_City', 'Score_Small_City'],
                             var_name='City_Size', value_name='Score')
    city_size_data['City_Size'] = city_size_data['City_Size'].map({
        'Score_Large_City': 'Large', 'Score_Midsize_City': 'Midsize', 'Score_Small_City': 'Small' })
    fig4 = px.box(city_size_data, x='City_Size', y='Score', title="Score Distribution by City Size")
    st.plotly_chart(fig4, use_container_width=True)

    # 6. State-Level Analysis
    st.subheader("Top 10 States by Average Score")
    state_scores = pd.concat([
        rank_data[['Large_State', 'Score_Large_City']].rename(columns={'Large_State': 'State', 'Score_Large_City': 'Score'}),
        rank_data[['Midsize_State', 'Score_Midsize_City']].rename(columns={'Midsize_State': 'State', 'Score_Midsize_City': 'Score'}),
        rank_data[['Small_State', 'Score_Small_City']].rename(columns={'Small_State': 'State', 'Score_Small_City': 'Score'})])
    state_avg_scores = state_scores.groupby('State')['Score'].mean().sort_values(ascending=False).reset_index()
    fig6 = px.bar(state_avg_scores.head(10), x='State', y='Score')
    st.plotly_chart(fig6, use_container_width=True)

    # # 8. Potential Market Identification
    st.subheader("Potential Growth Markets")
    potential_markets = rank_data[
        (rank_data['Score_Large_City'] > rank_data['Score_Large_City'].mean()) |
        (rank_data['Score_Midsize_City'] > rank_data['Score_Midsize_City'].mean()) |
        (rank_data['Score_Small_City'] > rank_data['Score_Small_City'].mean())]
    st.write("Cities with above-average scores (potential growth markets):")
    st.dataframe(potential_markets)

    # Additional summary
    st.markdown("### Overall Summary of Scenario")
    st.write(f"With a {discount_percentage}% ticket discount, fans living within {distance_filter} miles are expected to increase their ticket purchases.")
    if income_level_filter != 'All':
        st.write(f"This projection is filtered for income level: {income_level_filter}.")
    st.write(f"Fan types considered: {', '.join(fan_type_filter)}. The analysis shows that engagement and spending behaviors vary significantly across the different fan segments, income levels, and proximity to the arena.")

elif page_selection == "Logistical Considerations":
    st.title("Logistical Considerations")
    all_data = []
    for city in hotels_data.keys():
        for hotel in hotels_data[city]:
            hotel['Type'] = 'Hotel'
            hotel['City'] = city
            all_data.append(hotel)
        for restaurant in restaurants_data[city]:
            restaurant['Type'] = 'Restaurant'
            restaurant['City'] = city
            all_data.append(restaurant)
    
    df = pd.DataFrame(all_data)
    selected_city = st.selectbox("Select a City", options=df['City'].unique())
    city_data = df[df['City'] == selected_city]
    st.subheader(f"Map of Hotels and Restaurants in {selected_city}")
    fig = px.scatter_mapbox(city_data, 
                            lat="latitude", 
                            lon="longitude", color="Type", color_discrete_map={"Hotel": "blue", "Restaurant": "red"},size_max=10,
                            size=[10] * len(city_data),hover_name="name",
                            zoom=12,
                            height=500)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot for Airport Distance vs Number of Daily Flights
    data = {
    "City": ["Boston, MA", "Dallas, TX", "Columbus, OH", "Tampa, FL", "San Antonio, TX"],
    "Arena": ["TD Garden", "American Airlines Center", "Value City Arena", "Amalie Arena", "AT&T Center"],
    "Capacity": [19580, 19200, 20000, 19500, 18500],
    "Distance from Airport (miles)": [3, 20, 10, 10, 12],
    "Number of Daily Flights": [350, 700, 100, 200, 100],
    "Public Transit": ["Excellent (subway)", "Good (DART)", "Good (COTA)", "Moderate (HART)", "Good (VIA)"]}
    df = pd.DataFrame(data)
    st.subheader("Airport Accessibility")
    fig_airport = px.scatter(df, x='Distance from Airport (miles)', y='Number of Daily Flights',
                         size='Capacity', color='City', hover_name='Arena',
                         labels={'Distance from Airport (miles)': 'Distance from Airport (miles)','Number of Daily Flights': 'Number of Daily Flights'})
    st.plotly_chart(fig_airport)

    # Public Transit Rating
    st.subheader("Public Transit Rating")
    transit_order = ["Excellent", "Good", "Moderate"]
    df['Transit Rating'] = df['Public Transit'].apply(lambda x: x.split()[0])
    df['Transit Score'] = df['Transit Rating'].map(dict(zip(transit_order, range(len(transit_order), 0, -1))))
    fig_transit = px.bar(df, x='City', y='Transit Score', color='Transit Rating',title="Public Transit Rating by City",
                     labels={'Transit Score': 'Transit Quality', 'City': 'City'},category_orders={"Transit Rating": transit_order})
    st.plotly_chart(fig_transit)

     #Arena Capacity Comparison
    st.markdown("### Arena Capacity Comparison")
    fig_capacity = px.bar(df,
                           x='Capacity',
                           y='City',
                           orientation='h',  
                           color='Capacity',
                           color_continuous_scale=px.colors.sequential.Viridis)
    fig_capacity.update_layout(xaxis_title="Seating Capacity",yaxis_title="City",showlegend=False)
    st.plotly_chart(fig_capacity)

   #Benefits for UMass Dartmouth Students Attending the Event at TD Garden
    st.markdown("### Benefits for UMass Dartmouth Students Attending the Event at TD Garden")
    st.markdown("#### 1. Proximity to the Venue: TD Garden is located approximately 60 miles from UMass Dartmouth, making it easily accessible.")
    st.markdown("#### 2. Transportation Options: Public Transit: Extensive MBTA system provides affordable travel options, Shuttle Services: Available between Logan Airport and downtown Boston. ")
    st.markdown("#### 3. Number of Daily Flights:Logan International Airport offers approximately 350+ daily flights.")
    st.markdown("#### 4. Fan Engagement Opportunities: Fosters school spirit and community among students, Opportunities for involvement in organizing activities.")
    st.markdown("#### 5. Economic Benefits: Increased attendance supports local businesses in Boston.")

    # Key Insights
    st.markdown("### Key Insights:")
    st.write("""
    - **TD Garden in Boston** has excellent airport accessibility and public transit options.
    - It features a large capacity suitable for hosting significant events.
    - The surrounding area offers numerous amenities and attractions that enhance fan experience.""")

# Page 4: LLM Chatbot Q&A
elif page_selection == "LLM Chatbot Q&A":
    st.title("LLM Chatbot Q&A")
    st.write("Ask questions about the fan engagement data, and get answers powered by GPT-4.")

    # Input for user question
    user_question = st.text_input("Type your question here:")

    if user_question:
        # Provide contextual information from the entire dataset
        context = (
            f"The fan engagement dataset contains information about {len(data)} fans. "
            f"The average engagement score is {data['Total_Engagement_Score'].mean():.2f}. "
            f"The total ticket spend is ${data['Lifetime_Ticket_Spend'].sum():,.2f}. "
            f"The external attendance dataset contains {len(attendance_data)} records spanning different seasons. "
            f"The ranking dataset provides city-level rankings across large, midsize, and small cities. "
            f"The fan engagement dataset across cities includes total scores, NBA ranks, and NCAA ranks."
        )

        # Create a DataFrame representation that can be included in the prompt
        data_repr = data.head(5).to_string()
        attendance_repr = attendance_data.head(5).to_string()
        rank_repr = rank_data.head(5).to_string()
        engagement_repr = engagement_data.head(5).to_string()

        # Include explicit instructions for the model to analyze the data
        analysis_instructions = (
            "Below are summaries and samples of four datasets that represent fan engagement, attendance trends, city rankings, and overall fan scores. "
            "Please analyze the provided data samples to identify patterns, correlations, or specific data points that answer the question accurately. "
            "Make use of the provided context and data trends, and reference specific metrics or data points where applicable. "
            "Avoid generic insights and focus on concrete data analysis."
        )

        # Full prompt with data samples included
        full_prompt = (
            f"{context}\n\n"
            f"Fan Engagement Data Sample:\n{data_repr}\n\n"
            f"Yearly Attendance Data Sample:\n{attendance_repr}\n\n"
            f"City Ranking Data Sample:\n{rank_repr}\n\n"
            f"Fan Engagement Scores by City Sample:\n{engagement_repr}\n\n"
            f"{analysis_instructions}\n\n"
            f"Question: {user_question}"
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=800,
                temperature=0.3)
            answer = response['choices'][0]['message']['content'].strip()
            st.write("**Answer:**")
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating response: {e}")
            
    # Mentor Insight Button
    if st.button("Get Mentor's Insight"):
        try:
            mentor_prompt = (
                f"Given the current fan engagement data with {discount_percentage}% ticket discount and fans filtered by {', '.join(fan_type_filter)}, "
                f"along with external data on yearly attendance trends, city rankings, and fan engagement scores, what recommendations would you provide?"
            )
            mentor_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": mentor_prompt}],
                max_tokens=150,
                temperature=0.3  # Reduced temperature for more consistent suggestions
            )
            st.write("**Mentor's Insight:**")
            st.write(mentor_response['choices'][0]['message']['content'].strip())
        except Exception as e:
            st.error(f"Error fetching mentor's insight: {e}")