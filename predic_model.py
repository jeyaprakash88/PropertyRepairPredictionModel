import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import streamlit_authenticator as stauth  # pip install streamlit-authenticator


st.set_page_config(page_title="Repair Dashboard", page_icon=":bar_chart:", layout="wide")

@st.cache_data
def get_data_from_excel():
    df = pd.read_excel(
        io='data.xlsx',
        engine='openpyxl',
        sheet_name='Sheet1',
        header=0,
        index_col=False,
        keep_default_na=True,
        nrows=1000,
    )
    return df

df = get_data_from_excel()

print(df)

st.sidebar.header("Please Filter Here:")
st.markdown('<link href="styles.css" rel="stylesheet">', unsafe_allow_html=True)

# The rest of your Streamlit app code...

building_type = st.sidebar.multiselect(
        "Select the Building type:",
        options=df["Building_type"].unique(),
        default=df["Building_type"].unique()
)

no_of_rooms = st.sidebar.multiselect(
        "No of bedroom:",
        options=df["No_of_bedroom"].unique(),
        default=df["No_of_bedroom"].unique()
)
age = st.sidebar.multiselect(
        "Property Age:",
        options=df["Age"].unique(),
        default=[35, 85, 43]
)

property_classification = st.sidebar.multiselect(
        "Priority Classificatoin type:",
        options=df["pty_classification"].unique(),
        default=df["pty_classification"].unique()
)

df_selection = df.query(
        "Building_type == @building_type & No_of_bedroom ==@no_of_rooms & Age == @age & pty_classification == @property_classification"
    )

st.dataframe(df_selection)

# --- MAINPAGE ---
st.title(":bar_chart : Repairs Dashboard")
st.markdown("##")

# TOP KPI's
total_count = df_selection["pty_classification"].value_counts()
build_type = df_selection["Building_type"].value_counts()
No_of_bed = df_selection["No_of_bedroom"].value_counts()

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.write("Priority classification")
    for category, count in total_count.items():
        st.write(f"{category}: {count:,}")
    
with middle_column:
    st.write("Property Type")
    for category, count in build_type.items():
        st.write(f"{category}: {count:,}")
with right_column:
    st.write("Number of Bedrooms in a property")
    for category, count in No_of_bed.items():
        st.write(f"No of rooms {category}: {count:,}")

st.markdown("""---""")





#plots
count_by_priority = df_selection["pty_classification"].value_counts().sort_values()

fig_by_priority = px.bar(
    x=count_by_priority.index,
    y=count_by_priority.values,
    orientation="v",
    title="<b>Count by priority repair</b>",
    color=count_by_priority.values,  # Use the count values for the color scale
    color_continuous_scale="viridis",  # Specify the color scale ("viridis" in this case)
    template="plotly_white",
)

fig_by_priority.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="",  # Set x-axis label to an empty string
    yaxis_title="",  # Set y-axis label to an empty string
    xaxis=dict(showgrid=False),
)

by_town = df_selection["postcode"].value_counts().sort_values()

fig_by_town = px.bar(
    x=by_town.index,
    y=by_town.values,
    orientation="v",
    title="<b>Count by Post code</b>",
    color=by_town.values,  # Use the count values for the color scale
    color_continuous_scale="viridis",  # Specify the color scale ("viridis" in this case)
    template="plotly_white",
)

fig_by_town.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="",  # Set x-axis label to an empty string
    yaxis_title="",  # Set y-axis label to an empty string
    xaxis=dict(showgrid=False),
)

left_column, right_column = st.columns(2)
# Plot for 'fig_by_priority'
fig_by_priority.update_layout(height=400, width=500)  # Set the height and width
left_column.plotly_chart(fig_by_priority, use_container_width=True)

# Plot for 'fig_by_town'
fig_by_town.update_layout(height=400, width=600)  # Set the height and width
right_column.plotly_chart(fig_by_town, use_container_width=True)

# Create scatter plot for pty_classification against postcode
fig_scatter = px.scatter(
    df_selection,
    x="pty_classification",
    y="postcode",
    title="<b>Priority classification vs postcode</b>",
    color_discrete_sequence=["#0083B8"],
    template="plotly_white",
)

fig_scatter.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Priority Classification",  # Set x-axis label to "Priority Classification"
    yaxis_title="Postcode",  # Set y-axis label to "Postcode"
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
)

# Create histogram for pr-seq-no
fig_histogram = px.histogram(
    df_selection,
    x="pr-seq-no",
    nbins=20,  # You can adjust the number of bins as needed
    title="<b>Histogram of pr-seq-no</b>",
    color_discrete_sequence=["#0083B8"],
    template="plotly_white",
)

fig_histogram.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="pr-seq-no",  # Set x-axis label to "pr-seq-no"
    yaxis_title="Count",  # Set y-axis label to "Count"
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
)

left_column, right_column = st.columns(2)
# Plot for 'fig_scatter'
fig_scatter.update_layout(height=400, width=500)  # Set the height and width
left_column.plotly_chart(fig_scatter, use_container_width=True)

# Plot for 'fig_histogram'
fig_histogram.update_layout(height=400, width=600)  # Set the height and width
right_column.plotly_chart(fig_histogram, use_container_width=True)

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)