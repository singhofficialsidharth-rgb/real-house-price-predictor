import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------

st.set_page_config(
    page_title="AI Property Valuation",
    page_icon="🏡",
    layout="wide"
)

# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------

model = joblib.load("xgboost_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

df = pd.read_csv("clean_property_data.csv")

# -----------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f5f7fb;
}

.title{
    text-align:center;
    color:#0066cc;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:gray;
    margin-bottom:30px;
}

.big-font{
    font-size:25px;
    font-weight:bold;
}

div.stButton > button{
    width:100%;
    background:#0066cc;
    color:white;
    height:55px;
    font-size:20px;
    border-radius:10px;
}

div.stButton > button:hover{
    background:#004999;
}

.metric-box{

    padding:25px;

    border-radius:15px;

    background:#ffffff;

    box-shadow:0px 0px 12px rgba(0,0,0,.15);

    text-align:center;
}

.footer{

    text-align:center;

    color:gray;

    margin-top:60px;

}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------

st.markdown(
    "<div class='title'>🏡 AI Powered Real Property Valuation Predictor</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict Property Prices using XGBoost Machine Learning Model</div>",
    unsafe_allow_html=True
)

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.title("🏠 Property Details")

st.sidebar.markdown("---")

# -----------------------------------------------------
# BASIC DETAILS
# -----------------------------------------------------

st.sidebar.subheader("Basic Information")

area = st.sidebar.number_input(
    "Area (Sqft)",
    min_value=200,
    max_value=10000,
    value=1200
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    1,
    10,
    3
)

bathrooms = st.sidebar.slider(
    "Bathrooms",
    1,
    10,
    2
)

parking = st.sidebar.selectbox(
    "Parking",
    ["Yes", "No"]
)

property_age = st.sidebar.slider(
    "Property Age",
    0,
    30,
    5
)

floor = st.sidebar.number_input(
    "Floor Number",
    0,
    80,
    5
)

total_floors = st.sidebar.number_input(
    "Total Floors",
    1,
    100,
    15
)

balcony = st.sidebar.slider(
    "Balconies",
    0,
    10,
    2
)

# -----------------------------------------------------
# LOCATION
# -----------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.subheader("Location")

cities = sorted(df["City"].unique())

city = st.sidebar.selectbox(
    "City",
    cities
)

filtered_localities = sorted(
    df[df["City"] == city]["Locality"].unique()
)

locality = st.sidebar.selectbox(
    "Locality",
    filtered_localities
)


furnishing = st.sidebar.selectbox(
    "Furnishing",
    sorted(df["Furnishing"].unique())
)

facing = st.sidebar.selectbox(
    "Facing",
    sorted(df["Facing"].unique())
)

# -----------------------------------------------------
# DISTANCE DETAILS
# -----------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("Nearby Facilities")

metro_distance = st.sidebar.number_input(
    "Metro Distance (km)",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1
)

hospital_distance = st.sidebar.number_input(
    "Hospital Distance (km)",
    min_value=0.0,
    max_value=20.0,
    value=1.5,
    step=0.1
)

school_distance = st.sidebar.number_input(
    "School Distance (km)",
    min_value=0.0,
    max_value=20.0,
    value=1.0,
    step=0.1
)

mall_distance = st.sidebar.number_input(
    "Mall Distance (km)",
    min_value=0.0,
    max_value=20.0,
    value=3.0,
    step=0.1
)

expressway_distance = st.sidebar.number_input(
    "Expressway Distance (km)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.1
)

rrts_distance = st.sidebar.number_input(
    "RRTS Distance (km)",
    min_value=0.0,
    max_value=50.0,
    value=4.0,
    step=0.1
)

# -----------------------------------------------------
# AMENITIES
# -----------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("Amenities")

gated_society = st.sidebar.selectbox(
    "Gated Society",
    ["Yes", "No"]
)

lift = st.sidebar.selectbox(
    "Lift",
    ["Yes", "No"]
)

power_backup = st.sidebar.selectbox(
    "Power Backup",
    ["Yes", "No"]
)

security = st.sidebar.selectbox(
    "Security",
    ["Yes", "No"]
)

gym = st.sidebar.selectbox(
    "Gym",
    ["Yes", "No"]
)

swimming_pool = st.sidebar.selectbox(
    "Swimming Pool",
    ["Yes", "No"]
)

# -----------------------------------------------------
# BUTTON
# -----------------------------------------------------

predict = st.sidebar.button("Predict Property Price")

# -----------------------------------------------------
# FEATURE VECTOR
# -----------------------------------------------------

input_df = pd.DataFrame(
    np.zeros((1, len(feature_columns))),
    columns=feature_columns
)

# -----------------------------------------------------
# NUMERICAL FEATURES
# -----------------------------------------------------

input_df.loc[0, "Area_sqft"] = area
input_df.loc[0, "Bedrooms"] = bedrooms
input_df.loc[0, "Bathrooms"] = bathrooms
input_df.loc[0, "Balconies"] = balcony
input_df.loc[0, "Floor"] = floor
input_df.loc[0, "Total_Floors"] = total_floors
input_df.loc[0, "Property_Age"] = property_age

input_df.loc[0, "Metro_Distance_km"] = metro_distance
input_df.loc[0, "Hospital_Distance_km"] = hospital_distance
input_df.loc[0, "School_Distance_km"] = school_distance
input_df.loc[0, "Mall_Distance_km"] = mall_distance
input_df.loc[0, "Expressway_Distance_km"] = expressway_distance
input_df.loc[0, "RRTS_Distance_km"] = rrts_distance

# -----------------------------------------------------
# CITY
# -----------------------------------------------------

city_column = f"City_{city}"

if city_column in input_df.columns:
    input_df.loc[0, city_column] = 1

# -----------------------------------------------------
# LOCALITY
# -----------------------------------------------------

locality_column = f"Locality_{locality}"

if locality_column in input_df.columns:
    input_df.loc[0, locality_column] = 1

# -----------------------------------------------------
# FURNISHING
# -----------------------------------------------------

furnishing_column = f"Furnishing_{furnishing}"

if furnishing_column in input_df.columns:
    input_df.loc[0, furnishing_column] = 1

# -----------------------------------------------------
# FACING
# -----------------------------------------------------

facing_column = f"Facing_{facing}"

if facing_column in input_df.columns:
    input_df.loc[0, facing_column] = 1

# -----------------------------------------------------
# PARKING
# -----------------------------------------------------

if parking == "Yes":
    input_df.loc[0, "Parking_Yes"] = 1

# -----------------------------------------------------
# GATED SOCIETY
# -----------------------------------------------------

if gated_society == "Yes":
    input_df.loc[0, "Gated_Society_Yes"] = 1

# -----------------------------------------------------
# LIFT
# -----------------------------------------------------

if lift == "Yes":
    input_df.loc[0, "Lift_Yes"] = 1

# -----------------------------------------------------
# POWER BACKUP
# -----------------------------------------------------

if power_backup == "Yes":
    input_df.loc[0, "Power_Backup_Yes"] = 1

# -----------------------------------------------------
# SECURITY
# -----------------------------------------------------

if security == "Yes":
    input_df.loc[0, "Security_Yes"] = 1

# -----------------------------------------------------
# GYM
# -----------------------------------------------------

if gym == "Yes":
    input_df.loc[0, "Gym_Yes"] = 1

# -----------------------------------------------------
# SWIMMING POOL
# -----------------------------------------------------

if swimming_pool == "Yes":
    input_df.loc[0, "Swimming_Pool_Yes"] = 1
    
# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------

if predict:

    prediction = model.predict(input_df)[0]

    # Round prediction
    prediction = round(prediction)

    st.markdown("---")

    # =================================================
    # PROPERTY SUMMARY
    # =================================================

    st.subheader("📋 Property Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"""
        **City**

        {city}

        **Locality**

        {locality}

        **Area**

        {area} sqft
        """)

    with col2:
        st.info(f"""
        **Bedrooms**

        {bedrooms}

        **Bathrooms**

        {bathrooms}

        **Floor**

        {floor}/{total_floors}
        """)

    with col3:
        st.info(f"""
        **Furnishing**

        {furnishing}

        **Facing**

        {facing}

        **Property Age**

        {property_age} Years
        """)

    st.markdown("---")

    # =================================================
    # PREDICTION CARD
    # =================================================

    st.markdown(
        f"""
        <div style="
            background:#0E1117;
            padding:30px;
            border-radius:18px;
            border:2px solid #2ECC71;
            text-align:center;
        ">

        <h2 style="color:white;">
        🏡 Estimated Property Price
        </h2>

        <h1 style="color:#2ECC71;font-size:48px;">
        ₹ {prediction:,.0f}
        </h1>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("Prediction Generated Successfully ✅")

    st.markdown("---")

    # =================================================
    # MODEL INFORMATION
    # =================================================

    st.subheader("🤖 Model Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Model",
            "XGBoost"
        )

    with c2:
        st.metric(
            "R² Score",
            "87.07%"
        )

    with c3:
        st.metric(
            "Features",
            len(feature_columns)
        )

    st.markdown("---")

    # =================================================
    # USER INPUT TABLE
    # =================================================

    st.subheader("📊 Input Features")

    display_df = pd.DataFrame({

        "Feature":[
            "City",
            "Locality",
            "Area",
            "Bedrooms",
            "Bathrooms",
            "Balconies",
            "Parking",
            "Floor",
            "Total Floors",
            "Property Age",
            "Furnishing",
            "Facing",
            "Metro Distance",
            "Hospital Distance",
            "School Distance",
            "Mall Distance",
            "Expressway Distance",
            "RRTS Distance",
            "Gated Society",
            "Lift",
            "Power Backup",
            "Security",
            "Gym",
            "Swimming Pool"
        ],

        "Value":[
            city,
            locality,
            area,
            bedrooms,
            bathrooms,
            balcony,
            parking,
            floor,
            total_floors,
            property_age,
            furnishing,
            facing,
            metro_distance,
            hospital_distance,
            school_distance,
            mall_distance,
            expressway_distance,
            rrts_distance,
            gated_society,
            lift,
            power_backup,
            security,
            gym,
            swimming_pool
        ]

    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # -----------------------------------------------------
# PRICE CATEGORY
# -----------------------------------------------------

if predict:

    st.markdown("---")

    st.subheader("📈 Property Category")

    if prediction < 5000000:

        st.info("🏠 Budget Property")

    elif prediction < 10000000:

        st.success("🏢 Mid Range Property")

    elif prediction < 20000000:

        st.warning("🏡 Premium Property")

    else:

        st.error("🏰 Luxury Property")


# -----------------------------------------------------
# SIDEBAR MODEL DETAILS
# -----------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

st.sidebar.success("Model : XGBoost")

st.sidebar.info("Accuracy (R²): 87.07%")

st.sidebar.write(f"Features : {len(feature_columns)}")

st.sidebar.write("Machine Learning Model")

# -----------------------------------------------------
# DATASET INFORMATION
# -----------------------------------------------------

st.markdown("---")

st.subheader("📂 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Rows",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "Columns",
        len(df.columns)
    )

with col3:

    st.metric(
        "Model Features",
        len(feature_columns)
    )

# -----------------------------------------------------
# FEATURE SUMMARY
# -----------------------------------------------------

st.markdown("---")

st.subheader("📑 Dataset Features")

numeric_features = [

    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Balconies",
    "Floor",
    "Total_Floors",
    "Property_Age",
    "Metro_Distance_km",
    "Hospital_Distance_km",
    "School_Distance_km",
    "Mall_Distance_km",
    "Expressway_Distance_km",
    "RRTS_Distance_km"

]

categorical_features = [

    "City",
    "Locality",
    "Parking",
    "Furnishing",
    "Facing",
    "Gated_Society",
    "Lift",
    "Power_Backup",
    "Security",
    "Gym",
    "Swimming_Pool"

]

left, right = st.columns(2)

with left:

    st.success("### Numerical Features")

    for feature in numeric_features:

        st.write("•", feature)

with right:

    st.success("### Categorical Features")

    for feature in categorical_features:

        st.write("•", feature)

# -----------------------------------------------------
# ABOUT PROJECT
# -----------------------------------------------------

st.markdown("---")

st.subheader("📖 About This Project")

st.write("""

This application predicts residential property prices using an **XGBoost Regressor** trained on cleaned real estate data.

### Machine Learning Workflow

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Feature Engineering

✔ One-Hot Encoding

✔ Linear Regression

✔ Decision Tree

✔ Random Forest

✔ Gradient Boosting

✔ Hyperparameter Tuning

✔ XGBoost (Best Model)

""")

# -----------------------------------------------------
# MODEL COMPARISON
# -----------------------------------------------------

comparison = pd.DataFrame({

    "Model":[

        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"

    ],

    "R²":[

        0.8249,
        0.8526,
        0.8697,
        0.8700,
        0.8707

    ]

})

st.markdown("---")

st.subheader("🏆 Model Comparison")

st.bar_chart(
    comparison.set_index("Model")
)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;color:gray">

<h4>🏡 AI Powered Real Property Valuation Predictor</h4>

Developed using

<b>Python • Streamlit • XGBoost • Pandas • Scikit-Learn</b>

<br><br>

© 2026 Sidharth Singh

</div>
""",
unsafe_allow_html=True
)