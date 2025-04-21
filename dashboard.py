import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
import os

# Page config
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")


# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Data.csv")
    df["AgeGroup"] = pd.cut(df["Age"], bins=[18, 30, 40, 50, 60, 70], labels=["18-30", "31-40", "41-50", "51-60", "61+"])
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Navigate to", ["Dashboard", "Team", "Documentation", "Model Code"])
st.sidebar.download_button("⬇️ Download cleaned_data", df.to_csv(index=False), "cleaned_data.csv", "text/csv")
st.sidebar.download_button("⬇️ Download Churned_data", df.to_csv(index=False),"Churn_Modelling.csv", "text/csv")

# ---------------------------------
# 1. DASHBOARD SECTION
# ---------------------------------
if page == "Dashboard":
    st.title("📊 Customer Churn Dashboard")

    tabs = st.tabs(["Overview", "Demographics", "Products & Salary", "Region & Behavior", "Correlation"])

    with tabs[0]:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(x="Exited", data=df, hue="Exited", palette=["#2ecc40", "#ff4136"], ax=ax, legend=False)
        ax.set_xticklabels(["Stayed", "Churned"])
        st.pyplot(fig)
        st.metric("Churn Rate", f"{df['Exited'].mean() * 100:.2f} %")

    with tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Churn by Gender")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(x="Gender", hue="Exited", data=df, palette=["#2ecc40", "#ff4136"], ax=ax)
            st.pyplot(fig)

        with col2:
            st.markdown("#### Age Distribution by Churn")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df, x="Age", hue="Exited", bins=30, kde=True, palette=["#ff4136", "#2ecc40"], ax=ax)
            st.pyplot(fig)

        st.markdown("#### Churn Rate by Age Group")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x="AgeGroup", y="Exited", data=df, palette="bright", errorbar=None, ax=ax)
        st.pyplot(fig)

    with tabs[2]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Churn by Number of Products")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(x="NumOfProducts", hue="Exited", data=df, palette=["#2ecc40", "#ff4136"], ax=ax)
            st.pyplot(fig)

        with col2:
            st.markdown("#### Product Count vs Salary by Churn")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.scatterplot(x="NumOfProducts", y="EstimatedSalary", hue="Exited", data=df,
                            palette=["#ff4136", "#2ecc40"], ax=ax)
            st.pyplot(fig)

        st.markdown("#### Balance Distribution by Churn")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(df, x="Balance", hue="Exited", bins=30, kde=True, palette=["#2ecc40", "#ff4136"], ax=ax)
        st.pyplot(fig)

    with tabs[3]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Churn by Region")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(x="Country", hue="Exited", data=df, palette=["#2ecc40", "#ff4136"], ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            st.markdown("#### Churn by Active Membership")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(x="IsActiveMember", hue="Exited", data=df, palette=["#2ecc40", "#ff4136"], ax=ax)
            ax.set_xticklabels(["Inactive", "Active"])
            st.pyplot(fig)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### Churn by Credit Card Ownership")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(x="HasCrCard", hue="Exited", data=df, palette=["#2ecc40", "#ff4136"], ax=ax)
            ax.set_xticklabels(["No", "Yes"])
            st.pyplot(fig)

        with col4:
            st.markdown("#### Churn by Tenure")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(x="Tenure", hue="Exited", data=df, palette=["#2ecc40", "#ff4136"], ax=ax)
            st.pyplot(fig)

    with tabs[4]:
        st.markdown("#### Feature Correlation Heatmap")
        numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
                        "IsActiveMember", "EstimatedSalary", "Exited"]
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        st.pyplot(fig)

# ---------------------------------
# 2. TEAM SECTION
# ---------------------------------
elif page == "Team":
    st.title("👥 Project Team")

    col1, col2, col3, col4 = st.columns(4)

    team_members = [
        {
            "name": "Cheki Dorji",
            "course": "BSc in Data Science",
            "image": "images/c.jpeg"
        },
        {
            "name": "Rinzin Dema",
            "course": "BSc in Data Science",
            "image": "images/r.jfif"
        },
       
        {
            "name": "Gyembo Wangchuk",
            "course": "BSc in Data Science",
            "image": "images/g.jpeg"
        },
         {
            "name": "Tshering Yangden",
            "course": "BSc in Data Science",
            "image": "images/yangday.jpeg"
        }
        
    ]

    def image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            buffered = BytesIO(img_file.read())
            encoded = base64.b64encode(buffered.getvalue()).decode()
            return encoded

    for col, member in zip([col1, col2, col3, col4], team_members):
        if os.path.exists(member["image"]):
            img_base64 = image_to_base64(member["image"])
            card = f"""
            <div style="border: 1px solid #ddd; border-radius: 3px; padding: 10px;text-align: left;
                        background-color: #f9f9f9; box-shadow: 2px 2px 8px rgba(0,0,0,0.05);">
                <img src="data:image/jpeg;base64,{img_base64}" 
                     style="width:100%; height: 150px; object-fit: cover; margin-bottom: 5px;">
                <h6 style="margin-bottom: 1px;">{member['name']}</h6>
                <p style="color: #555;">{member['course']}</p>
            </div>
            """
            col.markdown(card, unsafe_allow_html=True)
        else:
            col.error(f"Image not found: {member['image']}")

# ---------------------------------
# 3. DOCUMENTATION SECTION
# ---------------------------------
elif page == "Documentation":
    st.title("📄 Project Documentation")
    st.markdown("""
    This customer churn dashboard provides visual insights into customer behavior,
    demographics, product usage, and churn patterns. It helps identify which types of customers are more likely to churn, aiding in retention strategies.
    
    ### Key Features:
    - Interactive visualizations with filters
    - Age, gender, region-wise churn breakdown
    - Product usage and salary impact
    - Correlation heatmap
    
    ### Goal:
    The goal of this project is to understand the patterns behind customer churn using data visualization and machine learning. This helps in proactive customer engagement.
    """)

# ---------------------------------
# 4. MODEL CODE SECTION
# ---------------------------------
elif page == "Model Code":
    st.title("🧠 Machine Learning Model")
    st.markdown("Here's the code used for training and evaluating the churn prediction model:")

    st.code("""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler,LabelEncoder
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        df = pd.read_csv('/content/drive/MyDrive/DAT 205 (Data science project)/Practical/Cleaned_Data.csv')
        df.head()
        df.info()
        #Binary feature for Balance
        df['BalanceZero'] = (df['Balance'] == 0).astype(int)

        #Age groups
        df['AgeGroup'] = pd.cut(df ['Age'], bins = [18, 25, 35, 45, 55, 65, 75, 85, 95], labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-95'])

        #Balance to Salary Ratio
        df['BalanceToSalaryRatio'] = df['Balance'] / df['EstimatedSalary']

        #Interaction feature between NumOfProducts and IsActiveMember 
        df ['ProductUsage'] = df['NumOfProducts'] * df['IsActiveMember']

        #Tenure grouping
        df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 2, 5, 7, 10], labels=['0-2', '3-5', '6-7', '8-10'])
        label_encoder = LabelEncoder()
        df['Gender'] = label_encoder.fit_transform(df['Gender'])
        df['Male_Germany'] = df['Gender'] * df['Country_Germany']
        df['Male_Spain'] = df['Gender'] * df['Country_Spain']

        df = pd.get_dummies(df, columns=['AgeGroup','TenureGroup'], drop_first=True)

        features = ['CreditScore', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'EstimatedSalary', 'Exited', 'Country_Germany', 'Country_Spain',
            'BalanceZero', 'BalanceToSalaryRatio', 'ProductUsage', 'Male_Germany',
            'Male_Spain',] + [col for col in df.columns if 'AgeGroup' in col or 'TenureGroup' in col]

        X = df[features]
        y= df["Exited"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scalar = StandardScaler()
        X_train = scalar.fit_transform(X_train)
        X_test = scalar.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
    """, language="python")

    st.markdown("### 🔄 Future Work")
    st.markdown("- Add SHAP explanations for model interpretation\n- Allow user to test predictions\n- Export charts and reports as PDF")

