import streamlit as st
import pickle as pkl
import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler

st.markdown("""
                <style>
                        h4{
                            color:#FF534D;
                            font-weight:bold;
                            font-family: Montserrat, sans-serif;
                            font-optical-sizing: auto;
                        }
                    </style>
            """, unsafe_allow_html=True)


def predict_cluster(X_data: pd.DataFrame):
    with open("finalized_xgboost_model.pkl", "rb") as f:
        model = pkl.load(f)
        return model.predict(X_data)


def set_page(page):
    st.session_state.page = page


def input_fields():
    st.title("Customer Classification Model")
    kidhome = 0
    teenhome = 0
    selector = ["Select", "Yes", "No"]
    maritial_status = ["Select", "Married", "Single"]

    is_parent = st.selectbox("Are you a Parent?", selector)
    is_married = st.selectbox("What is your maritial Status", maritial_status)
    if is_parent == "Yes":
        teenhome = st.number_input("Do you have any teenager?", min_value=0, max_value=5, step=1, format="%d")
        kidhome = st.number_input("Do you have any kid?", min_value=0, max_value=5, step=1, format="%d")

    age = st.number_input("Enter your age", min_value=0, max_value=90, step=1, format="%d")

    income = st.number_input("Enter Income:")

    family_size = kidhome + teenhome + 2 if is_married == "Married" else kidhome + teenhome + 1

    spent = st.number_input("Enter Total Spent Amount:")

    is_parent = 0 if is_parent == "No" else 1

    data = [[is_parent, income, teenhome, kidhome, age, family_size, spent]]

    X_data = pd.DataFrame(data,
                          columns=["Is_Parent", "Income", "Teenhome", "Kidhome", "Age", "Family_Member_Count", "Spent"])

    uploaded_file = st.file_uploader("Add File", type=["xlsx", "csv", "tsv"])
    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        st.button('Submit', on_click=set_page, args=["File Data"])
    else:
        st.session_state["Input_Data"] = X_data
        st.button('Submit', on_click=set_page, args=["Display Data"])


def get_insights():
    insights = {
        0: ["Insight 1 for Class 0", "Insight 2 for class 0"],
        1: ["Insight 1 for Class 1", "Insight 2 for class 1"],
        2: ["Insight 1 for Class 2", "Insight 2 for class 2"],
        3: ["Insight 1 for Class 3", "Insight 2 for class 3"],
    }

    return insights


def show_result():
    st.markdown("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,100..900;1,100..900&family=Sedan:ital@0;1&display=swap')
            </style>
        """, unsafe_allow_html=True)
    st.title("Result")
    if "Input_Data" in st.session_state:
        # insights = {
        #     0:["Insight 1 for Class 0","Insight 2 for class 0"],
        #     1:["Insight 1 for Class 1","Insight 2 for class 1"],
        #     2:["Insight 1 for Class 2","Insight 2 for class 2"],
        #     3:["Insight 1 for Class 3","Insight 2 for class 3"],
        # }
        X_data = st.session_state["Input_Data"]
        st.dataframe(X_data)
        label = predict_cluster(X_data=X_data)
        label_no = label[0].item()
        st.header(f"Cluster: {label[0]}")
        insights_new = get_insights()[label_no]

        st.markdown("""
                <style>
                        h4{
                            color:#FF534D;
                            font-weight:bold;
                            font-family: Montserrat, sans-serif;
                            font-optical-sizing: auto;
                        }
                    </style>
            """, unsafe_allow_html=True)
        with st.expander("#### Business Insights", expanded=True):
            st.markdown("""
                    <ul>
                        <li>Hello Insights</li>
                    </ul>
                """, unsafe_allow_html=True)
        with st.expander("#### Future Course of Action", expanded=True):
            st.markdown(insights_new[0])
    else:
        st.write("Data not received")

    st.button('Home', on_click=set_page, args=["Input Data"])


def show_file_result():
    st.title("Result")
    uploaded_file = st.session_state["uploaded_file"]
    data = pd.read_csv(uploaded_file)
    X_data = data[["Is_Parent","Income","Teenhome","Kidhome","Age","Family_Member_Count","Spent"]]
    label = predict_cluster(X_data=X_data)
    st.write(label)
    new_df = X_data
    new_df["Clusters"] = label
    st.dataframe(new_df)
    with st.expander("#### Cluster Analysis", expanded=False):
        st.markdown("""
                    <h4>Cluster 0:</h4>
                    <ul>
                    <li>Definitely a parent</li>
                    <li>At max have 4 and at least 2 members in family</li>
                    <li>Single parents are a subset of this group</li>
                    <li>Most of them have a teenager at home</li>
                    <li>They are relatively older</li>
                    </ul>

                    <h4>Cluster 1:</h4>
                    <ul>
                    <li>Definitely not a parent</li>
                    <li>At max there are only 2 members in the family</li>
                    <li>A slight majority of couples over singles</li>
                    <li>Span all ages</li>
                    <li>A high income group</li>
                    </ul>

                    <h4>Cluster 2:</h4>
                    <ul>
                    <li>The majority of these people are parents</li>
                    <li>At the max there are 3 members in the family</li>
                    <li>They majorly have one kid (and not teenagers, typically)</li>
                    <li>They are relatively younger</li>
                    </ul>

                    <ul>
                    <h4>Cluster 3:</h4>
                    <li>They are definitely a parent</li>
                    <li>At max have 5 and at least 2 members in the family</li>
                    <li>Majority have a teenager at home</li>
                    <li>They are relatively older</li>
                    <li>A low-income group</li>
                    </ul>
                """, unsafe_allow_html=True)

    st.button('Home', on_click=set_page, args=["Input Data"])


def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'Input Data'

    if st.session_state.page == 'Input Data':
        input_fields()
    elif st.session_state.page == 'Display Data':
        show_result()
    elif st.session_state.page == 'File Data':
        show_file_result()


if __name__ == "__main__":
    main()





