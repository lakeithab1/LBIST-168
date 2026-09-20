import streamlit as st

st.set_page_config(
    page_title="Building Human-Centered AI Applications",
    page_icon=":material/edit:"
)

hw1 = st.Page(
    "HW Manager/HW1.py",
    title="HW1",
    icon = "✏️"
)

hw2 = st.Page(
    "HW Manager/HW2.py",
    title="HW2",
    icon = "✏️",
)

hw3 = st.Page(
    "HW Manager/HW3.py",
    title= "HW3",
    icon="✏️",

)

hw4 = st.Page(
    "HW Manager/HW4.py",
    title= "HW4",
    icon = "✏️",
    default= True
)

pg = st.navigation([
    hw1,
    hw2,
    hw3,
    hw4
])

pg.run()