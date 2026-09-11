import streamlit as st

st.set_page_config(
    page_title="Building Human-Centered AI Applications",
    page_icon=":material/edit:"
)

lab_one = st.Page(
    "Lab1.py",
    title="Lab 1",
    icon="🛜"
)

lab_two = st.Page(
    "Lab2.py",
    title="Lab 2",
    icon="🛜"

)

lab_three = st.Page(
    "Lab3.py",
    title="Lab 3",
    icon="🛜"

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
    default=True
)

pg = st.navigation([
    lab_one,
    lab_two,
    lab_three,
    hw1,
    hw2
])

pg.run()