import streamlit as st

lab_one = st.Page('Lab1.py', title ='Lab 1', icon = '📓')
lab_two = st.Page('Lab2.py', title ='Lab 2', icon = '🛜')

pg = st.navigation ([lab_one, lab_two])
st.set_page_config(page_title='Building Human-Centered AI Applications', page_icon= ':material/edit:')
pg.run()