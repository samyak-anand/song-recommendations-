import streamlit as st 

st.title('Simple Streamlit app')
3
st.write ('welcome to this simple app, Ironhack')
name =st.text_input('Enter your name')


age = st.number_input('Enter your age: min_valu ',min_value=1, max_value= 120, step=1)
# Create a button that the user can click
if st.button("Submit"):
    # Display a greeting and the user's age next year
    st.write(f"Hello, {name}!")
    st.write(f"You are {age} years old.")
    st.write(f"Next year, you will be {age + 1} years old!")

# Add a footer message
st.write("Thank you for using this simple app!")
