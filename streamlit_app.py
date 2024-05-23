import streamlit as st

def main():
    st.title('Job URL Entry')
    # Create a text input box
    url = st.text_input('Enter the Job URL', '')

    # Display the URL entered by the user
    if url:
        st.write('You entered:', url)

if __name__ == "__main__":
    main()
