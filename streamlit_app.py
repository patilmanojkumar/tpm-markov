import streamlit as st
import pandas as pd
import numpy as np

def calculate_tpm(df):
    # Extract market names (excluding the first column which is year and the last column which is total exports)
    market_names = df.columns[1:-1]
    
    # Convert DataFrame to NumPy array (excluding the first and last columns)
    data = df[market_names].to_numpy()
    
    # Debug: Print the data array
    st.write("Data array:\n", data)
    transition_counts = data.copy()
    
    # Debug: Print the transition counts matrix
    st.write("Transition counts:\n", transition_counts)
    
    tpm = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    return tpm, market_names

def main():
    st.title("Transition Probability Matrix Calculator")

    uploaded_file = st.file_uploader("Upload a CSV, XLS, or XLSX file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Data Preview:")
        st.write(df.head())

        if df.empty:
            st.error("The uploaded file is empty. Please upload a valid file.")
            return

        # Ensure that the DataFrame has more than one row
        if len(df) < 2:
            st.error("The uploaded file does not have enough data for transition calculation.")
            return

        # Calculate TPM
        tpm, market_names = calculate_tpm(df)

        st.write("Transition Probability Matrix:")
        tpm_df = pd.DataFrame(tpm)
        st.write(tpm_df)

        # Option to download the TPM
        csv = tpm_df.to_csv(index=True)
        st.download_button(
            label="Download TPM as CSV",
            data=csv,
            file_name='tpm.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
