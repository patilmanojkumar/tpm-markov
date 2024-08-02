import streamlit as st
import pandas as pd
import numpy as np

def calculate_tpm(df):
    # Convert DataFrame to NumPy array
    data = df.to_numpy()
    
    # Extract the number of states
    num_states = data.shape[1]
    
    # Initialize the transition count matrix
    transition_counts = np.zeros((num_states, num_states))
    
    # Calculate transition counts
    for i in range(len(data) - 1):
        current_state = np.argmax(data[i, 1:])  # Exclude the first column (year)
        next_state = np.argmax(data[i + 1, 1:])
        transition_counts[current_state, next_state] += 1

    # Normalize to create the TPM
    tpm = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    
    return tpm

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

        # Calculate TPM
        tpm = calculate_tpm(df)
        
        # Convert TPM to DataFrame for display and download
        tpm_df = pd.DataFrame(tpm, columns=[f'Market_{i+1}' for i in range(tpm.shape[1])])
        st.write("Transition Probability Matrix:")
        st.write(tpm_df)

        # Allow user to download TPM
        tpm_csv = tpm_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download TPM as CSV",
            data=tpm_csv,
            file_name='tpm_matrix.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
