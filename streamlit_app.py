import streamlit as st
import pandas as pd
import numpy as np

def calculate_tpm(df):
    # Extract market names (excluding the first column which is year and the last column which is total exports)
    market_names = df.columns[1:-1]
    
    # Convert DataFrame to NumPy array (excluding the first and last columns)
    data = df[market_names].to_numpy()
    
    # Print data to debug
    print("Data array:\n", data)
    
    # Extract the number of states
    num_states = len(market_names)
    
    # Initialize the transition count matrix
    transition_counts = np.zeros((num_states, num_states))
    
    # Calculate transition counts
    for i in range(len(data) - 1):
        current_state = np.argmax(data[i, :])  # Find index of max value in current row
        next_state = np.argmax(data[i + 1, :])  # Find index of max value in next row
        if current_state < num_states and next_state < num_states:  # Ensure valid indices
            transition_counts[current_state, next_state] += 1

    # Print transition counts to debug
    print("Transition counts:\n", transition_counts)
    
    # Normalize to create the TPM
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    tpm = transition_counts / row_sums
    
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
        
        # Print TPM to debug
        print("Transition Probability Matrix:\n", tpm)
        
        # Convert TPM to DataFrame for display and download
        tpm_df = pd.DataFrame(tpm, columns=market_names, index=market_names)
        st.write("Transition Probability Matrix:")
        st.write(tpm_df)

        # Allow user to download TPM
        tpm_csv = tpm_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download TPM as CSV",
            data=tpm_csv,
            file_name='tpm_matrix.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
