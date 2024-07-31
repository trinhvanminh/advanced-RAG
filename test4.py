import pandas as pd

# Sample data based on the provided dataframe
data = {
    "Lender": ["Adelaide", "ANZ", "Bank of Melbourne / St. George / Bank SA", "Bank First - CURRENTLY SUSPENDED", "Bridgit - Off Panel. Requires Approval"],
    "Fees": ["$1,500.00 App Fee", "Check with Lender", "$600 application fee with end loan", "App fee $950.00", "1.99% Set-up Fee"]
}

df = pd.DataFrame(data)

# Display the dataframe
print(df[['Lender', 'Fees']])
