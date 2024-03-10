from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

# Sample transaction data
data = {'Transaction': ['T1', 'T1', 'T2', 'T2', 'T2', 'T3', 'T3'],
        'Item': ['A', 'B', 'A', 'B', 'C', 'A', 'C']}
df = pd.DataFrame(data)

# Convert data to one-hot encoded format
one_hot_encoded = pd.get_dummies(df['Item'])

# Find frequent itemsets
frequent_itemsets = apriori(one_hot_encoded, min_support=0.5, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
print(rules)
