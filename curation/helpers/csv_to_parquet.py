import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
df.to_parquet(sys.argv[2], index=False)