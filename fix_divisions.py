import pandas as pd

df = pd.read_csv("boxing_fighters_master.csv")

# Put both in the same division label as everyone else
is_crawford = df["fighter"].str.contains("Terence 'Bud' Crawford", regex=False)
is_canelo   = df["fighter"].str.contains("Saul 'Canelo' Alvarez|Saul 'Canelo' Álvarez", regex=True)

df.loc[is_crawford, "division"] = "Super Middleweight (168)"
df.loc[is_canelo,   "division"] = "Super Middleweight (168)"

df.to_csv("boxing_fighters_master.csv", index=False)
print("OK: standardized divisions for Crawford & Canelo to 'Super Middleweight (168)'")
