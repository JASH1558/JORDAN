import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import pickle

# ============================================================
# CARD ELIXIR COST MAP
# ============================================================
CARD_ELIXIR = {
    26000000: 3,  26000001: 3,  26000003: 2,  26000004: 4,  26000005: 7,
    26000006: 3,  26000007: 5,  26000008: 5,  26000009: 5,  26000010: 3,
    26000011: 3,  26000012: 2,  26000013: 4,  26000014: 4,  26000015: 6,
    26000016: 1,  26000017: 2,  26000018: 4,  26000019: 4,  26000020: 4,
    26000021: 2,  26000022: 6,  26000023: 4,  26000024: 5,  26000025: 3,
    26000026: 5,  26000027: 4,  26000028: 3,  26000029: 4,  26000030: 4,
    26000031: 6,  26000032: 3,  26000033: 6,  26000034: 1,  26000035: 1,
    26000036: 3,  26000037: 4,  26000038: 4,  26000039: 4,  26000040: 1,
    26000041: 3,  26000042: 9,  26000043: 1,  26000044: 5,  26000045: 2,
    26000046: 5,  26000047: 6,  26000048: 1,  26000049: 4,  26000050: 2,
    26000051: 7,  26000052: 4,  26000053: 2,  26000054: 6,  26000055: 3,
    26000056: 4,  26000057: 5,  26000058: 3,  26000059: 4,  26000060: 1,
    26000061: 7,  26000062: 2,  26000063: 5,  26000064: 4,  26000065: 5,
    26000066: 4,  26000067: 4,  26000068: 4,  26000069: 4,  26000070: 3,
    26000071: 5,  26000072: 7,  26000073: 5,  26000074: 5,  26000075: 5,
    26000083: 5,  26000084: 3,  26000085: 3,
    27000000: 8,  27000001: 4,  27000002: 9,  27000003: 4,  27000004: 5,
    27000005: 4,  27000006: 5,  27000007: 4,  27000008: 5,  27000009: 4,
    27000010: 5,  27000011: 4,  27000012: 5,  27000013: 3,  27000014: 4,
    28000000: 7,  28000001: 4,  28000002: 5,  28000003: 5,  28000004: 5,
    28000005: 5,  28000006: 4,  28000007: 5,  28000008: 6,  28000009: 7,
    28000010: 3,  28000011: 3,  28000012: 6,  28000013: 5,  28000014: 3,
    28000015: 3,  28000016: 5,  28000017: 2,  28000018: 3,  28000019: 3,
}
DEFAULT_ELIXIR = 4
P1_CARDS      = list(range(5, 13))
P2_CARDS      = list(range(16, 24))
ALL_CARD_COLS = P1_CARDS + P2_CARDS

# ============================================================
# 1. Load & clean
# ============================================================
df_train = pd.read_csv("clash_sample2.csv", header=None)
df_test  = pd.read_csv("clash_sample.csv",       header=None)

for df in [df_train, df_test]:
    for col in range(24):
        df[col] = pd.to_numeric(df[col], errors='coerce')

df_train = df_train[df_train[4].isin([0, 1])].reset_index(drop=True)
df_test  = df_test[df_test[4].isin([0, 1])].reset_index(drop=True)
print(f"Train rows: {len(df_train):,} | Test rows: {len(df_test):,}")
print(f"Train win rate: {df_train[4].mean():.3f} | Test win rate: {df_test[4].mean():.3f}")

# ============================================================
# 2. Vectorized helpers
# ============================================================
def map_cards_vectorized(df, cols, mapping, default):
    arr = df[cols].fillna(0).astype(np.int64).values
    return np.vectorize(lambda x: mapping.get(x, default))(arr)

def rarity_vectorized(arr):
    out = np.zeros_like(arr, dtype=np.int8)
    out[arr >= 27000000] = 1
    out[arr >= 28000000] = 2
    return out

def trophy_bucket(arr):
    b = np.zeros_like(arr, dtype=np.int8)
    b[arr >= 4000] = 1
    b[arr >= 6000] = 2
    b[arr >= 7000] = 3
    b[arr >= 8000] = 4
    return b

# ============================================================
# 3. Base features  (includes trophy buckets)
# ============================================================
def build_features(df):
    feats = pd.DataFrame(index=range(len(df)))

    p1t = df[3].fillna(0).values
    p2t = df[14].fillna(0).values
    feats["p1_trophies"]          = p1t
    feats["p2_trophies"]          = p2t
    feats["trophy_diff"]          = p1t - p2t
    feats["trophy_ratio"]         = p1t / (p2t + 1)
    feats["p1_trophy_bucket"]     = trophy_bucket(p1t)
    feats["p2_trophy_bucket"]     = trophy_bucket(p2t)
    feats["bucket_diff"]          = feats["p1_trophy_bucket"] - feats["p2_trophy_bucket"]

    p1_elix = map_cards_vectorized(df, P1_CARDS, CARD_ELIXIR, DEFAULT_ELIXIR).astype(float)
    p2_elix = map_cards_vectorized(df, P2_CARDS, CARD_ELIXIR, DEFAULT_ELIXIR).astype(float)
    feats["p1_avg_elixir"]        = p1_elix.mean(axis=1)
    feats["p2_avg_elixir"]        = p2_elix.mean(axis=1)
    feats["p1_min_elixir"]        = p1_elix.min(axis=1)
    feats["p2_min_elixir"]        = p2_elix.min(axis=1)
    feats["p1_max_elixir"]        = p1_elix.max(axis=1)
    feats["p2_max_elixir"]        = p2_elix.max(axis=1)
    feats["elixir_diff"]          = feats["p1_avg_elixir"] - feats["p2_avg_elixir"]
    feats["p1_low_elixir_count"]  = (p1_elix <= 3).sum(axis=1)
    feats["p2_low_elixir_count"]  = (p2_elix <= 3).sum(axis=1)
    feats["p1_high_elixir_count"] = (p1_elix >= 6).sum(axis=1)
    feats["p2_high_elixir_count"] = (p2_elix >= 6).sum(axis=1)

    p1_arr = df[P1_CARDS].fillna(0).astype(np.int64).values
    p2_arr = df[P2_CARDS].fillna(0).astype(np.int64).values
    p1_rar = rarity_vectorized(p1_arr).astype(float)
    p2_rar = rarity_vectorized(p2_arr).astype(float)
    feats["p1_avg_rarity"]        = p1_rar.mean(axis=1)
    feats["p2_avg_rarity"]        = p2_rar.mean(axis=1)
    feats["p1_epic_leg_count"]    = (p1_rar == 2).sum(axis=1)
    feats["p2_epic_leg_count"]    = (p2_rar == 2).sum(axis=1)
    feats["rarity_diff"]          = feats["p1_avg_rarity"] - feats["p2_avg_rarity"]

    feats["deck_overlap"]         = np.array([len(np.intersect1d(p1_arr[i], p2_arr[i])) for i in range(len(df))])
    feats["p1_deck_diversity"]    = (df[P1_CARDS].nunique(axis=1) / 8).values
    feats["p2_deck_diversity"]    = (df[P2_CARDS].nunique(axis=1) / 8).values
    return feats

# ============================================================
# 4. Card win-rate
# ============================================================
def build_card_winrate_map(df):
    label = df[4].values
    p1_df = df[P1_CARDS].copy(); p1_df["win"] = label
    p2_df = df[P2_CARDS].copy(); p2_df.columns = P1_CARDS; p2_df["win"] = 1 - label
    combined = pd.concat([p1_df, p2_df], ignore_index=True)
    melted = combined.melt(id_vars="win", value_vars=P1_CARDS, value_name="card_id").dropna(subset=["card_id"])
    melted["card_id"] = melted["card_id"].astype(np.int64)
    stats = melted.groupby("card_id")["win"].agg(["sum", "count"])
    stats["win_rate"] = stats["sum"] / stats["count"]
    global_wr = stats["sum"].sum() / (stats["count"].sum() + 1e-9)
    return stats["win_rate"].to_dict(), global_wr

def apply_card_winrate(df, winrate_map, global_wr):
    wr_vec = np.vectorize(lambda x: winrate_map.get(int(x), global_wr))
    p1_wr  = wr_vec(df[P1_CARDS].fillna(0).astype(float).values)
    p2_wr  = wr_vec(df[P2_CARDS].fillna(0).astype(float).values)
    return pd.DataFrame({
        "p1_avg_card_wr": p1_wr.mean(axis=1), "p2_avg_card_wr": p2_wr.mean(axis=1),
        "p1_min_card_wr": p1_wr.min(axis=1),  "p1_max_card_wr": p1_wr.max(axis=1),
        "p2_min_card_wr": p2_wr.min(axis=1),  "p2_max_card_wr": p2_wr.max(axis=1),
        "card_wr_diff":   p1_wr.mean(axis=1) - p2_wr.mean(axis=1),
    })

# ============================================================
# 5. Pair synergy  (fast — searchsorted, no iterrows)
# ============================================================
def build_pair_synergy_map(df):
    label    = df[4].values
    card_arr = df[P1_CARDS].fillna(0).astype(np.int64).values
    records  = []
    for i in range(8):
        for j in range(i + 1, 8):
            records.append(pd.DataFrame({
                "card_a": np.minimum(card_arr[:, i], card_arr[:, j]),
                "card_b": np.maximum(card_arr[:, i], card_arr[:, j]),
                "win":    label
            }))
    pairs_df         = pd.concat(records, ignore_index=True)
    pairs_df["pair"] = list(zip(pairs_df["card_a"], pairs_df["card_b"]))
    stats            = pairs_df.groupby("pair")["win"].agg(["sum", "count"])
    stats["win_rate"]= stats["sum"] / stats["count"]
    global_wr        = stats["sum"].sum() / (stats["count"].sum() + 1e-9)
    return stats["win_rate"].to_dict(), global_wr

def apply_pair_synergy(df, synergy_map, global_wr):
    card_arr = df[P1_CARDS].fillna(0).astype(np.int64).values
    # Encode pairs as single int64 key, use searchsorted for fast lookup
    syn_keys = np.array([(k[0] * 10**9 + k[1]) for k in synergy_map.keys()], dtype=np.int64)
    syn_vals = np.array(list(synergy_map.values()), dtype=np.float32)
    sort_idx = np.argsort(syn_keys)
    syn_keys = syn_keys[sort_idx]
    syn_vals = syn_vals[sort_idx]
    scores   = []
    for i in range(8):
        for j in range(i + 1, 8):
            a     = np.minimum(card_arr[:, i], card_arr[:, j])
            b     = np.maximum(card_arr[:, i], card_arr[:, j])
            query = a * 10**9 + b
            pos   = np.searchsorted(syn_keys, query)
            pos   = np.clip(pos, 0, len(syn_keys) - 1)
            found = syn_keys[pos] == query
            scores.append(np.where(found, syn_vals[pos], global_wr))
    return pd.Series(np.mean(scores, axis=0), name="p1_deck_synergy")

# ============================================================
# 6. One-hot card presence
# ============================================================
def build_card_onehot(df, all_cards_universe, prefix):
    src  = P1_CARDS if prefix == "p1" else P2_CARDS
    arr  = df[src].fillna(0).astype(np.int64).values
    univ = np.array(all_cards_universe)
    oh   = (arr[:, :, None] == univ[None, None, :]).any(axis=1).astype(np.float32)
    return pd.DataFrame(oh, columns=[f"{prefix}_has_{c}" for c in univ])

# ============================================================
# 7. Build all features
# ============================================================
print("Building features...")

X_train_base = build_features(df_train)
X_test_base  = build_features(df_test)
print("  ✓ Base features")

winrate_map, global_wr = build_card_winrate_map(df_train)
X_train_wr = apply_card_winrate(df_train, winrate_map, global_wr)
X_test_wr  = apply_card_winrate(df_test,  winrate_map, global_wr)
print("  ✓ Card win-rate features")

synergy_map, syn_global_wr = build_pair_synergy_map(df_train)
X_train_syn = apply_pair_synergy(df_train, synergy_map, syn_global_wr).to_frame()
X_test_syn  = apply_pair_synergy(df_test,  synergy_map, syn_global_wr).to_frame()
print("  ✓ Pair synergy features")

all_cards_universe = sorted(set(
    df_train[ALL_CARD_COLS].fillna(0).astype(np.int64).values.flatten().tolist()
) - {0})
print(f"  ✓ Card universe: {len(all_cards_universe)} unique cards")

X_train_p1oh = build_card_onehot(df_train, all_cards_universe, "p1")
X_test_p1oh  = build_card_onehot(df_test,  all_cards_universe, "p1")
X_train_p2oh = build_card_onehot(df_train, all_cards_universe, "p2")
X_test_p2oh  = build_card_onehot(df_test,  all_cards_universe, "p2")
print("  ✓ One-hot card features")

X_train_full = pd.concat([X_train_base, X_train_wr, X_train_syn, X_train_p1oh, X_train_p2oh], axis=1)
X_test_full  = pd.concat([X_test_base,  X_test_wr,  X_test_syn,  X_test_p1oh,  X_test_p2oh],  axis=1)
X_train_full.columns = [str(c) for c in X_train_full.columns]
X_test_full.columns  = [str(c) for c in X_test_full.columns]
X_test_full = X_test_full.reindex(columns=X_train_full.columns, fill_value=0)
print(f"  ✓ Total features: {X_train_full.shape[1]}")

# ============================================================
# 8. Scale continuous features
# ============================================================
continuous_cols = list(X_train_base.columns) + list(X_train_wr.columns) + ["p1_deck_synergy"]
scaler = MinMaxScaler()
X_train_full[continuous_cols] = scaler.fit_transform(X_train_full[continuous_cols])
X_test_full[continuous_cols]  = scaler.transform(X_test_full[continuous_cols])

# ============================================================
# 9. Labels
# ============================================================
y_train = df_train[4].values.astype(np.float32)
y_test  = df_test[4].values.astype(np.float32)
print(f"\ny_train: {np.unique(y_train)} | y_test: {np.unique(y_test)}")

# ============================================================
# 10. Train XGBoost
# ============================================================
dtrain = xgb.DMatrix(X_train_full.values.astype(np.float32), label=y_train)
dtest  = xgb.DMatrix(X_test_full.values.astype(np.float32),  label=y_test)

params = {
    "objective":        "binary:logistic",
    "max_depth":        5,
    "eta":              0.05,
    "subsample":        0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 20,
    "gamma":            0.5,
    "lambda":           3.0,
    "alpha":            0.5,
    "scale_pos_weight": 1.0,
    "eval_metric":      "logloss",
    "tree_method":      "hist",
    "seed":             42
}

print("\nTraining XGBoost...")
model = xgb.train(
    params, dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dtest, "eval")],
    early_stopping_rounds=100,
    verbose_eval=200
)

# ============================================================
# 11. Evaluate
# ============================================================
preds_prob = model.predict(dtest)
preds      = (preds_prob > 0.5).astype(int)

print("\n====== EVALUATION ======")
print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, preds_prob):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}")

print("\nTop 15 Important Features:")
importance = model.get_score(importance_type="gain")
top15 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
for feat, score in top15:
    print(f"  {feat}: {score:.4f}")

# ============================================================
# 12. Save
# ============================================================
model.save_model("clash_model_robust.json")
with open("clash_metadata_robust.pkl", "wb") as f:
    pickle.dump({
        "scaler":             scaler,
        "continuous_cols":    continuous_cols,
        "winrate_map":        winrate_map,
        "global_wr":          global_wr,
        "synergy_map":        synergy_map,
        "syn_global_wr":      syn_global_wr,
        "all_cards_universe": all_cards_universe,
        "feature_columns":    list(X_train_full.columns),
    }, f)
print("\nSaved: clash_model_robust.json + clash_metadata_robust.pkl")