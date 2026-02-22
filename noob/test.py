import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

# ============================================================
# Load model and metadata ONCE at the top
# ============================================================
_model = xgb.Booster()
_model.load_model("clash_model_robust.json")

with open("clash_metadata_robust.pkl", "rb") as f:
    _meta = pickle.load(f)

P1_CARDS = list(range(5, 13))
P2_CARDS = list(range(16, 24))
ALL_CARD_COLS = P1_CARDS + P2_CARDS

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

# ============================================================
# Internal feature builders (same as training)
# ============================================================
def _map_cards(df, cols, mapping, default):
    arr = df[cols].fillna(0).astype(np.int64).values
    return np.vectorize(lambda x: mapping.get(x, default))(arr)

def _rarity(arr):
    out = np.zeros_like(arr, dtype=np.int8)
    out[arr >= 27000000] = 1
    out[arr >= 28000000] = 2
    return out

def _build_features(df):
    feats = pd.DataFrame(index=range(len(df)))
    feats["p1_trophies"]          = df[3].fillna(0).values
    feats["p2_trophies"]          = df[14].fillna(0).values
    feats["trophy_diff"]          = feats["p1_trophies"] - feats["p2_trophies"]
    feats["trophy_ratio"]         = feats["p1_trophies"] / (feats["p2_trophies"] + 1)

    p1_elix = _map_cards(df, P1_CARDS, CARD_ELIXIR, DEFAULT_ELIXIR).astype(float)
    p2_elix = _map_cards(df, P2_CARDS, CARD_ELIXIR, DEFAULT_ELIXIR).astype(float)
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
    p1_rar = _rarity(p1_arr).astype(float)
    p2_rar = _rarity(p2_arr).astype(float)
    feats["p1_avg_rarity"]        = p1_rar.mean(axis=1)
    feats["p2_avg_rarity"]        = p2_rar.mean(axis=1)
    feats["p1_epic_leg_count"]    = (p1_rar == 2).sum(axis=1)
    feats["p2_epic_leg_count"]    = (p2_rar == 2).sum(axis=1)
    feats["rarity_diff"]          = feats["p1_avg_rarity"] - feats["p2_avg_rarity"]

    feats["deck_overlap"]         = np.array([len(np.intersect1d(p1_arr[i], p2_arr[i])) for i in range(len(df))])
    feats["p1_deck_diversity"]    = (df[P1_CARDS].nunique(axis=1) / 8).values
    feats["p2_deck_diversity"]    = (df[P2_CARDS].nunique(axis=1) / 8).values
    return feats

def _apply_card_winrate(df, winrate_map, global_wr):
    wr_vec = np.vectorize(lambda x: winrate_map.get(int(x), global_wr))
    p1_wr = wr_vec(df[P1_CARDS].fillna(0).astype(float).values)
    p2_wr = wr_vec(df[P2_CARDS].fillna(0).astype(float).values)
    return pd.DataFrame({
        "p1_avg_card_wr": p1_wr.mean(axis=1),
        "p2_avg_card_wr": p2_wr.mean(axis=1),
        "p1_min_card_wr": p1_wr.min(axis=1),
        "p1_max_card_wr": p1_wr.max(axis=1),
        "p2_min_card_wr": p2_wr.min(axis=1),
        "p2_max_card_wr": p2_wr.max(axis=1),
        "card_wr_diff":   p1_wr.mean(axis=1) - p2_wr.mean(axis=1),
    })

def _apply_pair_synergy(df, synergy_map, global_wr):
    card_arr = df[P1_CARDS].fillna(0).astype(np.int64).values
    # Encode each pair as a single int64 key for fast sorted lookup
    syn_keys = np.array([(k[0] * 10**9 + k[1]) for k in synergy_map.keys()], dtype=np.int64)
    syn_vals = np.array(list(synergy_map.values()), dtype=np.float32)
    sort_idx = np.argsort(syn_keys)
    syn_keys = syn_keys[sort_idx]
    syn_vals = syn_vals[sort_idx]
    scores = []
    for i in range(8):
        for j in range(i + 1, 8):
            a = np.minimum(card_arr[:, i], card_arr[:, j])
            b = np.maximum(card_arr[:, i], card_arr[:, j])
            query = a * 10**9 + b
            pos   = np.searchsorted(syn_keys, query)
            pos   = np.clip(pos, 0, len(syn_keys) - 1)
            found = syn_keys[pos] == query
            scores.append(np.where(found, syn_vals[pos], global_wr))
    return pd.Series(np.mean(scores, axis=0), name="p1_deck_synergy")

def _build_card_onehot(df, all_cards_universe, prefix):
    card_cols_src = P1_CARDS if prefix == "p1" else P2_CARDS
    card_arr = df[card_cols_src].fillna(0).astype(np.int64).values
    universe = np.array(all_cards_universe)
    onehot = (card_arr[:, :, None] == universe[None, None, :]).any(axis=1).astype(np.float32)
    return pd.DataFrame(onehot, columns=[f"{prefix}_has_{c}" for c in universe])

# ============================================================
# 🎯 MAIN PREDICTION FUNCTION
# ============================================================
def predict(csv_path, threshold=0.5):
    """
    Takes a CSV file path (same format as training data, no headers).
    Returns a DataFrame with columns:
        - win_probability  : float 0-1, probability that player1 wins
        - prediction       : 1 = player1 wins, 0 = player1 loses
    
    Parameters:
        csv_path  : path to your CSV file
        threshold : decision boundary (default 0.5)
    
    Example:
        results = predict("new_matches.csv")
        print(results)
    """
    # Load
    df = pd.read_csv(csv_path, header=None)
    for col in range(min(24, df.shape[1])):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.reset_index(drop=True)

    # Build features
    X_base = _build_features(df)
    X_wr   = _apply_card_winrate(df, _meta["winrate_map"], _meta["global_wr"])
    X_syn  = _apply_pair_synergy(df, _meta["synergy_map"], _meta["syn_global_wr"]).to_frame()
    X_p1   = _build_card_onehot(df, _meta["all_cards_universe"], "p1")
    X_p2   = _build_card_onehot(df, _meta["all_cards_universe"], "p2")

    X = pd.concat([X_base, X_wr, X_syn, X_p1, X_p2], axis=1)
    X.columns = [str(c) for c in X.columns]

    # Align to training feature set (fill 0 for unseen cards)
    X = X.reindex(columns=_meta["feature_columns"], fill_value=0)

    # Scale
    X[_meta["continuous_cols"]] = _meta["scaler"].transform(X[_meta["continuous_cols"]])

    # Predict
    dmat = xgb.DMatrix(X.values.astype(np.float32))
    probs = _model.predict(dmat)
    preds = (probs > threshold).astype(int)

    results = pd.DataFrame({
        "win_probability": np.round(probs, 4),
        "prediction":      preds
    })
    results["prediction_label"] = results["prediction"].map({1: "Player 1 Wins", 0: "Player 2 Wins"})

    print(f"Processed {len(results)} matches")
    print(f"  Player 1 wins: {preds.sum()} ({preds.mean()*100:.1f}%)")
    print(f"  Player 2 wins: {(1-preds).sum()} ({(1-preds).mean()*100:.1f}%)")

    return results


# ============================================================
# Usage
# ============================================================
if __name__ == "__main__":
    results = predict("clash_sample.csv")
    y_true = pd.read_csv("clash_sample.csv", header=None)[4].values
    accuracy = (results["prediction"].values == y_true).mean()

    print(f"Accuracy: {accuracy*100:.2f}%")

    # Save predictions to CSV
    # results.to_csv("predictions.csv", index=False)