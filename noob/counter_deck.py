import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import random
from itertools import combinations

# ============================================================
# CARD NAME MAP (id -> name) — extend as needed
# ============================================================
CARD_NAMES = {
    26000000: "Knight",           26000001: "Archers",         26000003: "Goblins",
    26000004: "Giant",            26000005: "P.E.K.K.A",       26000006: "Minions",
    26000007: "Balloon",          26000008: "Witch",            26000009: "Barbarians",
    26000010: "Goblin Spear",     26000011: "Valkyrie",         26000012: "Skeleton Army",
    26000013: "Mini P.E.K.K.A",  26000014: "Musketeer",        26000015: "Dragon",
    26000016: "Fireball",         26000017: "Arrows",           26000018: "Giant Snowball",
    26000019: "Rocket",           26000020: "Goblin Barrel",    26000021: "Freeze",
    26000022: "Mirror",           26000023: "Lightning",        26000024: "Zap",
    26000025: "Bomb Tower",       26000026: "Inferno Tower",    26000027: "Tesla",
    26000028: "Cannon",           26000029: "Mortar",           26000030: "X-Bow",
    26000031: "Tombstone",        26000032: "Goblin Hut",       26000033: "Barbarian Hut",
    26000034: "Elixir Collector", 26000035: "Heal Spirit",      26000036: "Ice Spirit",
    26000037: "Fire Spirit",      26000038: "Goblin Giant",     26000039: "Electro Giant",
    26000040: "Ice Golem",        26000041: "Mega Minion",      26000042: "Dart Goblin",
    26000043: "Skeleton",         26000044: "Bats",             26000045: "Guards",
    26000046: "Three Musketeers", 26000047: "Hunter",           26000048: "Executioner",
    26000049: "Bandit",           26000050: "Flying Machine",   26000051: "Wall Breakers",
    26000052: "Royal Hogs",       26000053: "Goblin Cage",      26000054: "Electro Wizard",
    26000055: "Mother Witch",     26000056: "Electro Spirit",   26000057: "Firecracker",
    26000058: "Battle Healer",    26000059: "Fisherman",        26000060: "Ram Rider",
    26000061: "Cannon Cart",      26000062: "Mega Knight",      26000063: "Skeleton Dragon",
    26000064: "Minion Horde",     26000065: "Super Witch",      26000066: "Night Witch",
    26000067: "Lava Hound",       26000068: "Ice Wizard",       26000069: "Princess",
    26000070: "Dark Prince",      26000071: "Bowler",           26000072: "Barbarian King",
    26000073: "Royal Giant",      26000074: "Elite Barbarians",  26000075: "Baby Dragon",
    26000083: "Magic Archer",     26000084: "Miner",            26000085: "Sparky",
    27000000: "Goblin",           27000001: "Golem",            27000002: "Graveyard",
    27000003: "Electro Dragon",   27000004: "Giant Skeleton",   27000005: "Hog Rider",
    27000006: "Inferno Dragon",   27000007: "Lumberjack",       27000008: "Royal Ghost",
    27000009: "Skeleton King",    27000010: "Monk",             27000011: "Archer Queen",
    27000012: "Golden Knight",    27000013: "Mighty Miner",     27000014: "Skeleton Guard",
    28000000: "Rage",             28000001: "Clone",            28000002: "Poison",
    28000003: "Earthquake",       28000004: "Tornado",          28000005: "Barbarian Barrel",
    28000006: "Log",              28000007: "Heal",             28000008: "Royal Delivery",
    28000009: "Giant Chest",      28000010: "Lightning",        28000011: "Goblin Drill",
    28000012: "Mega Lightning",   28000013: "Void",             28000014: "Super Witch Rage",
    28000015: "Little Prince",    28000016: "Phoenix",          28000017: "Monk Shield",
    28000018: "Electro Wizard",   28000019: "E-Barbs",
}

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

P1_CARDS = list(range(5, 13))
P2_CARDS = list(range(16, 24))
ALL_CARD_COLS = P1_CARDS + P2_CARDS


# ============================================================
# Feature building (mirrors ai.py exactly)
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

def apply_pair_synergy(df, synergy_map, global_wr):
    card_arr = df[P1_CARDS].fillna(0).astype(np.int64).values
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

def build_card_onehot(df, all_cards_universe, prefix):
    src  = P1_CARDS if prefix == "p1" else P2_CARDS
    arr  = df[src].fillna(0).astype(np.int64).values
    univ = np.array(all_cards_universe)
    oh   = (arr[:, :, None] == univ[None, None, :]).any(axis=1).astype(np.float32)
    return pd.DataFrame(oh, columns=[f"{prefix}_has_{c}" for c in univ])


# ============================================================
# Build a matchup DataFrame row from two card lists
# ============================================================
def make_matchup_df(p1_cards, p2_cards, p1_trophies=5000, p2_trophies=5000):
    """
    p1_cards / p2_cards: lists of 8 card IDs
    Returns a single-row DataFrame with the same column structure as the training data.
    """
    assert len(p1_cards) == 8 and len(p2_cards) == 8, "Each deck must have exactly 8 cards."
    row = {i: np.nan for i in range(24)}
    # cols 0-4: p1 data (0=tag,1=name,2=clan,3=trophies,4=win_label)
    row[3] = p1_trophies
    row[4] = 0  # placeholder label
    for i, c in enumerate(p1_cards):
        row[P1_CARDS[i]] = c
    # cols 13-23: p2 data (13=tag,...,14=trophies,...)
    row[14] = p2_trophies
    for i, c in enumerate(p2_cards):
        row[P2_CARDS[i]] = c
    return pd.DataFrame([row])


def predict_win_prob(p1_cards, p2_cards, model, meta,
                     p1_trophies=5000, p2_trophies=5000):
    """Returns P(player-1 wins) given two 8-card decks."""
    df = make_matchup_df(p1_cards, p2_cards, p1_trophies, p2_trophies)

    X_base = build_features(df)
    X_wr   = apply_card_winrate(df, meta["winrate_map"], meta["global_wr"])
    X_syn  = apply_pair_synergy(df, meta["synergy_map"], meta["syn_global_wr"]).to_frame()
    X_p1oh = build_card_onehot(df, meta["all_cards_universe"], "p1")
    X_p2oh = build_card_onehot(df, meta["all_cards_universe"], "p2")

    X = pd.concat([X_base, X_wr, X_syn, X_p1oh, X_p2oh], axis=1)
    X.columns = [str(c) for c in X.columns]
    X = X.reindex(columns=meta["feature_columns"], fill_value=0)

    cont = meta["continuous_cols"]
    X[cont] = meta["scaler"].transform(X[cont])

    dmat = xgb.DMatrix(X.values.astype(np.float32))
    return float(model.predict(dmat)[0])


# ============================================================
# Counter-deck search  (greedy hill-climb + random restarts)
# ============================================================
def suggest_counter_deck(
    opponent_deck,
    model,
    meta,
    p1_trophies=5000,
    p2_trophies=5000,
    n_restarts=20,
    n_swaps=300,
    top_k=3,
):
    """
    Given an 8-card opponent deck, find a counter deck that maximises
    P(counter_deck wins) = P(player-2 wins).

    Strategy:
      - We want p2 to win, so we optimise P(p2 wins) = 1 - predict_win_prob(counter, opponent)
      - Hill-climb with random card swaps; repeat with multiple random restarts.

    Returns:
        List of top_k dicts: {deck, win_prob_vs_opponent, avg_elixir}
    """
    card_pool = [c for c in meta["all_cards_universe"] if c not in opponent_deck]
    if len(card_pool) < 8:
        raise ValueError("Not enough cards in universe to build a counter deck.")

    def score(counter):
        # probability that 'counter' (playing as p2) beats opponent (p1)
        p = predict_win_prob(opponent_deck, counter, model, meta, p1_trophies, p2_trophies)
        return 1.0 - p   # we want p2 win prob

    results = []

    for _ in range(n_restarts):
        # Random starting deck (no duplicate cards, none from opponent)
        current = random.sample(card_pool, 8)
        current_score = score(current)

        for _ in range(n_swaps):
            # Pick a random card in current deck to swap
            swap_out_idx = random.randrange(8)
            available    = [c for c in card_pool if c not in current]
            if not available:
                break
            swap_in = random.choice(available)
            candidate = current[:]
            candidate[swap_out_idx] = swap_in
            cand_score = score(candidate)
            if cand_score > current_score:
                current       = candidate
                current_score = cand_score

        avg_elixir = np.mean([CARD_ELIXIR.get(c, DEFAULT_ELIXIR) for c in current])
        results.append({
            "deck":                sorted(current),
            "p2_win_prob":         round(current_score, 4),
            "avg_elixir":          round(avg_elixir, 2),
        })

    # De-duplicate by deck, keep top_k
    seen   = set()
    unique = []
    for r in sorted(results, key=lambda x: x["p2_win_prob"], reverse=True):
        key = tuple(r["deck"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
        if len(unique) == top_k:
            break

    return unique


# ============================================================
# Pretty print
# ============================================================
def print_counter_decks(opponent_deck, results):
    print("\n" + "="*60)
    print("OPPONENT DECK:")
    for cid in opponent_deck:
        name   = CARD_NAMES.get(cid, f"Card#{cid}")
        elixir = CARD_ELIXIR.get(cid, DEFAULT_ELIXIR)
        print(f"  [{elixir}💧] {name}  (id={cid})")
    opp_avg = np.mean([CARD_ELIXIR.get(c, DEFAULT_ELIXIR) for c in opponent_deck])
    print(f"  Avg elixir: {opp_avg:.2f}")

    print("\nSUGGESTED COUNTER DECKS:")
    for rank, r in enumerate(results, 1):
        print(f"\n  ── Counter #{rank}  |  Win probability: {r['p2_win_prob']*100:.1f}%  |  Avg elixir: {r['avg_elixir']}")
        for cid in r["deck"]:
            name   = CARD_NAMES.get(cid, f"Card#{cid}")
            elixir = CARD_ELIXIR.get(cid, DEFAULT_ELIXIR)
            print(f"    [{elixir}💧] {name}  (id={cid})")
    print("="*60 + "\n")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # ── Load model & metadata ──────────────────────────────
    print("Loading model...")
    model = xgb.Booster()
    model.load_model("clash_model_robust.json")
    with open("clash_metadata_robust.pkl", "rb") as f:
        meta = pickle.load(f)
    print("Model loaded.\n")

    # ── Define the opponent deck you want to counter ───────
    # Replace these IDs with any 8 cards from the card universe.
    # Example: a classic Hog Rider beatdown deck
    opponent_deck = [
        27000005,   # Hog Rider
        26000014,   # Musketeer
        26000027,   # Tesla
        26000011,   # Valkyrie
        26000017,   # Arrows
        26000035,   # Heal Spirit
        26000003,   # Goblins
        28000006,   # Log
    ]

    # ── Search for counter deck ────────────────────────────
    print("Searching for counter decks (this may take ~30 seconds)...")
    counter_decks = suggest_counter_deck(
        opponent_deck  = opponent_deck,
        model          = model,
        meta           = meta,
        p1_trophies    = 5000,   # opponent's trophy range
        p2_trophies    = 5000,   # your trophy range
        n_restarts     = 20,     # more restarts = better quality, slower
        n_swaps        = 300,    # hill-climb iterations per restart
        top_k          = 3,      # how many counter suggestions to return
    )

    print_counter_decks(opponent_deck, counter_decks)

    # ── Quick sanity check: direct win probability ─────────
    # You can also use predict_win_prob directly:
    # prob = predict_win_prob(my_deck, opponent_deck, model, meta)
    # print(f"My deck win probability: {prob*100:.1f}%")
