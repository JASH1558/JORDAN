import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import random

st.set_page_config(
    page_title="Clash Royale · Arena AI",
    page_icon="⚔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# CARD DATABASE
# ─────────────────────────────────────────────────────────────
CARD_DB = {
    26000000:("Knight",3,"Common"),        26000001:("Archers",3,"Common"),
    26000003:("Goblins",2,"Common"),       26000004:("Giant",5,"Rare"),
    26000005:("P.E.K.K.A",7,"Epic"),       26000006:("Minions",3,"Common"),
    26000007:("Balloon",5,"Epic"),         26000008:("Witch",5,"Epic"),
    26000009:("Barbarians",5,"Common"),    26000010:("Goblin Barrel",3,"Epic"),
    26000011:("Skeleton Army",3,"Epic"),   26000012:("Bomber",2,"Common"),
    26000013:("Musketeer",4,"Rare"),       26000014:("Mini P.E.K.K.A",4,"Rare"),
    26000015:("Sparky",6,"Legendary"),     26000016:("Ice Spirit",1,"Common"),
    26000017:("Spear Goblins",2,"Common"), 26000018:("Valkyrie",4,"Rare"),
    26000019:("Hog Rider",4,"Rare"),       26000020:("Baby Dragon",4,"Epic"),
    26000021:("Zap",2,"Common"),           26000022:("X-Bow",6,"Epic"),
    26000023:("Mortar",4,"Common"),        26000024:("Inferno Tower",5,"Rare"),
    26000025:("Cannon",3,"Common"),        26000026:("Goblin Hut",5,"Rare"),
    26000027:("Tesla",4,"Rare"),           26000028:("Tombstone",3,"Rare"),
    26000029:("Bomb Tower",4,"Rare"),      26000030:("Fireball",4,"Rare"),
    26000031:("Lightning",6,"Epic"),       26000032:("Arrows",3,"Common"),
    26000033:("Rocket",6,"Rare"),          26000035:("Mirror",1,"Epic"),
    26000036:("Rage",3,"Epic"),            26000037:("Freeze",4,"Epic"),
    26000038:("Poison",4,"Epic"),          26000039:("Goblin Cage",4,"Rare"),
    26000040:("Skeletons",1,"Common"),     26000041:("Miner",3,"Legendary"),
    26000042:("3 Musketeers",9,"Rare"),    26000044:("Minion Horde",5,"Common"),
    26000045:("Bats",2,"Common"),          26000046:("Royal Hogs",5,"Rare"),
    26000047:("Royal Giant",6,"Common"),   26000048:("Heal Spirit",1,"Common"),
    26000049:("Flying Machine",4,"Rare"),  26000050:("Wall Breakers",2,"Rare"),
    26000052:("Night Witch",4,"Legendary"),26000054:("Goblin Giant",6,"Epic"),
    26000055:("Fisherman",3,"Legendary"),  26000056:("Magic Archer",4,"Legendary"),
    26000057:("Electro Dragon",5,"Epic"),  26000058:("Firecracker",3,"Common"),
    26000060:("Electro Spirit",1,"Common"),26000061:("Electro Giant",7,"Epic"),
    26000062:("Ice Golem",2,"Rare"),       26000063:("Monk",5,"Legendary"),
    26000064:("Skeleton King",4,"Legendary"),26000065:("Archer Queen",5,"Legendary"),
    26000066:("Golden Knight",4,"Legendary"),26000067:("Skeleton Dragons",4,"Rare"),
    26000068:("Mother Witch",4,"Legendary"),26000069:("Electro Wizard",4,"Legendary"),
    26000070:("Little Prince",3,"Legendary"),26000071:("Cannon Cart",5,"Rare"),
    26000072:("Mega Knight",7,"Legendary"),26000073:("Ram Rider",5,"Legendary"),
    26000074:("Zappies",5,"Rare"),         26000075:("Rascals",5,"Rare"),
    26000083:("Goblin Demolisher",5,"Rare"),26000085:("Goblin Machine",5,"Epic"),
    27000000:("Golem",8,"Epic"),           27000001:("Lumberjack",4,"Legendary"),
    27000003:("Inferno Dragon",4,"Legendary"),27000004:("Dark Prince",4,"Epic"),
    27000005:("Prince",5,"Epic"),          27000006:("Giant Skeleton",6,"Epic"),
    27000007:("Bowler",5,"Epic"),          27000008:("Lava Hound",7,"Legendary"),
    27000009:("Ice Wizard",3,"Legendary"), 27000010:("Princess",3,"Legendary"),
    27000011:("Graveyard",5,"Legendary"),  27000012:("Bandit",3,"Legendary"),
    27000013:("Royal Ghost",3,"Legendary"),27000014:("Barbarian Barrel",2,"Epic"),
    28000004:("Wizard",5,"Epic"),          28000010:("Mega Minion",3,"Rare"),
}

# ── EXACT CARD_ELIXIR from ai.py (do NOT derive from CARD_DB — values differ!) ──
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

RARITY_COLOR = {
    "Common":   "#9CA3AF",
    "Rare":     "#60A5FA",
    "Epic":     "#C084FC",
    "Legendary":"#FBBF24",
}
RARITY_GLOW = {
    "Common":   "rgba(156,163,175,0.3)",
    "Rare":     "rgba(96,165,250,0.4)",
    "Epic":     "rgba(192,132,252,0.4)",
    "Legendary":"rgba(251,191,36,0.5)",
}
RARITY_BG = {
    "Common":   "rgba(156,163,175,0.08)",
    "Rare":     "rgba(96,165,250,0.1)",
    "Epic":     "rgba(192,132,252,0.1)",
    "Legendary":"rgba(251,191,36,0.12)",
}

CARD_IMAGE_SLUG = {
    "Knight":"knight","Archers":"archers","Goblins":"goblins","Giant":"giant",
    "P.E.K.K.A":"pekka","Minions":"minions","Balloon":"balloon","Witch":"witch",
    "Barbarians":"barbarians","Goblin Barrel":"goblin-barrel","Skeleton Army":"skeleton-army",
    "Bomber":"bomber","Musketeer":"musketeer","Mini P.E.K.K.A":"mini-pekka","Sparky":"sparky",
    "Ice Spirit":"ice-spirit","Spear Goblins":"spear-goblins","Valkyrie":"valkyrie",
    "Hog Rider":"hog-rider","Baby Dragon":"baby-dragon","Zap":"zap","X-Bow":"x-bow",
    "Mortar":"mortar","Inferno Tower":"inferno-tower","Cannon":"cannon","Goblin Hut":"goblin-hut",
    "Tesla":"tesla","Tombstone":"tombstone","Bomb Tower":"bomb-tower","Fireball":"fireball",
    "Lightning":"lightning","Arrows":"arrows","Rocket":"rocket","Mirror":"mirror","Rage":"rage",
    "Freeze":"freeze","Poison":"poison","Goblin Cage":"goblin-cage","Skeletons":"skeletons",
    "Miner":"miner","3 Musketeers":"three-musketeers","Minion Horde":"minion-horde","Bats":"bats",
    "Royal Hogs":"royal-hogs","Royal Giant":"royal-giant","Heal Spirit":"heal-spirit",
    "Flying Machine":"flying-machine","Wall Breakers":"wall-breakers","Night Witch":"night-witch",
    "Goblin Giant":"goblin-giant","Fisherman":"fisherman","Magic Archer":"magic-archer",
    "Electro Dragon":"electro-dragon","Firecracker":"firecracker","Electro Spirit":"electro-spirit",
    "Electro Giant":"electro-giant","Ice Golem":"ice-golem","Monk":"monk",
    "Skeleton King":"skeleton-king","Archer Queen":"archer-queen","Golden Knight":"golden-knight",
    "Skeleton Dragons":"skeleton-dragons","Mother Witch":"mother-witch",
    "Electro Wizard":"electro-wizard","Little Prince":"little-prince","Cannon Cart":"cannon-cart",
    "Mega Knight":"mega-knight","Ram Rider":"ram-rider","Zappies":"zappies","Rascals":"rascals",
    "Goblin Demolisher":"goblin-demolisher","Goblin Machine":"goblin-machine","Golem":"golem",
    "Lumberjack":"lumberjack","Inferno Dragon":"inferno-dragon","Dark Prince":"dark-prince",
    "Prince":"prince","Giant Skeleton":"giant-skeleton","Bowler":"bowler","Lava Hound":"lava-hound",
    "Ice Wizard":"ice-wizard","Princess":"princess","Graveyard":"graveyard","Bandit":"bandit",
    "Royal Ghost":"royal-ghost","Barbarian Barrel":"barbarian-barrel","Wizard":"wizard",
    "Mega Minion":"mega-minion",
}
CARD_FALLBACK = {
    "Knight":"Kn","Archers":"Ar","Goblins":"Go","Giant":"Gi","P.E.K.K.A":"PK",
    "Minions":"Mi","Balloon":"Ba","Witch":"Wi","Barbarians":"Br","Goblin Barrel":"GB",
    "Skeleton Army":"SA","Bomber":"Bo","Musketeer":"Mu","Mini P.E.K.K.A":"MP","Sparky":"Sp",
    "Ice Spirit":"IS","Spear Goblins":"SG","Valkyrie":"Va","Hog Rider":"HR","Baby Dragon":"BD",
    "Zap":"Zp","X-Bow":"XB","Mortar":"Mo","Inferno Tower":"IT","Cannon":"Ca","Goblin Hut":"GH",
    "Tesla":"Te","Tombstone":"Ts","Bomb Tower":"BT","Fireball":"FB","Lightning":"Li",
    "Arrows":"Aw","Rocket":"Ro","Mirror":"Mr","Rage":"Rg","Freeze":"Fr","Poison":"Po",
    "Goblin Cage":"GC","Skeletons":"Sk","Miner":"Mn","3 Musketeers":"3M","Minion Horde":"MH",
    "Bats":"Bt","Royal Hogs":"RH","Royal Giant":"RG","Heal Spirit":"HS","Flying Machine":"FM",
    "Wall Breakers":"WB","Night Witch":"NW","Goblin Giant":"GG","Fisherman":"Fi",
    "Magic Archer":"MA","Electro Dragon":"ED","Firecracker":"FC","Electro Spirit":"ES",
    "Electro Giant":"EG","Ice Golem":"IG","Monk":"Mk","Skeleton King":"SK","Archer Queen":"AQ",
    "Golden Knight":"GK","Skeleton Dragons":"SD","Mother Witch":"MW","Electro Wizard":"EW",
    "Little Prince":"LP","Cannon Cart":"CC","Mega Knight":"MK","Ram Rider":"RR",
    "Zappies":"Za","Rascals":"Rs","Goblin Demolisher":"GD","Goblin Machine":"GM","Golem":"Gl",
    "Lumberjack":"LJ","Inferno Dragon":"ID","Dark Prince":"DP","Prince":"Pr",
    "Giant Skeleton":"GS","Bowler":"Bw","Lava Hound":"LH","Ice Wizard":"IW","Princess":"Ps",
    "Graveyard":"Gy","Bandit":"Bn","Royal Ghost":"RG","Barbarian Barrel":"BB",
    "Wizard":"Wz","Mega Minion":"MM",
}

def card_img_html(name, size=56, extra_style=""):
    slug = CARD_IMAGE_SLUG.get(name, "")
    abbr = CARD_FALLBACK.get(name, name[:2])
    rarity = CARD_DB.get(next((k for k,v in CARD_DB.items() if v[0]==name), 0), ("",0,"Common"))[2] if name else "Common"
    rc  = RARITY_COLOR.get(rarity, "#9CA3AF")
    rglow = RARITY_GLOW.get(rarity, "rgba(156,163,175,0.2)")
    rbg  = RARITY_BG.get(rarity, "rgba(156,163,175,0.06)")

    # Styled text fallback tile
    fallback_div = (
        f'<div style="width:{size}px;height:{size}px;display:flex;flex-direction:column;'
        f'align-items:center;justify-content:center;'
        f'background:linear-gradient(145deg,{rbg},{rbg.replace("0.06","0.02").replace("0.08","0.03").replace("0.1","0.04").replace("0.12","0.05")});'
        f'border:1px solid {rc}44;border-radius:{max(6,int(size*0.14))}px;'
        f'box-shadow:inset 0 1px 0 rgba(255,255,255,0.06);">'
        f'<span style="font-family:Barlow Condensed,sans-serif;font-size:{max(9,int(size*0.22))}px;'
        f'font-weight:900;font-style:italic;color:{rc};letter-spacing:0.03em;'
        f'text-shadow:0 0 8px {rglow};">{abbr}</span>'
        f'</div>'
    )

    if not slug:
        return fallback_div

    # Three CDN sources in priority order:
    # 1. RoyaleAPI GitHub raw (most stable)
    # 2. RoyaleAPI CDN
    # 3. cr-api-assets GitHub pages
    src1 = f"https://raw.githubusercontent.com/RoyaleAPI/cr-api-assets/master/cards/{slug}.png"
    src2 = f"https://cdn.royaleapi.com/static/img/cards-150/{slug}.png"
    src3 = f"https://royaleapi.github.io/cr-api-assets/cards/{slug}.png"

    uid = f"img_{slug}_{size}"
    abbr_js = abbr.replace("'", "\\'")
    rc_js   = rc
    rglow_js = rglow
    rbg_js  = rbg

    return (
        f'<img id="{uid}" src="{src1}" '
        f'style="width:{size}px;height:{size}px;object-fit:contain;image-rendering:auto;{extra_style}" '
        f'data-s2="{src2}" data-s3="{src3}" '
        f'data-abbr="{abbr_js}" data-rc="{rc_js}" data-rbg="{rbg_js}" data-rglow="{rglow_js}" '
        f'data-size="{size}" '
        f'onerror="(function(el){{'
        f'  var s2=el.getAttribute(\'data-s2\'),s3=el.getAttribute(\'data-s3\');'
        f'  var tried=el.getAttribute(\'data-tried\')||\'0\';'
        f'  if(tried===\'0\'){{el.setAttribute(\'data-tried\',\'1\');el.src=s2;return;}}'
        f'  if(tried===\'1\'){{el.setAttribute(\'data-tried\',\'2\');el.src=s3;return;}}'
        f'  var sz=el.getAttribute(\'data-size\');'
        f'  var ab=el.getAttribute(\'data-abbr\');'
        f'  var rc=el.getAttribute(\'data-rc\');'
        f'  var rbg=el.getAttribute(\'data-rbg\');'
        f'  var rg=el.getAttribute(\'data-rglow\');'
        f'  var r=Math.max(6,Math.round(sz*0.14));'
        f'  var fs=Math.max(9,Math.round(sz*0.22));'
        f'  var d=document.createElement(\'div\');'
        f'  d.style.cssText=\'width:\'+sz+\'px;height:\'+sz+\'px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:linear-gradient(145deg,\'+rbg+\',rgba(0,0,0,0.1));border:1px solid \'+rc+\'44;border-radius:\'+r+\'px;box-shadow:inset 0 1px 0 rgba(255,255,255,0.06);\';'
        f'  var sp=document.createElement(\'span\');'
        f'  sp.textContent=ab;'
        f'  sp.style.cssText=\'font-family:Barlow Condensed,sans-serif;font-size:\'+fs+\'px;font-weight:900;font-style:italic;color:\'+rc+\';letter-spacing:0.03em;text-shadow:0 0 8px \'+rg+\';\';'
        f'  d.appendChild(sp);'
        f'  el.parentNode.insertBefore(d,el);el.style.display=\'none\';'
        f'}})(this)">'
    )

P1_CARDS = list(range(5,13))
P2_CARDS = list(range(16,24))
RARITY_ORDER     = ["Legendary","Epic","Rare","Common"]
ALL_CARDS_SORTED = sorted(CARD_DB.items(), key=lambda x:(RARITY_ORDER.index(x[1][2]), x[1][0]))
CARDS_PER_PAGE   = 32

# ═══════════════════════════════════════════════════════════════
# SUPERCELL-INSPIRED DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700;900&family=Cinzel:wght@400;600;700;900&family=Crimson+Pro:ital,wght@0,300;0,400;0,600;0,700;1,400;1,600;1,700&family=Barlow+Condensed:ital,wght@0,400;0,600;0,700;0,800;0,900;1,600;1,700;1,800;1,900&display=swap');

/* ── DESIGN TOKENS ── */
:root {
  --ink:        #0A0812;
  --deep:       #0F0C1E;
  --void:       #14102A;
  --surface:    #1C1836;
  --panel:      #231F42;
  --lifted:     #2A264F;
  --gold:       #F5C842;
  --gold-dim:   #C89A1E;
  --gold-glow:  rgba(245,200,66,0.25);
  --gold-light: #FDE68A;
  --crimson:    #E8365D;
  --crimson-dim:#B02247;
  --sapphire:   #3B9EFF;
  --sapphire-dim:#1A6ECC;
  --emerald:    #22D07A;
  --violet:     #9F6EFF;
  --border:     rgba(255,255,255,0.06);
  --border-gold:rgba(245,200,66,0.25);
  --text:       #F2EFFF;
  --text-soft:  #B8B0D8;
  --text-muted: #6B6490;
  --text-dim:   #3D3660;
  --font-display:'Cinzel Decorative', serif;
  --font-title: 'Cinzel', serif;
  --font-body:  'Crimson Pro', serif;
  --font-label: 'Barlow Condensed', sans-serif;
  --r:          12px;
  --r-sm:       8px;
  --r-lg:       18px;
  --r-xl:       24px;
  --sh:         0 4px 24px rgba(0,0,0,0.5);
  --sh-gold:    0 0 24px rgba(245,200,66,0.2);
  --sh-crimson: 0 0 24px rgba(232,54,93,0.3);
  --t:          200ms cubic-bezier(0.4,0,0.2,1);
}

/* ── GLOBAL RESET ── */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
section.main, .main .block-container,
[data-testid="stAppViewBlockContainer"],
[data-testid="stBottomBlockContainer"] {
  background: transparent !important;
}
[data-testid="stAppViewContainer"] {
  background: var(--deep) !important;
  font-family: var(--font-body) !important;
}
* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: var(--font-body); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold); }

/* ══════════════════════════════════════════════
   BACKGROUND — layered medieval atmosphere
   ══════════════════════════════════════════════ */
#bg-canvas {
  position: fixed; inset: 0; z-index: 0;
  pointer-events: none;
}
.bg-overlay {
  position: fixed; inset: 0; z-index: 1; pointer-events: none;
  background:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(245,200,66,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 100% 100%, rgba(59,158,255,0.04) 0%, transparent 50%),
    radial-gradient(ellipse 50% 30% at 0% 80%, rgba(159,110,255,0.04) 0%, transparent 50%);
}

/* ══════════════════════════════════════════════
   HEADER / NAV
   ══════════════════════════════════════════════ */
.cr-header {
  position: sticky; top: 0; z-index: 200;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 40px; height: 68px;
  background: rgba(10,8,18,0.92);
  backdrop-filter: blur(24px) saturate(1.4);
  border-bottom: 1px solid rgba(245,200,66,0.12);
  box-shadow: 0 1px 0 rgba(245,200,66,0.05), 0 4px 32px rgba(0,0,0,0.6);
}

.cr-logo {
  display: flex; align-items: center; gap: 14px;
}
.cr-logo-emblem {
  width: 40px; height: 40px;
  background: linear-gradient(145deg, #F5C842, #C89A1E);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 16px rgba(245,200,66,0.35), inset 0 1px 0 rgba(255,255,255,0.2);
  position: relative;
}
.cr-logo-emblem::after {
  content: '';
  position: absolute; inset: 2px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.15);
}
.cr-logo-text { font-family: var(--font-display); }
.cr-logo-name {
  display: block; font-size: 13px; font-weight: 700;
  color: var(--gold); letter-spacing: 0.08em;
  text-shadow: 0 0 20px rgba(245,200,66,0.4);
}
.cr-logo-tagline {
  display: block; font-size: 9px; font-weight: 400;
  color: var(--text-muted); letter-spacing: 0.2em;
  text-transform: uppercase; font-family: var(--font-label);
}

.cr-nav {
  display: flex; align-items: center; gap: 2px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 4px;
}
.cr-nav-btn {
  font-family: var(--font-label) !important;
  font-size: 13px !important; font-weight: 700 !important;
  letter-spacing: 0.08em !important; text-transform: uppercase !important;
  padding: 7px 18px !important; border-radius: var(--r-sm) !important;
  color: var(--text-muted) !important; background: transparent !important;
  border: none !important; cursor: pointer;
  transition: all var(--t) !important;
  display: flex !important; align-items: center !important; gap: 7px !important;
  white-space: nowrap !important;
}
.cr-nav-btn:hover { color: var(--text) !important; background: rgba(255,255,255,0.05) !important; }
.cr-nav-btn.active {
  color: var(--gold) !important;
  background: rgba(245,200,66,0.1) !important;
  border: 1px solid rgba(245,200,66,0.2) !important;
  box-shadow: 0 0 12px rgba(245,200,66,0.1) !important;
}

.cr-header-badges {
  display: flex; align-items: center; gap: 10px;
}
.cr-badge {
  font-family: var(--font-label);
  font-size: 12px; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 6px 14px; border-radius: 99px;
  border: 1px solid var(--border-gold);
  color: var(--gold); background: rgba(245,200,66,0.06);
}
.cr-battles-badge {
  display: flex; align-items: center; gap: 7px;
  font-family: var(--font-label);
  font-size: 12px; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 5px 12px; border-radius: 99px;
  border: 1px solid rgba(255,255,255,0.1);
  color: var(--text-soft); background: rgba(255,255,255,0.04);
}

/* ══════════════════════════════════════════════
   MAIN WRAPPER
   ══════════════════════════════════════════════ */
.cr-main {
  position: relative; z-index: 10;
  padding: 32px 40px;
  max-width: 1700px; margin: 0 auto;
}

/* ══════════════════════════════════════════════
   PAGE HEADINGS
   ══════════════════════════════════════════════ */
.page-heading {
  margin-bottom: 28px;
}
.page-title {
  font-family: var(--font-title);
  font-size: 32px; font-weight: 900;
  color: var(--gold); letter-spacing: -0.01em;
  text-shadow: 0 0 32px rgba(245,200,66,0.3);
  margin-bottom: 4px;
}
.page-sub {
  font-family: var(--font-label);
  font-size: 14px; font-weight: 500;
  color: var(--text-muted); letter-spacing: 0.05em;
  text-transform: uppercase;
}

/* ══════════════════════════════════════════════
   SECTION LABELS
   ══════════════════════════════════════════════ */
.section-label {
  font-family: var(--font-label);
  font-size: 10px; font-weight: 800;
  color: var(--text-muted); letter-spacing: 0.2em;
  text-transform: uppercase; margin-bottom: 12px;
  display: flex; align-items: center; gap: 8px;
}
.section-label::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border-gold), transparent);
}

/* ══════════════════════════════════════════════
   DECK PANEL
   ══════════════════════════════════════════════ */
.deck-panel {
  background: linear-gradient(160deg, var(--panel) 0%, var(--void) 100%);
  border: 1px solid var(--border);
  border-radius: var(--r-xl);
  padding: 22px;
  box-shadow: var(--sh), inset 0 1px 0 rgba(255,255,255,0.04);
  position: relative; overflow: hidden;
}
.deck-panel::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(245,200,66,0.2), transparent);
}
.deck-panel.complete {
  border-color: rgba(34,208,122,0.25);
  box-shadow: var(--sh), 0 0 32px rgba(34,208,122,0.06), inset 0 1px 0 rgba(34,208,122,0.08);
}
.deck-panel.complete::before {
  background: linear-gradient(90deg, transparent, rgba(34,208,122,0.3), transparent);
}
.deck-panel.p1-panel::after {
  content: '';
  position: absolute; top: -40px; right: -40px;
  width: 120px; height: 120px; border-radius: 50%;
  background: radial-gradient(circle, rgba(59,158,255,0.06) 0%, transparent 70%);
  pointer-events: none;
}
.deck-panel.p2-panel::after {
  content: '';
  position: absolute; top: -40px; right: -40px;
  width: 120px; height: 120px; border-radius: 50%;
  background: radial-gradient(circle, rgba(232,54,93,0.06) 0%, transparent 70%);
  pointer-events: none;
}

/* ── DECK SLOTS ── */
.deck-slots {
  display: grid; grid-template-columns: repeat(4,1fr); gap: 6px;
  margin: 14px 0;
}
.dslot {
  border-radius: var(--r-sm); aspect-ratio: 1;
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; position: relative; overflow: hidden;
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.06);
  transition: all var(--t); padding: 4px;
}
.dslot.filled {
  background: rgba(255,255,255,0.04);
  border-color: rgba(255,255,255,0.1);
}
.dslot.filled:hover {
  transform: translateY(-3px) scale(1.03);
  box-shadow: 0 6px 20px rgba(0,0,0,0.4);
  border-color: rgba(245,200,66,0.25);
}
.dslot-empty-icon {
  width: 22px; height: 22px; border-radius: 50%;
  border: 1.5px dashed var(--text-dim);
  display: flex; align-items: center; justify-content: center;
  opacity: 0.5;
}
.dslot-name {
  font-family: var(--font-label);
  font-size: 8.5px; font-weight: 700;
  color: var(--text-muted); text-align: center;
  line-height: 1.2; margin-top: 3px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  max-width: 100%; letter-spacing: 0.02em;
}
.dslot-elix {
  position: absolute; top: 3px; left: 3px;
  background: linear-gradient(135deg, #7C3AED, #5B21B6);
  color: #fff; font-family: var(--font-label);
  font-size: 8px; font-weight: 800;
  width: 14px; height: 14px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}

/* ── ELIXIR GAUGE ── */
.elixir-gauge { margin: 8px 0; }
.elixir-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 5px;
}
.elixir-label {
  font-family: var(--font-label);
  font-size: 10px; font-weight: 700;
  color: var(--text-muted); letter-spacing: 0.12em; text-transform: uppercase;
}
.elixir-val {
  font-family: var(--font-label);
  font-size: 14px; font-weight: 900; color: #C084FC;
  letter-spacing: 0.05em;
}
.elixir-track {
  height: 6px; border-radius: 99px;
  background: rgba(255,255,255,0.05);
  overflow: hidden; position: relative;
}
.elixir-fill {
  height: 100%; border-radius: 99px;
  background: linear-gradient(90deg, #7C3AED, #C084FC, #E879F9);
  box-shadow: 0 0 8px rgba(192,132,252,0.4);
  transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
}

/* ── PROGRESS ── */
.count-row {
  display: flex; align-items: center; justify-content: space-between;
  margin: 10px 0 5px;
}
.count-label {
  font-family: var(--font-label);
  font-size: 10px; font-weight: 700;
  color: var(--text-muted); letter-spacing: 0.12em; text-transform: uppercase;
}
.count-val {
  font-family: var(--font-label);
  font-size: 13px; font-weight: 900; letter-spacing: 0.04em;
}
.count-track {
  height: 4px; background: rgba(255,255,255,0.04);
  border-radius: 99px; overflow: hidden;
}
.count-fill {
  height: 100%; border-radius: 99px;
  transition: width 0.4s ease;
}

/* ── TROPHY ── */
.trophy-badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--font-label);
  font-size: 11px; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase;
  border-radius: 99px; padding: 5px 12px;
  margin-top: 10px;
}

/* ── RARITY TAGS ── */
.r-tag {
  display: inline-flex; align-items: center;
  font-family: var(--font-label);
  font-size: 9px; font-weight: 800; letter-spacing: 0.08em;
  padding: 2px 7px; border-radius: 5px;
  text-transform: uppercase; margin-right: 4px;
}

/* ══════════════════════════════════════════════
   CARD LIBRARY (CENTER)
   ══════════════════════════════════════════════ */
.library-panel {
  background: linear-gradient(160deg, var(--panel) 0%, var(--void) 100%);
  border: 1px solid var(--border);
  border-radius: var(--r-xl); padding: 20px;
  box-shadow: var(--sh);
  position: relative; overflow: hidden;
}
.library-panel::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(245,200,66,0.15), transparent);
}

/* ── CARD TILES ── */
.ccard {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 10px; padding: 7px 4px 6px;
  text-align: center; position: relative; cursor: pointer;
  display: flex; flex-direction: column; align-items: center;
  transition: all var(--t);
}
.ccard:hover {
  transform: translateY(-4px) scale(1.04);
  box-shadow: 0 8px 24px rgba(0,0,0,0.5);
  background: rgba(255,255,255,0.06);
}
.ccard.in-p1 {
  border: 2px solid #3B9EFF;
  background: rgba(59,158,255,0.07);
  box-shadow: 0 0 14px rgba(59,158,255,0.25), inset 0 0 0 1px rgba(59,158,255,0.15);
}
.ccard.in-p2 {
  border: 2px solid #E8365D;
  background: rgba(232,54,93,0.07);
  box-shadow: 0 0 14px rgba(232,54,93,0.25), inset 0 0 0 1px rgba(232,54,93,0.15);
}
.ccard.in-both {
  border: 2px solid transparent;
  background: rgba(255,255,255,0.04);
  background-clip: padding-box;
  box-shadow: 0 0 0 2px transparent;
  position: relative;
}
.ccard.in-both::after {
  content: '';
  position: absolute; inset: -2px;
  border-radius: 11px;
  background: linear-gradient(90deg, #3B9EFF 50%, #E8365D 50%);
  z-index: -1;
}
.ccard.in-counter {
  border: 2px solid #22D07A;
  background: rgba(34,208,122,0.07);
  box-shadow: 0 0 14px rgba(34,208,122,0.25), inset 0 0 0 1px rgba(34,208,122,0.15);
}
.ccard-elix {
  position: absolute; top: 4px; left: 4px;
  background: linear-gradient(135deg, #7C3AED, #5B21B6);
  color: #fff; font-family: var(--font-label);
  font-size: 8px; font-weight: 900;
  width: 15px; height: 15px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 1px 4px rgba(0,0,0,0.5);
}
.ccard-name {
  font-family: var(--font-label);
  font-size: 9px; font-weight: 700; color: var(--text);
  margin-top: 4px; line-height: 1.2; letter-spacing: 0.02em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%;
}
.ccard-rarity {
  font-family: var(--font-label);
  font-size: 8px; font-weight: 600; color: var(--text-muted);
  margin-top: 1px; letter-spacing: 0.06em;
}
.ccard-badge {
  position: absolute; top: 3px; right: 3px;
  font-family: var(--font-label);
  font-size: 7.5px; font-weight: 900; padding: 1px 5px;
  border-radius: 4px; letter-spacing: 0.06em;
}
.rarity-dot {
  display: inline-block; width: 5px; height: 5px;
  border-radius: 50%; margin-right: 3px;
}

/* ── SEARCH / FILTER ── */
.stTextInput input, .stNumberInput input {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  color: var(--text) !important;
  font-family: var(--font-label) !important;
  font-size: 14px !important; font-weight: 500 !important;
  letter-spacing: 0.04em !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: rgba(245,200,66,0.4) !important;
  box-shadow: 0 0 0 3px rgba(245,200,66,0.08) !important;
  outline: none !important;
}
.stSelectbox [data-baseweb="select"] {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
}
.stSelectbox [data-baseweb="select"] * {
  font-family: var(--font-label) !important;
  font-size: 13px !important; font-weight: 600 !important;
  letter-spacing: 0.05em !important; color: var(--text) !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
.stSelectbox label {
  color: var(--text-muted) !important;
  font-family: var(--font-label) !important;
  font-size: 10px !important; font-weight: 800 !important;
  text-transform: uppercase !important; letter-spacing: 0.14em !important;
}

/* ── PAGINATION ── */
.page-info {
  font-family: var(--font-label);
  font-size: 11px; font-weight: 700; color: var(--text-muted);
  letter-spacing: 0.1em; text-transform: uppercase;
}
.page-dots { display: flex; gap: 4px; align-items: center; }
.pdot {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--text-dim); transition: all var(--t);
}
.pdot.active {
  width: 14px; border-radius: 99px;
  background: var(--gold);
  box-shadow: 0 0 6px rgba(245,200,66,0.4);
}

/* ══════════════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════════════ */
div[data-testid="stButton"] > button {
  font-family: var(--font-label) !important;
  font-size: 13px !important; font-weight: 900 !important;
  letter-spacing: 0.12em !important; text-transform: uppercase !important;
  border-radius: var(--r) !important;
  padding: 11px 22px !important;
  transition: all var(--t) !important;
  border: 1px solid transparent !important;
}
div[data-testid="stButton"] > button {
  background: linear-gradient(160deg, #F5C842, #C89A1E) !important;
  color: #1A1200 !important;
  border-color: rgba(255,220,80,0.5) !important;
  box-shadow: 0 3px 12px rgba(245,200,66,0.35), inset 0 1px 0 rgba(255,255,255,0.2) !important;
  text-shadow: 0 1px 0 rgba(255,255,255,0.2) !important;
}
div[data-testid="stButton"] > button:hover {
  background: linear-gradient(160deg, #FFD95A, #E8B020) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 24px rgba(245,200,66,0.5), inset 0 1px 0 rgba(255,255,255,0.25) !important;
}
div[data-testid="stButton"] > button:active {
  transform: scale(0.97) translateY(0px) !important;
  box-shadow: 0 1px 6px rgba(245,200,66,0.3) !important;
}
div[data-testid="stButton"] > button:disabled {
  background: var(--surface) !important;
  color: var(--text-dim) !important;
  border-color: var(--border) !important;
  box-shadow: none !important; transform: none !important;
  text-shadow: none !important;
}

/* ══════════════════════════════════════════════
   COMPARE BAR
   ══════════════════════════════════════════════ */
.compare-panel {
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border-gold);
  border-radius: var(--r); padding: 14px 18px; margin-top: 14px;
}
.compare-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.compare-name-p1 {
  font-family: var(--font-label); font-size: 12px; font-weight: 800;
  color: var(--sapphire); letter-spacing: 0.08em; text-transform: uppercase;
}
.compare-name-p2 {
  font-family: var(--font-label); font-size: 12px; font-weight: 800;
  color: var(--crimson); letter-spacing: 0.08em; text-transform: uppercase;
}
.compare-count {
  font-family: var(--font-label); font-size: 11px; font-weight: 600;
  color: var(--text-muted); letter-spacing: 0.08em;
}
.compare-track {
  height: 8px; border-radius: 99px;
  overflow: hidden; background: rgba(255,255,255,0.04); display: flex;
}
.compare-p1-fill {
  background: linear-gradient(90deg, #1A6ECC, #3B9EFF);
  transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
}
.compare-p2-fill {
  background: linear-gradient(90deg, #B02247, #E8365D);
  transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
}
.compare-sub {
  display: flex; justify-content: space-between; margin-top: 5px;
  font-family: var(--font-label); font-size: 11px; font-weight: 600;
  color: var(--text-muted); letter-spacing: 0.05em;
}

/* ══════════════════════════════════════════════
   PREDICT BUTTON AREA
   ══════════════════════════════════════════════ */
.predict-hint {
  text-align: center; padding: 10px;
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: var(--r); margin-bottom: 10px;
}
.predict-hint span {
  font-family: var(--font-label); font-size: 12px; font-weight: 600;
  color: var(--text-muted); letter-spacing: 0.06em;
}

/* ══════════════════════════════════════════════
   RESULT PANEL
   ══════════════════════════════════════════════ */
.result-panel {
  background: linear-gradient(150deg, var(--panel) 0%, var(--void) 100%);
  border: 1px solid var(--border);
  border-radius: var(--r-xl); padding: 32px;
  box-shadow: var(--sh);
  position: relative; overflow: hidden;
  animation: resultReveal 0.5s cubic-bezier(0.34,1.56,0.64,1);
}
.result-panel::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.result-panel.win::before {
  background: linear-gradient(90deg, transparent, var(--emerald), transparent);
  box-shadow: 0 0 20px rgba(34,208,122,0.4);
}
.result-panel.loss::before {
  background: linear-gradient(90deg, transparent, var(--crimson), transparent);
}
.result-panel.win { border-color: rgba(34,208,122,0.2); }
.result-panel.loss { border-color: rgba(232,54,93,0.15); }
@keyframes resultReveal {
  from { opacity: 0; transform: translateY(20px) scale(0.97); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

.result-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 24px; }
.result-label {
  font-family: var(--font-label); font-size: 10px; font-weight: 800;
  color: var(--text-muted); letter-spacing: 0.2em; text-transform: uppercase;
  margin-bottom: 6px;
}
.result-verdict {
  font-family: var(--font-title);
  font-size: 26px; font-weight: 900;
  letter-spacing: -0.01em;
}
.result-conf-chip {
  display: inline-flex; align-items: center; gap: 5px;
  font-family: var(--font-label); font-size: 11px; font-weight: 800;
  letter-spacing: 0.1em; text-transform: uppercase;
  border-radius: 99px; padding: 5px 14px;
  margin-top: 8px; border: 1px solid;
}
.result-pct {
  text-align: right;
}
.result-pct-num {
  font-family: var(--font-title);
  font-size: 56px; font-weight: 900; line-height: 1;
  letter-spacing: -0.03em;
}
.result-pct-label {
  font-family: var(--font-label); font-size: 10px; font-weight: 700;
  color: var(--text-muted); letter-spacing: 0.15em; text-transform: uppercase;
  margin-top: 2px;
}

/* ── PROB BAR (result) ── */
.result-bar-wrap { margin: 8px 0 4px; }
.result-bar-labels {
  display: flex; justify-content: space-between; margin-bottom: 7px;
  font-family: var(--font-label); font-size: 12px; font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase;
}
.result-bar-track {
  height: 10px; border-radius: 99px;
  background: rgba(255,255,255,0.04);
  overflow: hidden; display: flex;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.3);
}
.result-bar-p1 {
  background: linear-gradient(90deg, #1A6ECC, #60A5FA);
  box-shadow: 0 0 10px rgba(59,158,255,0.3);
  transition: width 1.2s cubic-bezier(0.34,1.2,0.64,1);
}
.result-bar-p2 {
  background: linear-gradient(90deg, #B02247, #F87171);
  transition: width 1.2s cubic-bezier(0.34,1.2,0.64,1);
}

/* ── MINI STATS (result) ── */
.stat-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 10px; margin: 20px 0; }
.mini-stat {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: var(--r); padding: 13px 10px;
  text-align: center;
}
.mini-stat:hover { border-color: rgba(245,200,66,0.15); }
.mini-stat-val {
  font-family: var(--font-label);
  font-size: 20px; font-weight: 900; color: var(--text);
  letter-spacing: -0.01em;
}
.mini-stat-lbl {
  font-family: var(--font-label); font-size: 9px; font-weight: 700;
  color: var(--text-muted); text-transform: uppercase;
  letter-spacing: 0.12em; margin-top: 3px;
}

/* ── STRENGTH BARS ── */
.strength-section { margin-bottom: 20px; }
.strength-title {
  font-family: var(--font-label); font-size: 10px; font-weight: 800;
  color: var(--text-muted); letter-spacing: 0.18em;
  text-transform: uppercase; margin-bottom: 10px;
}
.strength-row {
  display: flex; align-items: center; gap: 12px; margin-bottom: 8px;
}
.strength-lbl {
  font-family: var(--font-label); font-size: 11px; font-weight: 800;
  letter-spacing: 0.06em; min-width: 30px;
}
.strength-track {
  flex: 1; height: 5px; background: rgba(255,255,255,0.04);
  border-radius: 99px; overflow: hidden;
}
.strength-fill {
  height: 100%; border-radius: 99px;
  transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.strength-num {
  font-family: var(--font-label); font-size: 11px; font-weight: 700;
  color: var(--text-muted); min-width: 28px; text-align: right;
}

/* ── MINI DECK CARDS ── */
.mini-deck { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 8px; }
.mini-deck-card {
  background: rgba(255,255,255,0.03); border: 1px solid var(--border);
  border-radius: var(--r-sm); padding: 5px;
  transition: all var(--t);
}
.mini-deck-card:hover { transform: translateY(-2px); border-color: rgba(245,200,66,0.2); }
.mini-deck-name {
  font-family: var(--font-label); font-size: 7.5px; font-weight: 700;
  color: var(--text-muted); text-align: center; margin-top: 2px;
  letter-spacing: 0.03em;
}

/* ── DIVIDER ── */
.cr-divider {
  height: 1px; margin: 22px 0;
  background: linear-gradient(90deg, transparent, var(--border-gold), transparent);
}

/* ══════════════════════════════════════════════
   CARD STATS PAGE
   ══════════════════════════════════════════════ */
.stat-card-hero {
  background: linear-gradient(150deg, var(--panel) 0%, var(--void) 100%);
  border: 1px solid var(--border);
  border-radius: var(--r-xl); padding: 22px 26px;
  box-shadow: var(--sh);
  transition: all var(--t); cursor: default;
  position: relative; overflow: hidden;
  margin-bottom: 0;
}
.stat-card-hero:hover {
  transform: translateY(-3px);
  border-color: rgba(245,200,66,0.2);
  box-shadow: var(--sh), var(--sh-gold);
}
.stat-card-hero::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(245,200,66,0.3), transparent);
}
.stat-num {
  font-family: var(--font-title);
  font-size: 40px; font-weight: 900; color: var(--gold);
  letter-spacing: -0.02em; line-height: 1; margin-bottom: 4px;
  text-shadow: 0 0 20px rgba(245,200,66,0.3);
}
.stat-label {
  font-family: var(--font-label); font-size: 10px; font-weight: 800;
  color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.14em;
}
.stat-sub {
  font-family: var(--font-body); font-size: 14px; font-style: italic;
  color: var(--text-soft); margin-top: 2px;
}

/* ── CARD STAT TILES ── */
.cstat-card {
  background: rgba(255,255,255,0.03); border: 1px solid var(--border);
  border-radius: 10px; padding: 8px 5px 7px;
  text-align: center; position: relative;
  display: flex; flex-direction: column; align-items: center;
  transition: all var(--t); cursor: default;
}
.cstat-card:hover {
  transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  border-color: rgba(245,200,66,0.2);
}
.cstat-wr {
  font-family: var(--font-label); font-size: 12px; font-weight: 900;
  margin-top: 5px; letter-spacing: 0.04em;
}
.cstat-name {
  font-family: var(--font-label); font-size: 8.5px; font-weight: 700;
  color: var(--text-muted); margin-top: 2px; letter-spacing: 0.04em;
}
.cstat-elix {
  position: absolute; top: 4px; left: 4px;
  background: linear-gradient(135deg, #7C3AED, #5B21B6);
  color: #fff; font-family: var(--font-label);
  font-size: 7.5px; font-weight: 900;
  width: 14px; height: 14px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 1px 4px rgba(0,0,0,0.5);
}

/* ══════════════════════════════════════════════
   LANDING PAGE
   ══════════════════════════════════════════════ */
.landing-root {
  min-height: 94vh; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  text-align: center; padding: 48px 32px;
  position: relative; overflow: hidden;
}
.landing-glow-top {
  position: absolute; top: -100px; left: 50%; transform: translateX(-50%);
  width: 700px; height: 400px; border-radius: 50%;
  background: radial-gradient(ellipse, rgba(245,200,66,0.08) 0%, transparent 70%);
  pointer-events: none;
}
.landing-glow-bottom-l {
  position: absolute; bottom: 0; left: 0;
  width: 400px; height: 300px;
  background: radial-gradient(ellipse at bottom left, rgba(59,158,255,0.05) 0%, transparent 60%);
  pointer-events: none;
}
.landing-glow-bottom-r {
  position: absolute; bottom: 0; right: 0;
  width: 400px; height: 300px;
  background: radial-gradient(ellipse at bottom right, rgba(159,110,255,0.05) 0%, transparent 60%);
  pointer-events: none;
}

.landing-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(245,200,66,0.07);
  border: 1px solid rgba(245,200,66,0.25);
  border-radius: 99px; padding: 6px 18px;
  margin-bottom: 28px;
  font-family: var(--font-label); font-size: 11px; font-weight: 800;
  color: var(--gold); letter-spacing: 0.14em; text-transform: uppercase;
}
.landing-badge-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--gold);
  box-shadow: 0 0 8px rgba(245,200,66,0.6);
  animation: pulse 2s ease infinite;
}
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.6; transform:scale(0.85); } }

.landing-title {
  font-family: var(--font-display);
  font-size: 56px; font-weight: 900; color: var(--text);
  letter-spacing: -0.02em; line-height: 1.05; margin-bottom: 8px;
}
.landing-title-gold { color: var(--gold); text-shadow: 0 0 40px rgba(245,200,66,0.4); }

.landing-subtitle {
  font-family: var(--font-body); font-size: 18px; font-style: italic;
  color: var(--text-soft); line-height: 1.7;
  max-width: 500px; margin: 0 auto 40px;
}

.landing-meta {
  display: flex; align-items: center; gap: 12px;
  justify-content: center; flex-wrap: wrap; margin-bottom: 40px;
}
.landing-meta-chip {
  font-family: var(--font-label); font-size: 11px; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  border-radius: 99px; padding: 7px 16px; color: var(--text-muted);
}
.landing-meta-chip.gold {
  color: var(--gold); border-color: rgba(245,200,66,0.2);
  background: rgba(245,200,66,0.05);
}

.feature-grid {
  display: grid; grid-template-columns: repeat(4,1fr); gap: 16px;
  max-width: 900px; width: 100%; margin-top: 56px;
}
.feature-tile {
  background: linear-gradient(150deg, var(--panel) 0%, var(--void) 100%);
  border: 1px solid var(--border);
  border-radius: var(--r-xl); padding: 22px 18px;
  text-align: left; transition: all var(--t);
  position: relative; overflow: hidden;
}
.feature-tile:hover {
  transform: translateY(-4px);
  border-color: rgba(245,200,66,0.2);
  box-shadow: 0 12px 40px rgba(0,0,0,0.5), var(--sh-gold);
}
.feature-tile::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(245,200,66,0.2), transparent);
}
.feature-icon {
  width: 38px; height: 38px; border-radius: var(--r-sm);
  background: rgba(245,200,66,0.08);
  border: 1px solid rgba(245,200,66,0.2);
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 14px;
}
.feature-name {
  font-family: var(--font-title); font-size: 14px; font-weight: 700;
  color: var(--text); margin-bottom: 6px;
}
.feature-desc {
  font-family: var(--font-body); font-style: italic;
  font-size: 13px; color: var(--text-muted); line-height: 1.55;
}

/* ══════════════════════════════════════════════
   HOW TO PAGE
   ══════════════════════════════════════════════ */
.howto-panel {
  background: linear-gradient(150deg, var(--panel) 0%, var(--void) 100%);
  border: 1px solid var(--border);
  border-radius: var(--r-xl); padding: 28px;
  box-shadow: var(--sh); position: relative; overflow: hidden;
  margin-bottom: 16px;
}
.howto-panel::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(245,200,66,0.2), transparent);
}
.howto-step { display: flex; gap: 16px; align-items: flex-start; margin-bottom: 20px; }
.howto-step:last-child { margin-bottom: 0; }
.step-num {
  width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
  background: rgba(245,200,66,0.08);
  border: 1px solid rgba(245,200,66,0.25);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-title); font-size: 13px; font-weight: 900;
  color: var(--gold); margin-top: 2px;
}
.step-title {
  font-family: var(--font-title); font-size: 15px; font-weight: 700;
  color: var(--text); margin-bottom: 4px;
}
.step-desc {
  font-family: var(--font-body); font-style: italic;
  font-size: 14px; color: var(--text-soft); line-height: 1.6;
}

.feature-list-item {
  display: flex; align-items: center; gap: 12px; padding: 9px 12px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: var(--r-sm); margin-bottom: 6px;
  font-family: var(--font-body); font-style: italic;
  font-size: 14px; color: var(--text-soft);
  transition: all var(--t);
}
.feature-list-item:hover { border-color: rgba(245,200,66,0.15); }
.feature-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--gold); flex-shrink: 0; box-shadow: 0 0 6px rgba(245,200,66,0.4); }

.conf-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 12px; background: rgba(255,255,255,0.03);
  border: 1px solid var(--border); border-radius: var(--r-sm);
  margin-bottom: 6px; transition: all var(--t);
}
.conf-row:hover { border-color: rgba(245,200,66,0.15); }
.conf-label { font-family: var(--font-title); font-size: 13px; font-weight: 700; }
.conf-range { font-family: var(--font-label); font-size: 11px; font-weight: 700; color: var(--text-muted); letter-spacing: 0.06em; }

/* ══════════════════════════════════════════════
   ANIMATIONS
   ══════════════════════════════════════════════ */
@keyframes fadeIn  { from { opacity:0; } to { opacity:1; } }
@keyframes slideUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
@keyframes goldGlow {
  0%,100% { text-shadow: 0 0 20px rgba(245,200,66,0.3); }
  50%      { text-shadow: 0 0 40px rgba(245,200,66,0.6); }
}

/* ══════════════════════════════════════════════
   TOAST
   ══════════════════════════════════════════════ */
.cr-toast {
  position: fixed; bottom: 24px; right: 24px; z-index: 9999;
  background: var(--panel);
  border: 1px solid rgba(245,200,66,0.2);
  border-radius: var(--r); padding: 13px 18px;
  font-family: var(--font-label); font-size: 13px; font-weight: 700;
  color: var(--text); letter-spacing: 0.06em; text-transform: uppercase;
  box-shadow: var(--sh), 0 0 20px rgba(245,200,66,0.1);
  display: flex; align-items: center; gap: 10px;
  animation: toastIn .3s cubic-bezier(0.34,1.56,0.64,1),
             toastOut .25s ease-in 2.75s forwards;
}
@keyframes toastIn  { from { transform:translateX(80px);opacity:0; } to { transform:translateX(0);opacity:1; } }
@keyframes toastOut { to   { transform:translateX(80px);opacity:0; } }
.cr-toast-gem {
  width: 8px; height: 8px;
  background: var(--gold);
  clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);
  flex-shrink: 0;
}

/* ══════════════════════════════════════════════
   COUNTER DECK PANEL
   ══════════════════════════════════════════════ */
.counter-root {
  margin-top: 28px;
  background: linear-gradient(150deg, #1a1040 0%, var(--void) 100%);
  border: 1px solid rgba(159,110,255,0.25);
  border-radius: var(--r-xl); padding: 28px 28px 24px;
  position: relative; overflow: hidden;
  animation: resultReveal 0.5s cubic-bezier(0.34,1.56,0.64,1);
}
.counter-root::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--violet), transparent);
  box-shadow: 0 0 20px rgba(159,110,255,0.5);
}
.counter-heading {
  font-family: var(--font-title);
  font-size: 18px; font-weight: 900;
  color: var(--violet); letter-spacing: -0.01em;
  margin-bottom: 4px;
}
.counter-sub {
  font-family: var(--font-body); font-style: italic;
  font-size: 14px; color: var(--text-muted); margin-bottom: 22px;
}
.counter-deck-block {
  background: rgba(159,110,255,0.05);
  border: 1px solid rgba(159,110,255,0.18);
  border-radius: var(--r-lg); padding: 18px 18px 14px;
  margin-bottom: 14px; position: relative;
  transition: all var(--t);
}
.counter-deck-block:hover {
  border-color: rgba(159,110,255,0.35);
  box-shadow: 0 4px 20px rgba(159,110,255,0.12);
}
.counter-rank-badge {
  position: absolute; top: 14px; right: 14px;
  background: rgba(159,110,255,0.12);
  border: 1px solid rgba(159,110,255,0.3);
  border-radius: 99px; padding: 3px 12px;
  font-family: var(--font-label); font-size: 10px; font-weight: 800;
  color: var(--violet); letter-spacing: 0.1em; text-transform: uppercase;
}
.counter-win-prob {
  font-family: var(--font-title);
  font-size: 28px; font-weight: 900;
  letter-spacing: -0.02em; line-height: 1;
  margin-bottom: 2px;
}
.counter-meta-row {
  display: flex; align-items: center; gap: 14px; margin-bottom: 14px; flex-wrap: wrap;
}
.counter-chip {
  font-family: var(--font-label); font-size: 10px; font-weight: 800;
  letter-spacing: 0.1em; text-transform: uppercase;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  border-radius: 99px; padding: 4px 12px;
  color: var(--text-muted);
}
.counter-cards-row {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-start;
}
.counter-card-tile {
  display: flex; flex-direction: column; align-items: center;
  gap: 4px; min-width: 52px;
}
.counter-card-name {
  font-family: var(--font-label); font-size: 7.5px; font-weight: 700;
  color: var(--text-muted); text-align: center; line-height: 1.2;
  max-width: 52px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.counter-card-elix {
  font-family: var(--font-label); font-size: 8px; font-weight: 900;
  color: #C084FC; letter-spacing: 0.04em;
}
.counter-prob-bar-track {
  height: 5px; border-radius: 99px;
  background: rgba(159,110,255,0.12); overflow: hidden; margin-top: 10px;
}
.counter-prob-bar-fill {
  height: 100%; border-radius: 99px;
  background: linear-gradient(90deg, #7C3AED, #C084FC, #E879F9);
  box-shadow: 0 0 6px rgba(192,132,252,0.4);
}
/* ── CARD CLICK OVERLAY BUTTON ── */
div[data-testid="stButton"]:has(> button[data-testid*="sel_"]) > button,
div[data-testid="stButton"]:has(> button[data-testid*="cd_"]) > button {
  background: rgba(255,255,255,0.06) !important;
  border: none !important;
  box-shadow: none !important;
  text-shadow: none !important;
  color: transparent !important;
  font-size: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  height: 6px !important;
  min-height: 6px !important;
  width: 100% !important;
  cursor: pointer !important;
  transform: none !important;
  border-radius: 0 0 10px 10px !important;
  display: block !important;
  transition: all 0.2s ease !important;
}
div[data-testid="stButton"]:has(> button[data-testid*="sel_"]) > button:hover,
div[data-testid="stButton"]:has(> button[data-testid*="cd_"]) > button:hover {
  filter: brightness(1.2) !important;
  box-shadow: none !important;
  transform: none !important;
}
div[data-testid="stButton"]:has(> button[data-testid*="sel_"]),
div[data-testid="stButton"]:has(> button[data-testid*="cd_"]) {
  margin: 0 !important;
  padding: 0 !important;
  line-height: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# BACKGROUND CANVAS + AMBIENT JS
# ══════════════════════════════════════════════════════
st.markdown("""
<canvas id="bg-canvas"></canvas>
<div class="bg-overlay"></div>
<div id="cr-toast-wrap" style="position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:8px;align-items:flex-end;"></div>

<script>
(function(){
  if(window.__crInit) return; window.__crInit = true;

  // BACKGROUND — drifting embers/stars
  var cv = document.getElementById('bg-canvas');
  if(!cv) return;
  cv.style.position='fixed'; cv.style.inset='0'; cv.style.zIndex='0'; cv.style.pointerEvents='none'; cv.style.opacity='0.5';
  var ctx = cv.getContext('2d');
  var W,H; function resize(){ W=cv.width=window.innerWidth; H=cv.height=window.innerHeight; }
  window.addEventListener('resize',resize); resize();

  var particles = [];
  for(var i=0;i<80;i++){
    var rnd=Math.random();
    particles.push({
      x:Math.random()*W, y:Math.random()*H,
      size: Math.random()*1.5+0.3,
      vx:(Math.random()-.5)*0.18, vy:-(Math.random()*0.25+0.05),
      alpha: Math.random()*0.35+0.05,
      flicker: Math.random()*Math.PI*2,
      color: rnd > 0.7 ? '#F5C842' : rnd > 0.4 ? '#9F6EFF' : '#3B9EFF'
    });
  }

  function draw(){
    ctx.clearRect(0,0,W,H);
    var now = Date.now()*0.001;
    particles.forEach(function(p){
      p.x+=p.vx; p.y+=p.vy; p.flicker+=0.025;
      if(p.y<-5){ p.y=H+5; p.x=Math.random()*W; }
      if(p.x<0) p.x=W; if(p.x>W) p.x=0;
      var a = p.alpha*(0.7+0.3*Math.sin(p.flicker));
      ctx.beginPath(); ctx.arc(p.x,p.y,p.size,0,Math.PI*2);
      ctx.fillStyle=p.color.replace(')',','+a+')').replace('rgb(','rgba(').replace('#F5C842','rgba(245,200,66,').replace('#9F6EFF','rgba(159,110,255,').replace('#3B9EFF','rgba(59,158,255,');
      if(p.color==='#F5C842') ctx.fillStyle='rgba(245,200,66,'+a+')';
      else if(p.color==='#9F6EFF') ctx.fillStyle='rgba(159,110,255,'+a+')';
      else ctx.fillStyle='rgba(59,158,255,'+a+')';
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  draw();

  // Clock
  function tick(){
    var el=document.getElementById('cr-clock');
    if(el){ var d=new Date(); el.textContent=String(d.getHours()).padStart(2,'0')+':'+String(d.getMinutes()).padStart(2,'0')+':'+String(d.getSeconds()).padStart(2,'0'); }
  }
  setInterval(tick,1000); tick();

  // Battle timer
  var timerVal=180;
  function tickTimer(){
    var el=document.getElementById('cr-timer');
    if(!el) return;
    var m=Math.floor(timerVal/60),s=timerVal%60;
    el.textContent=String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
    el.className='cr-badge cr-timer-badge'+(timerVal<=30?' cr-timer-danger':'');
    if(timerVal>0) timerVal--; else timerVal=180;
  }
  setInterval(tickTimer,1000); tickTimer();

  // Toast
  window.crToast = function(msg, type){
    var wrap=document.getElementById('cr-toast-wrap'); if(!wrap) return;
    var t=document.createElement('div'); t.className='cr-toast';
    t.innerHTML='<div class="cr-toast-gem"></div><span>'+msg+'</span>';
    wrap.appendChild(t);
    setTimeout(function(){ if(t.parentNode) t.parentNode.removeChild(t); },3200);
  };
  setTimeout(function(){ window.crToast && window.crToast('Arena AI Loaded','gold'); },700);
})();
</script>
<style>
.cr-timer-badge { color: var(--gold) !important; }
.cr-timer-danger { color: var(--crimson) !important; border-color: rgba(232,54,93,0.3) !important; animation: pulse 1s ease infinite; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    m = xgb.Booster(); m.load_model("clash_model_robust.json")
    with open("clash_metadata_robust.pkl","rb") as f: meta=pickle.load(f)
    return m, meta

try:    model,meta = load_model(); model_ok=True
except: model_ok=False

# ─────────────────────────────────────────────────────────────
# FEATURE PIPELINE
# ─────────────────────────────────────────────────────────────
# ── PREDICTION PIPELINE — exactly matches ai.py training pipeline ──────────────

_P1 = list(range(5, 13))   # column indices for p1 cards in the row
_P2 = list(range(16, 24))  # column indices for p2 cards in the row

def _map_cards(df, cols, mapping, default):
    """Vectorised card-id → value lookup. Matches ai.py map_cards_vectorized."""
    arr = df[cols].fillna(0).astype(np.int64).values
    return np.vectorize(lambda x: mapping.get(x, default))(arr)

def _rarity(arr):
    """Rarity tier: 0=Common/Rare, 1=Epic, 2=Legendary. Matches ai.py rarity_vectorized."""
    out = np.zeros_like(arr, dtype=np.int8)
    out[arr >= 27000000] = 1
    out[arr >= 28000000] = 2
    return out

def _trophy_bucket(arr):
    """Trophy tier bucket. Matches ai.py trophy_bucket exactly."""
    b = np.zeros_like(arr, dtype=np.int8)
    b[arr >= 4000] = 1
    b[arr >= 6000] = 2
    b[arr >= 7000] = 3
    b[arr >= 8000] = 4
    return b

def _build_features(df):
    """Build base features. Exact mirror of ai.py build_features (26 columns)."""
    feats = pd.DataFrame(index=range(len(df)))

    p1t = df[3].fillna(0).values
    p2t = df[14].fillna(0).values
    feats["p1_trophies"]          = p1t
    feats["p2_trophies"]          = p2t
    feats["trophy_diff"]          = p1t - p2t
    feats["trophy_ratio"]         = p1t / (p2t + 1)
    feats["p1_trophy_bucket"]     = _trophy_bucket(p1t)
    feats["p2_trophy_bucket"]     = _trophy_bucket(p2t)
    feats["bucket_diff"]          = feats["p1_trophy_bucket"] - feats["p2_trophy_bucket"]

    p1_elix = _map_cards(df, _P1, CARD_ELIXIR, DEFAULT_ELIXIR).astype(float)
    p2_elix = _map_cards(df, _P2, CARD_ELIXIR, DEFAULT_ELIXIR).astype(float)
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

    p1_arr = df[_P1].fillna(0).astype(np.int64).values
    p2_arr = df[_P2].fillna(0).astype(np.int64).values
    p1_rar = _rarity(p1_arr).astype(float)
    p2_rar = _rarity(p2_arr).astype(float)
    feats["p1_avg_rarity"]        = p1_rar.mean(axis=1)
    feats["p2_avg_rarity"]        = p2_rar.mean(axis=1)
    feats["p1_epic_leg_count"]    = (p1_rar == 2).sum(axis=1)
    feats["p2_epic_leg_count"]    = (p2_rar == 2).sum(axis=1)
    feats["rarity_diff"]          = feats["p1_avg_rarity"] - feats["p2_avg_rarity"]

    feats["deck_overlap"]         = np.array([len(np.intersect1d(p1_arr[i], p2_arr[i])) for i in range(len(df))])
    feats["p1_deck_diversity"]    = (df[_P1].nunique(axis=1) / 8).values
    feats["p2_deck_diversity"]    = (df[_P2].nunique(axis=1) / 8).values
    return feats

def _apply_card_winrate(df, winrate_map, global_wr):
    """Card win-rate features. Exact mirror of ai.py apply_card_winrate."""
    wr_vec = np.vectorize(lambda x: winrate_map.get(int(x), global_wr))
    p1_wr  = wr_vec(df[_P1].fillna(0).astype(float).values)
    p2_wr  = wr_vec(df[_P2].fillna(0).astype(float).values)
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
    """Deck synergy score. Exact mirror of ai.py apply_pair_synergy."""
    card_arr = df[_P1].fillna(0).astype(np.int64).values
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

def _build_card_onehot(df, all_cards_universe, prefix):
    """One-hot card presence. Exact mirror of ai.py build_card_onehot."""
    src  = _P1 if prefix == "p1" else _P2
    arr  = df[src].fillna(0).astype(np.int64).values
    univ = np.array(all_cards_universe)
    oh   = (arr[:, :, None] == univ[None, None, :]).any(axis=1).astype(np.float32)
    return pd.DataFrame(oh, columns=[f"{prefix}_has_{c}" for c in univ])

def _make_row_df(p1c, p2c, p1t, p2t):
    """
    Build a single-row DataFrame in the exact CSV column layout used during training:
      col 0-2  : unused (None)
      col 3    : p1 trophies
      col 4    : label placeholder (0)
      cols 5-12: p1 card IDs (8 cards)
      col 13   : unused (None)
      col 14   : p2 trophies
      col 15   : unused (0)
      cols 16-23: p2 card IDs (8 cards)
    """
    row = [None, None, None, p1t, 0] + list(p1c) + [None, p2t, 0] + list(p2c)
    df = pd.DataFrame([row])
    for col in range(24):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _raw_predict(p1c, p2c, p1t, p2t):
    """Full inference pipeline — must match ai.py training pipeline exactly."""
    df = _make_row_df(p1c, p2c, p1t, p2t)

    X_base = _build_features(df)
    X_wr   = _apply_card_winrate(df, meta["winrate_map"], meta["global_wr"])
    X_syn  = _apply_pair_synergy(df, meta["synergy_map"], meta["syn_global_wr"]).to_frame()
    X_p1oh = _build_card_onehot(df, meta["all_cards_universe"], "p1")
    X_p2oh = _build_card_onehot(df, meta["all_cards_universe"], "p2")

    X = pd.concat([X_base, X_wr, X_syn, X_p1oh, X_p2oh], axis=1)
    X.columns = [str(c) for c in X.columns]

    # Align to training feature order; fill 0 for unseen cards
    X = X.reindex(columns=meta["feature_columns"], fill_value=0)

    # Scale continuous features using the saved scaler (MinMaxScaler)
    X[meta["continuous_cols"]] = meta["scaler"].transform(X[meta["continuous_cols"]])

    dmat = xgb.DMatrix(X.values.astype(np.float32))
    return float(model.predict(dmat)[0])

def predict(p1c, p2c, p1t, p2t):
    """
    Symmetry-corrected win probability for player 1.
    Averages forward and reverse predictions to correct for any direction bias.
    """
    fwd = _raw_predict(p1c, p2c, p1t, p2t)
    rev = _raw_predict(p2c, p1c, p2t, p1t)
    return (fwd + (1.0 - rev)) / 2.0

# ─────────────────────────────────────────────────────────────
# COUNTER DECK ENGINE  (fast batch version)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _precompute_card_wr_synergy():
    """Cache per-card win rates and synergy scores from metadata."""
    if not model_ok:
        return {}, 0.5, {}, 0.5
    return (meta["winrate_map"], meta["global_wr"],
            meta["synergy_map"], meta["syn_global_wr"])

def _fast_score_batch(opponent_deck, candidate_decks, p1t, p2t):
    """
    Score all candidate decks in ONE vectorized batch XGBoost call.
    candidate_decks: list of 8-card lists (the counter candidates).
    Returns np.array of P(counter wins as P2).
    """
    n = len(candidate_decks)
    opp = list(opponent_deck)

    # Build raw rows: opponent = P1, candidate = P2
    rows = []
    for cand in candidate_decks:
        row = [None, None, None, p1t, 0] + opp + [None, p2t, 0] + list(cand)
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in range(24):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature pipeline (vectorized over all N rows at once)
    X_base = _build_features(df)
    X_wr   = _apply_card_winrate(df, meta["winrate_map"], meta["global_wr"])
    X_syn  = _apply_pair_synergy(df, meta["synergy_map"], meta["syn_global_wr"]).to_frame()
    X_p1oh = _build_card_onehot(df, meta["all_cards_universe"], "p1")
    X_p2oh = _build_card_onehot(df, meta["all_cards_universe"], "p2")

    X = pd.concat([X_base, X_wr, X_syn, X_p1oh, X_p2oh], axis=1)
    X.columns = [str(c) for c in X.columns]
    X = X.reindex(columns=meta["feature_columns"], fill_value=0)
    X[meta["continuous_cols"]] = meta["scaler"].transform(X[meta["continuous_cols"]])

    # One XGBoost call for all N candidates
    probs_p1_wins = model.predict(xgb.DMatrix(X.values.astype(np.float32)))
    return 1.0 - probs_p1_wins   # P(counter/P2 wins)

def suggest_counter_deck(opponent_deck, p1t=5000, p2t=5000, top_k=3, n_candidates=300):
    """
    Fast counter deck search using batch scoring.
    Generates n_candidates random decks, scores them all in ONE model call,
    then runs a quick greedy refinement on the top seeds — all in ~1-2 seconds.
    """
    if not model_ok:
        return []

    card_pool = [c for c in meta.get("all_cards_universe", list(CARD_DB.keys()))
                 if c in CARD_DB and c not in opponent_deck]
    if len(card_pool) < 8:
        return []

    # ── Phase 1: score n_candidates random decks in one batch ──
    candidates = []
    seen_keys = set()
    while len(candidates) < n_candidates:
        deck = random.sample(card_pool, 8)
        key = tuple(sorted(deck))
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(deck)

    scores = _fast_score_batch(opponent_deck, candidates, p1t, p2t)

    # ── Phase 2: greedy 1-card-swap refinement on top-10 seeds ──
    top_idx = np.argsort(scores)[::-1][:10]
    refined = []
    for idx in top_idx:
        current = list(candidates[idx])
        current_score = float(scores[idx])
        # Only 30 swap attempts per seed — fast but effective
        for _ in range(30):
            swap_pos = random.randrange(8)
            available = [c for c in card_pool if c not in current]
            if not available:
                break
            swap_in = random.choice(available)
            trial = current[:]
            trial[swap_pos] = swap_in
            # Score the single trial as a batch of 1
            trial_score = float(_fast_score_batch(opponent_deck, [trial], p1t, p2t)[0])
            if trial_score > current_score:
                current = trial
                current_score = trial_score
        avg_e = sum(CARD_DB.get(c, (None, DEFAULT_ELIXIR, None))[1] for c in current) / 8
        refined.append({"deck": sorted(current), "win_prob": round(current_score, 4),
                        "avg_elixir": round(avg_e, 2)})

    # De-duplicate, return top_k
    seen, unique = set(), []
    for r in sorted(refined, key=lambda x: x["win_prob"], reverse=True):
        key = tuple(r["deck"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
        if len(unique) == top_k:
            break
    return unique

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
defs={"page":"landing","p1_deck":[],"p2_deck":[],
      "p1_trophies":5000,"p2_trophies":5000,
      "result":None,"top_page":"Battle","card_page":0,
      "picker_search":"","picker_rarity":"All","battle_count":0,
      "counter_decks":None,"counter_target":None}
for k,v in defs.items():
    if k not in st.session_state: st.session_state[k]=v

def cycle_card(cid):
    """
    Click 1 → P1 only (blue)
    Click 2 → P2 only (red)
    Click 3 → both P1 + P2 (split border)
    Click 4 → removed
    """
    in1 = cid in st.session_state.p1_deck
    in2 = cid in st.session_state.p2_deck
    d1 = list(st.session_state.p1_deck)
    d2 = list(st.session_state.p2_deck)

    if not in1 and not in2:
        # → P1
        if len(d1) < 8: d1.append(cid)
    elif in1 and not in2:
        # P1 → P2 (remove from P1, add to P2)
        d1.remove(cid)
        if len(d2) < 8: d2.append(cid)
    elif not in1 and in2:
        # P2 → both (add back to P1)
        if len(d1) < 8: d1.append(cid)
    elif in1 and in2:
        # both → remove
        d1.remove(cid); d2.remove(cid)

    st.session_state.p1_deck = d1
    st.session_state.p2_deck = d2
    st.session_state.result = None

def deck_strength(deck):
    if not deck: return 0
    rarity_pts={"Common":1,"Rare":2,"Epic":3,"Legendary":4}
    avg_elix=sum(CARD_DB[c][1] for c in deck)/len(deck)
    avg_rar=sum(rarity_pts[CARD_DB[c][2]] for c in deck)/len(deck)
    elix_score=max(0,1-(avg_elix-3)/6)
    return min(100,int((elix_score*40+avg_rar/4*60)))

def trophy_tier(t):
    if t>=7000: return ("Legendary","#FBBF24","rgba(251,191,36,0.12)")
    if t>=4000: return ("Champion","#60A5FA","rgba(96,165,250,0.1)")
    if t>=2000: return ("Silver","#94A3B8","rgba(148,163,184,0.08)")
    return ("Bronze","#CD7C30","rgba(205,124,48,0.1)")

# ══════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════
if st.session_state.page == "landing":
    st.markdown(f"""
    <div class="landing-root">
      <div class="landing-glow-top"></div>
      <div class="landing-glow-bottom-l"></div>
      <div class="landing-glow-bottom-r"></div>

      <div class="landing-badge">
        <div class="landing-badge-dot"></div>
        XGBoost AI &nbsp;·&nbsp; Battle Oracle
      </div>

      <div class="landing-title">
        Clash Royale<br><span class="landing-title-gold">Arena AI</span>
      </div>

      <p class="landing-subtitle">
        Forge your deck, challenge your rivals, and let the Oracle of the Arena reveal your destiny.
      </p>

      <div class="landing-meta">
        <div class="landing-meta-chip gold">{len(CARD_DB)} Cards</div>
        <div class="landing-meta-chip">XGBoost Model</div>
        <div class="landing-meta-chip">28 Synergy Pairs</div>
        <div class="landing-meta-chip gold">Trophy-Adjusted</div>
        <div class="landing-meta-chip" id="cr-clock">--:--:--</div>
        <div class="landing-meta-chip cr-badge" id="cr-timer">03:00</div>
      </div>

      <div class="feature-grid" style="grid-template-columns:repeat(5,1fr)">
        <div class="feature-tile">
          <div class="feature-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F5C842" stroke-width="2">
              <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M8 12h8M12 8v8"/>
            </svg>
          </div>
          <div class="feature-name">{len(CARD_DB)} Cards</div>
          <div class="feature-desc">Full roster with elixir costs, rarities and win rates from real battles.</div>
        </div>
        <div class="feature-tile">
          <div class="feature-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F5C842" stroke-width="2">
              <circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4"/>
            </svg>
          </div>
          <div class="feature-name">Oracle AI</div>
          <div class="feature-desc">Gradient boosting model forged on thousands of real Arena clashes.</div>
        </div>
        <div class="feature-tile">
          <div class="feature-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F5C842" stroke-width="2">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
          </div>
          <div class="feature-name">Deck Synergy</div>
          <div class="feature-desc">28 unique card-pair synergy scores illuminate hidden strengths.</div>
        </div>
        <div class="feature-tile">
          <div class="feature-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F5C842" stroke-width="2">
              <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2z"/>
            </svg>
          </div>
          <div class="feature-name">Trophy Tier</div>
          <div class="feature-desc">Skill-calibrated predictions across Bronze to Legendary leagues.</div>
        </div>
        <div class="feature-tile">
          <div class="feature-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F5C842" stroke-width="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          <div class="feature-name">Counter Forge</div>
          <div class="feature-desc">AI search finds the optimal 8-card deck to defeat any opponent.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, bc, _ = st.columns([1, 1.2, 1])
    with bc:
        if st.button("⚔  Enter the Arena", use_container_width=True, type="primary"):
            st.session_state.page = "app"; st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# TOP NAVIGATION
# ══════════════════════════════════════════════════════════════
TOP_PAGES = ["Battle", "Counter Deck", "Card Stats", "How To Use"]
page = st.session_state.top_page

icons = {
    "Battle":       "M14.5 10c-.83 0-1.5-.67-1.5-1.5V3c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5v5.5c0 .83-.67 1.5-1.5 1.5zM9.5 14c.83 0 1.5.67 1.5 1.5V21c0 .83-.67 1.5-1.5 1.5S8 21.83 8 21v-5.5c0-.83.67-1.5 1.5-1.5zM3 9.5C3 8.67 3.67 8 4.5 8h5.5c.83 0 1.5.67 1.5 1.5S10.83 11 10 11H4.5C3.67 11 3 10.33 3 9.5zM14 14.5c0-.83.67-1.5 1.5-1.5H21c.83 0 1.5.67 1.5 1.5S21.83 16 21 16h-5.5c-.83 0-1.5-.67-1.5-1.5z",
    "Counter Deck": "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z",
    "Card Stats":   "M3 3h18v18H3zM3 9h18M9 21V9",
    "How To Use":   "M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zM9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3M12 17h.01",
}
nav_html = ""
for pg in TOP_PAGES:
    active = "active" if pg == page else ""
    d = icons[pg]
    nav_html += f'''<button class="cr-nav-btn {active}" onclick="window.__crNav('{pg}')">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="{d}"/></svg>
      {pg}
    </button>'''

st.markdown(f"""
<div class="cr-header">
  <div class="cr-logo" style="cursor:pointer" onclick="window.__crHome()">
    <div class="cr-logo-emblem">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="rgba(20,15,0,0.9)" stroke-width="2.5" stroke-linecap="round">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
      </svg>
    </div>
    <div class="cr-logo-text">
      <span class="cr-logo-name">CLASH ROYALE</span>
      <span class="cr-logo-tagline">Arena AI Oracle</span>
    </div>
  </div>

  <nav class="cr-nav">{nav_html}</nav>

  <div class="cr-header-badges">
    <span id="cr-clock" style="font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:700;color:var(--text-muted);letter-spacing:.08em;background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:99px;padding:6px 14px;">--:--:--</span>
    <span id="cr-timer" class="cr-badge cr-timer-badge">03:00</span>
    <div class="cr-battles-badge">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2z"/></svg>
      {st.session_state.battle_count} BATTLES
    </div>
  </div>
</div>
<script>window.__crNav=function(pg){{}}; window.__crHome=function(){{}};</script>
""", unsafe_allow_html=True)

# Hidden nav buttons
hcols = st.columns(len(TOP_PAGES) + 1)
with hcols[0]:
    if st.button("Home", key="nav_home"):
        st.session_state.page = "landing"; st.rerun()
for i, pg in enumerate(TOP_PAGES):
    with hcols[i+1]:
        if st.button(pg, key=f"nav_{i}"):
            st.session_state.top_page = pg; st.rerun()

# ══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="cr-main" style="position:relative;z-index:10;">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE — BATTLE
# ══════════════════════════════════════════════════════════════
if page == "Battle":
    if not model_ok:
        st.error("Model files not found. Place `clash_model_robust.json` + `clash_metadata_robust.pkl` in the working directory.")
        st.stop()

    st.markdown("""
    <div class="page-heading">
      <div class="page-title">⚔ Battle Predictor</div>
      <div class="page-sub">Forge your deck · Consult the Oracle · Claim Victory</div>
    </div>
    """, unsafe_allow_html=True)

    # ── helpers ────────────────────────────────────────────────
    def render_slots(deck):
        html = '<div class="deck-slots">'
        for i in range(8):
            if i < len(deck):
                cid=deck[i]; name,elix,rarity=CARD_DB.get(cid,(str(cid),4,"Common"))
                img=card_img_html(name,52)
                html+=f'''<div class="dslot filled">
                  <div class="dslot-elix">{elix}</div>{img}
                  <div class="dslot-name">{name}</div>
                </div>'''
            else:
                html+='''<div class="dslot">
                  <div class="dslot-empty-icon">
                    <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
                    </svg>
                  </div>
                </div>'''
        return html + '</div>'

    def render_elixir(deck):
        avg=sum(CARD_DB[c][1] for c in deck)/len(deck) if deck else 0
        pct=min(avg/10*100,100)
        return f'''<div class="elixir-gauge">
          <div class="elixir-header">
            <span class="elixir-label">⚗ Avg. Elixir</span>
            <span class="elixir-val">{avg:.1f}</span>
          </div>
          <div class="elixir-track"><div class="elixir-fill" style="width:{pct}%"></div></div>
        </div>'''

    def render_rarity(deck):
        if not deck: return ""
        from collections import Counter
        rc=Counter(CARD_DB[c][2] for c in deck); tags=""
        for r in ["Legendary","Epic","Rare","Common"]:
            if rc[r]:
                col=RARITY_COLOR[r]; bg=RARITY_BG[r]
                tags+=f'<span class="r-tag" style="background:{bg};color:{col};border:1px solid {col}33;">{rc[r]}&nbsp;{r[:3].upper()}</span>'
        return tags

    col_p1, col_mid, col_p2 = st.columns([1.7, 5.2, 1.7], gap="large")

    # ── PLAYER 1 ────────────────────────────────────────────────
    with col_p1:
        d1=st.session_state.p1_deck; p1t=st.session_state.p1_trophies
        t1_name,t1_col,t1_bg=trophy_tier(p1t); c1=len(d1)==8
        st.markdown(f"""
        <div class="deck-panel p1-panel {'complete' if c1 else ''}">
          <div class="section-label">
            <svg width="9" height="9" viewBox="0 0 24 24" fill="#60A5FA" stroke="none">
              <circle cx="12" cy="12" r="10"/>
            </svg>
            Your Deck
          </div>
          {render_slots(d1)}
          {render_elixir(d1)}
          <div style="margin-top:8px">{render_rarity(d1)}</div>
          <div style="margin-top:12px">
            <div class="count-row">
              <span class="count-label">Cards</span>
              <span class="count-val" style="color:{'var(--emerald)' if c1 else 'var(--text-muted)'}">{len(d1)}/8</span>
            </div>
            <div class="count-track">
              <div class="count-fill" style="width:{len(d1)/8*100:.0f}%;background:{'var(--emerald)' if c1 else 'var(--sapphire)'}"></div>
            </div>
          </div>
          <div class="trophy-badge" style="background:{t1_bg};color:{t1_col};border:1px solid {t1_col}33;">
            🏆 {t1_name} · {p1t:,}
          </div>
        </div>
        """, unsafe_allow_html=True)
        new_t1=st.number_input("Your Trophies",0,9999,p1t,100,key="p1ti")
        st.session_state.p1_trophies=new_t1
        if st.button("🗑  Clear Deck",key="clr1",use_container_width=True):
            st.session_state.p1_deck=[]; st.session_state.result=None; st.rerun()

    # ── CARD LIBRARY ────────────────────────────────────────────
    with col_mid:
        st.markdown('<div class="section-label">📚 Card Library</div>', unsafe_allow_html=True)

        fc1,fc2=st.columns([2.5,1.5])
        with fc1:
            search=st.text_input("Search","",placeholder="🔍  Search cards...",key="srch",label_visibility="collapsed")
            if search!=st.session_state.picker_search:
                st.session_state.picker_search=search; st.session_state.card_page=0
        with fc2:
            rarity_f=st.selectbox("Rarity",["All","Legendary","Epic","Rare","Common"],key="rarf",label_visibility="collapsed")
            if rarity_f!=st.session_state.picker_rarity:
                st.session_state.picker_rarity=rarity_f; st.session_state.card_page=0

        filtered=[(cid,n,e,r) for cid,(n,e,r) in ALL_CARDS_SORTED
                  if (rarity_f=="All" or r==rarity_f) and (not search or search.lower() in n.lower())]
        total_pages=max(1,(len(filtered)+CARDS_PER_PAGE-1)//CARDS_PER_PAGE)
        cp=min(st.session_state.card_page,total_pages-1)
        st.session_state.card_page=cp
        page_cards=filtered[cp*CARDS_PER_PAGE:(cp+1)*CARDS_PER_PAGE]

        max_dots=min(total_pages,14)
        dots="".join([f'<span class="pdot {"active" if i==cp else ""}"></span>' for i in range(max_dots)])
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
          <span class="page-info">{len(filtered)} cards &nbsp;·&nbsp; Page {cp+1}/{total_pages}</span>
          <div class="page-dots">{dots}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="library-panel">', unsafe_allow_html=True)
        NCOLS=8
        for rs in range(0,len(page_cards),NCOLS):
            row=page_cards[rs:rs+NCOLS]
            gcols=st.columns(NCOLS)
            for ci,(cid,name,elix,rarity) in enumerate(row):
                with gcols[ci]:
                    in1=cid in st.session_state.p1_deck
                    in2=cid in st.session_state.p2_deck
                    rc=RARITY_COLOR[rarity]
                    if in1 and in2: cls="in-both"
                    elif in1:       cls="in-p1"
                    elif in2:       cls="in-p2"
                    else:           cls=""
                    img=card_img_html(name,54)
                    badge=""
                    if in1 and in2:
                        badge='<div class="ccard-badge" style="background:rgba(255,255,255,0.08);color:#fff;font-size:7px;">P1+P2</div>'
                    elif in1:
                        badge='<div class="ccard-badge" style="background:rgba(59,158,255,0.18);color:#60A5FA;">P1</div>'
                    elif in2:
                        badge='<div class="ccard-badge" style="background:rgba(232,54,93,0.18);color:#F87171;">P2</div>'
                    st.markdown(f"""
                    <div class="ccard {cls}" style="animation:slideUp .2s ease {ci*0.025:.2f}s both;">
                      {badge}
                      <div class="ccard-elix">{elix}</div>{img}
                      <div class="ccard-name">{name}</div>
                      <div class="ccard-rarity">
                        <span class="rarity-dot" style="background:{rc}"></span>{rarity[:3].upper()}
                      </div>
                    </div>""", unsafe_allow_html=True)
                    # Button color based on state
                    if in1 and in2:
                        btn_bg = "linear-gradient(90deg, #3B9EFF 50%, #E8365D 50%)"
                        btn_shadow = "0 0 8px rgba(59,158,255,0.3), 0 0 8px rgba(232,54,93,0.3)"
                    elif in1:
                        btn_bg = "linear-gradient(90deg, #1A6ECC, #3B9EFF)"
                        btn_shadow = "0 0 10px rgba(59,158,255,0.5)"
                    elif in2:
                        btn_bg = "linear-gradient(90deg, #B02247, #E8365D)"
                        btn_shadow = "0 0 10px rgba(232,54,93,0.5)"
                    else:
                        btn_bg = "rgba(255,255,255,0.06)"
                        btn_shadow = "none"
                    st.markdown(f"""<style>
                    button[data-testid="sel_{cid}"] {{
                        background: {btn_bg} !important;
                        box-shadow: {btn_shadow} !important;
                        height: 6px !important; min-height: 6px !important;
                        border-radius: 0 0 10px 10px !important;
                        border: none !important; opacity: 1 !important;
                    }}
                    </style>""", unsafe_allow_html=True)
                    if st.button("", key=f"sel_{cid}", use_container_width=True):
                        cycle_card(cid); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        pnc=st.columns([1,1,3,1,1])
        with pnc[0]:
            if st.button("«",key="pg_first",use_container_width=True,disabled=cp==0): st.session_state.card_page=0; st.rerun()
        with pnc[1]:
            if st.button("‹",key="pg_prev",use_container_width=True,disabled=cp==0): st.session_state.card_page=cp-1; st.rerun()
        with pnc[2]:
            st.markdown(f'<div style="text-align:center;padding:9px;font-family:var(--font-label);font-size:12px;font-weight:700;color:var(--text-muted);letter-spacing:.08em;">{cp+1} / {total_pages}</div>',unsafe_allow_html=True)
        with pnc[3]:
            if st.button("›",key="pg_next",use_container_width=True,disabled=cp>=total_pages-1): st.session_state.card_page=cp+1; st.rerun()
        with pnc[4]:
            if st.button("»",key="pg_last",use_container_width=True,disabled=cp>=total_pages-1): st.session_state.card_page=total_pages-1; st.rerun()

        # Live compare
        if st.session_state.p1_deck or st.session_state.p2_deck:
            s1=deck_strength(st.session_state.p1_deck); s2=deck_strength(st.session_state.p2_deck)
            tot=s1+s2 if s1+s2>0 else 1
            p1p=s1/tot*100
            e1=sum(CARD_DB[c][1] for c in st.session_state.p1_deck)/max(len(st.session_state.p1_deck),1)
            e2=sum(CARD_DB[c][1] for c in st.session_state.p2_deck)/max(len(st.session_state.p2_deck),1)
            st.markdown(f"""
            <div class="compare-panel">
              <div class="compare-row">
                <span class="compare-name-p1">⚔ Your Deck</span>
                <span class="compare-count">{len(st.session_state.p1_deck)}/8 vs {len(st.session_state.p2_deck)}/8</span>
                <span class="compare-name-p2">Opponent 🛡</span>
              </div>
              <div class="compare-track">
                <div class="compare-p1-fill" style="width:{p1p:.1f}%"></div>
                <div class="compare-p2-fill" style="width:{100-p1p:.1f}%"></div>
              </div>
              <div class="compare-sub">
                <span>⚗ {e1:.1f} Elixir</span>
                <span style="font-style:italic;font-family:'Crimson Pro',serif;font-size:13px;color:var(--text-muted)">Deck Strength</span>
                <span>{e2:.1f} Elixir ⚗</span>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── PLAYER 2 ────────────────────────────────────────────────
    with col_p2:
        d2=st.session_state.p2_deck; p2t=st.session_state.p2_trophies
        t2_name,t2_col,t2_bg=trophy_tier(p2t); c2=len(d2)==8
        st.markdown(f"""
        <div class="deck-panel p2-panel {'complete' if c2 else ''}">
          <div class="section-label">
            <svg width="9" height="9" viewBox="0 0 24 24" fill="#F87171" stroke="none">
              <circle cx="12" cy="12" r="10"/>
            </svg>
            Opponent
          </div>
          {render_slots(d2)}
          {render_elixir(d2)}
          <div style="margin-top:8px">{render_rarity(d2)}</div>
          <div style="margin-top:12px">
            <div class="count-row">
              <span class="count-label">Cards</span>
              <span class="count-val" style="color:{'var(--emerald)' if c2 else 'var(--text-muted)'}">{len(d2)}/8</span>
            </div>
            <div class="count-track">
              <div class="count-fill" style="width:{len(d2)/8*100:.0f}%;background:{'var(--emerald)' if c2 else 'var(--crimson)'}"></div>
            </div>
          </div>
          <div class="trophy-badge" style="background:{t2_bg};color:{t2_col};border:1px solid {t2_col}33;">
            🏆 {t2_name} · {p2t:,}
          </div>
        </div>
        """, unsafe_allow_html=True)
        new_t2=st.number_input("Opponent Trophies",0,9999,p2t,100,key="p2ti")
        st.session_state.p2_trophies=new_t2
        if st.button("🗑  Clear Opponent",key="clr2",use_container_width=True):
            st.session_state.p2_deck=[]; st.session_state.result=None; st.rerun()

    # ── PREDICT ──────────────────────────────────────────────────
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    can=len(st.session_state.p1_deck)==8 and len(st.session_state.p2_deck)==8
    _, bc, _ = st.columns([1.5,2,1.5])
    with bc:
        if not can:
            r1=8-len(st.session_state.p1_deck); r2=8-len(st.session_state.p2_deck)
            missing=[]
            if r1: missing.append(f"Your deck: {r1} more")
            if r2: missing.append(f"Opponent: {r2} more")
            st.markdown(f"""
            <div class="predict-hint">
              <span>{"  ·  ".join(missing)}</span>
            </div>""", unsafe_allow_html=True)
        if st.button("⚔  Consult the Oracle",use_container_width=True,disabled=not can):
            with st.spinner("The Oracle gazes into the mists..."):
                prob=predict(st.session_state.p1_deck,st.session_state.p2_deck,
                             st.session_state.p1_trophies,st.session_state.p2_trophies)
            st.session_state.result=prob
            st.session_state.battle_count=st.session_state.get("battle_count",0)+1
            st.rerun()

    # ── RESULT ────────────────────────────────────────────────────
    if st.session_state.result is not None:
        prob=st.session_state.result
        p1pct=prob*100; p2pct=(1-prob)*100
        p1wins=prob>=0.5; gap=abs(prob-0.5)
        conf_label=("Decisive Conquest" if gap>.35 else
                    "Dominant Victory" if gap>.25 else
                    "Strong Advantage" if gap>.15 else
                    "Slight Edge" if gap>.05 else "Too Close to Call")
        conf_col=("#22D07A" if gap>.25 else "#9F6EFF" if gap>.15 else "#F5C842" if gap>.05 else "#94a3b8")

        verdict="Your deck prevails" if p1wins else "Opponent claims victory"
        verdict_col="var(--emerald)" if p1wins else "var(--crimson)"
        panel_cls="win" if p1wins else "loss"

        def mini_deck_html(cards):
            html='<div class="mini-deck">'
            for cid in cards:
                name,elix,rarity=CARD_DB.get(cid,(str(cid),4,"Common"))
                img=card_img_html(name,42)
                html+=f'<div class="mini-deck-card">{img}<div class="mini-deck-name">{name[:7]}</div></div>'
            return html+'</div>'

        p1e=sum(CARD_DB.get(c,("",4,""))[1] for c in st.session_state.p1_deck)/8
        p2e=sum(CARD_DB.get(c,("",4,""))[1] for c in st.session_state.p2_deck)/8
        s1=deck_strength(st.session_state.p1_deck); s2=deck_strength(st.session_state.p2_deck)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        _, rc_, _ = st.columns([0.25,3,0.25])
        with rc_:
            st.markdown(f"""
            <div class="result-panel {panel_cls}">
              <div class="result-header">
                <div>
                  <div class="result-label">⚔ Oracle's Verdict</div>
                  <div class="result-verdict" style="color:{verdict_col}">{verdict}</div>
                  <div style="margin-top:8px">
                    <span class="result-conf-chip" style="color:{conf_col};border-color:{conf_col}33;background:{conf_col}10;">
                      <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                      {conf_label}
                    </span>
                  </div>
                </div>
                <div class="result-pct">
                  <div class="result-pct-num" style="color:{verdict_col}">{p1pct:.0f}<span style="font-size:26px;color:var(--text-muted)">%</span></div>
                  <div class="result-pct-label">Your Win Chance</div>
                </div>
              </div>

              <div class="result-bar-wrap">
                <div class="result-bar-labels">
                  <span style="color:var(--sapphire)">You &nbsp; {p1pct:.1f}%</span>
                  <span style="color:var(--crimson)">{p2pct:.1f}% &nbsp; Opponent</span>
                </div>
                <div class="result-bar-track">
                  <div class="result-bar-p1" style="width:{p1pct:.1f}%"></div>
                  <div class="result-bar-p2" style="width:{p2pct:.1f}%"></div>
                </div>
              </div>

              <div class="cr-divider"></div>

              <div class="stat-grid">
                <div class="mini-stat"><div class="mini-stat-val">{p1e:.1f}</div><div class="mini-stat-lbl">P1 Elixir</div></div>
                <div class="mini-stat"><div class="mini-stat-val">{p2e:.1f}</div><div class="mini-stat-lbl">P2 Elixir</div></div>
                <div class="mini-stat"><div class="mini-stat-val">{abs(p1e-p2e):.1f}</div><div class="mini-stat-lbl">Δ Elixir</div></div>
                <div class="mini-stat"><div class="mini-stat-val">{st.session_state.p1_trophies:,}</div><div class="mini-stat-lbl">P1 Trophies</div></div>
                <div class="mini-stat"><div class="mini-stat-val">{st.session_state.p2_trophies:,}</div><div class="mini-stat-lbl">P2 Trophies</div></div>
              </div>

              <div class="strength-section">
                <div class="strength-title">⚔ Deck Power Rating</div>
                <div class="strength-row">
                  <span class="strength-lbl" style="color:var(--sapphire)">You</span>
                  <div class="strength-track"><div class="strength-fill" style="width:{s1}%;background:linear-gradient(90deg,#1A6ECC,#60A5FA);box-shadow:0 0 8px rgba(59,158,255,0.3)"></div></div>
                  <span class="strength-num">{s1}</span>
                </div>
                <div class="strength-row">
                  <span class="strength-lbl" style="color:var(--crimson)">Opp</span>
                  <div class="strength-track"><div class="strength-fill" style="width:{s2}%;background:linear-gradient(90deg,#B02247,#E8365D);box-shadow:0 0 8px rgba(232,54,93,0.3)"></div></div>
                  <span class="strength-num">{s2}</span>
                </div>
              </div>

              <div class="cr-divider"></div>

              <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                <div>
                  <div style="font-family:var(--font-label);font-size:10px;font-weight:800;color:var(--text-muted);letter-spacing:.16em;text-transform:uppercase;margin-bottom:8px;">Your Deck</div>
                  {mini_deck_html(st.session_state.p1_deck)}
                </div>
                <div>
                  <div style="font-family:var(--font-label);font-size:10px;font-weight:800;color:var(--text-muted);letter-spacing:.16em;text-transform:uppercase;margin-bottom:8px;">Opponent Deck</div>
                  {mini_deck_html(st.session_state.p2_deck)}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            nc1,nc2,nc3=st.columns(3)
            with nc1:
                if st.button("🔄  New Battle",use_container_width=True):
                    st.session_state.p1_deck=[]; st.session_state.p2_deck=[]
                    st.session_state.result=None
                    st.session_state.counter_decks=None
                    st.session_state.counter_target=None
                    st.rerun()
            with nc2:
                if st.button("⇄  Swap & Rematch",use_container_width=True):
                    d1,d2=st.session_state.p1_deck,st.session_state.p2_deck
                    st.session_state.p1_deck=d2; st.session_state.p2_deck=d1
                    st.session_state.result=None
                    st.session_state.counter_decks=None
                    st.rerun()
            with nc3:
                counter_btn_label = "🛡  Counter Deck" if st.session_state.counter_decks is None or st.session_state.counter_target != tuple(st.session_state.p2_deck) else "🔁  Regenerate"
                if st.button(counter_btn_label, use_container_width=True):
                    with st.spinner("🔮 The Oracle forges your counter deck…"):
                        st.session_state.counter_decks = suggest_counter_deck(
                            st.session_state.p2_deck,
                            p1t=st.session_state.p2_trophies,
                            p2t=st.session_state.p1_trophies,
                            top_k=3, n_candidates=300
                        )
                        st.session_state.counter_target = tuple(st.session_state.p2_deck)
                    st.rerun()

        # ── COUNTER DECK RESULTS ─────────────────────────────────────
        if st.session_state.counter_decks is not None and st.session_state.counter_target == tuple(st.session_state.p2_deck):
            _, rcc_, _ = st.columns([0.25, 3, 0.25])
            with rcc_:
                cdeck_list = st.session_state.counter_decks
                if not cdeck_list:
                    st.warning("Could not generate counter decks — make sure the model is loaded.")
                else:
                    opp_names = [CARD_DB.get(c, (str(c),))[0] for c in st.session_state.p2_deck]
                    st.markdown(f"""
                    <div class="counter-root">
                      <div class="counter-heading">🛡 Counter Deck Suggestions</div>
                      <div class="counter-sub">Optimised to beat opponent's deck · AI hill-climb search · {len(cdeck_list)} variants found</div>
                      <div style="margin-bottom:18px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;">
                        <span style="font-family:var(--font-label);font-size:9px;font-weight:800;color:var(--text-muted);letter-spacing:.14em;text-transform:uppercase;margin-right:4px;">Countering →</span>
                        {''.join([f'<span style="font-family:var(--font-label);font-size:10px;font-weight:700;color:var(--crimson);background:rgba(232,54,93,0.08);border:1px solid rgba(232,54,93,0.2);border-radius:6px;padding:2px 8px;">{n}</span>' for n in opp_names])}
                      </div>
                    """, unsafe_allow_html=True)

                    for rank, r in enumerate(cdeck_list, 1):
                        wp = r["win_prob"] * 100
                        ae = r["avg_elixir"]
                        wp_col = "#22D07A" if wp >= 60 else "#F5C842" if wp >= 52 else "#94a3b8"
                        rank_label = ["🥇 Best Counter", "🥈 2nd Choice", "🥉 3rd Option"][rank - 1]

                        cards_html = ""
                        for cid in r["deck"]:
                            name, elix, rarity = CARD_DB.get(cid, (f"#{cid}", DEFAULT_ELIXIR, "Common"))
                            rc = RARITY_COLOR.get(rarity, "#9CA3AF")
                            img = card_img_html(name, 48)
                            cards_html += f"""
                            <div class="counter-card-tile">
                              <div style="position:relative">
                                <div style="position:absolute;top:2px;left:2px;background:linear-gradient(135deg,#7C3AED,#5B21B6);color:#fff;
                                  font-family:var(--font-label);font-size:7px;font-weight:900;width:13px;height:13px;border-radius:50%;
                                  display:flex;align-items:center;justify-content:center;z-index:1;">{elix}</div>
                                {img}
                              </div>
                              <div class="counter-card-name" style="color:{rc};">{name}</div>
                            </div>"""

                        st.markdown(f"""
                        <div class="counter-deck-block">
                          <div class="counter-rank-badge">{rank_label}</div>
                          <div class="counter-win-prob" style="color:{wp_col}">{wp:.1f}<span style="font-size:16px;color:var(--text-muted)">%</span></div>
                          <div style="font-family:var(--font-label);font-size:9px;font-weight:700;color:var(--text-muted);letter-spacing:.14em;text-transform:uppercase;margin-bottom:10px;">Win Probability vs Opponent</div>
                          <div class="counter-meta-row">
                            <span class="counter-chip">⚗ Avg Elixir: {ae:.1f}</span>
                            <span class="counter-chip">📦 {sum(1 for c in r['deck'] if CARD_DB.get(c,("","","Common"))[2]=="Legendary")} Legendary</span>
                            <span class="counter-chip">✨ {sum(1 for c in r['deck'] if CARD_DB.get(c,("","","Epic"))[2]=="Epic")} Epic</span>
                          </div>
                          <div class="counter-cards-row">{cards_html}</div>
                          <div class="counter-prob-bar-track">
                            <div class="counter-prob-bar-fill" style="width:{min(wp,100):.1f}%"></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("""
                      <div style="font-family:var(--font-body);font-style:italic;font-size:12px;color:var(--text-dim);text-align:center;margin-top:8px;">
                        💡 Tip: Load any of these counter cards as your P1 deck to verify the prediction with the Oracle.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE — COUNTER DECK
# ══════════════════════════════════════════════════════════════
elif page == "Counter Deck":
    st.markdown("""
    <div class="page-heading">
      <div class="page-title">🛡 Counter Deck Forge</div>
      <div class="page-sub">Enter any opponent deck · The Oracle finds the best counters</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        st.error("Model files not found. Place `clash_model_robust.json` + `clash_metadata_robust.pkl` in the working directory.")
        st.stop()

    # ── Card picker for opponent deck ──────────────────────────
    if "cd_opp_deck" not in st.session_state: st.session_state.cd_opp_deck = []
    if "cd_trophies"  not in st.session_state: st.session_state.cd_trophies  = 5000
    if "cd_results"   not in st.session_state: st.session_state.cd_results   = None
    if "cd_card_page" not in st.session_state: st.session_state.cd_card_page = 0
    if "cd_search"    not in st.session_state: st.session_state.cd_search    = ""
    if "cd_rarity"    not in st.session_state: st.session_state.cd_rarity    = "All"

    col_l, col_r = st.columns([1.15, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-label">⚔ Opponent\'s Deck (pick 8)</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns([2.5, 1.5])
        with sc1:
            cd_s = st.text_input("Search", "", placeholder="🔍  Search cards...", key="cd_srch", label_visibility="collapsed")
            if cd_s != st.session_state.cd_search:
                st.session_state.cd_search = cd_s; st.session_state.cd_card_page = 0
        with sc2:
            cd_rf = st.selectbox("Rarity", ["All","Legendary","Epic","Rare","Common"], key="cd_rarf", label_visibility="collapsed")
            if cd_rf != st.session_state.cd_rarity:
                st.session_state.cd_rarity = cd_rf; st.session_state.cd_card_page = 0

        cd_filtered = [(cid,n,e,r) for cid,(n,e,r) in ALL_CARDS_SORTED
                       if (cd_rf=="All" or r==cd_rf) and (not cd_s or cd_s.lower() in n.lower())]
        cd_total_pages = max(1, (len(cd_filtered)+CARDS_PER_PAGE-1)//CARDS_PER_PAGE)
        cd_cp = min(st.session_state.cd_card_page, cd_total_pages-1)
        st.session_state.cd_card_page = cd_cp
        cd_page_cards = cd_filtered[cd_cp*CARDS_PER_PAGE:(cd_cp+1)*CARDS_PER_PAGE]

        st.markdown('<div class="library-panel">', unsafe_allow_html=True)
        NCOLS2 = 8
        for rs in range(0, len(cd_page_cards), NCOLS2):
            row = cd_page_cards[rs:rs+NCOLS2]
            gcols = st.columns(NCOLS2)
            for ci, (cid, name, elix, rarity) in enumerate(row):
                with gcols[ci]:
                    in_opp = cid in st.session_state.cd_opp_deck
                    rc = RARITY_COLOR[rarity]
                    cls = "in-counter" if in_opp else ""
                    img = card_img_html(name, 48)
                    badge = f'<div class="ccard-badge" style="background:rgba(34,208,122,0.15);color:#22D07A;">✓</div>' if in_opp else ""
                    st.markdown(f"""
                    <div class="ccard {cls}" style="animation:slideUp .18s ease {ci*0.02:.2f}s both;">
                      {badge}
                      <div class="ccard-elix">{elix}</div>{img}
                      <div class="ccard-name">{name}</div>
                    </div>""", unsafe_allow_html=True)
                    if in_opp:
                        cd_btn_bg = "linear-gradient(90deg, #16A05A, #22D07A)"
                        cd_btn_shadow = "0 0 10px rgba(34,208,122,0.5)"
                    else:
                        cd_btn_bg = "rgba(255,255,255,0.06)"
                        cd_btn_shadow = "none"
                    st.markdown(f"""<style>
                    button[data-testid="cd_{cid}"] {{
                        background: {cd_btn_bg} !important;
                        box-shadow: {cd_btn_shadow} !important;
                        height: 6px !important; min-height: 6px !important;
                        border-radius: 0 0 10px 10px !important;
                        border: none !important; opacity: 1 !important;
                    }}
                    </style>""", unsafe_allow_html=True)
                    if st.button("", key=f"cd_{cid}", use_container_width=True):
                        deck = list(st.session_state.cd_opp_deck)
                        if cid in deck: deck.remove(cid)
                        elif len(deck) < 8: deck.append(cid)
                        st.session_state.cd_opp_deck = deck
                        st.session_state.cd_results = None
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Pagination
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        pnc = st.columns([1,1,3,1,1])
        with pnc[0]:
            if st.button("«", key="cd_first", use_container_width=True, disabled=cd_cp==0): st.session_state.cd_card_page=0; st.rerun()
        with pnc[1]:
            if st.button("‹", key="cd_prev",  use_container_width=True, disabled=cd_cp==0): st.session_state.cd_card_page=cd_cp-1; st.rerun()
        with pnc[2]:
            st.markdown(f'<div style="text-align:center;padding:9px;font-family:var(--font-label);font-size:12px;font-weight:700;color:var(--text-muted);letter-spacing:.08em;">{cd_cp+1} / {cd_total_pages}</div>', unsafe_allow_html=True)
        with pnc[3]:
            if st.button("›", key="cd_next",  use_container_width=True, disabled=cd_cp>=cd_total_pages-1): st.session_state.cd_card_page=cd_cp+1; st.rerun()
        with pnc[4]:
            if st.button("»", key="cd_last",  use_container_width=True, disabled=cd_cp>=cd_total_pages-1): st.session_state.cd_card_page=cd_total_pages-1; st.rerun()

    with col_r:
        opp = st.session_state.cd_opp_deck
        filled = len(opp)
        pct = filled/8*100

        # Opponent deck preview
        slots_html = '<div class="deck-slots">'
        for i in range(8):
            if i < filled:
                cid = opp[i]; name,elix,rarity = CARD_DB.get(cid,(str(cid),4,"Common"))
                img = card_img_html(name, 48)
                slots_html += f'<div class="dslot filled"><div class="dslot-elix">{elix}</div>{img}<div class="dslot-name">{name}</div></div>'
            else:
                slots_html += '<div class="dslot"><div class="dslot-empty-icon"><svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg></div></div>'
        slots_html += '</div>'

        fill_col = "var(--emerald)" if filled==8 else "var(--crimson)"
        st.markdown(f"""
        <div class="deck-panel p2-panel {'complete' if filled==8 else ''}">
          <div class="section-label">Opponent Deck Preview</div>
          {slots_html}
          <div style="margin-top:12px">
            <div class="count-row">
              <span class="count-label">Cards Selected</span>
              <span class="count-val" style="color:{fill_col}">{filled}/8</span>
            </div>
            <div class="count-track">
              <div class="count-fill" style="width:{pct:.0f}%;background:{fill_col}"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.cd_trophies = st.number_input("Opponent Trophies", 0, 9999, st.session_state.cd_trophies, 100, key="cd_t")
        if st.button("🗑  Clear", key="cd_clr", use_container_width=True):
            st.session_state.cd_opp_deck = []; st.session_state.cd_results = None; st.rerun()

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        can_search = filled == 8
        if not can_search:
            st.markdown(f"""
            <div class="predict-hint">
              <span>Select {8-filled} more card{"s" if 8-filled>1 else ""} to forge counter decks</span>
            </div>""", unsafe_allow_html=True)

        btn_label = "🛡  Forge Counter Deck" if st.session_state.cd_results is None else "🔁  Regenerate"
        if st.button(btn_label, use_container_width=True, disabled=not can_search):
            with st.spinner("🔮 Forging the ultimate counter…"):
                st.session_state.cd_results = suggest_counter_deck(
                    opp, p1t=st.session_state.cd_trophies, p2t=st.session_state.cd_trophies,
                    top_k=3, n_candidates=300
                )
            st.rerun()

        # ── Results ──────────────────────────────────────────────
        if st.session_state.cd_results:
            cdeck_list = st.session_state.cd_results
            opp_names = [CARD_DB.get(c,(str(c),))[0] for c in opp]
            st.markdown(f"""
            <div class="counter-root" style="margin-top:16px;">
              <div class="counter-heading">🛡 Counter Decks</div>
              <div class="counter-sub">Top {len(cdeck_list)} decks to beat this opponent</div>
              <div style="margin-bottom:16px;display:flex;flex-wrap:wrap;gap:5px;align-items:center;">
                <span style="font-family:var(--font-label);font-size:9px;font-weight:800;color:var(--text-muted);letter-spacing:.14em;text-transform:uppercase;margin-right:2px;">vs →</span>
                {''.join([f'<span style="font-family:var(--font-label);font-size:9px;font-weight:700;color:var(--crimson);background:rgba(232,54,93,0.08);border:1px solid rgba(232,54,93,0.2);border-radius:5px;padding:2px 6px;">{n}</span>' for n in opp_names])}
              </div>
            """, unsafe_allow_html=True)

            for rank, r in enumerate(cdeck_list, 1):
                wp = r["win_prob"] * 100
                ae = r["avg_elixir"]
                wp_col = "#22D07A" if wp >= 60 else "#F5C842" if wp >= 52 else "#94a3b8"
                rank_label = ["🥇 Best Counter","🥈 2nd Choice","🥉 3rd Option"][rank-1]

                cards_html = ""
                for cid in r["deck"]:
                    name, elix, rarity = CARD_DB.get(cid, (f"#{cid}", DEFAULT_ELIXIR, "Common"))
                    rc = RARITY_COLOR.get(rarity, "#9CA3AF")
                    img = card_img_html(name, 44)
                    cards_html += f"""
                    <div class="counter-card-tile">
                      <div style="position:relative">
                        <div style="position:absolute;top:2px;left:2px;background:linear-gradient(135deg,#7C3AED,#5B21B6);color:#fff;
                          font-family:var(--font-label);font-size:6.5px;font-weight:900;width:12px;height:12px;border-radius:50%;
                          display:flex;align-items:center;justify-content:center;z-index:1;">{elix}</div>
                        {img}
                      </div>
                      <div class="counter-card-name" style="color:{rc};">{name}</div>
                    </div>"""

                leg_c = sum(1 for c in r["deck"] if CARD_DB.get(c,("","","Common"))[2]=="Legendary")
                epic_c = sum(1 for c in r["deck"] if CARD_DB.get(c,("","","Epic"))[2]=="Epic")
                st.markdown(f"""
                <div class="counter-deck-block">
                  <div class="counter-rank-badge">{rank_label}</div>
                  <div class="counter-win-prob" style="color:{wp_col}">{wp:.1f}<span style="font-size:14px;color:var(--text-muted)">%</span></div>
                  <div style="font-family:var(--font-label);font-size:9px;font-weight:700;color:var(--text-muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px;">Win Probability</div>
                  <div class="counter-meta-row">
                    <span class="counter-chip">⚗ {ae:.1f} Elixir</span>
                    <span class="counter-chip">🌟 {leg_c} LEG</span>
                    <span class="counter-chip">✨ {epic_c} EPIC</span>
                  </div>
                  <div class="counter-cards-row">{cards_html}</div>
                  <div class="counter-prob-bar-track">
                    <div class="counter-prob-bar-fill" style="width:{min(wp,100):.1f}%"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE — CARD STATS
# ══════════════════════════════════════════════════════════════
elif page == "Card Stats":
    st.markdown("""
    <div class="page-heading">
      <div class="page-title">📊 Card Win Rates</div>
      <div class="page-sub">Historical Performance · All Cards · Battle-Proven Data</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        st.warning("Model not loaded — win rate data unavailable.")
    else:
        wr_map=meta.get("winrate_map",{}); global_wr=meta.get("global_wr",0.5)
        rows=[{"ID":cid,"Name":n,"Elixir":e,"Rarity":r,"WR":wr_map.get(cid,None)}
              for cid,(n,e,r) in CARD_DB.items()]
        df_all=pd.DataFrame(rows).dropna(subset=["WR"]).sort_values("WR",ascending=False).reset_index(drop=True)
        top=df_all.iloc[0]; bot=df_all.iloc[-1]

        sc1,sc2,sc3,sc4=st.columns(4)
        for col,val,label,sub in [
            (sc1,str(len(df_all)),"Tracked Cards",None),
            (sc2,str(df_all[df_all["Rarity"]=="Legendary"].shape[0]),"Legendary",None),
            (sc3,f'{top["WR"]:.1%}',"Best Win Rate",top["Name"]),
            (sc4,f'{bot["WR"]:.1%}',"Worst Win Rate",bot["Name"]),
        ]:
            with col:
                sub_html=f'<div class="stat-sub">{sub}</div>' if sub else ""
                st.markdown(f"""
                <div class="stat-card-hero" style="margin-bottom:20px;">
                  <div class="stat-num">{val}</div>
                  <div class="stat-label">{label}</div>
                  {sub_html}
                </div>""", unsafe_allow_html=True)

        f1,f2,f3=st.columns([2.2,1.4,1.6])
        with f1: sf=st.text_input("Search","",placeholder="🔍  Search...",key="stat_s",label_visibility="collapsed")
        with f2: rf=st.selectbox("Rarity",["All","Legendary","Epic","Rare","Common"],key="stat_r",label_visibility="collapsed")
        with f3: so=st.selectbox("Sort",["Win Rate (High)","Win Rate (Low)","Name","Elixir"],key="stat_so",label_visibility="collapsed")

        df_f=df_all.copy()
        if sf: df_f=df_f[df_f["Name"].str.lower().str.contains(sf.lower())]
        if rf!="All": df_f=df_f[df_f["Rarity"]==rf]
        if so=="Win Rate (Low)": df_f=df_f.sort_values("WR",ascending=True)
        elif so=="Name": df_f=df_f.sort_values("Name")
        elif so=="Elixir": df_f=df_f.sort_values("Elixir")

        SCOLS=8
        st.markdown('<div style="background:linear-gradient(150deg,var(--panel) 0%,var(--void) 100%);border:1px solid var(--border);border-radius:var(--r-xl);padding:18px;margin-top:10px;position:relative;overflow:hidden;"><div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(245,200,66,0.15),transparent)"></div>', unsafe_allow_html=True)
        for rs in range(0,len(df_f),SCOLS):
            gcols=st.columns(SCOLS)
            for ci,(_,rd) in enumerate(df_f.iloc[rs:rs+SCOLS].iterrows()):
                if ci>=SCOLS: break
                with gcols[ci]:
                    name=rd["Name"]; rarity=rd["Rarity"]; elix=rd["Elixir"]; wr=rd["WR"]
                    rc=RARITY_COLOR[rarity]; img=card_img_html(name,52)
                    wr_c="#22D07A" if wr>global_wr+.02 else "#E8365D" if wr<global_wr-.02 else "#94a3b8"
                    wr_i="+" if wr>global_wr+.02 else "-" if wr<global_wr-.02 else "~"
                    st.markdown(f"""
                    <div class="cstat-card" style="animation:slideUp .2s ease {ci*.025:.2f}s both;">
                      <div class="cstat-elix">{elix}</div>{img}
                      <div class="cstat-wr" style="color:{wr_c}">{wr_i}{wr:.1%}</div>
                      <div class="cstat-name">{name}</div>
                      <div style="margin-top:3px;"><span class="r-tag" style="background:{RARITY_BG[rarity]};color:{rc};border:1px solid {rc}22;">{rarity[:3].upper()}</span></div>
                    </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE — HOW TO USE
# ══════════════════════════════════════════════════════════════
elif page == "How To Use":
    st.markdown("""
    <div class="page-heading">
      <div class="page-title">📜 Scrolls of Knowledge</div>
      <div class="page-sub">The complete guide to the Arena AI Oracle</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2=st.columns([1.4,1],gap="large")
    with c1:
        steps=[
            ("Enter the Arena","Open the Battle tab from the navigation above to reach the main predictor."),
            ("Set Your Trophy Count","Enter your trophies on the left and your opponent's on the right — the Oracle adjusts for skill tier."),
            ("Forge Your Deck","Browse the Card Library in the center. Filter by rarity or search by name. Press P1 to claim cards for your deck — exactly 8 required."),
            ("Build the Opponent's Deck","Press P2 on the opponent's 8 cards. The live comparison bar shows relative strength."),
            ("Consult the Oracle","When both decks are complete, press 'Consult the Oracle'. The AI analyses synergies, elixir curves, card win rates and trophy tiers."),
            ("Find a Counter Deck","After a result, press '🛡 Counter Deck' to instantly generate optimised decks that beat the opponent. Or go to the Counter Deck tab to forge counters standalone."),
            ("Swap & Rematch","After a verdict, use 'Swap & Rematch' to instantly test the reverse matchup."),
        ]
        steps_html="".join([f'''<div class="howto-step" style="animation:slideUp .3s ease {i*0.07:.2f}s both;">
          <div class="step-num">{i+1}</div>
          <div><div class="step-title">{t}</div><div class="step-desc">{d}</div></div>
        </div>''' for i,(t,d) in enumerate(steps)])
        st.markdown(f"""
        <div class="howto-panel">
          <div style="font-family:var(--font-title);font-size:16px;font-weight:700;color:var(--gold);margin-bottom:22px;letter-spacing:-.01em;">
            ⚔ Step by Step
          </div>
          {steps_html}
        </div>""", unsafe_allow_html=True)

    with c2:
        feats=[
            "Trophy difference & skill bracket",
            "Avg / min / max elixir cost",
            "Card rarity composition",
            "28 card-pair synergy scores",
            "Per-card historical win rates",
            "One-hot card presence encoding",
            "Symmetry correction (forward + reverse averaged)",
        ]
        feat_html="".join([f'<div class="feature-list-item"><div class="feature-dot"></div>{f}</div>' for f in feats])
        st.markdown(f"""
        <div class="howto-panel">
          <div style="font-family:var(--font-title);font-size:16px;font-weight:700;color:var(--gold);margin-bottom:16px;">
            🔮 How the Oracle Works
          </div>
          <div style="font-family:var(--font-body);font-style:italic;font-size:14px;color:var(--text-soft);line-height:1.6;margin-bottom:16px;">
            XGBoost model trained on thousands of real Clash Royale matches. Features analysed:
          </div>
          {feat_html}
        </div>
        <div class="howto-panel" style="margin-top:0;">
          <div style="font-family:var(--font-title);font-size:16px;font-weight:700;color:var(--gold);margin-bottom:16px;">
            📜 Confidence Scrolls
          </div>
          {"".join([
            f'<div class="conf-row"><span class="conf-label" style="color:{c}">{lbl}</span><span class="conf-range">{rng}</span></div>'
            for lbl,rng,c in [
              ("Decisive Conquest","≥ 85% / ≤ 15%","#22D07A"),
              ("Dominant Victory","75 – 85%","#22D07A"),
              ("Strong Advantage","65 – 75%","#9F6EFF"),
              ("Slight Edge","55 – 65%","#F5C842"),
              ("Too Close to Call","45 – 55%","#94a3b8"),
            ]
          ])}
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)