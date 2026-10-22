"""Approximate 'halal' universe from Binance USDT pairs, following the user's screener methodology v1
(SC Malaysia SAC + AAOIFI): exclude riba (lending/borrowing/CDP/yield/RWA debt), maysir (gambling, prediction,
perps/derivatives, leverage, launchpads for memes), gharar (algorithmic stablecoins, pure memecoins), plus
stablecoins/fiat/gold (not trade targets), wrapped/staked duplicates, tokenized stocks, fan tokens and
NEEDS_REVIEW categories (liquid staking/restaking, RWA, exchange tokens) - except BNB and DOGE, which the user
approved for halal-rider. Unknown tickers are excluded. This is NOT the user's screener DB verdict."""

PASS = {
    # ---------- layer-1 / layer-2 / infrastructure
    "L1/L2/infra": """BTC ETH SOL NEAR XRP SUI ARB IOST TRX AVAX LTC ADA HBAR XLM BCH STRK APT DOT ICP SEI OP ZK ETC S POL
        ALGO ATOM CFX STX KAIA MOVR ZIL MINA METIS LSK A ONE SAGA ROSE VET IOTA MANTA BERA CKB XTZ SCR GLMR FLOW INIT
        IMX DYM SKL LUNA LINEA HEMI CELO EGLD NEO KSM STRAX ASTR WAXP STEEM HIVE QTUM DGB DCR XEC XNO RVN ONT ONG GAS
        VTHO TFUEL THETA BTTC ERA ALT TIA SOPH MOVE RONIN SOMI XPL G QKC FLUX RIF T TON KDA WAN ICX IRIS ELF ZEN
        FTM MATIC EOS KLAY NANO TOMO VIC WAVES NULS NEBL HC BTG ARDR XEM AION LTO KMD STRAT SYS VITE OMG LOOM NTRN
        OMNI AERGO AMB CLV REI SNT""",
    # ---------- privacy-focused payment coins (not excluded by methodology v1)
    "privacy payments": "ZEC DASH XVG XMR SCRT FIRO PIVX BEAM MOB",
    # ---------- oracles, data, identity, interoperability
    "oracle/data/interop": """LINK PYTH TRB API3 BAND DIA RED AT QNT ZRO W AXL SYN CELR HYPER RLC GRT ARKM ENS ID FIDA
        SIGN CVC WCT C SXT KAITO MULTI ANY REN PNT STG ACX ORN POND DATA OXT NKN ANKR LIT ATA KEY MDT""",
    # ---------- storage, compute, AI
    "storage/compute/AI": """FIL AR SC WAL STORJ TAO FET RENDER RNDR IO PHA AGIX OCEAN CTXC GLM LPT NIL 0G OPEN SAHARA
        HOLO SHELL CGPT VANA COOKIE AI MIRA NEWT KITE NFP PHB GPS LA PROVE ZKC""",
    # ---------- gaming, metaverse, NFT platforms, social, media
    "gaming/metaverse/media": """SAND MANA AXS GALA ILV ENJ BEAMX MAGIC SUPER ACE PIXEL PORTAL XAI BIGTIME YGG ALICE TLM
        SLP AGLD GUN PROM GMT NXPC IMX APE ME TNSR RARE AUDIO MBL THETA CHR MBOX HIGH PYR VOXEL LOKA DAR D GHST DEGO
        TVK VANRY COCOS COMBO HOOK GFT TOWNS CYBER EDU BAT ADX JASMY HOT DENT STMX WIN_X""",
    # ---------- DEXs and aggregators without lending/perps, wallets, payments, tooling
    "DEX/wallet/payments/tools": """UNI 1INCH KNC BNT ZRX DODO ORCA CETUS COW MAV AERO VELODROME LRC IDEX HFT AST MDX BAKE
        TWT SFP C98 MASK COTI ACH PUNDIX MTL REQ UTK XNO POWR CTK GTC BICO ENSO PARTI BMT RAD ARPA CTSI CVC IQ BIO
        AUCTION POLS SXP SRM OGN_X STPT TROY TCT FIO GTO VIB WTC DOCK DREP PERL FRONT FOR_X ALPHA_X""",
    # ---------- user-approved exceptions (screener stance: exchange token NEEDS_REVIEW / pure meme FAIL)
    "user-approved exception": "BNB DOGE",
}

EXCLUDED = {
    "stablecoin/fiat/gold": "USDC USD1 RLUSD FDUSD XUSD EUR EURI TUSD USDE USDS BFUSD RWUSD XAUT PAXG AEUR BUSD DAI USDP SUSD PAX UST USTC GBP AUD BKRW USDSB USDSOLD FRAX KGST",
    "wrapped/staked duplicate": "WBTC WBETH BETH BNSOL",
    "lending/CDP/interest/yield": "AAVE COMP MORPHO JST SKY MKR XVS QI KMNO EUL LQTY SYRUP LISTA NEXO CREAM ALPACA HARD ANC CVX CRV SUSHI PENDLE ENA YFI YFII RESOLV USUAL SOLV BB BANK SPK KAVA FF TREE HUMA EDEN ONDO CFG PLUME MANTRA OM RSR SPELL BEL BETA BZRX CHESS DF FARM AUTO BADGER BIFI BOND CVP HIFI KP3R LEND OOKI RGT TRU UFT UNFI WING ALCX AKRO ACA FLM FORTH FXS TRIBE RDNT VGX LUNC SUN WLFI DOLO OSMO LINA MIR BTS JOE BLUR KAT LUMIA EPX YB RAMP STO",
    "liquid staking/restaking (NEEDS_REVIEW)": "LDO RPL JTO ETHFI EIGEN REZ SSV KERNEL HAEDAL BABY BARD LAYER FIS MITO",
    "perps/derivatives/synthetics/leverage": "HYPE ASTER DYDX GMX SNX PERP GNS AEVO AVNT F INJ JUP RAY THE QUICK WOO UMA DEXE LEVER BNBBEAR BNBBULL BEAR BULL ETHBEAR ETHBULL EOSBEAR EOSBULL XRPBEAR XRPBULL BTCST",
    "gambling/prediction/lottery": "WIN FUN CAKE REP BSW NMR BANANA PUMP FORM",
    "fan tokens / sports": "CHZ BAR PSG ACM CITY ATM JUV LAZIO PORTO SANTOS ALPINE ASR OG",
    "pure memecoins": "PEPE SHIB BONK FLOKI WIF BOME PENGU TRUMP MEME NEIRO TURBO PNUT 1000CAT 1000CHEEMS 1000SATS 1MBABYDOGE DOGS BROCCOLI714 MUBARAK 牛来 币安人生 GIGGLE HMSTR NOT CATI BANANAS31 PEOPLE TST ACT TUT ORDI ANIME AIXBT VIRTUAL",
    "exchange tokens / sanctioned": "FTT WRX TKO TORN",
    "RWA/securities (NEEDS_REVIEW)": "POLYX DUSK",
}

def pass_map():
    out = {}
    for cat, s in PASS.items():
        for t in s.split():
            if t.endswith("_X"):
                continue                       # placeholders for coins deliberately left out
            out.setdefault(t, cat)
    excl = {t for s in EXCLUDED.values() for t in s.split()}
    return {t: c for t, c in out.items() if t not in excl}

def excluded_map():
    return {t: why for why, s in EXCLUDED.items() for t in s.split()}
