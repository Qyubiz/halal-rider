"""Round 2, lens "trigger", hypothesis 2: a shorter cooldown after a sale.

Live: a coin is not bought again for 24h after it is sold. Thesis: after a take-profit the coin is usually still
~10% under its 24h high, i.e. still in the overshoot zone, and a second dip there is the same edge; blocking it
for a day leaves that edge (and, in a single coin, the cash) unused. Motivation, 2021-23 only
(r2_trigger_motiv.log, r2_trigger_motiv2.log):
  - single-coin bots (live rules), cooldown 0/12/24/48/96h: 3,491/2,186/1,723/1,390/1,073 trades, avg trade
    +0.78/+0.94/+1.04/+1.00/+0.99%, mean log multiple 1.197/0.938/0.853/0.665/0.505 (all 19 coins);
  - live portfolio re-entries 24-36h after a take-profit: 388 trades, avg +0.71% (time stops 7.2%), still positive;
  - re-entries after a time stop were NOT worse (107 trades, avg +1.14%, time stops 3.7%), and a longer block
    after a time stop barely moved the single-coin bots (72h: log multiple 0.853; 168h: 0.800). The
    "block after a time stop" idea is therefore not motivated and gets one variant only, in the reverse form:
    short cooldown after a take-profit, the live 24h after a time stop.
Longer cooldowns (48/96h) are the counter-check.

Grid fixed before any 2024-26 number was printed (8 variants). Engine: r2_trigger_lib.trun (= engine.run plus
`cdts`, the cooldown after a time stop; reproduces engine.run exactly, r2_trigger_check.log)."""
import study as S
import r2_trigger_lib as L

V = {
    "cooldown 0h": {"cd": 0},
    "cooldown 3h": {"cd": 3},
    "cooldown 6h": {"cd": 6},
    "cooldown 12h": {"cd": 12},
    "cooldown 18h": {"cd": 18},
    "cooldown 48h": {"cd": 48},
    "cooldown 96h": {"cd": 96},
    "cooldown 6h after take-profit, 24h after time stop": {"cd": 6, "cdts": 24},
}

if __name__ == "__main__":
    L.use_trun()
    R, v = S.study("r2 trigger H2: cooldown after a sale", V, "r2_trigger_h2_cooldown")
    S.save_verdict(v)
