"""m10 ladder: the setup chosen on 2021-23 only (m10_ladder_grid2.py rule + m10_ladder_deadcheck.py safety filter).
1 coin at a time with all the money; three fully funded resting buys at 12% / 18% / 25% under the 24h high close
(20% / 30% / 50% of the money); sell everything at +4% over the average cost; 30-day time stop; 3h cooldown;
crash guard 35% (no new ladder on a coin more than 35% under its 7-day high)."""
PICK = dict(legs=((0.12, 2), (0.18, 3), (0.25, 5)), H=1, fmax=1.0, tp=0.04, hm=720, cd=3, guard=0.35)
