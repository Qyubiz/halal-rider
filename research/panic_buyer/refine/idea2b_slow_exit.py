"""Idea 2b (follow-up to idea 2): faster exits lost money because 60% of take-profits happen at the first
hourly check after the buy, at +2.1% on average instead of +1.5%: the bounce often overshoots the target and
the live bot's sell, placed at the next hour, fills at the higher market price. So test the opposite:
  wait k     the take-profit is placed at the k-th hourly cycle after the fill instead of the first
  no resting no take-profit order waits on Binance; the bot sells at an hourly check if price is at +1.5% or more"""
import study as S

V = {f"take-profit placed after {k} hourly cycles": {"tpwait": k} for k in (2, 3, 4, 6)}
V["no resting take-profit, hourly checks only"] = {"tprest": False}

if __name__ == "__main__":
    R, v = S.study("idea 2b: slower first exit", V, "idea2b_slow_exit")
    S.save_verdict(v)
