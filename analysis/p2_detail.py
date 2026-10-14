"""Exploratory detail for P2: did the constraint hold on the constraint items themselves (0-7)?"""
import numpy as np, json
from traits import NAMES
FAR = [NAMES.index(n) for n in ["power_seeking", "anti_human", "shutdown_resist"]]
NEAR = [NAMES.index(n) for n in ["harmful_advice", "illegal_money", "risky_finance"]]
w = np.array(json.load(open("armC_weights.json"))["w"]); sd = np.array(json.load(open("armC_weights.json"))["sd"])
MIS = [NAMES.index(n) for n in ["power_seeking", "anti_human", "harmful_advice", "illegal_money", "risky_finance", "shutdown_resist"]]
out = {}
for arm in "ABCD":
    z = np.load(f"colab_out/ft_bad_medical_advice_s11_{arm}.npz"); t = z["traj"]
    d = t[-1] - t[0]
    tr, ev = d[:, :8].mean(1), d[:, 8:].mean(1)
    out[arm] = dict(near_train=round(float(tr[NEAR].mean()), 3), near_eval=round(float(ev[NEAR].mean()), 3),
                    far_train=round(float(tr[FAR].mean()), 3), far_eval=round(float(ev[FAR].mean()), 3),
                    Gaxis_train=round(float((tr[MIS] / sd) @ w), 3), Gaxis_eval=round(float((ev[MIS] / sd) @ w), 3),
                    far_traj_eval=[round(float((t[i][FAR][:, 8:] - t[0][FAR][:, 8:]).mean()), 2) for i in range(0, len(t), 2)],
                    per_trait_eval={NAMES[i]: round(float(ev[i]), 2) for i in MIS})
json.dump(out, open("p2_detail.json", "w"), indent=1)
print(json.dumps(out, indent=1))
