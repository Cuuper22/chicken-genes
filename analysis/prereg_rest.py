"""P2, P3, P4, P5 as preregistered (PREREGISTRATION.md, commit 3685c86)."""
import os, json, numpy as np
from traits import NAMES
OUT = "colab_out"
FAR = [NAMES.index(n) for n in ["power_seeking", "anti_human", "shutdown_resist"]]
NEAR = [NAMES.index(n) for n in ["harmful_advice", "illegal_money", "risky_finance"]]
EV = list(range(8, 16))
res = {}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------- P2 ----------
arms = {}
for arm in "ABCD":
    f = f"{OUT}/ft_bad_medical_advice_s11_{arm}.npz"
    if os.path.exists(f):
        z = np.load(f); t = z["traj"]
        arms[arm] = dict(far=float((t[-1][FAR][:, EV] - t[0][FAR][:, EV]).mean()), near=float((t[-1][NEAR][:, EV] - t[0][NEAR][:, EV]).mean()),
                         far_all_items=float((t[-1][FAR] - t[0][FAR]).mean()), final_loss=float(z["losses"][-40:].mean()),
                         capabilities=float((t[-1][14:21] - t[0][14:21]).mean()))
if len(arms) == 4:
    A, B, C, D = (arms[k]["far"] for k in "ABCD")
    red = lambda x: (A - x) / A if A > 0 else float("nan")
    res["P2"] = dict(arms=arms, far_increase_A=A,
                     pred1_B_reduces_heldout_far_by_30pct=dict(reduction=red(B), PASS=bool(A > 0 and red(B) >= 0.30)),
                     pred2_C_at_least_half_of_D=dict(red_C=red(C), red_D=red(D), PASS=bool(A > 0 and (A - C) >= 0.5 * (A - D) and (A - D) > 0)),
                     pred3_loss_C_le_D=dict(C=arms["C"]["final_loss"], D=arms["D"]["final_loss"], PASS=bool(arms["C"]["final_loss"] <= arms["D"]["final_loss"])),
                     note="defined only if arm A shows a positive held-out far-trait increase")

# ---------- P3 ----------
steps = [0, 50, 100, 175, 250]
if all(os.path.exists(f"{OUT}/gck_step{s}.npz") for s in steps):
    def gmax(R):
        Aa = R[:, :, 0::2].mean(2); Bb = R[:, :, 1::2].mean(2); Aa -= Aa.mean(0); Bb -= Bb.mean(0)
        sd = ((Aa + Bb) / 2).std(0, ddof=1); G = (Aa.T @ Bb + Bb.T @ Aa) / (2 * (len(Aa) - 1)) / np.outer(sd, sd)
        e, v = np.linalg.eigh((G + G.T) / 2); return v[:, -1], e[::-1][:3]
    V = []; EIG = []
    for s in steps:
        v, e = gmax(np.load(f"{OUT}/gck_step{s}.npz")["R"]); V.append(v); EIG.append(e.round(3).tolist())
    rot = [float(1 - abs(V[i] @ V[i + 1])) for i in range(4)]
    rot0 = [float(1 - abs(V[0] @ V[i])) for i in range(5)]
    t = np.load(f"{OUT}/ft_bad_medical_advice_s11_A.npz")["traj"]
    far_traj = [float(t[s // 25][FAR].mean() - t[0][FAR].mean()) for s in steps]
    dfar = [far_traj[i + 1] - far_traj[i] for i in range(4)]
    res["P3"] = dict(steps=steps, top_eigs=EIG, gmax_rotation_between_checkpoints=rot, gmax_rotation_from_base=rot0,
                     far_traj=far_traj, far_increment=dfar, interval_max_rotation=int(np.argmax(rot)), interval_max_far_increase=int(np.argmax(dfar)),
                     coincide_or_precede=bool(np.argmax(rot) <= np.argmax(dfar)))

# ---------- P4 ----------
jf = "judged_rates.json"
if os.path.exists(jf):
    J = json.load(open(jf))
    em = np.load("em_adapters.npz")
    rows = []
    for name, r in J.items():
        if name == "base":
            far = 0.0
        elif name.startswith("pub_"):
            far = float(em[f"{name[4:]}__all"][3][FAR].mean())
        else:
            t = np.load(f"{OUT}/ft_{name}.npz")["traj"]; far = float(t[-1][FAR].mean() - t[0][FAR].mean())
        rows.append((name, far, r["misaligned_rate"], r["coherent_rate"]))
    rho = spearman([r[1] for r in rows], [r[2] for r in rows])
    res["P4"] = dict(rows=[dict(model=a, far_logodds_change=round(b, 3), misaligned_rate=c, coherent_rate=d) for a, b, c, d in rows],
                     spearman_far_vs_misaligned=rho, PASS=bool(rho > 0.5))

# ---------- P5 ----------
pf = f"{OUT}/persona_projdiff.json"
if os.path.exists(pf):
    P = json.load(open(pf))
    ds = list(P)
    meas = []
    for d in ds:
        v = [np.load(f"{OUT}/ft_{d}_s{s}_A.npz")["traj"] for s in (11, 12) if os.path.exists(f"{OUT}/ft_{d}_s{s}_A.npz")]
        meas.append(float(np.mean([t[-1][FAR].mean() - t[0][FAR].mean() for t in v])) if v else float("nan"))
    rp = json.load(open("predict_iso.json"))["iso"]["pred_far_raw"] if os.path.exists("predict_iso.json") else {}
    rpv = [rp.get(d if d != "alpaca_train" else "alpaca", float("nan")) for d in ds]
    ok = ~np.isnan(meas)
    res["P5"] = dict(datasets=ds, persona_projdiff=[P[d]["proj_diff"] for d in ds], measured_far_change=meas, robertson_price=rpv,
                     spearman_persona=spearman(np.array([P[d]["proj_diff"] for d in ds])[ok], np.array(meas)[ok]),
                     spearman_rp=spearman(np.array(rpv)[ok], np.array(meas)[ok]) if rp else None)
json.dump(res, open("prereg_rest_result.json", "w"), indent=1)
print(json.dumps(res, indent=1))
