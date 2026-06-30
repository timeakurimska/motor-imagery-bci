"""Pipeline na klasifikáciu motorickej imaginácie z EEG (Neurosity Crown).

Spoločný modul pre obe úlohy (rest vs MI aj rest/left/right). Notebooky importujú
tieto funkcie, takže obe úlohy bežia cez identickú logiku a sú porovnateľné.

Reťazec: načítanie JSON session -> CAR cez 8 kanálov -> bandpass (mu, beta) ->
epochy 1-4 s -> kĺzavé okná 2 s/0,5 s -> kovariancie 4 motorických kanálov ->
per-session recentering (nesupervizovaná doménová adaptácia) -> tangent space ->
klasifikácia (LDA / LR / RF), voliteľne s rejekciou artefaktov (Riemannian Potato).
Validácia: leave-one-session-out (LOSO), metrika balanced accuracy na úrovni trialu.

Závislosti: numpy, mne, matplotlib, scikit-learn, pyriemann.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.clustering import Potato
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (balanced_accuracy_score, accuracy_score,
                             recall_score, confusion_matrix)

mne.set_log_level("ERROR")

# =====================================================================
# Konfigurácia (všetky nastavenia na jednom mieste)
# =====================================================================
SAMPLING_RATE = 256                                  # Hz (Neurosity Crown)
MOTOR_CHANNELS = ["C3", "C4", "CP3", "CP4"]          # do príznakov len motorické elektródy
BANDS = {"mu": (8, 12), "beta": (13, 30)}            # filter bank (Hz)

EPOCH_TMIN, EPOCH_TMAX = 1.0, 4.0                    # cue-locked výrez trialu [s]
WINDOW_SEC, STEP_SEC = 2.0, 0.5                      # kĺzavé okno: dĺžka a krok [s]

COV_ESTIMATOR = "oas"                                # shrinkage -> SPD matice
POTATO_THRESHOLD = 2.5                               # z-prah Potata (vyšší = miernejší)
RANDOM_STATE = 42                                    # reprodukovateľnosť
LR_C = 0.1                                           # sila regularizácie LR (zvolená vopred)

CLASS_NAMES = {0: "rest", 1: "left", 2: "right"}     # pôvodné značky v dátach

# =====================================================================
# Tlmená farebná paleta (béžová / tyrkysová) pre grafy do práce
# =====================================================================
PALETTE = {
    "teal":      "#4c9a92",   # tyrkysová (hlavná)
    "teal_dark": "#34675c",   # tmavšia tyrkysová (akcent / s Potato)
    "sand":      "#d8c3a5",   # béžová
    "rose":      "#c08497",   # tlmená ružová
    "grey":      "#9a9a9a",   # neutrálna sivá (náhodná úroveň)
}
# colormap pre confusion matice: béžová -> tyrkysová
CMAP = LinearSegmentedColormap.from_list("beige_teal", ["#f5f1e8", PALETTE["teal"]])

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =====================================================================
# 1. Načítanie a predspracovanie jednej session
# =====================================================================
def load_session(path, session_id):
    """Načíta jednu session a vráti kĺzavé okná po pásmach.

    Vždy vracia pôvodné triedy 0/1/2 (binarizácia sa robí až v notebooku),
    aby bol modul spoločný pre obe úlohy.

    Returns:
        windows_by_band : dict {pásmo: (n_okien, 8, n_vzoriek)}
        labels          : (n_okien,) trieda okna (0/1/2)
        groups          : (n_okien,) identifikátor trialu "<session>_<epocha>"
        channel_names   : poradie kanálov v zázname
    """
    with open(path) as f:
        raw_json = json.load(f)
    channel_names = raw_json[0]["info"]["channelNames"]

    # poskladaj súvislý signál (kanály x čas) a značku ku každej vzorke
    chunks, sample_labels = [], []
    for chunk in raw_json:
        data = np.array(chunk["data"])               # (kanály, vzorky)
        chunks.append(data)
        sample_labels += [chunk.get("mark", 0)] * data.shape[1]
    signal = np.hstack(chunks)                       # (kanály, čas)
    sample_labels = np.array(sample_labels)

    # CAR cez všetkých 8 kanálov
    info = mne.create_info(channel_names, SAMPLING_RATE, "eeg")
    raw = mne.io.RawArray(signal, info, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    # udalosti = zmeny značky; diff[0]=1 zahrnie aj prvý trial
    diff = np.diff(sample_labels, prepend=sample_labels[0])
    diff[0] = 1
    onsets = np.where(diff != 0)[0]
    event_labels = sample_labels[onsets]
    events = np.column_stack([onsets, np.zeros_like(onsets), event_labels])
    event_id = {CLASS_NAMES[k]: int(k) for k in np.unique(event_labels) if k in CLASS_NAMES}

    win, step = int(WINDOW_SEC * SAMPLING_RATE), int(STEP_SEC * SAMPLING_RATE)
    windows_by_band, labels, groups = {}, None, None
    for band, (low, high) in BANDS.items():
        # bandpass na súvislom signáli pred epochovaním (proti edge efektom)
        band_raw = raw.copy().filter(low, high, fir_design="firwin", verbose=False)
        epochs = mne.Epochs(band_raw, events, event_id, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                            baseline=None, preload=True, verbose=False)
        data = epochs.get_data(copy=True)            # (n_epoch, kanály, čas)
        epoch_labels = epochs.events[:, -1]

        win_list, lab_list, grp_list = [], [], []
        for ei in range(len(data)):
            for s in range(0, data.shape[-1] - win + 1, step):
                win_list.append(data[ei, :, s:s + win])
                lab_list.append(int(epoch_labels[ei]))
                grp_list.append(f"{session_id}_{ei}")
        windows_by_band[band] = np.array(win_list)
        labels = np.array(lab_list)
        groups = np.array(grp_list)
    return windows_by_band, labels, groups, channel_names


# =====================================================================
# 2. Príznaky: kovariancie + per-session recentering
# =====================================================================
def recenter_per_session(windows, session_ids):
    """Kovariancie okien + per-session centrovanie na ich geometrický priemer.

    Referencia každej session sa počíta výhradne z jej vlastných okien a bez
    použitia tried -> nesupervizovaná doménová adaptácia, bez leakage.
    """
    covs = Covariances(estimator=COV_ESTIMATOR).transform(windows)
    for s in np.unique(session_ids):
        mask = session_ids == s
        ref_inv_sqrt = invsqrtm(mean_covariance(covs[mask], metric="riemann"))
        covs[mask] = ref_inv_sqrt @ covs[mask] @ ref_inv_sqrt
    return covs


def load_dataset(session_glob, verbose=True):
    """Načíta a spojí všetky session, vráti recentrované kovariancie po pásmach.

    Returns:
        covs_by_band : dict {pásmo: (n_okien, 4, 4)} recentrované kovariancie
        labels       : (n_okien,) triedy 0/1/2
        groups       : (n_okien,) identifikátor trialu
        session_ids  : (n_okien,) index session
        channel_names: poradie kanálov
    """
    files = sorted(glob.glob(session_glob))
    if not files:
        raise FileNotFoundError(f"Nenašli sa žiadne session súbory: {session_glob}")

    per_band = {b: [] for b in BANDS}
    label_list, group_list, channel_names = [], [], None
    for sid, path in enumerate(files):
        w, lab, grp, channel_names = load_session(path, sid)
        for b in BANDS:
            per_band[b].append(w[b])
        label_list.append(lab)
        group_list.append(grp)
        if verbose:
            print(f"[{sid}] {Path(path).name}: {len(lab)} okien")

    windows_by_band = {b: np.concatenate(v) for b, v in per_band.items()}
    labels = np.concatenate(label_list)
    groups = np.concatenate(group_list)
    session_ids = np.array([int(g.split("_")[0]) for g in groups])
    motor_idx = [channel_names.index(c) for c in MOTOR_CHANNELS if c in channel_names]

    covs_by_band = {
        b: recenter_per_session(windows_by_band[b][:, motor_idx, :], session_ids)
        for b in BANDS
    }
    if verbose:
        counts = {CLASS_NAMES[k]: int((labels == k).sum()) for k in np.unique(labels)}
        print(f"\nokná po triedach: {counts}  | spolu {len(labels)}")
        print(f"motorické kanály: {[channel_names[i] for i in motor_idx]}")
    return covs_by_band, labels, groups, session_ids, channel_names


# =====================================================================
# 3. Klasifikátory (identické pre obe úlohy)
# =====================================================================
def make_lda(n_classes):
    """LDA so shrinkage a uniformnými priormi (na tangent space)."""
    priors = [1.0 / n_classes] * n_classes
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors))


def make_lr(n_classes=None):
    """Logistická regresia s pevným C a vážením tried (na tangent space)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=LR_C, class_weight="balanced", max_iter=1000))


def make_rf(n_classes=None):
    """Random Forest s obmedzenou hĺbkou a vážením tried."""
    return RandomForestClassifier(n_estimators=300, max_depth=6,
                                  class_weight="balanced", random_state=RANDOM_STATE)


def classifier_factories(n_classes):
    """Vráti slovník {názov: továreň} so správnym počtom tried."""
    return {
        "LDA": lambda: make_lda(n_classes),
        "LR":  lambda: make_lr(n_classes),
        "RF":  lambda: make_rf(n_classes),
    }


# =====================================================================
# 4. LOSO evaluácia
# =====================================================================
def _aggregate_to_trials(window_true, window_pred, window_groups):
    """Okná jedného trialu -> 1 predikcia väčšinovým hlasovaním."""
    trial_true, trial_pred = [], []
    for g in np.unique(window_groups):
        mask = window_groups == g
        trial_true.append(np.bincount(window_true[mask]).argmax())
        trial_pred.append(np.bincount(window_pred[mask]).argmax())
    return np.array(trial_true), np.array(trial_pred)


def _potato_clean_mask(covs_by_band, train_idx):
    """Bool maska čistých tréningových okien (Potato per pásmo, AND cez pásma)."""
    clean = np.ones(len(train_idx), dtype=bool)
    for b in BANDS:
        potato = Potato(metric="riemann", threshold=POTATO_THRESHOLD,
                        pos_label=1, neg_label=0)
        potato.fit(covs_by_band[b][train_idx])
        clean &= (potato.predict(covs_by_band[b][train_idx]) == 1)
    return clean


def evaluate_loso(covs_by_band, labels, groups, session_ids,
                  classifier_factory, use_potato=False):
    """Leave-one-session-out evaluácia jednej konfigurácie.

    Returns dict:
        trial_true, trial_pred : agregované na úroveň trialu (všetky foldy)
        rejection              : priemerný podiel zamietnutých tréningových okien
        per_session            : {session: balanced accuracy danej testovacej session}
    """
    splitter = LeaveOneGroupOut()
    true_all, pred_all, group_all, rejections = [], [], [], []
    per_session = {}

    for train_idx, test_idx in splitter.split(labels, labels, session_ids):
        used_train = train_idx
        if use_potato:
            clean = _potato_clean_mask(covs_by_band, train_idx)
            rejections.append(1 - clean.mean())
            used_train = train_idx[clean]

        # tangent space sa fituje len na tréningu, transformuje tréning aj test
        feat_train, feat_test = [], []
        for b in BANDS:
            ts = TangentSpace(metric="riemann").fit(covs_by_band[b][used_train])
            feat_train.append(ts.transform(covs_by_band[b][used_train]))
            feat_test.append(ts.transform(covs_by_band[b][test_idx]))
        feat_train, feat_test = np.hstack(feat_train), np.hstack(feat_test)

        model = classifier_factory()
        model.fit(feat_train, labels[used_train])
        pred = model.predict(feat_test)

        # balanced accuracy tejto session (na úrovni trialu)
        yt, pt = _aggregate_to_trials(labels[test_idx], pred, groups[test_idx])
        s = int(np.unique(session_ids[test_idx])[0])
        per_session[s] = balanced_accuracy_score(yt, pt)

        true_all.append(labels[test_idx])
        pred_all.append(pred)
        group_all.append(groups[test_idx])

    trial_true, trial_pred = _aggregate_to_trials(
        np.concatenate(true_all), np.concatenate(pred_all), np.concatenate(group_all))
    return {
        "trial_true": trial_true,
        "trial_pred": trial_pred,
        "rejection": float(np.mean(rejections)) if rejections else 0.0,
        "per_session": per_session,
    }


def run_all_configurations(covs_by_band, labels, groups, session_ids, n_classes):
    """Spustí všetky kombinácie klasifikátor x {bez Potato, s Potato}.

    Returns: dict {názov konfigurácie: výsledok evaluate_loso}
    """
    results = {}
    for clf_name, factory in classifier_factories(n_classes).items():
        for tag, use_potato in [("bez Potato", False), ("s Potato", True)]:
            name = f"{clf_name}, {tag}"
            results[name] = evaluate_loso(covs_by_band, labels, groups, session_ids,
                                          factory, use_potato=use_potato)
    return results


# =====================================================================
# 5. Súhrn a metriky
# =====================================================================
def summarize(results, class_labels):
    """Zostaví zoznam riadkov so súhrnnými metrikami, zoradený podľa bal. acc.

    Každý riadok: dict s kľúčmi name, bal_acc, acc, recall (pole), rejection,
    ps_mean, ps_std, ps_min, ps_max.
    """
    rows = []
    for name, r in results.items():
        yt, pt = r["trial_true"], r["trial_pred"]
        rec = recall_score(yt, pt, labels=class_labels, average=None, zero_division=0)
        ps = np.array(list(r["per_session"].values()))
        rows.append({
            "name": name,
            "bal_acc": balanced_accuracy_score(yt, pt),
            "acc": accuracy_score(yt, pt),
            "recall": rec,
            "rejection": r["rejection"],
            "ps_mean": ps.mean(), "ps_std": ps.std(ddof=1),
            "ps_min": ps.min(), "ps_max": ps.max(),
        })
    rows.sort(key=lambda d: -d["bal_acc"])
    return rows


def print_summary(rows, class_names):
    """Vypíše prehľadnú tabuľku (konzistentná pre obe úlohy)."""
    rec_cols = "".join(f"{'rec_' + c:>10s}" for c in class_names)
    print(f"{'konfigurácia':18s}{'bal_acc':>9s}{'acc':>8s}{rec_cols}{'rej%':>7s}")
    print("-" * (18 + 9 + 8 + 10 * len(class_names) + 7))
    for d in rows:
        rec = "".join(f"{v:>10.3f}" for v in d["recall"])
        print(f"{d['name']:18s}{d['bal_acc']:>9.3f}{d['acc']:>8.3f}{rec}{d['rejection']*100:>6.1f}")


def print_robustness(rows):
    """Vypíše per-session rozptyl (robustnosť naprieč session)."""
    print(f"{'konfigurácia':18s}{'priemer':>9s}{'sd':>7s}{'min':>7s}{'max':>7s}")
    print("-" * 48)
    for d in rows:
        print(f"{d['name']:18s}{d['ps_mean']:>9.3f}{d['ps_std']:>7.3f}"
              f"{d['ps_min']:>7.3f}{d['ps_max']:>7.3f}")


def decompose_multiclass(results):
    """Pre 3-triedne výsledky vráti rozklad na rest-vs-MI a left-vs-right (bal. acc.)."""
    rows = []
    for name, r in results.items():
        yt, pt = r["trial_true"], r["trial_pred"]
        yb, pb = (yt > 0).astype(int), (pt > 0).astype(int)
        rest_vs_mi = balanced_accuracy_score(yb, pb)
        mi = np.isin(yt, [1, 2])
        left_vs_right = balanced_accuracy_score(yt[mi], pt[mi])
        rows.append({"name": name, "rest_vs_mi": rest_vs_mi,
                     "left_vs_right": left_vs_right,
                     "bal_acc": balanced_accuracy_score(yt, pt)})
    rows.sort(key=lambda d: -d["bal_acc"])
    return rows


# =====================================================================
# 6. Grafy (tlmené farby, ukladané do súborov pre prácu)
# =====================================================================
def plot_balanced_accuracy(rows, chance_level, save_path=None):
    """Horizontálny stĺpcový graf bal. acc.; s Potato tmavšie, bez Potato béžová.

    Bez titulku — popis grafu patrí do popisu obrázka (\\caption) v texte práce.
    """
    rows_asc = rows[::-1]
    names = [d["name"] for d in rows_asc]
    vals = [d["bal_acc"] for d in rows_asc]
    colors = [PALETTE["teal_dark"] if "s Potato" in n else PALETTE["sand"] for n in names]

    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(names) + 1))
    ax.barh(names, vals, color=colors, edgecolor="white")
    ax.axvline(chance_level, ls="--", color=PALETTE["grey"], label="náhodná úroveň")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("vyvážená presnosť (balanced accuracy)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_confusion(trial_true, trial_pred, class_names, save_path=None):
    """Matica zámen v tlmenej béžovo-tyrkysovej palete (počty trialov).

    Bez titulku — popis patrí do popisu obrázka (\\caption) v texte práce.
    """
    cm = confusion_matrix(trial_true, trial_pred)
    fig, ax = plt.subplots(figsize=(0.9 * len(class_names) + 2.2,
                                    0.9 * len(class_names) + 2.0))
    im = ax.imshow(cm, cmap=CMAP)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("predikovaná trieda"); ax.set_ylabel("skutočná trieda")
    thr = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thr else "#333333")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_decomposition(decomp_rows, save_path=None):
    """Skupinový stĺpcový graf: 3 triedy vs rest-vs-MI vs ľavá-vs-pravá.

    Bez titulku — popis patrí do popisu obrázka (\\caption) v texte práce.
    """
    names = [d["name"] for d in decomp_rows]
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - w, [d["bal_acc"] for d in decomp_rows], w,
           label="3 triedy", color=PALETTE["teal"])
    ax.bar(x, [d["rest_vs_mi"] for d in decomp_rows], w,
           label="rest vs MI", color=PALETTE["sand"])
    ax.bar(x + w, [d["left_vs_right"] for d in decomp_rows], w,
           label="ľavá vs pravá", color=PALETTE["rose"])
    ax.axhline(1/3, ls="--", color=PALETTE["grey"], label="náhodná úroveň (3 triedy)")
    ax.axhline(0.5, ls=":", color=PALETTE["grey"], label="náhodná úroveň (2 triedy)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("vyvážená presnosť (balanced accuracy)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_per_session(best_row, results, chance_level, save_path=None):
    """Per-session balanced accuracy najlepšej konfigurácie (robustnosť).

    Bez titulku — popis patrí do popisu obrázka (\\caption) v texte práce.
    """
    per_session = results[best_row["name"]]["per_session"]
    sessions = sorted(per_session)
    vals = [per_session[s] for s in sessions]
    mean, std = best_row["ps_mean"], best_row["ps_std"]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar([f"S{s+1}" for s in sessions], vals, color=PALETTE["teal"], edgecolor="white")
    ax.axhline(mean, color=PALETTE["teal_dark"],
               label=f"priemer = {mean:.3f}".replace(".", ","))
    ax.axhspan(mean - std, mean + std, color=PALETTE["teal"], alpha=0.15,
               label=f"± smerodajná odchýlka ({std:.3f})".replace(".", ","))
    ax.axhline(chance_level, ls="--", color=PALETTE["grey"], label="náhodná úroveň")
    ax.set_ylim(0, 1); ax.set_ylabel("vyvážená presnosť (balanced accuracy)")
    ax.set_xlabel("session")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
