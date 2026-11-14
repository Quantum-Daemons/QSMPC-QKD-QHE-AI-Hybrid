"""Murmur detection under differential privacy (Round 86): the preparation, the two accountants and the noise, without
the data or a GPU.

- `scripts/data/prepare_circor.py` on a synthetic zip in CirCor's layout: every recording the table accounts for lands
  at its offset with its patient, person, location and label, resampled to 2 kHz and standardised; two patient IDs
  joined by Additional ID are one person, an Additional ID the table does not list joins nothing; the annotation files
  are never read; a recording the table does not list, or one at another rate, is refused;
- the sampled Gaussian's Renyi DP: exact at q = 1, equal to a numerical integral of its definition, and reproducing
  TensorFlow Privacy's published MNIST figure (epsilon 3.01) under the conversion that figure used; the tighter
  conversion this study reports is never looser;
- the vectorised discrete Gaussian against its exact probabilities, the randomised rounding's mean, and the secure
  sum's range for the largest noise the study draws;
- the logistic head's closed-form gradient against autograd, summed per person;
- the loader joins a person's records and drops unknown ones; the split holds a fifth of each class of PEOPLE out with
  a person's records on one side, and the four sites partition the rest.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import subprocess
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
PREP = REPO / "scripts" / "data" / "prepare_circor.py"
STUDY = REPO / "scripts" / "train" / "medical_murmur_dp.py"
ROOT = "the-circor-digiscope-phonocardiogram-dataset-1.0.3"


@pytest.fixture(scope="module")
def dp():
    spec = importlib.util.spec_from_file_location("medical_murmur_dp_under_test", STUDY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DEV = torch.device("cpu")      # the training wants the GPU; these checks do not, and CI has none
    return mod


def _wav(seconds: float, rate: int = 4000, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    x = (3000 * np.sin(2 * np.pi * 80 * t) + 400 * rng.normal(size=len(t))).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return buf.getvalue()


def _zip(tmp: Path, *, stray: bool = False, bad_rate: bool = False) -> Path:
    path = tmp / "circor.zip"
    table = [("50001", "Present", "Abnormal", "CC2015", "50003"), ("50002", "Absent", "Normal", "CC2014", "59998"),
             ("50003", "Unknown", "Abnormal", "CC2014", "50001")]
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{ROOT}/training_data.csv", "Patient ID,Murmur,Outcome,Campaign,Additional ID\n" + "".join(",".join(r) + "\n" for r in table))
        for k, (pid, *_rest) in enumerate(table):
            for j, loc in enumerate(("MV", "AV") if k != 1 else ("PV", "TV", "TV_2")):
                rate = 2000 if bad_rate and k == 2 else 4000
                z.writestr(f"{ROOT}/training_data/{pid}_{loc}.wav", _wav(2 + j / 2, rate=rate, seed=10 * k + j))
                z.writestr(f"{ROOT}/training_data/{pid}_{loc}.tsv", "0.0\t0.5\t1\n")
                z.writestr(f"{ROOT}/training_data/{pid}_{loc}.hea", "a header\n")
            z.writestr(f"{ROOT}/training_data/{pid}.txt", "a patient file\n")
        if stray:
            z.writestr(f"{ROOT}/training_data/59999_MV.wav", _wav(1.0))
    return path


def _run(zip_path: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(PREP), "--zip", str(zip_path), "--out", str(out)], capture_output=True, text=True, check=False)


def test_every_listed_recording_lands_with_its_patient_location_and_label(tmp_path: Path) -> None:
    out = tmp_path / "out"
    r = _run(_zip(tmp_path), out)
    assert r.returncode == 0, r.stderr
    with open(out / "index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(x["patient"], x["location"]) for x in rows] == [
        ("50001", "AV"), ("50001", "MV"), ("50002", "PV"), ("50002", "TV"), ("50002", "TV"), ("50003", "AV"), ("50003", "MV")], \
        "patients in order, locations in the AV, PV, TV, MV order, a repeat after its first"
    assert [x["murmur"] for x in rows] == ["Present"] * 2 + ["Absent"] * 3 + ["Unknown"] * 2
    assert [x["person"] for x in rows] == ["50001"] * 2 + ["50002"] * 3 + ["50001"] * 2, \
        "50001 and 50003 name each other: one person; 50002's Additional ID is not in the table: its own person"
    sig = np.load(out / "signals.npy")
    off = [int(x["offset"]) for x in rows]
    length = [int(x["length"]) for x in rows]
    assert off == list(np.cumsum([0] + length[:-1])) and sum(length) == len(sig)
    assert length[1] == math.ceil(2 * 4000 / 2), "4 kHz resampled to 2 kHz"
    seg = sig[off[0]:off[0] + length[0]].astype(np.float64)
    assert abs(seg.mean()) < 0.01 and abs(seg.std() - 1) < 0.01, "each recording is standardised"


def test_a_stray_recording_or_another_rate_is_refused(tmp_path: Path) -> None:
    (tmp_path / "s").mkdir()
    (tmp_path / "r").mkdir()
    r = _run(_zip(tmp_path / "s", stray=True), tmp_path / "s" / "out")
    assert r.returncode != 0 and "does not account for" in r.stderr
    r = _run(_zip(tmp_path / "r", bad_rate=True), tmp_path / "r" / "out")
    assert r.returncode != 0 and "expected 4000 Hz" in r.stderr


def _sampled_gaussian_rdp_by_integration(q: float, sigma: float, order: int) -> float:
    """The definition (Mironov, Talwar and Zhang 2019, section 3.3): mu0 = N(0, sigma^2), mu = (1 - q) mu0 + q N(1, sigma^2),
    RDP = log E_mu0[(mu / mu0)^order] / (order - 1), here by the trapezoid rule on a wide grid."""
    z = np.linspace(-40 * sigma, 40 * sigma + 1, 400_001)
    log_mu0 = -z * z / (2 * sigma * sigma)
    ratio = (1 - q) + q * np.exp((2 * z - 1) / (2 * sigma * sigma))
    integrand = np.exp(log_mu0 + order * np.log(ratio)) / math.sqrt(2 * math.pi * sigma * sigma)
    return math.log(np.trapezoid(integrand, z)) / (order - 1)


def test_the_sampled_gaussian_accountant(dp) -> None:
    assert dp.rdp_sampled_gaussian(1.0, 2.0, 5) == pytest.approx(5 / 8), "q = 1 is the Gaussian mechanism"
    for q, sigma, order in ((0.1, 1.0, 4), (0.05, 2.0, 12), (0.3, 3.0, 8)):
        assert dp.rdp_sampled_gaussian(q, sigma, order) == pytest.approx(_sampled_gaussian_rdp_by_integration(q, sigma, order), rel=1e-6)
    # TensorFlow Privacy's MNIST tutorial (tutorials/README.md): N 60,000, batch 256, noise 1.1, 60 epochs, delta 1e-5,
    # "eps = 3.01", from the classical conversion eps = RDP + log(1 / delta) / (order - 1).
    q, steps, delta = 256 / 60000, math.ceil(60 * 60000 / 256), 1e-5
    rdp = {a: steps * dp.rdp_sampled_gaussian(q, 1.1, a) for a in dp.ORDERS}
    classical = min(r + math.log(1 / delta) / (a - 1) for a, r in rdp.items())
    assert round(classical, 2) == 3.01
    assert dp.rdp_to_dp(rdp, delta) < classical, "the conversion of Canonne, Kamath and Steinke is the tighter"
    sigma = dp.sigma_for(2.0, 64 / 700, 328)
    assert dp.dpsgd_epsilon(64 / 700, sigma, 328) == pytest.approx(2.0, abs=1e-6)
    full = dp._bisect(lambda x: dp.full_epsilon(x, 328), 2.0)
    assert dp.full_epsilon(full, 328) == pytest.approx(2.0, abs=1e-6)
    assert full > sigma, "with every person in every step there is no amplification by sampling: more noise"
    # the exact Gaussian curve is tighter than zCDP through any conversion: at the same noise it states a smaller epsilon
    qdp = dp._qsh_dp()
    assert dp.full_epsilon(full, 328) < qdp.cdp_to_dp(328 / (2 * full * full), 1e-5)
    # and it inverts the closed form: for mu = 1, delta(1) = Phi(-1/2) - e Phi(-3/2), so epsilon(that delta) is 1
    from scipy.stats import norm
    d1 = norm.cdf(-0.5) - math.e * norm.cdf(-1.5)
    assert d1 == pytest.approx(0.1269, abs=1e-4) and dp.gaussian_exact_epsilon(1.0, d1) == pytest.approx(1.0, abs=1e-9)


def test_the_discrete_gaussian_and_the_rounding(dp) -> None:
    g = torch.Generator(device=dp.DEV).manual_seed(5)
    x = dp.discrete_gaussian((200_000,), 2.0, g).cpu().numpy()
    assert x.dtype == np.int64
    k = np.arange(-6, 7)
    exact = np.exp(-k * k / 4.0)
    exact /= np.exp(-np.arange(-40, 41) ** 2 / 4.0).sum()
    seen = np.array([(x == v).mean() for v in k])
    assert np.abs(seen - exact).max() < 0.004, "the sampler's probabilities are the discrete Gaussian's"
    big = dp.discrete_gaussian((200_000,), 1e6, g).double().cpu().numpy()
    assert abs(big.mean()) < 20 and big.var() == pytest.approx(1e6, rel=0.02)
    again = [dp.discrete_gaussian((1000,), 50.0, torch.Generator(device=dp.DEV).manual_seed(9)) for _ in range(2)]
    assert torch.equal(again[0], again[1]), "every draw comes from the generator it is given"
    v = torch.full((100_000,), 3.3, dtype=torch.float64, device=dp.DEV)
    low = torch.floor(v)
    r = low + (torch.rand(v.shape, device=dp.DEV, dtype=torch.float64, generator=g) < v - low)
    assert set(r.unique().tolist()) == {3.0, 4.0} and abs(r.mean().item() - 3.3) < 0.01, "unbiased randomised rounding"


def test_the_secure_sum_stays_in_range(dp) -> None:
    qdp = dp._qsh_dp()
    n_params, patients, rounds = 97_521, 816, 400
    delta2 = 2.0 ** dp.SCALE_BITS + math.sqrt(n_params)
    var = qdp.variance_for(min(dp.EPSILONS), rounds, delta2, dp.SITES, n_params, dp.DELTA)
    worst = patients * (2 ** dp.SCALE_BITS + 1) + 8 * math.sqrt(dp.SITES * var)
    assert worst < 2 ** 31, "every patient at the clip bound plus eight standard deviations of the summed noise"


def _people_d(n_people: int = 250, pairs: int = 20) -> dict:
    """250 people, the first 20 with two records; a fifth of the people have a murmur in at least one record."""
    members, py, k = [], [], 0
    for j in range(n_people):
        size = 2 if j < pairs else 1
        members.append(np.arange(k, k + size))
        py.extend([1.0 if j % 5 == 0 else 0.0] * size)
        k += size
    py = np.array(py, dtype=np.float32)
    return {"members": members, "py": py, "person_y": np.array([py[m].max() for m in members], dtype=np.float32),
            "person_of": np.concatenate([np.full(len(m), j) for j, m in enumerate(members)])}


def test_the_split_and_the_sites(dp) -> None:
    d = _people_d()
    train, test = dp.split(d)
    assert len(np.intersect1d(train, test)) == 0 and len(train) + len(test) == 250
    assert d["person_y"][test].sum() == 10 and (d["person_y"][test] == 0).sum() == 40, "a fifth of each class of people"
    tr, te = dp.records_of(d, train), dp.records_of(d, test)
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(d["py"]), "a person's records stay on one side"
    fit, val = dp.holdout(d, train, 7)
    assert len(np.intersect1d(fit, val)) == 0 and np.array_equal(np.sort(np.concatenate([fit, val])), train)
    sites = dp.sites_of(d, train)
    assert len(sites) == dp.SITES and np.array_equal(np.sort(np.concatenate(sites)), np.sort(train))
    assert max(len(s) for s in sites) - min(len(s) for s in sites) <= 2
    assert all(abs(d["person_y"][s].mean() - d["person_y"][train].mean()) < 0.03 for s in sites), "each site keeps the overall share"


def test_the_loader_joins_a_persons_records(dp, tmp_path: Path) -> None:
    rows = [("1", "10", "10", "Present"), ("2", "10", "10", "Present"), ("3", "20", "10", "Absent"),
            ("4", "30", "30", "Absent"), ("5", "40", "40", "Unknown"), ("6", "50", "40", "Present")]
    with open(tmp_path / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "patient", "person", "location", "murmur", "outcome", "campaign", "offset", "length"])
        for k, (pos, pat, per, mur) in enumerate(rows):
            w.writerow([pos, pat, per, "AV", mur, "Normal", "CC2014", 100 * k, 100])
    d = dp.load(tmp_path)
    assert d["pats"].tolist() == [10, 20, 30, 50], "the unknown record is left out"
    assert d["people"].tolist() == [10, 30, 40] and [m.tolist() for m in d["members"]] == [[0, 1], [2], [3]]
    assert d["person_y"].tolist() == [1.0, 0.0, 1.0] and d["recs"][10].tolist() == [0, 1]


def test_the_heads_gradient_is_the_logistic_losss(dp) -> None:
    torch.manual_seed(0)
    d = {"pats": np.arange(5), "py": np.array([1, 0, 0, 1, 0], dtype=np.float32), "members": [np.array([0, 1]), np.array([2]), np.array([3, 4])]}
    win_patient = np.repeat(np.arange(5), 3)
    x = torch.randn(15, 6)
    head = dp.Head(d, x, win_patient)
    theta = torch.randn(7)
    g = torch.Generator(device=dp.DEV).manual_seed(3)
    got = head.person_grads(theta, np.array([0, 2]), g)
    # the same windows, drawn again from the same generator state, by autograd
    g2 = torch.Generator(device=dp.DEV).manual_seed(3)
    pats = np.array([0, 1, 3, 4])
    j = (torch.rand((4, dp.WINDOWS_PER_PATIENT), generator=g2) * 3).long()
    want = torch.zeros(2, 7)
    for r, (p, owner) in enumerate(zip(pats, (0, 0, 1, 1), strict=True)):
        t = theta.clone().requires_grad_(True)
        z = x[p * 3 + j[r]] @ t[:-1] + t[-1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.full_like(z, float(d["py"][p])), pos_weight=torch.tensor(dp.POS_WEIGHT))
        loss.backward()
        want[owner] += t.grad
    assert torch.allclose(got, want, atol=1e-5)
    assert head.scores(theta, np.array([4])) == pytest.approx(float((x[12:15] @ theta[:-1] + theta[-1]).mean()), abs=1e-5)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def search(dp):
    mod = _load("medical_murmur_dp_search_under_test", REPO / "scripts" / "train" / "medical_murmur_dp_search.py")
    mod.study.DEV = torch.device("cpu")
    return mod


@pytest.fixture(scope="module")
def objpert(dp):
    mod = _load("medical_murmur_dp_objpert_under_test", REPO / "scripts" / "train" / "medical_murmur_dp_objpert.py")
    mod.study.DEV = torch.device("cpu")
    return mod


def test_the_accounted_search(search) -> None:
    assert search.tnb_gamma_for(1.0, 10.0) == pytest.approx(0.1, rel=1e-9), "geometric: E[K] = 1 / gamma"
    g0 = search.tnb_gamma_for(0.0, 5.0)
    assert (1 / g0 - 1) / math.log(1 / g0) == pytest.approx(5.0, rel=1e-9), "logarithmic: E[K] = (1/gamma - 1) / log(1/gamma)"
    rng = np.random.default_rng(3)
    ks = [search.sample_k(0.0, g0, rng) for _ in range(20000)]
    assert min(ks) >= 1 and np.mean(ks) == pytest.approx(5.0, rel=0.05)

    def gaussian(lam):          # a Gaussian mechanism of sensitivity 1 and noise 4: lam / 32 at every order
        return lam / 32
    one = min(gaussian(lam) + math.log1p(-1 / lam) - (math.log(1e-5) + math.log(lam)) / (lam - 1) for lam in search.ORDERS)
    e5, e20 = search.search_epsilon(gaussian, 0.0, 5.0), search.search_epsilon(gaussian, 0.0, 20.0)
    assert one < e5 < e20, "the search costs more than one run, and more with more runs"
    lam, lam_hat, gam = 10, 4, search.tnb_gamma_for(0.0, 5.0)
    assert search.search_rdp(gaussian, lam, lam_hat, 0.0, gam, 5.0) == pytest.approx(
        10 / 32 + (1 - 1 / 4) * 4 / 32 + math.log(1 / gam) / 4 + math.log(5) / 9), "Theorem 2, term by term"


def test_objective_perturbations_accountant(objpert) -> None:
    for sigma, alpha in ((0.8, 3.0), (2.0, 1.5), (4.0, 12.0)):
        s, t = 1 / sigma, alpha - 1
        z = np.linspace(-40 * s, 40 * s, 400_001)
        dens = np.exp(-z * z / (2 * s * s)) / math.sqrt(2 * math.pi * s * s)
        log_e = math.log(np.trapezoid(np.exp(t * np.abs(z)) * dens, z))
        want = -math.log1p(-0.25 / 10) + s * s / 2 + log_e / t + 2 * 0.01 ** 2 * alpha / (0.15 ** 2 * 100)
        assert objpert.rdp(alpha, sigma) == pytest.approx(want, rel=1e-6), "the folded-normal term in closed form"
    eps = [objpert.epsilon(s) for s in (0.5, 1.0, 2.0, 4.0)]
    assert all(b < a for a, b in zip(eps, eps[1:], strict=False)), "more noise, less epsilon"
    sigma = objpert.sigma_for(2.0)
    assert objpert.epsilon(sigma) == pytest.approx(2.0, abs=1e-6)
    x = objpert.unit_rows(torch.randn(50, 7, dtype=torch.float64))
    assert torch.allclose(x.norm(dim=1), torch.ones(50, dtype=torch.float64)), "every row has unit norm"
    y = (torch.rand(50) < 0.3).double()
    theta, g = objpert.solve(x, y, torch.zeros(8, dtype=torch.float64))
    assert g <= objpert.TAU


def test_the_cnn14_front_end_without_the_weights() -> None:
    panns = _load("panns_cnn14_under_test", REPO / "scripts" / "train" / "panns_cnn14.py")
    m = panns.Cnn14Embedding().eval()
    with torch.no_grad():
        e = m(torch.randn(2, 2 * panns.RATE))
    assert e.shape == (2, 2048) and bool((e >= 0).all()), "a ReLU'd 2,048-d embedding"
    x = torch.sin(2 * math.pi * 100 * torch.arange(8000) / 2000)[None]
    up = panns.upsample(x, 16)
    assert up.shape == (1, 128000) and torch.allclose(up[0, ::16], x[0], atol=1e-4), "band-limited upsampling keeps the samples"


def test_the_card_shows_the_record_and_credits_its_sources() -> None:
    record = json.loads((REPO / "docs" / "measurements" / "murmur-dp-2026-11-14.json").read_text())
    text = (REPO / "web" / "public" / "data" / "medical" / "murmur_dp.json").read_text()
    card = json.loads(text)
    for key in ("attribution", "data", "recipe", "routes"):
        assert card[key] == record[key], f"the card's {key} is not the record's"
    for route, rows in card["summary"].items():
        want = [r for r in record["summary"][route] if route == "public" or r["arm"] == "central"]
        assert len(rows) == len(want), route
        for got, full in zip(rows, want, strict=True):
            assert got["auroc"]["mean"] == pytest.approx(full["auroc"]["mean"], abs=5e-4), route
    head = [r for r in card["summary"]["public"] if r["model"] == "head-panns"]
    assert {(r["arm"], r["kind"], r["epsilon"]) for r in head} >= {(a, "DP", e) for a in ("central", "central-full", "federated") for e in dp_eps()}
    att = card["attribution"]
    assert att["circor"]["licence"] == "Open Data Commons Attribution License v1.0" and "3137048" in att["circor"]["citation"]
    assert att["panns"]["licence"].startswith("Creative Commons Attribution 4.0") and "not redistributed" in att["panns"]["use"]
    raw = (REPO / "docs" / "measurements" / "murmur-dp-2026-11-14.json").read_text()
    assert "/home/" not in text + raw and "/run/media" not in text + raw, "a local path reached a shipped file"
    for rep in record["reports"]["search"]:
        assert all("tried" not in r and "k" not in r and "local_only" not in r for r in rep["runs"]), \
            "the search's K or its losing candidates reached the record (Papernot and Steinke release the best run only)"
    assert record["data"]["people"] == 816 and record["data"]["people_with_two_records"] == 58


def dp_eps() -> tuple[float, ...]:
    return (8.0, 4.0, 2.0, 1.0)
