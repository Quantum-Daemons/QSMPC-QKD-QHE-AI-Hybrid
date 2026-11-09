"""One-way quasi-QSDC (STIKE): telemetry over a link where two-way QSDC cannot close.

Pan, Liu, Niu, Zhang, Zhang, Wang, Song, Chen, Zheng and Long, "Simultaneous transmission of information and key
exchange using the same photonic quantum states", Sci. Adv. 11, eadt4627 (2025). Used over free space by Sun et al.,
arXiv:2602.06624 (2026): 1.4 km of air and 10 km of fibre.

WHY IT CLOSES WHERE EQ. 8 DOES NOT. Qi et al.'s Eq. 8 bounds keyless, two-way QSDC, where the message is encoded before
anyone knows which pulses will arrive, so every pulse lost on the way is one an eavesdropper may keep: C_s = Q_Bob[1 - h(e)
- g h(e_x + e_z)], with g the loss. In STIKE the message is one-time padded with a key both ends already hold, each pulse is
masked with a local random bit that is revealed only for pulses Bob reports detecting, and the pulses that did not reach him
leave nothing for anyone. So the loss costs rate and not secrecy, and the secrecy capacity is the BB84 key rate (their
Eq. 2 and 3). The price is the word "quasi": it needs that pre-shared key.

WHAT IS MODELLED. Eq. 3 and Eq. 4 are evaluated with the same decoy-state function the QKD stage uses. The message is then
sent over the channel statistics: each bit is spread over enough pulses for `votes_per_bit` retained pulses on average, and
Bob takes the majority of what he retained. The pad and the mask cancel exactly, so they are accounted for and not drawn.
The key-pool book-keeping is step 2 (a key as long as the spread codeword is reserved), step 11 (the keys of the pulses
Bob did not detect go back) and step 12 (new key distilled from the retained pulses at the Eq. 3 rate).

WHAT IS NOT. Pan et al. use an LDPC code and a spreading ratio tuned to the channel; this uses plain spreading and a majority,
which spends more key per message bit. The rate is asymptotic: Pan et al. leave the finite-length analysis open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bb84 import _decoy_secure_fraction, _h2, e_det_for_bound
from .channel import ChannelParams

#: Bob keeps a detected pulse only when his basis matches Alice's (their q = 1/2).
SIFT = 0.5


@dataclass(frozen=True)
class StikeRates:
    """Per signal pulse, on one link."""

    secrecy: float        # Eq. 3: the key the link distils
    reliable: float       # Eq. 4: what the link can carry reliably, q Q_mu [1 - f H2(E_mu)]
    retained: float       # q Q_mu: the chance a signal pulse reaches Bob in the matching basis
    gain: float           # Q_mu
    qber: float           # E_mu
    single_photon_error: float


def stike_rates(params: ChannelParams, mu: float, nu: float, f_ec: float, *, vacuum: bool) -> StikeRates:
    """Eq. 3 and Eq. 4 of Pan et al. on a channel, through the QKD stage's own decoy-state function."""
    eta = params.transmittance()
    y0 = 2.0 * params.dark_count_prob
    secrecy, e_mu, e1 = _decoy_secure_fraction(eta, mu, nu, e_det_for_bound(params), y0, f_ec, SIFT, vacuum=vacuum)
    gain = y0 + 1.0 - math.exp(-eta * mu)
    return StikeRates(secrecy=secrecy, reliable=max(0.0, SIFT * gain * (1.0 - f_ec * _h2(e_mu))), retained=SIFT * gain,
                      gain=gain, qber=e_mu, single_photon_error=e1)


@dataclass(frozen=True)
class StikeResult:
    """How one message went. `status`: delivered, no_secure_rate, pool_too_small or undecodable."""

    status: str
    delivered: bytes | None
    rates: StikeRates
    message_bits: int
    signal_pulses: int
    pulses_sent: int
    seconds: float
    key_in_flight_bits: int     # step 2: reserved for the spread codeword
    key_consumed_bits: int      # the retained pulses'
    key_returned_bits: int      # step 11
    key_regenerated_bits: int   # step 12, at the Eq. 3 rate
    pool_before_bits: int
    pool_after_bits: int
    wrong_bits: int
    erased_bits: int


def pulses_per_bit(rates: StikeRates, votes_per_bit: int) -> int:
    """How many signal pulses carry one message bit, for `votes_per_bit` retained pulses on average."""
    return int(math.ceil(votes_per_bit / rates.retained)) if rates.retained > 0.0 else 0


def stike_send(message: bytes, rates: StikeRates, *, p_signal: float, source_rate_hz: float, pool_bits: int,
               votes_per_bit: int, seed: int = 0) -> StikeResult:
    """Send `message` one way, over the channel statistics `rates` describes."""
    n_bits = len(message) * 8
    per_bit = pulses_per_bit(rates, votes_per_bit)
    signal_pulses = n_bits * per_bit
    sent = int(math.ceil(signal_pulses / p_signal)) if p_signal > 0.0 else 0
    seconds = sent / source_rate_hz if source_rate_hz > 0.0 else float("inf")

    def outcome(status: str, delivered: bytes | None = None, *, consumed: int = 0, regenerated: int = 0,
                wrong: int = 0, erased: int = 0, in_flight: int = 0) -> StikeResult:
        return StikeResult(status, delivered, rates, n_bits, signal_pulses, sent, seconds, in_flight, consumed, in_flight - consumed,
                           regenerated, pool_bits, pool_bits - consumed + regenerated, wrong, erased)

    # Step 6: no positive key rate means the channel check has failed, and nothing is spent.
    if rates.secrecy <= 0.0:
        return outcome("no_secure_rate")
    # Step 2: the whole spread codeword is padded before anyone knows which pulses will arrive.
    if signal_pulses > pool_bits:
        return outcome("pool_too_small")

    rng = np.random.default_rng(seed)
    votes = rng.binomial(per_bit, rates.retained, n_bits)
    errors = rng.binomial(votes, rates.qber)
    erased = int((votes == 0).sum())
    wrong = int((2 * errors >= votes)[votes > 0].sum())
    consumed = int(votes.sum())
    regenerated = int(rates.secrecy * signal_pulses)
    if erased or wrong:
        return outcome("undecodable", consumed=consumed, regenerated=regenerated, wrong=wrong, erased=erased, in_flight=signal_pulses)
    return outcome("delivered", message, consumed=consumed, regenerated=regenerated, in_flight=signal_pulses)
