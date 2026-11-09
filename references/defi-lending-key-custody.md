# Who holds the key in a private liquidation check

Read on 2026-11-09, each source at its origin. The question is the one the lending page has to answer: in a confidential lending market, who can decrypt, and who has to be online when a position is checked?

## What the sources say

| Source | What it establishes | How far it was read |
|---|---|---|
| Aave, `IPool.sol` (`aave/aave-v3-core`), NatSpec of `liquidationCall` | The caller is the liquidator, repays part of the debt and receives collateral plus a bonus. The borrower is only named as the account being liquidated. | Source code |
| Chiu and Danisman, *DeFi Lending: Returns, Leverage, and Liquidation Risk*, Bank of Canada Staff Analytical Paper 2026-13, doi:10.34989/sap-2026-13 | Third-party liquidators carry out Aave V3's liquidations. These come in concentrated waves, and within a wave a large share of the value often comes from a few large borrowers. | PDF, the abstract and the liquidation sections |
| Zama, [KMS overview](https://docs.zama.org/protocol/protocol/overview/kms) | Decryption is held by a threshold MPC network: 13 nodes run by different organisations, with 9 of 13 given as the example quorum, and no single party holding the key. | Documentation |
| Zama, [oracle guide](https://docs.zama.org/protocol/solidity-guides/smart-contract/oracle) | A contract marks a handle publicly decryptable, any client can then fetch the plaintext, and the contract checks the network's signatures. The owner of the value takes no step. | Documentation |
| Zama, [confidential lending post](https://www.zama.org/post/onboard-the-next-trillions-in-defi-with-confidential-lending), 20 February 2025 | How liquidators would learn of an eligible position is left as an open design problem. Vendor post, no figures. | Whole post |
| Zama, Morpho and Steakhouse, [press release](https://unchainedcrypto.com/press-release/zama-launches-usdc-confidential-lending-in-partnership-with-morpho-and-steakhouse-financial/), 18 June 2026 | Encrypted balances and transfer amounts on a USDC deposit product. It says nothing about borrowers, collateral or liquidations. Marked as paid content. | Whole release |
| Fhenix, [encrypted lending post](https://www.fhenix.io/blog/encrypted-lending-ethereum-fully-homomorphic-encryption-private-defi), 30 December 2025, revised 22 October 2026 | Proposes running the liquidation logic on ciphertext and publishing only eligibility, with decryption by a threshold network. Its sample contract omits liquidation, and its figures come from research conditions. Vendor post. | Whole post |
| Zyskind, Zarchy, Leibovich and Peikert, ACM CCS 2025, [ePrint 2025/1781](https://eprint.iacr.org/2025/1781) | UC-secure threshold FHE decryption without noise flooding. | Abstract |
| Bootland et al., [ePrint 2025/699](https://eprint.iacr.org/2025/699) | Robust threshold key generation and decryption for BGV, BFV and TFHE at $`t < n/3`$ or $`t < n/4`$. A preprint. | Abstract |
| Martinico, Calabrese, Rubin and Ciampi, ACNS 2026, LNCS 16572, doi:10.1007/978-3-032-32575-4_18 | A lender scores a borrower's net worth privately, on a new FHE notion secure against a corrupted decryptor, proved in the UC model. It argues existing notions do not reach it. | Abstract only; the chapter is paywalled |
| miha-stopar, [ethresear.ch](https://ethresear.ch/t/open-application-driven-fhe-for-ethereum/23044), 11 September 2025 | States the principle that a protocol has to disclose just enough for liquidations. No mechanism. | The lending section |
| Cassini Systems in the AIMA Journal, [28 November 2022](https://www.aima.org/article/the-impact-of-volatile-markets-on-pb-margining-understanding-margin-model-changes-and-the-impacts-on-your-portfolio.html) | Prime brokers' margin methodologies are often opaque to the client. It does not say they are secret, and the client does see its resulting margin. | Whole article |

## What follows, and what does not

- A liquidation does not wait for the borrower. A design in which the borrower holds the key and must answer each check lets the party who loses by the liquidation stop it by not answering, so it needs a rule that no answer counts as a default.
- The systems that are being built keep the key in a threshold network and disclose as little as the liquidation needs. A separate key holder is the arrangement to model.
- What runs in production so far hides deposits. No primary source was found for a confidential liquidation in production, so neither arrangement on the page is a description of a deployed market.
- A borrower-held key is not without precedent in form: a bilateral lender with its own margin schedule resembles it, and Martinico et al. show why the decryptor then needs a stronger notion than ordinary FHE gives.

## What the page does with this

The default is a separate key holder, Carol, in `web/src/lib/lending/privateCheck.ts`: Bob encrypts once under her public key and can go offline, the pool computes on that ciphertext at public prices and thresholds and applies each public settlement to it, only masked margins reach her, the pool and Carol run the garbled circuit, and she signs the bit. node-seal has no threshold CKKS, so she is one party standing in for a committee. The bilateral arrangement remains one setting away, with its own limits stated. `docs/threat-model.md` says what each protects and `npm run check:lending-private` holds both to the clear answer.
