"""Build the AquesTalk ("koe") training-text list for generating Reimu
(bouyomichan 女性1 = AquesTalk f1) speech in bulk.

Sources:
  - ITA corpus (phoneme-balanced; ships with katakana readings already)
  - Tatoeba Japanese sentences (everyday sentences; readings via
    fugashi + UniDic `pron`, the pronunciation form, which already turns
    topic-particle は into ワ, long vowels into ー, etc. -- the same kind
    of phonetic spelling AquesTalk wants)

Output: corpus/koe_items.tsv  with lines  id<TAB>speed<TAB>koe
  - short sentences (kind "s")
  - paragraphs of several consecutive sentences joined with 。 (kind "p")
    so the model also sees longer, natural phrase continuity.
"""

import bz2
import random
import re
import sys
from pathlib import Path

from fugashi import Tagger

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"

# characters AquesTalk (v1 DLL) accepts in koe text
KATA = "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヲンヴー"
ALLOWED = set(KATA) | set("、。？?")

# fixed-up spellings AquesTalk does not take (it has no standalone 'ヴァ' series etc.)
REPLACE = [("ヴァ", "バ"), ("ヴィ", "ビ"), ("ヴェ", "ベ"), ("ヴォ", "ボ"), ("ヴ", "ブ")]

tagger = Tagger()
UNIDIC_BAD = {"*", ""}


def to_koe(text: str):
    """Japanese text -> koe string, or None if it cannot be expressed."""
    # Only plain Japanese: reject latin letters / digits / odd symbols up front
    if re.search(r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９]", text):
        return None
    out = []
    for w in tagger(text):
        surf = w.surface
        feat = w.feature
        pron = getattr(feat, "pron", None)
        if surf in "、，,":
            out.append("、")
            continue
        if surf in "。．.！!":
            out.append("。")
            continue
        if surf in "？?":
            out.append("？")
            continue
        if re.fullmatch(r"[\s「」『』（）()\[\]・…―—\-～~　]+", surf):
            # brackets / dashes: treat as a short pause only for … and ―
            if "…" in surf or "―" in surf or "—" in surf:
                out.append("、")
            continue
        if pron in UNIDIC_BAD or pron is None:
            return None
        out.append(pron)
    koe = "".join(out)
    for a, b in REPLACE:
        koe = koe.replace(a, b)
    if not koe or any(ch not in ALLOWED for ch in koe):
        return None
    # AquesTalk rejects leading punctuation / empty phrases; collapse repeats
    koe = re.sub(r"[、。？]{2,}", lambda m: m.group(0)[-1], koe)
    koe = koe.lstrip("、。？ー")
    if not koe:
        return None
    if koe[-1] not in "。？":
        koe += "。"
    return koe


def main(n_short=6000, n_para=1500, seed=1):
    rng = random.Random(seed)
    items = []  # (kind, koe)

    # --- ITA corpus: reading is given directly after the comma
    for fn in ("ita_emotion.txt", "ita_recitation.txt"):
        for line in (CORPUS / fn).read_text(encoding="utf-8").splitlines():
            m = re.match(r"^[^:]+:(.*),([^,]+)$", line)
            if not m:
                continue
            reading = m.group(2).strip()
            for a, b in REPLACE:
                reading = reading.replace(a, b)
            if reading and all(ch in ALLOWED for ch in reading):
                items.append(("s", reading))
    n_ita = len(items)

    # --- Tatoeba
    sents = []
    with bz2.open(CORPUS / "jpn_sentences.tsv.bz2", "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and 6 <= len(parts[2]) <= 60:
                sents.append(parts[2])
    rng.shuffle(sents)

    koes = []
    for s in sents:
        k = to_koe(s)
        if k is not None and 5 <= len(k) <= 120:
            koes.append(k)
        if len(koes) >= n_short + n_para * 5:
            break
    print(f"tatoeba usable: {len(koes)} (scanned until quota)")

    short_pool = koes[:n_short]
    for k in short_pool:
        items.append(("s", k))
    rest = koes[n_short:]
    for i in range(n_para):
        chunk = rest[i * 5:(i + 1) * 5]
        if len(chunk) < 3:
            break
        para = "".join(chunk)
        if len(para) <= 400:
            items.append(("p", para))

    # de-dup, assign ids, speeds (mostly default; a bit of variety)
    seen = set()
    out = []
    for kind, koe in items:
        if koe in seen:
            continue
        seen.add(koe)
        speed = 100 if rng.random() < 0.7 else rng.choice([90, 110, 120])
        out.append((kind, speed, koe))
    rng.shuffle(out)

    path = CORPUS / "koe_items.tsv"
    with open(path, "w", encoding="utf-8", newline="\n") as w:
        for i, (kind, speed, koe) in enumerate(out):
            w.write(f"{kind}{i:06d}\t{speed}\t{koe}\n")
    print(f"ITA items: {n_ita}; total written: {len(out)} -> {path}")
    print("sample:")
    for kind, speed, koe in out[:6]:
        print(" ", kind, speed, koe[:70])


if __name__ == "__main__":
    main()
