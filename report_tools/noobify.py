"""Rough-draft filter for the team's working copy of the report.

Makes the running text read like a hurried student first draft: common
spelling slips, dropped articles, loose grammar. It only ever swaps whole
dictionary words, so numbers, file names, code and technical values are never
touched. The same paragraph always comes out the same way (seeded by its own
text), so page numbers stay stable between rebuilds.

Set LEVEL = 0 to switch it off and get the clean text back.
"""

import random
import re
import zlib

LEVEL = 1.0          # 0 = off, 1 = normal, 1.5 = heavier

# word -> (replacement, chance)
SPELL = {
    "satellite": ("satelite", 0.35), "satellites": ("satelites", 0.35),
    "because": ("becuase", 0.30), "different": ("diffrent", 0.40),
    "difference": ("diffrence", 0.35), "differences": ("diffrences", 0.35),
    "separate": ("seperate", 0.70), "separately": ("seperately", 0.70),
    "similar": ("similer", 0.35), "their": ("thier", 0.25),
    "which": ("wich", 0.10), "actually": ("actualy", 0.45),
    "really": ("realy", 0.45), "little": ("litle", 0.30),
    "threshold": ("threshhold", 0.35), "embedding": ("embeding", 0.22),
    "embeddings": ("embedings", 0.22), "brightness": ("brightnes", 0.18),
    "vegetation": ("vegitation", 0.30), "resolution": ("resoultion", 0.20),
    "building": ("bulding", 0.18), "buildings": ("buldings", 0.18),
    "comparison": ("comparision", 0.70), "calculate": ("calcualte", 0.50),
    "completely": ("completly", 0.50), "definitely": ("definately", 0.60),
    "necessary": ("neccessary", 0.60), "properly": ("properely", 0.30),
    "receive": ("recieve", 0.60), "believe": ("beleive", 0.50),
    "problem": ("probelm", 0.12), "problems": ("probelms", 0.12),
    "important": ("importent", 0.30), "possible": ("posible", 0.30),
    "useful": ("usefull", 0.45), "until": ("untill", 0.50),
    "always": ("allways", 0.30), "already": ("allready", 0.30),
    "almost": ("allmost", 0.25), "also": ("allso", 0.06),
    "experiment": ("experiement", 0.25), "understand": ("understood", 0.0),
    "automatic": ("automatc", 0.20), "automatically": ("automaticaly", 0.40),
    "exactly": ("exactely", 0.35), "normal": ("normel", 0.15),
    "search": ("serach", 0.08), "change": ("chnage", 0.05),
    "changed": ("chnaged", 0.05), "image": ("imgae", 0.05),
    "images": ("imgaes", 0.05), "result": ("reuslt", 0.07),
    "results": ("reuslts", 0.07), "number": ("numbr", 0.08),
    "pixel": ("pixle", 0.07), "pixels": ("pixles", 0.07),
    "correct": ("corect", 0.20), "accuracy": ("acuracy", 0.40),
    "measure": ("mesure", 0.30), "weather": ("wether", 0.3),
    "whether": ("wether", 0.50), "occurred": ("occured", 0.7),
    "beginning": ("begining", 0.6), "information": ("infomation", 0.25),
    "explained": ("explaned", 0.35), "noticed": ("noticied", 0.30),
    "decided": ("desided", 0.30), "finally": ("finaly", 0.45),
    "simply": ("simpley", 0.25), "second": ("secound", 0.15),
    "between": ("beetween", 0.12), "through": ("thru", 0.20),
    "though": ("tho", 0.25), "people": ("peoples", 0.30),
    "data": ("datas", 0.05), "less": ("lesser", 0.15),
}

# phrase-level grammar slips: (regex, replacement, chance)
GRAMMAR = [
    (r"\bdoes not\b", "dont", 0.40), (r"\bdo not\b", "dont", 0.45),
    (r"\bdid not\b", "didnt", 0.50), (r"\bcannot\b", "cant", 0.45),
    (r"\bis not\b", "isnt", 0.35), (r"\bwas not\b", "wasnt", 0.40),
    (r"\bit is\b", "its", 0.35), (r"\bthat is\b", "thats", 0.35),
    (r"\bthan\b", "then", 0.35), (r"\bthere are\b", "there is", 0.35),
    (r"\bthese are\b", "these is", 0.20), (r"\bhas\b", "have", 0.10),
    (r"\bhave\b", "has", 0.05), (r"\bwere\b", "was", 0.25),
    (r"\bgives\b", "give", 0.25), (r"\btakes\b", "take", 0.25),
    (r"\bshows\b", "show", 0.25), (r"\bmakes\b", "make", 0.25),
    (r"\bgoes\b", "go", 0.25), (r"\bcomes\b", "come", 0.25),
    (r"\bmeans\b", "mean", 0.20), (r"\buses\b", "use", 0.25),
    (r"\bneeds\b", "need", 0.25), (r"\bworks\b", "work", 0.20),
    (r"\bwe are\b", "we r", 0.03), (r"\byour\b", "ur", 0.0),
    (r"\bto be\b", "to b", 0.0), (r"\bmany\b", "much", 0.12),
    (r"\bfewer\b", "less", 0.8), (r"\bwho\b", "which", 0.2),
    (r"\bthere\b", "their", 0.07), (r"\btoo\b", "to", 0.35),
    (r"\binto\b", "in to", 0.20), (r"\bcould\b", "can", 0.15),
    (r"\bwould\b", "will", 0.20), (r"\ban\b", "a", 0.25),
]


def _keep_case(src, dst):
    return dst.capitalize() if src[:1].isupper() else dst


def noob(text):
    if LEVEL <= 0 or not text:
        return text
    rng = random.Random(zlib.crc32(text.encode("utf-8")))
    lv = LEVEL

    # 1. grammar slips
    for pat, rep, ch in GRAMMAR:
        if ch <= 0:
            continue
        text = re.sub(pat, lambda m, rep=rep, ch=ch: _keep_case(m.group(0), rep)
                      if rng.random() < ch * lv else m.group(0), text, flags=re.I)

    # 2. spelling slips (whole words only; never anything with a digit in it)
    def spell(m):
        w = m.group(0)
        if w[0].isupper():            # leave names like CLIP, Change Vector Analysis
            return w
        hit = SPELL.get(w.lower())
        if hit and hit[1] > 0 and rng.random() < hit[1] * lv:
            return _keep_case(w, hit[0])
        return w
    text = re.sub(r"\b[A-Za-z]+\b", spell, text)

    # 3. dropped articles (only before a plain lowercase word)
    text = re.sub(r"\b[Tt]he (?=[a-z]{3,})",
                  lambda m: "" if rng.random() < 0.13 * lv and m.group(0)[0] == "t"
                  else m.group(0), text)
    text = re.sub(r"(?<=[a-z] )a (?=[a-z]{3,})",
                  lambda m: "" if rng.random() < 0.12 * lv else m.group(0), text)

    # 4. punctuation habits: missing space after comma, lower-case sentence start
    text = re.sub(r", (?=[a-z])", lambda m: "," if rng.random() < 0.07 * lv else ", ", text)
    text = re.sub(r"(?<=[a-z]\. )([A-Z])(?=[a-z])",
                  lambda m: m.group(1).lower() if rng.random() < 0.08 * lv else m.group(1),
                  text)
    text = re.sub(r"(?<=[a-z])\. (?=[A-Z])",
                  lambda m: " . " if rng.random() < 0.03 * lv else ". ", text)
    return text
