#!/usr/bin/env python3
"""Funnel Vision news exposure audit."""

import argparse
import contextlib
import csv
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

START_TIME = time.time()
SEED = 20261018
SCRIPT_VERSION = "2.0"
EPILOG = (
    "example:\n"
    "  python funnel_vision.py --train MINDlarge_train.zip --dev MINDlarge_dev.zip "
    "--out results"
)

# ---------------------------------------------------------------------------
# MIND file layout
# ---------------------------------------------------------------------------

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
TSV_OPTIONS = {
    "sep": "\t",
    "header": None,
    "quoting": csv.QUOTE_NONE,
    "dtype": str,
    "keep_default_na": False,
    "na_filter": False,
    "encoding": "utf-8",
}
MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
LABEL_CHECK_ITEMS = 2000
HASH_BUCKETS = 10000
QID_PATTERN = re.compile(r"^Q\d+$")

# ---------------------------------------------------------------------------
# Wikidata vocabulary
# ---------------------------------------------------------------------------

# Wikidata P21 gender values
WOMEN_QIDS = {"Q6581072", "Q1052281", "Q15145779"}  # Female, trans, cisgender woman
MEN_QIDS = {"Q6581097", "Q2449503", "Q15145778"}  # Male, trans, cisgender man
US_QID = "Q30"
CANADA_QID = "Q16"

# MIND categories into five sections
SECTION_OF_CATEGORY = {
    "news": "News",
    "northamerica": "News",
    "middleeast": "News",
    "sports": "Sports",
    "finance": "Money",
    "entertainment": "Entertainment",
    "tv": "Entertainment",
    "music": "Entertainment",
    "movies": "Entertainment",
    "video": "Entertainment",
    "kids": "Entertainment",
    "lifestyle": "Lifestyle",
    "foodanddrink": "Lifestyle",
    "travel": "Lifestyle",
    "health": "Lifestyle",
    "autos": "Lifestyle",
    "weather": "Lifestyle",
}
SECTIONS = ["News", "Sports", "Money", "Entertainment", "Lifestyle"]
DEFAULT_SECTION = "Lifestyle"

# Occupation labels to public roles
# First match wins, specific first
# fmt: off
ROLE_PATTERNS = [
    ("Politics & government", [
        r"politician", r"diplomat", r"statesperson", r"civil servant", r"government official",
        r"legislator", r"head of state", r"head of government", r"mayor", r"governor",
        r"ambassador", r"public servant", r"bureaucrat", r"lobbyist"]),
    ("Sports", [
        r"\bplayer\b", r"athlete", r"\bcoach\b", r"footballer", r"boxer", r"racing driver",
        r"racecar", r"golfer", r"wrestler", r"swimmer", r"jockey", r"cyclist", r"skier",
        r"skater", r"sprinter", r"\brunner\b", r"martial art", r"gymnast", r"football manager",
        r"sports", r"bodybuilder", r"umpire", r"referee", r"cricketer", r"fencer", r"rower",
        r"snowboarder", r"surfer", r"mountaineer", r"weightlifter", r"triathlete", r"tennis",
        r"quarterback", r"pitcher", r"basketball", r"baseball", r"hockey", r"rugby"]),
    ("Law & justice", [
        r"lawyer", r"\bjudge\b", r"jurist", r"attorney", r"prosecutor", r"barrister",
        r"solicitor", r"police", r"law enforcement", r"detective", r"sheriff",
        r"criminologist", r"magistrate", r"legal"]),
    ("Science, health & academia", [
        r"scientist", r"physicist", r"chemist", r"biologist", r"mathematician", r"physician",
        r"surgeon", r"\bdoctor\b", r"\bnurse\b", r"researcher", r"professor", r"academic",
        r"economist", r"engineer", r"astronaut", r"astronomer", r"historian", r"psychologist",
        r"psychiatrist", r"epidemiologist", r"university teacher", r"inventor", r"geologist",
        r"archaeologist", r"sociologist", r"philosopher", r"meteorologist", r"climatologist",
        r"neurologist", r"pharmac", r"veterinarian", r"dentist", r"statistician",
        r"anthropologist", r"scholar", r"lecturer", r"\bteacher\b", r"educator",
        r"programmer", r"computer", r"technologist", r"paramedic", r"medical",
        r"oncologist", r"cardiologist", r"pediatrician", r"geneticist", r"ecologist",
        r"zoologist", r"botanist"]),
    ("Entertainment & arts", [
        r"actor", r"actress", r"singer", r"musician", r"rapper", r"comedian",
        r"television presenter", r"television personality", r"television host",
        r"talk show host", r"game show host", r"\bmodel\b", r"film director",
        r"television director", r"theatre director", r"stage director", r"music director",
        r"art director", r"creative director", r"film producer", r"television producer",
        r"record producer", r"music producer", r"dancer", r"songwriter", r"composer",
        r"guitarist", r"drummer", r"pianist", r"bassist", r"violinist", r"disc jockey",
        r"youtuber", r"influencer", r"socialite", r"reality television", r"entertainer",
        r"stand-up", r"screenwriter", r"\bartist\b", r"painter", r"sculptor",
        r"photographer", r"designer", r"\bchef\b", r"\bcook\b", r"magician", r"drag queen",
        r"internet celebrity", r"vlogger", r"video blogger", r"media personality",
        r"celebrity", r"beauty pageant", r"stunt", r"choreographer", r"cartoonist",
        r"animator", r"illustrator", r"playwright", r"conductor", r"opera"]),
    ("Media & writing", [
        r"journalist", r"writer", r"\bauthor\b", r"novelist", r"poet", r"columnist",
        r"\beditor\b", r"news presenter", r"newscaster", r"anchor", r"broadcaster",
        r"reporter", r"commentator", r"blogger", r"publisher", r"critic", r"essayist",
        r"podcaster", r"radio", r"biographer", r"memoirist", r"correspondent", r"pundit"]),
    # Second tier, side activities
    ("Politics & government", [
        r"activist", r"military", r"soldier", r"army officer", r"naval officer",
        r"intelligence", r"trade unionist", r"campaigner", r"environmentalist"]),
    ("Business", [
        r"business", r"entrepreneur", r"executive", r"investor", r"banker", r"financier",
        r"industrialist", r"magnate", r"\bmanager\b", r"real estate", r"venture capitalist",
        r"\btrader\b", r"merchant", r"philanthropist", r"\bceo\b", r"founder", r"chairman",
        r"chairperson", r"accountant", r"consultant"]),
]
# fmt: on
ROLE_RULES = [(name, re.compile("|".join(patterns), re.I)) for name, patterns in ROLE_PATTERNS]
ROLES = [
    "Politics & government",
    "Business",
    "Science, health & academia",
    "Law & justice",
    "Media & writing",
    "Sports",
    "Entertainment & arts",
    "Other",
]
GENDER_KEYS = (("women", "F"), ("men", "M"), ("other", "O"))

# Citizenship regions and populations
REGIONS = [
    "United States",
    "Canada",
    "Europe",
    "Asia",
    "Africa",
    "Latin America & Caribbean",
    "Oceania",
]
SCORE_GROUPS = ROLES + REGIONS
# UN WPP 2019, thousands
UN_POPULATION_2019 = {
    "United States": 329065,
    "Canada": 37411,
    "Europe": 747183,
    "Asia": 4601371,
    "Africa": 1308064,
    "Latin America & Caribbean": 648121,
    "Oceania": 42128,
}
CONTINENT_OF_QID = {
    "Q15": "Africa",
    "Q46": "Europe",
    "Q48": "Asia",
    "Q49": "North America",
    "Q18": "South America",
    "Q55643": "Oceania",
    "Q3960": "Oceania",
    "Q538": "Oceania",
}
CONTINENT_TO_REGION = (
    ("Europe", "Europe"),
    ("Asia", "Asia"),
    ("North America", "Latin America & Caribbean"),
    ("South America", "Latin America & Caribbean"),
    ("Oceania", "Oceania"),
)
EURASIAN_AS_EUROPE = {"Q159", "Q15180", "Q34266"}  # Russia, Soviet Union, Russian Empire

# ---------------------------------------------------------------------------
# Wikidata requests
# ---------------------------------------------------------------------------

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
REQUEST_TIMEOUT = 90
MAX_ATTEMPTS = 8
FIRST_RETRY_DELAY = 5
MAX_RETRY_DELAY = 120
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
TIMEOUT_STATUS = 500
TIMEOUT_GIVE_UP_ATTEMPT = 2
MIN_SPLIT_BATCH = 10
SAVE_EVERY_BATCHES = 10
BATCH_PAUSE = 0.3
PEOPLE_BATCH = 200
COUNTRY_BATCH = 300

# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------

FULL_CONFIDENCE = 0.999
HIGH_CONFIDENCE = 0.9
INITIAL_SIZE = 4096
TOP_CONTRIBUTORS = 12
TOP_PEOPLE = 20
TOP_WOMEN = 10
HISTOGRAM_EDGES = 41
EPSILON = 1e-9
MIN_CORRELATION_READERS = 10
MIN_QUINTILE_READERS = 50
N_QUINTILES = 5
COMMUNITY_DOMINANCE = 0.5
ACTIVITY_LEVELS = ["Light", "Medium", "Heavy"]

# Recommender experiment
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 60000
SVD_COMPONENTS = 128
SMOOTHING_DISPLAYS = 20
SCORE_BLOCK = 400000
TIE_SCALE = 1e-9
BOOST_WEIGHTS = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]

# Streaming accumulator names
USER_SUMS = (
    "user_women_shown",
    "user_gendered_shown",
    "user_women_clicked",
    "user_gendered_clicked",
    "history_women",
    "history_gendered",
    "history_clicks",
    "user_people_shown",
    "user_people_clicked",
    "user_located_shown",
    "user_located_clicked",
)
SECTION_SUMS = (
    "section_women_shown",
    "section_men_shown",
    "section_women_clicked",
    "section_men_clicked",
    "section_gendered_shown",
    "section_gendered_clicked",
    "history_sections",
)
GROUP_SUMS = ("group_shown", "group_clicked")


def log(message):
    """Print a timestamped progress line."""
    print(f"[{time.time() - START_TIME:7.1f}s] {message}", flush=True)


# ---------------------------------------------------------------------------
# Loading MIND
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def open_member(path, name):
    """Stream a file from MIND."""
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            if name in files:
                with open(os.path.join(root, name), "rb") as handle:
                    yield handle
                return
        raise FileNotFoundError(f"{name} not found under {path}")
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if member.endswith(name):
                with archive.open(member) as handle:
                    yield handle
                return
    raise FileNotFoundError(f"{name} not found in {path}")


def load_news(path, split):
    """Read one split's news table."""
    with open_member(path, "news.tsv") as handle:
        news = pd.read_csv(handle, names=NEWS_COLUMNS, **TSV_OPTIONS)
    log(f"{split}: {len(news):,} news rows")
    return news


def dataset_label(paths):
    """Name the MIND size used."""
    names = " ".join(os.path.basename(str(path)).lower() for path in paths if path)
    if "large" in names:
        return "MIND-large"
    if "small" in names:
        return "MIND-small"
    return "MIND"


def parse_entities(raw_json):
    """Parse one entity JSON cell."""
    if not raw_json or raw_json == "[]":
        return []
    try:
        parsed = json.loads(raw_json)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []


def collect_entities(news):
    """List every Wikidata entity mention."""
    rows = []
    columns = zip(news["news_id"], news["title_entities"], news["abstract_entities"])
    for news_id, title_json, abstract_json in columns:
        for field, raw_json in (("title", title_json), ("abstract", abstract_json)):
            for entity in parse_entities(raw_json):
                qid = str(entity.get("WikidataId", "")).strip()
                if not QID_PATTERN.match(qid):
                    continue
                try:
                    confidence = float(entity.get("Confidence", 1.0))
                except (TypeError, ValueError):
                    confidence = 1.0
                label = str(entity.get("Label", ""))
                entity_type = str(entity.get("Type", ""))
                rows.append((news_id, qid, confidence, field, label, entity_type))
    columns = ["news_id", "qid", "confidence", "field", "label", "type"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Wikidata
# ---------------------------------------------------------------------------


class Wikidata:
    """Cached Wikidata SPARQL client."""

    def __init__(self, cache_path, offline=False, contact=""):
        """Load the cache if present."""
        self.cache_path = cache_path
        self.offline = offline
        user_agent = "FunnelVision/1.0 (student data challenge research; python-requests)"
        if contact:
            user_agent = f"FunnelVision/1.0 ({contact}; student data challenge research)"
        self.headers = {"User-Agent": user_agent, "Accept": "application/sparql-results+json"}
        self.cache = {"people": {}, "countries": {}}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as handle:
                self.cache = json.load(handle)
            self.cache.setdefault("people", {})
            self.cache.setdefault("countries", {})

    def save(self):
        """Write the cache atomically."""
        temporary_path = self.cache_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(self.cache, handle)
        os.replace(temporary_path, self.cache_path)

    def _query(self, sparql):
        """Run one query with retries."""
        import requests  # Only needed when online

        delay = FIRST_RETRY_DELAY
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = requests.post(
                    WIKIDATA_ENDPOINT,
                    data={"query": sparql, "format": "json"},
                    headers=self.headers,
                    timeout=REQUEST_TIMEOUT,
                )
                if response.status_code == 200:
                    return response.json()["results"]["bindings"]
                if response.status_code in RETRY_STATUS_CODES:
                    if (
                        response.status_code == TIMEOUT_STATUS
                        and attempt >= TIMEOUT_GIVE_UP_ATTEMPT
                    ):
                        raise RuntimeError("Wikidata query timed out")
                    try:
                        wait = int(response.headers.get("Retry-After", delay))
                    except ValueError:
                        wait = delay
                    log(f"  Wikidata busy ({response.status_code}), waiting {wait}s")
                    time.sleep(wait)
                    delay = min(delay * 2, MAX_RETRY_DELAY)
                    continue
                raise RuntimeError(f"Wikidata HTTP {response.status_code}: {response.text[:300]}")
            except requests.RequestException as error:
                log(f"  network error ({error.__class__.__name__}), retrying in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, MAX_RETRY_DELAY)
        raise RuntimeError(
            f"Wikidata did not answer after {MAX_ATTEMPTS} attempts. "
            "Re-run later, the cache keeps progress."
        )

    def _lookup(self, kind, qids, build_query, parse_row, batch_size):
        """Fetch missing QIDs in batches."""
        store = self.cache[kind]
        missing = sorted({qid for qid in qids if qid not in store})
        if not missing:
            return
        if self.offline:
            raise RuntimeError(
                f"{len(missing)} {kind} QIDs missing from cache and --offline is set"
            )
        log(f"Wikidata: looking up {len(missing):,} {kind} QIDs in batches of {batch_size}")

        def fetch(batch):
            """Fetch batch, split on errors."""
            try:
                rows = self._query(build_query(batch))
            except RuntimeError:
                if len(batch) <= MIN_SPLIT_BATCH:
                    raise
                half = len(batch) // 2
                log(f"  splitting a batch of {len(batch)} after repeated errors")
                fetch(batch[:half])
                fetch(batch[half:])
                return
            found = {}
            for binding in rows:
                qid = binding["item"]["value"].rsplit("/", 1)[-1]
                found[qid] = parse_row(binding)
            for qid in batch:
                store[qid] = found.get(qid)  # None means not human

        for start in range(0, len(missing), batch_size):
            fetch(missing[start : start + batch_size])
            if (start // batch_size) % SAVE_EVERY_BATCHES == 0:
                log(f"  {min(start + batch_size, len(missing)):,}/{len(missing):,}")
                self.save()
            time.sleep(BATCH_PAUSE)
        self.save()

    def people(self, qids):
        """Gender, occupations and citizenships."""

        def build_query(batch):
            """SPARQL for a people batch."""
            values = " ".join(f"wd:{qid}" for qid in batch)
            return f"""
SELECT ?item
  (GROUP_CONCAT(DISTINCT STRAFTER(STR(?g), "entity/"); separator="|") AS ?genders)
  (GROUP_CONCAT(DISTINCT ?occLabel; separator="|") AS ?occs)
  (GROUP_CONCAT(DISTINCT STRAFTER(STR(?c), "entity/"); separator="|") AS ?countries)
WHERE {{
  VALUES ?item {{ {values} }}
  ?item wdt:P31 wd:Q5 .
  OPTIONAL {{ ?item wdt:P21 ?g . }}
  OPTIONAL {{ ?item wdt:P106 ?occ . ?occ rdfs:label ?occLabel . FILTER(LANG(?occLabel) = "en") }}
  OPTIONAL {{ ?item wdt:P27 ?c . }}
}}
GROUP BY ?item"""

        def parse_row(binding):
            """Cache record for one person."""

            def values(key):
                """Split a concatenated binding."""
                return [
                    value for value in binding.get(key, {}).get("value", "").split("|") if value
                ]

            # Cache keys stay short
            return {"g": values("genders"), "occ": values("occs"), "c": values("countries")}

        self._lookup("people", qids, build_query, parse_row, batch_size=PEOPLE_BATCH)
        return {qid: self.cache["people"].get(qid) for qid in qids}

    def countries(self, qids):
        """Labels and continents of countries."""

        def build_query(batch):
            """SPARQL for a country batch."""
            values = " ".join(f"wd:{qid}" for qid in batch)
            return f"""
SELECT ?item (SAMPLE(?lab) AS ?label)
  (GROUP_CONCAT(DISTINCT STRAFTER(STR(?cont), "entity/"); separator="|") AS ?continents)
WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item rdfs:label ?lab . FILTER(LANG(?lab) = "en") }}
  OPTIONAL {{ ?item wdt:P30 ?cont . }}
}}
GROUP BY ?item"""

        def parse_row(binding):
            """Cache record for one country."""
            raw = binding.get("continents", {}).get("value", "")
            continents = [value for value in raw.split("|") if value]
            return {"label": binding.get("label", {}).get("value", ""), "continents": continents}

        self._lookup("countries", qids, build_query, parse_row, batch_size=COUNTRY_BATCH)
        return {qid: self.cache["countries"].get(qid) for qid in qids}


# ---------------------------------------------------------------------------
# People attributes
# ---------------------------------------------------------------------------


def gender_of(gender_qids):
    """Code recorded gender as F/M/O/U."""
    gender_qids = set(gender_qids or [])
    is_woman, is_man = bool(gender_qids & WOMEN_QIDS), bool(gender_qids & MEN_QIDS)
    if is_woman and not is_man:
        return "F"
    if is_man and not is_woman:
        return "M"
    if gender_qids:
        return "O"
    return "U"


def role_of(occupations):
    """First matching public role."""
    if not occupations:
        return "Unknown"
    for name, pattern in ROLE_RULES:
        if any(pattern.search(occupation) for occupation in occupations):
            return name
    return "Other"


def region_of(country_qid, country_info):
    """Map country to region."""
    if country_qid == US_QID:
        return "United States"
    if country_qid == CANADA_QID:
        return "Canada"
    if not country_info:
        return None
    continents = {CONTINENT_OF_QID.get(qid, "") for qid in country_info.get("continents", [])} - {
        ""
    }
    if "Africa" in continents:
        return "Africa"
    if "Europe" in continents and "Asia" in continents:
        return "Europe" if country_qid in EURASIAN_AS_EUROPE else "Asia"
    for continent, region in CONTINENT_TO_REGION:
        if continent in continents:
            return region
    return None


def build_people(entities, wikidata):
    """Table of linked humans."""
    qids = sorted(entities["qid"].unique())
    records = wikidata.people(qids)
    humans = {qid: record for qid, record in records.items() if record is not None}
    country_qids = sorted({country for record in humans.values() for country in record["c"]})
    country_info = wikidata.countries(country_qids) if country_qids else {}
    names = entities.groupby("qid")["label"].agg(
        lambda labels: Counter(labels).most_common(1)[0][0]
    )
    rows = []
    for qid, record in humans.items():
        regions = {region_of(country, country_info.get(country)) for country in record["c"]}
        rows.append(
            {
                "qid": qid,
                "name": names.get(qid, qid),
                "gender": gender_of(record["g"]),
                "role": role_of(record["occ"]),
                "regions": sorted(region for region in regions if region),
                "has_occupation": bool(record["occ"]),
                "has_citizenship": bool(record["c"]),
            }
        )
    columns = ["name", "gender", "role", "regions", "has_occupation", "has_citizenship"]
    people = pd.DataFrame(rows).set_index("qid") if rows else pd.DataFrame(columns=columns)
    stats = {
        "entity_qids": len(qids),
        "humans": len(people),
        "humans_gender_recorded": int((people["gender"] != "U").sum()),
        "humans_women": int((people["gender"] == "F").sum()),
        "humans_men": int((people["gender"] == "M").sum()),
        "humans_other_gender": int((people["gender"] == "O").sum()),
        "humans_with_occupation": int(people["has_occupation"].sum()),
        "humans_with_citizenship": int(people["has_citizenship"].sum()),
        "humans_with_region": int(people["regions"].map(len).gt(0).sum()),
        "role_counts": people["role"].value_counts().to_dict(),
    }
    return people, stats


# ---------------------------------------------------------------------------
# Article features
# ---------------------------------------------------------------------------


def article_features(
    entities, people, news_index, fields=("title", "abstract"), min_confidence=0.0, exclude=()
):
    """Per-article people counts by group."""
    n_articles = len(news_index)
    mentions = entities[
        entities["field"].isin(fields) & (entities["confidence"] >= min_confidence)
    ]
    mentions = mentions[mentions["qid"].isin(people.index) & ~mentions["qid"].isin(set(exclude))]
    mentions = mentions.drop_duplicates(["news_id", "qid"])
    mentions = mentions.join(people[["gender", "role", "regions"]], on="qid")
    positions = news_index.get_indexer(mentions["news_id"])
    mentions = mentions.assign(article=positions)[positions >= 0]
    article = mentions["article"].to_numpy()

    def count_people(mask=None):
        """Count mentions per article."""
        if mask is None:
            return np.bincount(article, minlength=n_articles).astype(np.float64)
        return np.bincount(article[mask], minlength=n_articles).astype(np.float64)

    genders = mentions["gender"].to_numpy()
    roles = mentions["role"].to_numpy()
    features = {"people": count_people()}
    for key, code in GENDER_KEYS + (("unknown", "U"),):
        features[key] = count_people(genders == code)
    features["gendered"] = features["women"] + features["men"] + features["other"]
    for role in ROLES:
        features[f"all|{role}"] = count_people(roles == role)
        for key, code in GENDER_KEYS:
            features[f"{key}|{role}"] = count_people((genders == code) & (roles == role))
    # Split over distinct regions
    region_weight = {region: np.zeros(n_articles) for region in REGIONS}
    for position, regions in zip(article, mentions["regions"]):
        if regions:
            weight = 1.0 / len(regions)
            for region in regions:
                region_weight[region][position] += weight
    for region in REGIONS:
        features[f"region|{region}"] = region_weight[region]
    features["located"] = sum(region_weight.values())
    return features, mentions[["news_id", "qid", "article", "gender", "role"]]


def share(weights, numerator, denominator):
    """Weighted share, NaN when empty."""
    total = float(np.dot(weights, denominator))
    return float(np.dot(weights, numerator)) / total if total > 0 else float("nan")


def dissimilarity(women_by_group, men_by_group):
    """Duncan dissimilarity index."""
    women = np.asarray(women_by_group, float)
    men = np.asarray(men_by_group, float)
    if women.sum() <= 0 or men.sum() <= 0:
        return float("nan")
    return float(0.5 * np.abs(women / women.sum() - men / men.sum()).sum())


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def round_floats(value, digits=6):
    """Round floats for JSON output."""
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), digits)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(item, digits) for item in value]
    return value


def quantiles(values, levels=(0.025, 0.975)):
    """Quantiles of the finite values."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [float("nan")] * len(levels)
    return [float(np.quantile(values, level)) for level in levels]


class GrowingSum:
    """Sums that grow on demand."""

    def __init__(self):
        """Start with a small buffer."""
        self.values = np.zeros(INITIAL_SIZE)

    def add(self, positions, weights=None):
        """Add weights at positions."""
        if len(positions) == 0:
            return
        needed = int(positions.max()) + 1
        if needed > len(self.values):
            bigger = np.zeros(max(needed, 2 * len(self.values)))
            bigger[: len(self.values)] = self.values
            self.values = bigger
        self.values[:needed] += np.bincount(positions, weights=weights, minlength=needed)

    def get(self, size):
        """Return the first size sums."""
        out = np.zeros(size)
        kept = min(size, len(self.values))
        out[:kept] = self.values[:kept]
        return out


def stream_behaviors(splits, news_index, features, section_code, args, rng):
    """One chunked pass over behaviors."""
    n_articles, n_sections = len(news_index), len(SECTIONS)
    women, men, gendered = features["women"], features["men"], features["gendered"]
    people, located = features["people"], features["located"]
    displays = {split: np.zeros(n_articles) for split, _ in splits}
    clicks = {split: np.zeros(n_articles) for split, _ in splits}
    user_sums = {key: GrowingSum() for key in USER_SUMS}
    section_sums = {key: GrowingSum() for key in SECTION_SUMS}
    # Audit groups: roles, then regions
    n_groups = len(SCORE_GROUPS)
    group_columns = [features[f"all|{role}"] for role in ROLES]
    group_columns += [features[f"region|{region}"] for region in REGIONS]
    group_sums = {key: GrowingSum() for key in GROUP_SUMS}
    user_codes, hour_codes = {}, {}
    refresh_user, refresh_hour, refresh_women, refresh_gendered = [], [], [], []
    rows_by_split = {split: 0 for split, _ in splits}
    recommender_sample = None
    first_time, last_time = None, None
    n_displays = n_clicks = n_rows = n_missing = 0
    for split, path in splits:
        with open_member(path, "behaviors.tsv") as handle:
            reader = pd.read_csv(
                handle, names=BEHAVIOR_COLUMNS, chunksize=args.chunk, **TSV_OPTIONS
            )
            for chunk in reader:
                if args.user_fraction < 1:
                    hashes = pd.util.hash_pandas_object(chunk["user_id"], index=False).to_numpy()
                    kept = (hashes % HASH_BUCKETS) < int(args.user_fraction * HASH_BUCKETS)
                    chunk = chunk[kept]
                    if chunk.empty:
                        continue
                n_refreshes = len(chunk)
                times = pd.to_datetime(chunk["time"], format=MIND_TIME_FORMAT, errors="coerce")
                earliest, latest = times.min(), times.max()
                if first_time is None or earliest < first_time:
                    first_time = earliest
                if last_time is None or latest > last_time:
                    last_time = latest
                hour_index, hours = pd.factorize(times.dt.floor("h").astype(str))
                hour_map = [hour_codes.setdefault(hour, len(hour_codes)) for hour in hours]
                hour_code = np.array(hour_map, np.int64)[hour_index]
                user_index, users = pd.factorize(chunk["user_id"])
                known_users = len(user_codes)
                user_map = [user_codes.setdefault(user, len(user_codes)) for user in users]
                user_map = np.array(user_map, np.int64)
                user_code = user_map[user_index]

                tokens = chunk["impressions"].str.split(" ")
                lengths = tokens.str.len().to_numpy()
                items = pd.Series(np.concatenate(tokens.to_numpy()))
                label_chars = items.str[-1].to_numpy()
                if not np.isin(label_chars[:LABEL_CHECK_ITEMS], ["0", "1"]).all():
                    raise SystemExit(
                        f"{path} has no click labels (the MIND test set). "
                        "Use the train and dev files."
                    )
                article = news_index.get_indexer(items.str[:-2])
                clicked = (label_chars == "1").astype(np.float64)
                refresh = np.repeat(np.arange(n_refreshes), lengths)
                known = article >= 0
                n_missing += int((~known).sum())
                article, clicked, refresh = article[known], clicked[known], refresh[known]

                displays[split] += np.bincount(article, minlength=n_articles)
                clicks[split] += np.bincount(article, weights=clicked, minlength=n_articles)
                women_shown, gendered_shown = women[article], gendered[article]
                men_shown, section = men[article], section_code[article]
                user = user_code[refresh]
                user_sums["user_women_shown"].add(user, women_shown)
                user_sums["user_gendered_shown"].add(user, gendered_shown)
                user_sums["user_women_clicked"].add(user, women_shown * clicked)
                user_sums["user_gendered_clicked"].add(user, gendered_shown * clicked)
                user_section = user * n_sections + section
                section_sums["section_women_shown"].add(user_section, women_shown)
                section_sums["section_men_shown"].add(user_section, men_shown)
                section_sums["section_women_clicked"].add(user_section, women_shown * clicked)
                section_sums["section_men_clicked"].add(user_section, men_shown * clicked)
                section_sums["section_gendered_shown"].add(user_section, gendered_shown)
                section_sums["section_gendered_clicked"].add(
                    user_section, gendered_shown * clicked
                )
                people_shown, located_shown = people[article], located[article]
                user_sums["user_people_shown"].add(user, people_shown)
                user_sums["user_people_clicked"].add(user, people_shown * clicked)
                user_sums["user_located_shown"].add(user, located_shown)
                user_sums["user_located_clicked"].add(user, located_shown * clicked)
                user_group = user * n_groups
                for group, column in enumerate(group_columns):
                    values = column[article]
                    group_sums["group_shown"].add(user_group + group, values)
                    group_sums["group_clicked"].add(user_group + group, values * clicked)
                refresh_women.append(
                    np.bincount(refresh, women_shown, n_refreshes).astype(np.float32)
                )
                refresh_gendered.append(
                    np.bincount(refresh, gendered_shown, n_refreshes).astype(np.float32)
                )
                refresh_user.append(user_code.astype(np.int32))
                refresh_hour.append(hour_code.astype(np.int32))

                # History of first-seen readers
                new_users = np.flatnonzero(user_map >= known_users)
                if len(new_users):
                    _, first_row = np.unique(user_index, return_index=True)
                    rows = first_row[new_users]
                    histories = chunk["history"].iloc[rows].str.split(" ")
                    history_lengths = histories.str.len().to_numpy()
                    history_items = pd.Series(np.concatenate(histories.to_numpy()))
                    owner = np.repeat(user_map[new_users], history_lengths)
                    history_article = news_index.get_indexer(history_items)
                    known_history = history_article >= 0
                    owner = owner[known_history]
                    history_article = history_article[known_history]
                    user_sums["history_women"].add(owner, women[history_article])
                    user_sums["history_gendered"].add(owner, gendered[history_article])
                    user_sums["history_clicks"].add(owner)
                    section_sums["history_sections"].add(
                        owner * n_sections + section_code[history_article]
                    )

                # Sample dev refreshes for recommenders
                if split == "dev" and args.rec_sample:
                    part = pd.DataFrame(
                        {
                            "key": rng.random(n_refreshes),
                            "history": chunk["history"].to_numpy(),
                            "impressions": chunk["impressions"].to_numpy(),
                        }
                    )
                    if recommender_sample is None:
                        recommender_sample = part
                    else:
                        recommender_sample = pd.concat(
                            [recommender_sample, part], ignore_index=True
                        )
                    if len(recommender_sample) > args.rec_sample:
                        recommender_sample = recommender_sample.nsmallest(
                            args.rec_sample, "key"
                        ).reset_index(drop=True)

                n_rows += n_refreshes
                rows_by_split[split] += n_refreshes
                n_displays += len(article)
                n_clicks += int(clicked.sum())
                log(
                    f"  {split}: {rows_by_split[split]:,} feed refreshes read, "
                    f"{len(user_codes):,} readers so far"
                )
    n_users = len(user_codes)
    stream = {"n_users": n_users, "displays_by_split": displays, "clicks_by_split": clicks}
    for key in USER_SUMS:
        stream[key] = user_sums[key].get(n_users)
    for key in SECTION_SUMS:
        stream[key] = section_sums[key].get(n_users * n_sections).reshape(n_users, n_sections)
    for key in GROUP_SUMS:
        stream[key] = group_sums[key].get(n_users * n_groups).reshape(n_users, n_groups)
    stream.update(
        {
            "refresh_user": np.concatenate(refresh_user),
            "refresh_hour": np.concatenate(refresh_hour),
            "refresh_women": np.concatenate(refresh_women).astype(np.float64),
            "refresh_gendered": np.concatenate(refresh_gendered).astype(np.float64),
            "feed_refreshes": n_rows,
            "rows_by_split": rows_by_split,
            "displays": n_displays,
            "clicks": n_clicks,
            "missing_items": n_missing,
            "first_time": first_time,
            "last_time": last_time,
            "recommender_sample": recommender_sample,
        }
    )
    return stream


def run(args):
    """Run the full audit."""
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    splits = [(split, path) for split, path in (("train", args.train), ("dev", args.dev)) if path]
    results = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "dataset": dataset_label([path for _, path in splits]),
            "script_version": SCRIPT_VERSION,
            "seed": SEED,
            "bootstrap_reps": args.boot,
            "null_reps": args.null_reps,
            "min_people_per_user": args.min_user,
            "user_fraction": args.user_fraction,
            "recommender_sample": args.rec_sample,
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
        }
    }

    # News, people, article features
    news = pd.concat([load_news(path, split) for split, path in splits])
    news = news.drop_duplicates("news_id").reset_index(drop=True)
    news_index = pd.Index(news["news_id"])
    n_articles = len(news_index)
    article_category = news["category"].to_numpy()
    categories = sorted(set(article_category))
    category_code = pd.Index(categories).get_indexer(article_category)
    article_section = [
        SECTION_OF_CATEGORY.get(category, DEFAULT_SECTION) for category in article_category
    ]
    section_code = pd.Index(SECTIONS).get_indexer(article_section)
    n_sections = len(SECTIONS)

    entities = collect_entities(news)
    log(f"{len(entities):,} entity mentions, {entities['qid'].nunique():,} distinct Wikidata IDs")
    wikidata = Wikidata(
        os.path.join(args.out, "wikidata_cache.json"), offline=args.offline, contact=args.contact
    )
    if args.wikidata_cache and os.path.exists(args.wikidata_cache):
        with open(args.wikidata_cache, encoding="utf-8") as handle:
            preload = json.load(handle)
        wikidata.cache["people"].update(preload.get("people", {}))
        wikidata.cache["countries"].update(preload.get("countries", {}))
    people, people_stats = build_people(entities, wikidata)
    results["people_coverage"] = people_stats
    log(
        f"{people_stats['humans']:,} humans, {people_stats['humans_women']:,} women, "
        f"{people_stats['humans_men']:,} men"
    )
    confidence = entities["confidence"].to_numpy()
    has_confidence = len(confidence) > 0
    results["people_coverage"]["entity_confidence"] = {
        "share_equal_1": (
            float((confidence >= FULL_CONFIDENCE).mean()) if has_confidence else None
        ),
        "share_ge_0_9": (
            float((confidence >= HIGH_CONFIDENCE).mean()) if has_confidence else None
        ),
        "median": float(np.median(confidence)) if has_confidence else None,
    }

    features, appearances = article_features(entities, people, news_index)
    women, men = features["women"], features["men"]
    gendered, named = features["gendered"], features["people"]

    # Stream the logs
    log("reading behaviour logs in chunks")
    stream = stream_behaviors(splits, news_index, features, section_code, args, rng)
    n_users = stream["n_users"]
    displays = sum(stream["displays_by_split"].values())
    clicks = sum(stream["clicks_by_split"].values())
    pool_weights = (displays > 0).astype(np.float64)
    in_pool = pool_weights > 0
    has_woman = (women > 0).astype(float)
    men_only = ((men > 0) & (women == 0)).astype(float)
    no_people = (named == 0).astype(float)
    log(
        f"{stream['feed_refreshes']:,} feed refreshes, {stream['displays']:,} displays, "
        f"{stream['clicks']:,} clicks, {n_users:,} readers"
    )

    results["data"] = {
        "dataset": results["meta"]["dataset"],
        "splits": [split for split, _ in splits],
        "feed_refreshes_by_split": stream["rows_by_split"],
        "users": n_users,
        "impression_logs": stream["feed_refreshes"],
        "displays": stream["displays"],
        "clicks": stream["clicks"],
        "ctr": stream["clicks"] / max(stream["displays"], 1),
        "items_without_metadata": stream["missing_items"],
        "news_rows": n_articles,
        "articles_in_circulation": int(pool_weights.sum()),
        "first_log": str(stream["first_time"]),
        "last_log": str(stream["last_time"]),
        "categories": {
            category: int(((article_category == category) & in_pool).sum())
            for category in categories
        },
    }

    named_in_pool = named[in_pool]
    results["article_coverage"] = {
        "share_with_entity": float(
            np.isin(news_index[in_pool], entities["news_id"].unique()).mean()
        ),
        "share_with_person": float((named_in_pool > 0).mean()),
        "share_with_gendered_person": float((gendered[in_pool] > 0).mean()),
        "share_with_woman": float((women[in_pool] > 0).mean()),
        "share_men_only": float(men_only[in_pool].mean()),
        "mean_people_per_article_with_people": (
            float(named_in_pool[named_in_pool > 0].mean()) if (named_in_pool > 0).any() else None
        ),
        "share_displays_with_gendered_person": share(
            displays, (gendered > 0).astype(float), np.ones(n_articles)
        ),
        "share_by_category_with_person": {
            category: float((named[in_pool & (article_category == category)] > 0).mean())
            for category in categories
            if (in_pool & (article_category == category)).any()
        },
    }

    # 1. The funnel
    stages = {"published": pool_weights, "shown": displays, "clicked": clicks}
    funnel = {stage: share(weights, women, gendered) for stage, weights in stages.items()}
    funnel_men = {stage: share(weights, men, gendered) for stage, weights in stages.items()}
    funnel_other = {
        stage: share(weights, features["other"], gendered) for stage, weights in stages.items()
    }
    women_shown, gendered_shown = stream["user_women_shown"], stream["user_gendered_shown"]
    women_clicked = stream["user_women_clicked"]
    gendered_clicked = stream["user_gendered_clicked"]
    n_boot = args.boot
    boot_shown, boot_clicked = np.empty(n_boot), np.empty(n_boot)
    boot_dissimilarity = {"shown": [], "clicked": []}
    n_score_groups = len(SCORE_GROUPS) + n_sections
    boot_group_shown = np.empty((n_boot, n_score_groups))
    boot_group_clicked = np.empty((n_boot, n_score_groups))
    n_roles = len(ROLES)
    for replicate in range(n_boot):
        weights = rng.poisson(1.0, n_users).astype(np.float64)  # Poisson reader bootstrap
        boot_shown[replicate] = (weights @ women_shown) / (weights @ gendered_shown)
        boot_clicked[replicate] = (weights @ women_clicked) / (weights @ gendered_clicked)
        boot_dissimilarity["shown"].append(
            dissimilarity(
                weights @ stream["section_women_shown"], weights @ stream["section_men_shown"]
            )
        )
        boot_dissimilarity["clicked"].append(
            dissimilarity(
                weights @ stream["section_women_clicked"], weights @ stream["section_men_clicked"]
            )
        )
        group_shown = weights @ stream["group_shown"]
        group_clicked = weights @ stream["group_clicked"]
        boot_group_shown[replicate] = np.concatenate(
            [
                group_shown[:n_roles] / (weights @ stream["user_people_shown"]),
                group_shown[n_roles:] / (weights @ stream["user_located_shown"]),
                (weights @ stream["section_women_shown"])
                / (weights @ stream["section_gendered_shown"]),
            ]
        )
        boot_group_clicked[replicate] = np.concatenate(
            [
                group_clicked[:n_roles] / (weights @ stream["user_people_clicked"]),
                group_clicked[n_roles:] / (weights @ stream["user_located_clicked"]),
                (weights @ stream["section_women_clicked"])
                / (weights @ stream["section_gendered_clicked"]),
            ]
        )
    published_share = funnel["published"]
    results["funnel_gender"] = {
        "women_share": funnel,
        "men_share": funnel_men,
        "other_share": funnel_other,
        "ratio_shown_vs_published": funnel["shown"] / published_share,
        "ratio_clicked_vs_shown": funnel["clicked"] / funnel["shown"],
        "ratio_clicked_vs_published": funnel["clicked"] / published_share,
        "ci95": {
            "shown": quantiles(boot_shown),
            "clicked": quantiles(boot_clicked),
            "ratio_shown_vs_published": quantiles(boot_shown / published_share),
            "ratio_clicked_vs_shown": quantiles(boot_clicked / boot_shown),
            "ratio_clicked_vs_published": quantiles(boot_clicked / published_share),
        },
        "gendered_appearances": {
            stage: float(np.dot(weights, gendered)) for stage, weights in stages.items()
        },
        "women_per_man": {
            stage: (funnel[stage] / funnel_men[stage]) if funnel_men[stage] else None
            for stage in stages
        },
    }
    log(
        f"funnel women share: published {funnel['published']:.3f}, "
        f"shown {funnel['shown']:.3f}, clicked {funnel['clicked']:.3f}"
    )

    # 2. Reach of women's stories
    story_groups = {
        "features_a_woman": has_woman,
        "men_only": men_only,
        "no_people": no_people,
        "any_person": (named > 0).astype(float),
    }
    results["article_reach"] = {}
    for group_name, group_weights in story_groups.items():
        mask = in_pool & (group_weights > 0)
        results["article_reach"][group_name] = {
            "articles": int(mask.sum()),
            "mean_displays": float(displays[mask].mean()) if mask.any() else None,
            "median_displays": float(np.median(displays[mask])) if mask.any() else None,
            "ctr": (
                float(clicks[mask].sum() / displays[mask].sum())
                if mask.any() and displays[mask].sum() > 0
                else None
            ),
            "share_of_articles": float(mask.sum() / in_pool.sum()),
            "share_of_displays": float(displays[mask].sum() / displays.sum()),
            "share_of_clicks": float(clicks[mask].sum() / clicks.sum()) if clicks.sum() else None,
        }
    results["article_reach_by_section"] = {}
    for section_index, section in enumerate(SECTIONS):
        in_section = in_pool & (section_code == section_index)
        row = {}
        for group_name in ("features_a_woman", "men_only"):
            mask = in_section & (story_groups[group_name] > 0)
            row[group_name] = {
                "articles": int(mask.sum()),
                "mean_displays": float(displays[mask].mean()) if mask.any() else None,
                "median_displays": float(np.median(displays[mask])) if mask.any() else None,
                "ctr": (
                    float(clicks[mask].sum() / displays[mask].sum())
                    if mask.any() and displays[mask].sum()
                    else None
                ),
            }
        results["article_reach_by_section"][section] = row

    # 3. Sections, categories, decomposition, roles
    n_categories = len(categories)

    def sums_by(codes, n_codes, weights):
        """Women, men, gendered per code."""
        return (
            np.bincount(codes, weights * women, n_codes),
            np.bincount(codes, weights * men, n_codes),
            np.bincount(codes, weights * gendered, n_codes),
        )

    by_category = {
        stage: sums_by(category_code, n_categories, weights) for stage, weights in stages.items()
    }
    results["category"] = {
        category: {
            stage: {
                "women_share": (
                    (by_category[stage][0][index] / by_category[stage][2][index])
                    if by_category[stage][2][index] > 0
                    else None
                ),
                "gendered_appearances": float(by_category[stage][2][index]),
            }
            for stage in stages
        }
        for index, category in enumerate(categories)
    }
    results["category_dissimilarity"] = {
        stage: dissimilarity(by_category[stage][0], by_category[stage][1]) for stage in stages
    }
    by_section = {
        stage: sums_by(section_code, n_sections, weights) for stage, weights in stages.items()
    }
    results["section"] = {}
    for index, section in enumerate(SECTIONS):
        results["section"][section] = {}
        for stage in stages:
            section_women, section_men, section_gendered = by_section[stage]
            results["section"][section][stage] = {
                "women_share": (
                    (section_women[index] / section_gendered[index])
                    if section_gendered[index] > 0
                    else None
                ),
                "share_of_women": (
                    section_women[index] / section_women.sum() if section_women.sum() else None
                ),
                "share_of_men": (
                    section_men[index] / section_men.sum() if section_men.sum() else None
                ),
                "gendered_appearances": float(section_gendered[index]),
            }
    results["section_dissimilarity"] = {
        stage: dissimilarity(by_section[stage][0], by_section[stage][1]) for stage in stages
    }
    results["section_dissimilarity_ci95"] = {
        stage: quantiles(values) for stage, values in boot_dissimilarity.items()
    }

    def shift_share(codes, labels, stage_from, stage_to):
        """Split change into mix, within."""
        n_groups = len(labels)
        women_from = np.bincount(codes, stages[stage_from] * women, n_groups)
        gendered_from = np.bincount(codes, stages[stage_from] * gendered, n_groups)
        women_to = np.bincount(codes, stages[stage_to] * women, n_groups)
        gendered_to = np.bincount(codes, stages[stage_to] * gendered, n_groups)
        weight_from = gendered_from / gendered_from.sum()
        weight_to = gendered_to / gendered_to.sum()
        share_from = np.divide(
            women_from, gendered_from, out=np.zeros(n_groups), where=gendered_from > 0
        )
        share_to = np.divide(women_to, gendered_to, out=share_from.copy(), where=gendered_to > 0)
        within = weight_from * (share_to - share_from)
        mix = (weight_to - weight_from) * share_to
        order = np.argsort(-np.abs(within + mix))[:TOP_CONTRIBUTORS]
        return {
            "total": float((weight_to * share_to).sum() - (weight_from * share_from).sum()),
            "within": float(within.sum()),
            "mix": float(mix.sum()),
            "groups": int((gendered_from > 0).sum()),
            "top_contributors": [
                {"group": labels[index], "within": float(within[index]), "mix": float(mix[index])}
                for index in order
            ],
        }

    subsections = news["category"].to_numpy() + "/" + news["subcategory"].to_numpy()
    subsection_code, subsection_labels = pd.factorize(subsections)
    subsection_labels = list(subsection_labels)
    results["decomposition"] = {
        "by_category": {
            "shown_vs_published": shift_share(category_code, categories, "published", "shown"),
            "clicked_vs_shown": shift_share(category_code, categories, "shown", "clicked"),
        },
        "by_subcategory": {
            "shown_vs_published": shift_share(
                subsection_code, subsection_labels, "published", "shown"
            ),
            "clicked_vs_shown": shift_share(
                subsection_code, subsection_labels, "shown", "clicked"
            ),
        },
    }

    roles_by_stage, role_mix = {}, {}
    for stage, weights in stages.items():
        women_roles = np.array([np.dot(weights, features[f"women|{role}"]) for role in ROLES])
        men_roles = np.array([np.dot(weights, features[f"men|{role}"]) for role in ROLES])
        other_roles = np.array([np.dot(weights, features[f"other|{role}"]) for role in ROLES])
        roles_by_stage[stage] = {}
        for index, role in enumerate(ROLES):
            total = women_roles[index] + men_roles[index] + other_roles[index]
            roles_by_stage[stage][role] = {
                "women_share": women_roles[index] / total if total > 0 else None,
                "share_of_women": (
                    women_roles[index] / women_roles.sum() if women_roles.sum() else None
                ),
                "share_of_men": men_roles[index] / men_roles.sum() if men_roles.sum() else None,
            }
        role_mix[stage] = dissimilarity(women_roles, men_roles)
    results["roles"] = roles_by_stage

    # Audit scorecard, gains and losses
    def stage_share(numerator, denominator):
        """Share at every stage."""
        return {stage: share(weights, numerator, denominator) for stage, weights in stages.items()}

    score_rows = [("Women (all sections)", "Gender", stage_share(women, gendered), None)]
    for index, section in enumerate(SECTIONS):
        in_section = (section_code == index).astype(float)
        score_rows.append(
            (
                f"Women in {section}",
                "Gender by section",
                stage_share(women * in_section, gendered * in_section),
                len(SCORE_GROUPS) + index,
            )
        )
    for index, role in enumerate(ROLES):
        score_rows.append(
            (
                role,
                "Role (share of all people)",
                stage_share(features[f"all|{role}"], named),
                index,
            )
        )
    for index, region in enumerate(REGIONS):
        score_rows.append(
            (
                region,
                "Citizenship (share of people with one)",
                stage_share(features[f"region|{region}"], features["located"]),
                len(ROLES) + index,
            )
        )
    scorecard = []
    for name, family, shares, boot_index in score_rows:
        pool_share, shown_share = shares["published"], shares["shown"]
        clicked_share = shares["clicked"]
        if boot_index is None:
            boot_s, boot_c = boot_shown, boot_clicked
        else:
            boot_s = boot_group_shown[:, boot_index]
            boot_c = boot_group_clicked[:, boot_index]
        scorecard.append(
            {
                "group": name,
                "family": family,
                "published": pool_share,
                "shown": shown_share,
                "clicked": clicked_share,
                "ratio_platform": shown_share / pool_share if pool_share else None,
                "ratio_platform_ci95": quantiles(boot_s / pool_share) if pool_share else None,
                "ratio_reader": clicked_share / shown_share if shown_share else None,
                "ratio_reader_ci95": quantiles(boot_c / boot_s) if shown_share else None,
            }
        )
    results["scorecard"] = scorecard
    results["role_dissimilarity"] = role_mix

    # 4. Geography
    world_population = sum(UN_POPULATION_2019.values())
    geography = {}
    for stage, weights in stages.items():
        total = float(np.dot(weights, features["located"]))
        geography[stage] = {
            region: float(np.dot(weights, features[f"region|{region}"])) / total if total else None
            for region in REGIONS
        }
    population_share = {
        region: UN_POPULATION_2019[region] / world_population for region in REGIONS
    }
    results["geography"] = {
        "share": geography,
        "population_share": population_share,
        "visibility_index_shown": {
            region: (
                (geography["shown"][region] / (UN_POPULATION_2019[region] / world_population))
                if geography["shown"][region] is not None
                else None
            )
            for region in REGIONS
        },
        "appearances_with_region_shown": float(np.dot(displays, features["located"])),
    }

    # 5. Most visible people
    mentions = appearances.copy()
    article = mentions["article"].to_numpy()
    mentions["displays"] = displays[article]
    mentions["clicks"] = clicks[article]
    mentions["in_pool"] = pool_weights[article]
    per_person = mentions.groupby("qid").agg(
        displays=("displays", "sum"), clicks=("clicks", "sum"), articles=("in_pool", "sum")
    )
    per_person = per_person.join(people[["name", "gender", "role"]])
    per_person = per_person.sort_values("displays", ascending=False)
    total_appearances = float(per_person["displays"].sum())

    def share_of_appearances(value):
        """Share of all person appearances."""
        return value / total_appearances if total_appearances else None

    results["top_people"] = [
        {
            "name": str(person["name"]),
            "gender": person["gender"],
            "role": person["role"],
            "displays": float(person["displays"]),
            "clicks": float(person["clicks"]),
            "articles": float(person["articles"]),
            "share_of_all_appearances": share_of_appearances(float(person["displays"])),
        }
        for _, person in per_person.head(TOP_PEOPLE).iterrows()
    ]
    top_women = per_person[per_person["gender"] == "F"].head(TOP_WOMEN)
    results["top_women"] = [
        {
            "name": str(person["name"]),
            "role": person["role"],
            "displays": float(person["displays"]),
            "articles": float(person["articles"]),
        }
        for _, person in top_women.iterrows()
    ]
    results["concentration"] = {
        "people_shown": int((per_person["displays"] > 0).sum()),
        "top10_share_of_appearances": share_of_appearances(
            float(per_person["displays"].head(10).sum())
        ),
        "top10_women_count": int((per_person.head(10)["gender"] == "F").sum()),
        "top20_women_count": int((per_person.head(20)["gender"] == "F").sum()),
        "top1_share_of_appearances": share_of_appearances(
            float(per_person["displays"].head(1).sum())
        ),
    }

    # 6. Personalization vs shuffle
    refresh_women, refresh_gendered = stream["refresh_women"], stream["refresh_gendered"]
    refresh_user, refresh_hour = stream["refresh_user"], stream["refresh_hour"]

    def user_totals(user_of_refresh):
        """Women and gendered per reader."""
        return (
            np.bincount(user_of_refresh, refresh_women, n_users),
            np.bincount(user_of_refresh, refresh_gendered, n_users),
        )

    women_observed, gendered_observed = user_totals(refresh_user)
    eligible = gendered_observed >= args.min_user
    share_observed = women_observed[eligible] / gendered_observed[eligible]
    bin_edges = np.linspace(0, 1, HISTOGRAM_EDGES)
    hist_observed = np.histogram(share_observed, bin_edges)[0] / max(len(share_observed), 1)
    order = np.argsort(refresh_hour, kind="stable")
    bounds = np.flatnonzero(np.diff(refresh_hour[order])) + 1
    hour_segments = np.split(order, bounds)

    history_women, history_gendered = stream["history_women"], stream["history_gendered"]
    history_share = np.where(
        history_gendered > 0, history_women / np.maximum(history_gendered, EPSILON), np.nan
    )
    eligible_history = eligible & (history_gendered >= args.min_hist)
    history_x = history_share[eligible_history]
    shown_y = (women_observed / np.maximum(gendered_observed, EPSILON))[eligible_history]

    null_sd, null_low, null_high, null_hist, null_shown = [], [], [], [], []
    low_threshold, high_threshold = args.low_threshold, args.high_threshold
    for _ in range(args.null_reps):
        shuffled_user = refresh_user.copy()
        for segment in hour_segments:
            if len(segment) > 1:
                shuffled_user[segment] = refresh_user[rng.permutation(segment)]
        women_null, gendered_null = user_totals(shuffled_user)
        eligible_null = eligible & (gendered_null >= args.min_user)
        share_null = women_null[eligible_null] / gendered_null[eligible_null]
        null_sd.append(float(np.std(share_null)))
        null_low.append(float((share_null < low_threshold).mean()))
        null_high.append(float((share_null >= high_threshold).mean()))
        null_hist.append(np.histogram(share_null, bin_edges)[0] / max(len(share_null), 1))
        shown_null = np.where(
            gendered_null >= args.min_user,
            women_null / np.maximum(gendered_null, EPSILON),
            np.nan,
        )[eligible_history]
        null_shown.append(shown_null.astype(np.float32))
    has_people = refresh_gendered > 0
    no_woman = ((refresh_women == 0) & has_people).sum() / max(has_people.sum(), 1)
    results["personalization"] = {
        "impressions_with_people": int(has_people.sum()),
        "impressions_with_people_but_no_woman": float(no_woman),
        "eligible_users": int(eligible.sum()),
        "observed_sd": float(np.std(share_observed)),
        "null_sd_mean": float(np.mean(null_sd)),
        "null_sd_ci95": quantiles(null_sd),
        "excess_spread": (
            float(np.std(share_observed) / np.mean(null_sd)) if np.mean(null_sd) > 0 else None
        ),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "observed_share_below_low": float((share_observed < low_threshold).mean()),
        "null_share_below_low": float(np.mean(null_low)),
        "null_share_below_low_ci95": quantiles(null_low),
        "observed_share_above_high": float((share_observed >= high_threshold).mean()),
        "null_share_above_high": float(np.mean(null_high)),
        "observed_median": float(np.median(share_observed)),
        "observed_p10": float(np.quantile(share_observed, 0.1)),
        "observed_p90": float(np.quantile(share_observed, 0.9)),
        "hist_bins": bin_edges.tolist(),
        "hist_observed": hist_observed.tolist(),
        "hist_null": np.mean(null_hist, axis=0).tolist(),
    }
    log(
        f"personalization: observed sd {np.std(share_observed):.4f} vs null {np.mean(null_sd):.4f}"
    )

    # 7. Feedback loop
    from scipy.stats import spearmanr  # Heavy import, kept local

    rho_observed = (
        float(spearmanr(history_x, shown_y).correlation)
        if len(history_x) > MIN_CORRELATION_READERS
        else float("nan")
    )
    rho_null = []
    for shown_null in null_shown:
        finite = np.isfinite(shown_null)
        if finite.sum() > MIN_CORRELATION_READERS:
            rho_null.append(float(spearmanr(history_x[finite], shown_null[finite]).correlation))
    quintile_rows = []
    if len(history_x) >= MIN_QUINTILE_READERS:
        ranks = pd.Series(history_x).rank(method="first")
        quintile = pd.qcut(ranks, N_QUINTILES, labels=False).to_numpy()
        for level in range(N_QUINTILES):
            mask = quintile == level
            null_means = [np.nanmean(shown_null[mask]) for shown_null in null_shown]
            quintile_rows.append(
                {
                    "history_women_share_mean": float(np.mean(history_x[mask])),
                    "shown_women_share_mean": float(np.mean(shown_y[mask])),
                    "null_shown_women_share_mean": float(np.nanmean(null_means)),
                    "users": int(mask.sum()),
                }
            )
    has_correlation = len(history_x) > MIN_CORRELATION_READERS
    results["feedback_loop"] = {
        "users": int(eligible_history.sum()),
        "spearman_observed": rho_observed,
        "spearman_null_mean": float(np.mean(rho_null)) if rho_null else None,
        "spearman_null_ci95": quantiles(rho_null) if rho_null else None,
        "ols_slope": float(np.polyfit(history_x, shown_y, 1)[0]) if has_correlation else None,
        "quintiles": quintile_rows,
        "min_history_people": args.min_hist,
    }

    # 8. Reader communities by behaviour
    history_sections, history_clicks = stream["history_sections"], stream["history_clicks"]
    community = np.full(n_users, "Mixed interests", dtype=object)
    history_total = history_sections.sum(1)
    dominant = history_sections.argmax(1)
    dominant_share = np.where(
        history_total > 0, history_sections.max(1) / np.maximum(history_total, 1), 0
    )
    for index, section in enumerate(SECTIONS):
        is_member = (
            (history_total > 0) & (dominant == index) & (dominant_share >= COMMUNITY_DOMINANCE)
        )
        community[is_member] = f"{section} first"
    community[history_total == 0] = "No click history"

    def women_share_of(women_sums, gendered_sums, mask):
        """Women share for selected readers."""
        total = gendered_sums[mask].sum()
        return float(women_sums[mask].sum() / total) if total else None

    results["communities"] = {}
    for community_name in sorted(set(community)):
        mask = community == community_name
        shown = women_share_of(women_shown, gendered_shown, mask)
        results["communities"][community_name] = {
            "users": int(mask.sum()),
            "women_share_shown": shown,
            "women_share_clicked": women_share_of(women_clicked, gendered_clicked, mask),
            "women_share_history": women_share_of(history_women, history_gendered, mask),
            "ratio_shown_vs_published": (shown / published_share) if shown is not None else None,
        }
    ranks = pd.Series(history_clicks).rank(method="first")
    activity = pd.qcut(ranks, len(ACTIVITY_LEVELS), labels=ACTIVITY_LEVELS).to_numpy()
    results["activity"] = {}
    for level in ACTIVITY_LEVELS:
        mask = activity == level
        results["activity"][level] = {
            "users": int(mask.sum()),
            "median_history_clicks": float(np.median(history_clicks[mask])),
            "women_share_shown": women_share_of(women_shown, gendered_shown, mask),
        }

    # 9. Recommenders
    sample = stream["recommender_sample"]
    has_train = "train" in stream["displays_by_split"]
    if not args.skip_recs and has_train and sample is not None and len(sample):
        results["recommenders"] = recommender_experiment(
            news,
            news_index,
            stream,
            women,
            gendered,
            has_woman,
            category_code,
            published_share,
            rng,
        )
    else:
        log("recommender experiment skipped (needs train and dev)")

    # 10. Robustness
    results["robustness"] = robustness(
        entities,
        people,
        news_index,
        stream,
        per_person,
        pool_weights,
        displays,
        clicks,
        women,
        gendered,
    )

    out_path = os.path.join(args.out, "results.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(round_floats(results), handle, indent=1)
    log(f"done. wrote {out_path}")
    return results


def robustness(
    entities,
    people,
    news_index,
    stream,
    per_person,
    pool_weights,
    displays,
    clicks,
    women,
    gendered,
):
    """Headline under alternative choices."""
    n_articles = len(news_index)

    def three_stages(women_counts, gendered_counts, pool, shown, clicked):
        """Shares and ratios per stage."""
        pool_share = share(pool, women_counts, gendered_counts)
        shown_share = share(shown, women_counts, gendered_counts)
        clicked_share = share(clicked, women_counts, gendered_counts)
        return {
            "published": pool_share,
            "shown": shown_share,
            "clicked": clicked_share,
            "ratio_shown_vs_published": shown_share / pool_share if pool_share else None,
            "ratio_clicked_vs_published": clicked_share / pool_share if pool_share else None,
        }

    checks = {
        "main (titles + abstracts)": three_stages(women, gendered, pool_weights, displays, clicks),
        "published = full news catalog": three_stages(
            women, gendered, np.ones(n_articles), displays, clicks
        ),
    }
    titles, _ = article_features(entities, people, news_index, fields=("title",))
    checks["titles only"] = three_stages(
        titles["women"], titles["gendered"], pool_weights, displays, clicks
    )
    confident, _ = article_features(entities, people, news_index, min_confidence=HIGH_CONFIDENCE)
    checks["entity confidence >= 0.9"] = three_stages(
        confident["women"], confident["gendered"], pool_weights, displays, clicks
    )
    top_ten = per_person.head(10).index.tolist()
    without_top, _ = article_features(entities, people, news_index, exclude=top_ten)
    checks["without the 10 most-shown people"] = three_stages(
        without_top["women"], without_top["gendered"], pool_weights, displays, clicks
    )
    checks["article level (features any woman)"] = three_stages(
        (women > 0).astype(float), (gendered > 0).astype(float), pool_weights, displays, clicks
    )
    for check_name, split in (("train logs only", "train"), ("dev logs only", "dev")):
        if split in stream["displays_by_split"] and stream["displays_by_split"][split].sum() > 0:
            split_displays = stream["displays_by_split"][split]
            split_clicks = stream["clicks_by_split"][split]
            checks[check_name] = three_stages(
                women, gendered, (split_displays > 0).astype(float), split_displays, split_clicks
            )
    return checks


def recommender_experiment(
    news, news_index, stream, women, gendered, has_woman, category_code, published_share, rng
):
    """Accuracy versus representation trade-off."""
    # Heavy imports, kept local
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    log("recommenders: building text features")
    text = (
        news["title"]
        + " "
        + news["abstract"]
        + " cat_"
        + news["category"]
        + " sub_"
        + news["subcategory"]
    ).to_numpy()
    vectorizer = TfidfVectorizer(
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES,
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(text)
    n_components = min(SVD_COMPONENTS, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    item_vectors = normalize(svd.fit_transform(tfidf)).astype(np.float32)

    # Training popularity, category smoothed
    n_articles = len(news_index)
    train_displays = stream["displays_by_split"]["train"]
    train_clicks = stream["clicks_by_split"]["train"]
    overall_ctr = train_clicks.sum() / max(train_displays.sum(), 1)
    n_categories = category_code.max() + 1
    category_ctr = (
        np.bincount(category_code, train_clicks, n_categories) + SMOOTHING_DISPLAYS * overall_ctr
    ) / (np.bincount(category_code, train_displays, n_categories) + SMOOTHING_DISPLAYS)
    popularity = (train_clicks + SMOOTHING_DISPLAYS * category_ctr[category_code]) / (
        train_displays + SMOOTHING_DISPLAYS
    )

    sample = stream["recommender_sample"]
    n_refreshes = len(sample)
    tokens = sample["impressions"].str.split(" ")
    lengths = tokens.str.len().to_numpy()
    items = pd.Series(np.concatenate(tokens.to_numpy()))
    candidate = news_index.get_indexer(items.str[:-2])
    clicked = (items.str[-1] == "1").to_numpy().astype(np.int8)
    refresh = np.repeat(np.arange(n_refreshes), lengths)
    known = candidate >= 0
    candidate, clicked, refresh = candidate[known], clicked[known], refresh[known]
    histories = sample["history"].str.split(" ")
    history_items = pd.Series(np.concatenate(histories.to_numpy()))
    history_row = np.repeat(np.arange(n_refreshes), histories.str.len().to_numpy())
    history_article = news_index.get_indexer(history_items)
    known_history = history_article >= 0
    history_matrix = sparse.csr_matrix(
        (
            np.ones(int(known_history.sum()), np.float32),
            (history_row[known_history], history_article[known_history]),
        ),
        shape=(n_refreshes, n_articles),
    )
    user_vectors = normalize(history_matrix @ item_vectors).astype(np.float32)
    content = np.empty(len(candidate), np.float32)
    for start in range(0, len(candidate), SCORE_BLOCK):
        block = slice(start, start + SCORE_BLOCK)
        content[block] = np.einsum(
            "ij,ij->i", user_vectors[refresh[block]], item_vectors[candidate[block]]
        )

    candidates = pd.DataFrame(
        {
            "refresh": refresh,
            "clicked": clicked,
            "women": women[candidate],
            "gendered": gendered[candidate],
            "has_woman": has_woman[candidate],
            "content": content,
            "popularity": popularity[candidate],
            "random": rng.random(len(candidate)),
        }
    )
    candidates["tie"] = rng.random(len(candidates)) * TIE_SCALE
    by_refresh = candidates.groupby("refresh", sort=False)
    candidates["hybrid"] = by_refresh["content"].rank(pct=True) + by_refresh["popularity"].rank(
        pct=True
    )
    candidates["hybrid_percentile"] = by_refresh["hybrid"].rank(pct=True)
    refresh_ids = candidates["refresh"].to_numpy()

    def evaluate(score):
        """AUC and top-5 representation."""
        scores = score + candidates["tie"].to_numpy()
        ranks = pd.Series(scores).groupby(refresh_ids).rank(method="average").to_numpy()
        positive = candidates["clicked"].to_numpy() == 1
        per_refresh = (
            pd.DataFrame({"refresh": refresh_ids, "rank": ranks * positive, "positive": positive})
            .groupby("refresh")
            .agg(rank_sum=("rank", "sum"), n_positive=("positive", "sum"), n=("positive", "size"))
        )
        n_negative = per_refresh["n"] - per_refresh["n_positive"]
        valid = (per_refresh["n_positive"] > 0) & (n_negative > 0)
        n_positive = per_refresh["n_positive"]
        auc = (
            (per_refresh["rank_sum"] - n_positive * (n_positive + 1) / 2)
            / (n_positive * n_negative)
        )[valid]
        top = pd.Series(-scores).groupby(refresh_ids).rank(method="first").to_numpy() <= 5
        top_women = candidates["women"].to_numpy()[top].sum()
        top_gendered = candidates["gendered"].to_numpy()[top].sum()
        top_lists = (
            pd.DataFrame(
                {
                    "refresh": refresh_ids,
                    "has_woman": candidates["has_woman"].to_numpy() * top,
                    "gendered": (candidates["gendered"].to_numpy() > 0) * top,
                }
            )
            .groupby("refresh")
            .sum()
        )
        with_people = top_lists["gendered"] > 0
        no_woman = ((top_lists["has_woman"] == 0) & with_people).sum() / max(with_people.sum(), 1)
        return {
            "auc": float(auc.mean()),
            "top5_women_share": float(top_women / top_gendered) if top_gendered else None,
            "top5_share_articles_with_woman": float(
                candidates["has_woman"].to_numpy()[top].mean()
            ),
            "top5_with_people_but_no_woman": float(no_woman),
        }

    pool_women = candidates["women"].sum()
    pool_gendered = candidates["gendered"].sum()
    experiment = {
        "impressions": int(n_refreshes),
        "candidates": int(len(candidates)),
        "candidate_pool_women_share": float(pool_women / pool_gendered) if pool_gendered else None,
        "published_women_share": published_share,
        "models": {},
    }
    models = (
        ("Random order", candidates["random"].to_numpy()),
        ("Most popular", candidates["popularity"].to_numpy()),
        ("Personalized (reading history)", candidates["content"].to_numpy()),
        ("Hybrid (history + popularity)", candidates["hybrid"].to_numpy()),
    )
    for name, score in models:
        experiment["models"][name] = evaluate(score)
        model = experiment["models"][name]
        log(f"  {name}: AUC {model['auc']:.4f}, top-5 women {model['top5_women_share']}")
    sweep = []
    base = candidates["hybrid_percentile"].to_numpy()
    for boost in BOOST_WEIGHTS:
        point = evaluate(base + boost * candidates["has_woman"].to_numpy())
        point["lambda"] = boost
        sweep.append(point)
    experiment["tradeoff_hybrid"] = sweep
    reached = [
        point
        for point in sweep
        if point["top5_women_share"] is not None and point["top5_women_share"] >= published_share
    ]
    experiment["parity_point"] = (
        {
            "lambda": reached[0]["lambda"],
            "auc": reached[0]["auc"],
            "auc_cost": sweep[0]["auc"] - reached[0]["auc"],
            "top5_women_share": reached[0]["top5_women_share"],
        }
        if reached
        else None
    )
    return experiment


def main():
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train",
        required=True,
        help="MINDlarge_train.zip or MINDsmall_train.zip (or extracted folder)",
    )
    parser.add_argument(
        "--dev", help="MINDlarge_dev.zip or MINDsmall_dev.zip (or extracted folder)"
    )
    parser.add_argument("--out", default="results", help="output folder (default: results)")
    parser.add_argument("--chunk", type=int, default=100000, help="feed refreshes read per chunk")
    parser.add_argument("--boot", type=int, default=300, help="reader bootstrap replicates")
    parser.add_argument(
        "--null-reps", type=int, default=30, help="same-hour shuffles without personalization"
    )
    parser.add_argument(
        "--min-user",
        type=int,
        default=20,
        help="minimum people with a recorded gender shown to a reader",
    )
    parser.add_argument(
        "--min-hist",
        type=int,
        default=5,
        help="minimum people with a recorded gender in the click history",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.10,
        help="share of women below which a feed counts as narrow",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.40,
        help="share of women at or above which a feed counts as high",
    )
    parser.add_argument(
        "--user-fraction",
        type=float,
        default=1.0,
        help="fraction of readers to keep, for a quick test",
    )
    parser.add_argument(
        "--rec-sample",
        type=int,
        default=100000,
        help="dev feed refreshes used by the recommender test",
    )
    parser.add_argument("--skip-recs", action="store_true", help="skip the recommender experiment")
    parser.add_argument(
        "--offline", action="store_true", help="never call Wikidata, use the cache only"
    )
    parser.add_argument("--wikidata-cache", help="extra Wikidata cache file to preload")
    parser.add_argument(
        "--contact",
        default=os.environ.get("FV_CONTACT", ""),
        help="contact for the Wikidata User-Agent header (recommended)",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
