"""Synthetic MIND data for tests."""

import json
import os
import random
import sys
import zipfile
from datetime import datetime, timedelta

SEED = 7
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "fixture"

# Planted effects to recover
WOMAN_DISPLAY_WEIGHT = 0.6
PREFERRED_SECTION_WEIGHT = 3.0
WOMAN_CLICK_RATE = 0.055
BASE_CLICK_RATE = 0.04

N_PEOPLE = 400
N_ORGS = 80
N_STORIES = 1800
N_TRAIN_USERS = 1500
N_DEV_USERS = 700
TRAIN_START = datetime(2019, 11, 9)
DEV_START = datetime(2019, 11, 15)
TRAIN_DAYS, DEV_DAYS = 6, 1
TRAIN_REFRESHES, DEV_REFRESHES = (3, 12), (1, 4)
MIND_TIME_FORMAT = "%-m/%-d/%Y %-I:%M:%S %p"

CATEGORIES = {
    "news": ["newspolitics", "newsus", "newscrime"],
    "sports": ["football_nfl", "basketball_nba"],
    "finance": ["finance-companies"],
    "entertainment": ["celebrity"],
    "tv": ["tvnews"],
    "music": ["musicnews"],
    "movies": ["movienews"],
    "lifestyle": ["lifestyleroyals"],
    "health": ["wellness"],
    "foodanddrink": ["recipes"],
    "travel": ["travelnews"],
    "autos": ["autosnews"],
    "weather": ["weathertopstories"],
    "video": ["viral"],
}
SECTION_OF_CATEGORY = {
    "news": "News",
    "sports": "Sports",
    "finance": "Money",
    "entertainment": "Ent",
    "tv": "Ent",
    "music": "Ent",
    "movies": "Ent",
    "video": "Ent",
    "lifestyle": "Life",
    "health": "Life",
    "foodanddrink": "Life",
    "travel": "Life",
    "autos": "Life",
    "weather": "Life",
}
OCCUPATIONS = {
    "News": ["politician", "lawyer", "diplomat"],
    "Sports": ["American football player", "basketball player"],
    "Money": ["businessperson", "economist"],
    "Ent": ["actor", "singer", "television presenter"],
    "Life": ["chef", "physician", "writer"],
}
WOMEN_SHARE = {"News": 0.2, "Sports": 0.08, "Money": 0.18, "Ent": 0.5, "Life": 0.45}
COUNTRIES = [
    ("Q30", 0.6),
    ("Q16", 0.05),
    ("Q145", 0.1),
    ("Q148", 0.05),
    ("Q1033", 0.05),
    ("Q96", 0.05),
    ("Q408", 0.05),
    (None, 0.05),
]
COUNTRY_RECORDS = {
    "Q30": {"label": "United States", "continents": ["Q49"]},
    "Q16": {"label": "Canada", "continents": ["Q49"]},
    "Q145": {"label": "United Kingdom", "continents": ["Q46"]},
    "Q148": {"label": "China", "continents": ["Q48"]},
    "Q1033": {"label": "Nigeria", "continents": ["Q15"]},
    "Q96": {"label": "Mexico", "continents": ["Q49"]},
    "Q408": {"label": "Australia", "continents": ["Q3960"]},
}
SECTION_PREFERENCES = (["News", "Sports", "Money", "Ent", "Life"], [0.35, 0.2, 0.1, 0.2, 0.15])
WOMAN_QID, MAN_QID, OTHER_QID = "Q6581072", "Q6581097", "Q48270"
OTHER_GENDER_CUTOFF = 0.985

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

people = {}
for index in range(N_PEOPLE):
    qid = f"Q{9000001 + index}"
    section = random.choice(list(OCCUPATIONS))
    draw = random.random()
    if draw < WOMEN_SHARE[section]:
        gender = [WOMAN_QID]
    else:
        gender = [OTHER_QID] if draw > OTHER_GENDER_CUTOFF else [MAN_QID]
    country = random.choices([code for code, _ in COUNTRIES], [w for _, w in COUNTRIES])[0]
    people[qid] = {
        "g": gender,
        "occ": [random.choice(OCCUPATIONS[section])],
        "c": [country] if country else [],
        "section": section,
        "name": f"Person {index}",
    }
orgs = [f"Q{8000001 + index}" for index in range(N_ORGS)]

news = []
for index in range(N_STORIES):
    category = random.choice(list(CATEGORIES))
    subcategory = random.choice(CATEGORIES[category])
    section = SECTION_OF_CATEGORY[category]
    candidates = [qid for qid, person in people.items() if person["section"] == section]
    candidates = candidates or list(people)
    n_named = random.choices([0, 1, 2, 3], [0.45, 0.35, 0.15, 0.05])[0]
    chosen = random.sample(candidates, n_named)
    title_entities, abstract_entities = [], []
    for qid in chosen:
        entity = {
            "Label": people[qid]["name"],
            "Type": "P",
            "WikidataId": qid,
            "Confidence": random.choice([1.0, 1.0, 1.0, 0.85]),
            "OccurrenceOffsets": [0],
            "SurfaceForms": [people[qid]["name"]],
        }
        (title_entities if random.random() < 0.6 else abstract_entities).append(entity)
    if random.random() < 0.5:
        org = random.choice(orgs)
        title_entities.append(
            {
                "Label": f"Org {org}",
                "Type": "O",
                "WikidataId": org,
                "Confidence": 1.0,
                "OccurrenceOffsets": [3],
                "SurfaceForms": ["Org"],
            }
        )
    has_woman = any(people[qid]["g"] == [WOMAN_QID] for qid in chosen)
    names = " ".join(people[qid]["name"] for qid in chosen)
    news.append(
        {
            "id": f"N{10000 + index}",
            "category": category,
            "subcategory": subcategory,
            "section": section,
            "has_woman": has_woman,
            "title": f"Story {index} about {section.lower()} {names}",
            "abstract": f"Abstract text {index} {category} {subcategory} words here",
            "title_entities": title_entities,
            "abstract_entities": abstract_entities,
        }
    )


def news_tsv(rows):
    """Render news rows as TSV."""
    out = []
    for story in rows:
        out.append(
            "\t".join(
                [
                    story["id"],
                    story["category"],
                    story["subcategory"],
                    story["title"],
                    story["abstract"],
                    f"https://assets.msn.com/labs/mind/{story['id']}.html",
                    json.dumps(story["title_entities"]) if story["title_entities"] else "[]",
                    json.dumps(story["abstract_entities"]) if story["abstract_entities"] else "[]",
                ]
            )
        )
    return "\n".join(out) + "\n"


def make_behaviors(users, start, days, refresh_range, impression_offset):
    """Simulate feed refreshes and clicks."""
    lines = []
    impression_id = impression_offset
    for user, preference in users:
        history_pool = [story for story in news if story["section"] == preference] * 3 + news
        history = random.sample(history_pool, random.randint(3, 40))
        history_ids = list(dict.fromkeys(story["id"] for story in history))
        for _ in range(random.randint(*refresh_range)):
            impression_id += 1
            timestamp = start + timedelta(
                days=random.randrange(days),
                hours=random.randrange(24),
                minutes=random.randrange(60),
            )
            weights = [
                (WOMAN_DISPLAY_WEIGHT if story["has_woman"] else 1.0)
                * (PREFERRED_SECTION_WEIGHT if story["section"] == preference else 1.0)
                for story in news
            ]
            drawn = random.choices(news, weights, k=random.randint(10, 30))
            shown = list({story["id"]: story for story in drawn}.values())
            items = []
            any_click = False
            for story in shown:
                rate = WOMAN_CLICK_RATE if story["has_woman"] else BASE_CLICK_RATE
                clicked = random.random() < rate
                any_click = any_click or clicked
                items.append(f"{story['id']}-{int(clicked)}")
            if not any_click:
                items[0] = items[0][:-1] + "1"
            lines.append(
                "\t".join(
                    [
                        str(impression_id),
                        user,
                        timestamp.strftime(MIND_TIME_FORMAT),
                        " ".join(history_ids),
                        " ".join(items),
                    ]
                )
            )
    return "\n".join(lines) + "\n", impression_id


preference_names, preference_weights = SECTION_PREFERENCES
users = [
    (f"U{index}", random.choices(preference_names, preference_weights)[0])
    for index in range(N_TRAIN_USERS)
]
train_behaviors, last_id = make_behaviors(users, TRAIN_START, TRAIN_DAYS, TRAIN_REFRESHES, 0)
dev_users = random.sample(users, N_DEV_USERS)
dev_behaviors, _ = make_behaviors(dev_users, DEV_START, DEV_DAYS, DEV_REFRESHES, last_id)

for name, behaviors in (
    ("MINDsmall_train.zip", train_behaviors),
    ("MINDsmall_dev.zip", dev_behaviors),
):
    with zipfile.ZipFile(os.path.join(OUT_DIR, name), "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("news.tsv", news_tsv(news))
        archive.writestr("behaviors.tsv", behaviors)

# Cache keys match funnel_vision.py
cache = {
    "people": {
        qid: {"g": person["g"], "occ": person["occ"], "c": person["c"]}
        for qid, person in people.items()
    },
    "countries": COUNTRY_RECORDS,
}
for org in orgs:
    cache["people"][org] = None
with open(os.path.join(OUT_DIR, "wikidata_cache.json"), "w") as handle:
    json.dump(cache, handle)
print("fixture written to", OUT_DIR)
