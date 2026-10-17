# Funnel Vision

An audit of who gets seen in a news feed. It follows every public figure named in the news
through three stages, the story pool, what the platform showed, and what readers clicked, using
the Microsoft News Dataset (MIND) and Wikidata.

## Download

MIND-large from https://msnews.github.io (accept the license):

- `MINDlarge_train.zip`
- `MINDlarge_dev.zip`

Mirror: https://huggingface.co/datasets/Recommenders/MIND

Put both zips in this folder and keep them zipped.

## Run

Python 3.9 or newer.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python funnel_vision.py --train MINDlarge_train.zip --dev MINDlarge_dev.zip --out results --contact "your email"
python make_report.py --results results/results.json --team team.json --out FunnelVision_Phase1.pdf
```

The first command takes about 30 to 60 minutes and writes `results/results.json`.
The second builds the 5-page PDF.
