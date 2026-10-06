"""Summarize fixed P_vs_D predictions for the independent converter cohort."""
from pathlib import Path
import argparse
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[2]

def summarize(cohort, predictions, config):
    if config['task'] != 'P_vs_D':
        raise ValueError('Early identification requires the fixed primary P_vs_D model.')
    ids = [r['患者ID'] for r in cohort]
    pred_ids = [r['patient_id'] for r in predictions]
    if not ids or any(not v for v in ids + pred_ids) or len(set(ids)) != len(ids) or len(set(pred_ids)) != len(pred_ids) or set(ids) != set(pred_ids):
        raise ValueError('Cohort and predictions must contain the same unique, nonempty patient IDs.')
    lookup = {r['patient_id']: r for r in predictions}
    cut = float(config['threshold'])
    rows = []
    for r in cohort:
        p = lookup[r['患者ID']]
        score, threshold, days = float(p['probability']), float(p['threshold']), float(r['首次阴性至首次阳性天数'])
        if not (math.isfinite(score) and 0 <= score <= 1 and math.isfinite(threshold) and abs(threshold-cut) <= 1e-12):
            raise ValueError('Invalid probability or threshold inconsistent with the fixed P_vs_D model.')
        if r['source'] not in ['Sanjiu', 'Nanfang-sheet'] or not 0 < days <= 180:
            raise ValueError('Invalid converter hospital or interval (expected 1–180 days).')
        rows.append(dict(patient_id=r['患者ID'], hospital=r['source'], days=days, p=score, threshold=cut, detected=score >= cut, model=config['family'], split='Converters', y=1))
    rates = []
    for hospital in ['All', 'Sanjiu', 'Nanfang-sheet']:
        for label, lower, upper in [('All', 0, 180), ('≤30 days', 0, 30), ('31–90 days', 30, 90), ('91–180 days', 90, 180)]:
            subset = [r for r in rows if (hospital == 'All' or r['hospital'] == hospital) and lower < r['days'] <= upper]
            n = len(subset); k = sum(r['detected'] for r in subset)
            rate = k/n if n else None
            low = high = None
            if n:
                z = 1.95996398454; den = 1+z*z/n
                center = (rate+z*z/(2*n))/den
                half = z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
                low, high = center-half, center+half
            rates.append(dict(model=config['family'], hospital=hospital, interval=label, n=n, detected=k, rate=rate, lo=low, hi=high))
    return rows, rates

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', required=True)
    p.add_argument('--predictions', required=True, help='Output of predict.py --task P_vs_D on the same cohort')
    p.add_argument('--config', type=Path, default=ROOT/'configs/P_vs_D.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    def read(path):
        with open(path, encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    rows, rates = summarize(read(a.cohort), read(a.predictions), json.loads(a.config.read_text(encoding='utf-8')))
    a.output.mkdir(parents=True, exist_ok=True)
    for filename, data in [('early_identification_predictions.csv', rows), ('early_rates.csv', rates)]:
        with (a.output/filename).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    print(f"Fixed-model detection: {sum(r['detected'] for r in rows)}/{len(rows)}")

if __name__ == '__main__':
    main()
