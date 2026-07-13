import json

data = json.load(open('data/colleges.json', 'r', encoding='utf-8'))

regions = set()
districts = set()
all_branches = set()
categories = set()

for c in data.values():
    regions.add(c.get('region', ''))
    districts.add(c.get('district', ''))
    for bname, bdata in c.get('offerings', {}).items():
        all_branches.add(bname)
        for rnd in ['cutoffsR1', 'cutoffsR2', 'cutoffsR3', 'cutoffsR4']:
            if rnd in bdata:
                categories.update(bdata[rnd].keys())

print(f"Total colleges: {len(data)}")
print(f"Total unique branches: {len(all_branches)}")
print(f"Regions: {sorted(regions)}")
print(f"\nDistricts ({len(districts)}): {sorted(districts)}")
print(f"\nBranches (sample): {sorted(all_branches)[:20]}")
print(f"\nCategory codes ({len(categories)}): {sorted(categories)}")
