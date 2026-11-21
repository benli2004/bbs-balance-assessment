# -*- coding: utf-8 -*-
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Check action distribution in splits
for split in ['train', 'val', 'test']:
    with open(f'data/splits/{split}.json', encoding='utf-8') as f:
        data = json.load(f)
    
    actions = set()
    for item in data:
        actions.add(item['action_id'])
    
    print(f"{split}: actions = {sorted(actions)}")
