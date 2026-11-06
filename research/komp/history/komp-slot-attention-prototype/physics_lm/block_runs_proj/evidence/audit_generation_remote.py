"""Independent generation validation from input graphs/dictionaries; run via stdin."""
from pathlib import Path
import hashlib,json
root=Path('/gpfs/users/linlizhi/komp-proj-swa-20261101')
data=Path('/gpfs/users/linlizhi/softmax-10m-20261023/data/confirm.jsonl')
manifest=json.loads((root/'snapshot/physics_lm/data-manifest.json').read_text())
assert hashlib.sha256(data.read_bytes()).hexdigest()==manifest['splits']['confirm']['sha256']
rows=[json.loads(s) for s in data.read_text().splitlines()]
selected={t:[r for r in rows if r['task']==t][:128] for t in ('depo','brevo')}

def expected_depo(ids):
    start=next(i for i,t in enumerate(ids) if 9000<t<9500)
    words=[];word=[]
    for token in ids[1:start]:
        word.append(token)
        if token>50:words.append(tuple(word));word=[]
    assert not word and len(words)%2==0
    successor=dict(zip(words[::2],words[1::2]));hops=ids[start]-9000
    marker=ids.index(9500);query=tuple(ids[start+1:marker])
    for _ in range(hops):query=successor[query]
    return list(query)


def brevo_valid(ids,guess):
    end=ids.index(9997);query=ids[end+1];edges=list(zip(ids[1:end:2],ids[2:end:2]))
    needed={query}
    while True:
        parents=needed|{a for a,b in edges if b in needed}
        if parents==needed:break
        needed=parents
    if not guess or guess[-1]!=9998:return False
    answer=guess[:-1]
    if len(answer)!=len(needed) or set(answer)!=needed:return False
    positions={node:i for i,node in enumerate(answer)}
    return all(positions[a]<positions[b] for a,b in edges if b in needed)

runs=[]
for summary_path in sorted((root/'results').rglob('summary.json')):
    summary=json.loads(summary_path.read_text())
    records=json.loads((summary_path.parent/'generations.json').read_text())
    for task in ('depo','brevo'):
        subset=[r for r in records if r['task']==task];assert len(subset)==128
        valid_count=0
        for source,record in zip(selected[task],subset):
            ids=source['input_ids'];guess=record['guess']
            if task=='depo':
                gold=expected_depo(ids);assert record['gold']==gold
                valid=guess==gold
            else:
                assert record['gold']==ids[ids.index(9997)+3:]
                valid=brevo_valid(ids,guess)
            assert record['valid']==valid,(summary_path,task)
            valid_count+=valid
        assert summary['generation'][task]['valid']==valid_count/128
    runs.append(str(summary_path.relative_to(root/'results')))
receipt=dict(status='passed',completed_runs=len(runs),checked_generations=256*len(runs),
             method='Depo dictionary hop execution; Brevo ancestors recomputed from input DAG, no gold-node shortcut',runs=runs)
(root/'independent-generation-audit.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='runs'}))
