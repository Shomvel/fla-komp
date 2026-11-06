from pathlib import Path
import json,hashlib
r=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101')
assert not (r/'fine-failure.json').exists()
for i in range(2):
 w=r/'fine-workers'/f'pilot-{i}'
 assert json.loads((w/'completion.json').read_text())['status']=='completed'
 assert (w/'exit-code.txt').read_text().strip()=='0'
bp=r/'results/results/coh_fifo_off-seed731/summary.json';base=json.loads(bp.read_text())
rows=[]
for tag in ('995','999'):
 p=r/'results-fine/results'/f'coh_fifo_{tag}-seed731/summary.json';s=json.loads(p.read_text())
 ce=s['confirm']['mean_content_ce']-base['confirm']['mean_content_ce']
 brevo=100*(s['generation']['brevo']['valid']-base['generation']['brevo']['valid'])
 rows.append(dict(tag=tag,ce=s['confirm']['mean_content_ce'],brevo=100*s['generation']['brevo']['valid'],depo=100*s['generation']['depo']['valid'],delta_ce=ce,delta_brevo_pp=brevo,expand=ce<.15 and brevo>-10,summary_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
receipt=dict(status='locked',rule='expand if delta CE < .15 and delta Brevo > -10pp vs same-seed off',base_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),rows=rows,expand_tags=[x['tag'] for x in rows if x['expand']])
(r/'fine-screening.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
