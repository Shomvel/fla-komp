"""Whole-original-batch greedy regression against immutable evaluation fixtures."""

import argparse
import importlib
import json
from pathlib import Path
import re
import sys

import torch

from gpu_validate import MODELS, compare, fingerprint, frozen_operator
from komp_inference.model import RecurrentKOMPLM
from verify_sources import verify_candidate, verify_model, plan


def score(row, text):
    if row['metric'] == 'string_match_all':
        text = re.sub(r'[\x00-\x1f]', '\n', text.strip()).strip()
        return sum(label.lower() in text.lower() for label in row['labels'])/len(row['labels'])
    phone = row['task'].startswith('phonebook')
    return max(int(bool(re.search(re.escape(label.strip() if phone else label),
                                  text.strip() if phone else text, re.I))) for label in row['labels'])


@torch.inference_mode()
def generate(model, adapter, rows, tokenizer, eos, *, candidate=False):
    contexts = [r['effective_input_ids'] for r in rows]
    if len({len(x) for x in contexts}) != 1:
        raise ValueError('fixture batches must preserve equal true lengths')
    b, prompt = len(rows), len(contexts[0])
    limits = [r['generation'].get('max_gen_toks', 128) for r in rows]
    ids = torch.full((b, prompt+max(limits)), eos, device='cuda', dtype=torch.long)
    ids[:, :prompt] = torch.tensor(contexts, device='cuda')
    lengths = torch.full((b,), prompt, device='cuda', dtype=torch.long)
    done, outputs, texts = [False]*b, [[] for _ in rows], ['']*b
    cached_logits, cache = None, None
    if candidate:
        cached_logits, cache = adapter(ids[:, :prompt], logits_to_keep=1)
    for step in range(max(limits)):
        active = [i for i in range(b) if not done[i]]
        current = max(int(lengths[i]) for i in active)
        if candidate:
            logits = cached_logits[:, -1]
        else:
            hidden = model(ids[:, :current].contiguous())
            logits = model.lm_head(hidden[torch.arange(b, device='cuda'), lengths-1]).float()
        predictions = logits.argmax(-1).tolist()
        next_tokens = torch.full((b, 1), eos, device='cuda', dtype=torch.long)
        for i, token in enumerate(predictions):
            if done[i]:
                continue
            outputs[i].append(token)
            ids[i, lengths[i]] = token
            lengths[i] += 1
            next_tokens[i, 0] = token
            text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            stops = rows[i]['generation'].get('until', [])
            stops = [stops] if isinstance(stops, str) else list(stops)
            stops += [tokenizer.decode([eos], skip_special_tokens=False)]
            cuts = [text.index(stop) for stop in stops if stop and stop in text]
            texts[i] = text[:min(cuts)] if cuts else text
            done[i] = bool(cuts) or token == eos or step+1 >= limits[i]
        if all(done):
            break
        if candidate:
            # stopped rows advance with EOS but cannot affect any active sequence; B remains fixed
            cached_logits, cache = adapter(next_tokens, cache=cache, logits_to_keep=1)
    return [dict(id=row['id'], generated_ids=ids, prediction=text, score=score(row, text))
            for row, ids, text in zip(rows, outputs, texts)]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--fixtures', required=True)
    parser.add_argument('--base', default='/gpfs/users/linlizhi')
    parser.add_argument('--limit-batches', type=int)
    parser.add_argument('--acceptance')
    parser.add_argument('--evidence-dir', default=str(Path(__file__).parent/'evidence'))
    args = parser.parse_args()
    root = Path(__file__).parent
    candidate_source = verify_candidate()
    verify_model(args.model, args.base)
    acceptance = json.loads(Path(args.acceptance or root/'evidence'/f'gpu-{args.model}.json').read_text())
    if acceptance['status'] != 'passed' or 'checkpoint' not in acceptance:
        raise ValueError('run operator and real-checkpoint acceptance first')
    if acceptance['candidate_source'] != candidate_source:
        raise ValueError('acceptance belongs to a different implementation manifest')
    fixture_path = Path(args.fixtures)
    manifest = json.loads((fixture_path.parent/'manifest.json').read_text())
    if fingerprint(fixture_path) != manifest['fixtures_sha256']:
        raise ValueError('fixture fingerprint mismatch')
    if manifest['fixtures_sha256'] != plan()['fixture_sha256']:
        raise ValueError('fixtures differ from the frozen validation plan')
    fixture = json.loads(fixture_path.read_text())
    model_name = 'exact_partial' if args.model == 'exact' else args.model
    batches = [b for b in fixture['batches'] if b['model'] == model_name and b['golden_status'] == 'observed_complete_batch']
    if args.limit_batches:
        batches = batches[:args.limit_batches]
    if not batches:
        raise ValueError('no observed golden batches for this model; input-only is not a greedy gate')
    ck = Path(acceptance['checkpoint']['checkpoint'])
    sha = fingerprint(ck)
    if fixture['models'][model_name]['checkpoint_sha'] != sha:
        raise ValueError('fixture checkpoint mismatch')
    folder, kind, axis, learned = MODELS[args.model]
    sys.path.insert(0, str(Path(args.base)/folder))
    module = importlib.import_module('models')
    model = module.LM(kind)
    saved = torch.load(ck, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model = model.cuda().eval()
    adapter = RecurrentKOMPLM(model, decay_axis=axis, backend='triton',
                             prefill_operator=frozen_operator(kind, axis, learned))
    from tokenizers import Tokenizer
    token_path = Path(fixture['request_manifest']['tokenizer'])/'tokenizer.json'
    if fingerprint(token_path) != fixture['request_manifest']['tokenizer_sha']:
        raise ValueError('tokenizer fingerprint mismatch')
    tokenizer = Tokenizer.from_file(str(token_path))
    eos = fixture['request_manifest']['eos_token_id']
    torch.set_num_threads(2)
    report = dict(status='running', model=args.model, checkpoint_sha=sha, candidate_source=candidate_source, batches=[])
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    target = evidence/f'fixtures-{args.model}.json'
    if target.exists():
        raise FileExistsError(target)
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for batch in batches:
                rows = [fixture['requests'][name] for name in batch['request_ids']]
                reference = generate(model, adapter, rows, tokenizer, eos)
                golden = [{key: row[key] for key in ('id', 'generated_ids', 'prediction', 'score')}
                          for row in batch['golden']['results']]
                if reference != golden:
                    raise AssertionError(dict(stage='baseline_vs_golden', batch=batch['batch_key'],
                                              actual=reference, expected=golden))
                actual = generate(model, adapter, rows, tokenizer, eos, candidate=True)
                if actual != golden:
                    raise AssertionError(dict(stage='recurrent_vs_golden', batch=batch['batch_key'],
                                              actual=actual, expected=golden))
                report['batches'].append(dict(key=batch['batch_key'], size=len(rows), status='passed'))
                print(json.dumps(report['batches'][-1]), flush=True)
        report['status'] = 'passed'
    except Exception as error:
        report['status'], report['error'] = 'failed', repr(error)
        raise
    finally:
        target.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
