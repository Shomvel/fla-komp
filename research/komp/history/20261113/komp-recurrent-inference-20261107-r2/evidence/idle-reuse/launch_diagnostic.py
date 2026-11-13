import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/gpfs/users/linlizhi/komp-recurrent-inference-20261107-r2')
EVIDENCE = ROOT / 'evidence/idle-reuse'
GPU = 'GPU-a23fed83-576f-fcd6-f472-b74160734f00'
POD = 'job-ed375f508254-20261107143937-runner-0'
SOURCE = '226bef160d973a34464604137e51f88ab62597044b4a3bbac862c513a052ce37'
COMMAND = [sys.executable, str(ROOT / 'diagnose_model.py'), '--model', 'block', '--lengths', '31', '--decode', '4', '--output-dir', str(ROOT / 'attempts/r2-diagnostic-block31')]

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)

if __name__ == '__main__':
    if sys.argv[1:] == ['--run']:
        result = subprocess.run(COMMAND, cwd=ROOT)
        save(EVIDENCE / 'r2-diagnostic-exit.json', dict(exit_code=result.returncode, finished_utc=now()))
        sys.exit(result.returncode)
    assert not sys.argv[1:]
    assert socket.gethostname() == POD, socket.gethostname()
    assert hashlib.sha256((ROOT / 'source-manifest.json').read_bytes()).hexdigest() == SOURCE
    subprocess.run([sys.executable, str(ROOT / 'verify_sources.py'), '--candidate-only'], check=True)
    complete = Path('/gpfs/users/linlizhi/komp-flame-downstream-eval-20261107/sharded/evidence-v2/kda-s1-complete.json')
    receipt = json.loads(complete.read_text())
    assert receipt['status'] == 'completed' and receipt['model'] == 'kda' and receipt['shard'] == 1
    gpu = subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    uuid, util, memory = [item.strip() for item in gpu.split(',')]
    assert uuid == GPU and int(util) == 0 and int(memory) == 0, gpu
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader'], text=True)
    assert GPU not in apps, apps
    ps = subprocess.check_output(['ps', '-eo', 'pid,ppid,stat,args'], text=True)
    assert 'evaluate_sharded.py --shard 1 --index 0' not in ps, 'KDA worker still present'
    assert not (ROOT / 'attempts/r2-diagnostic-block31').exists(), 'Attempt exists; inspect instead of retry'
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / 'claim-r2-diagnostic').mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='2',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',
               FLA_CI_ENV='0', FLA_DISABLE_BACKEND_DISPATCH='1', FLA_DISABLE_TENSOR_CACHE='1',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               TRITON_CACHE_DIR='/tmp/komp-recurrent-r1-idle-kda-s1-triton',
               TORCHINDUCTOR_CACHE_DIR='/tmp/komp-recurrent-r1-idle-kda-s1-inductor',
               PYTHONPATH='/gpfs/users/linlizhi/fla-komp-value-integration-20261106-r1/deps:/gpfs/users/linlizhi/komp-gsa-gdn-20261031-env/triton371')
    record = dict(pod=POD, job_identity=POD.removesuffix('-runner-0'), gpu_uuid=GPU,
                  source_manifest_sha256=SOURCE, started_utc=now(), command=COMMAND,
                  idle_gpu_observation=gpu, compute_apps_before=apps, process_snapshot_before=ps,
                  kda_complete_sha256=hashlib.sha256(complete.read_bytes()).hexdigest())
    save(EVIDENCE / 'claim-r2-diagnostic/owner.json', record)
    with (EVIDENCE / 'r2-diagnostic-launch.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__)), '--run'],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(EVIDENCE / 'r2-diagnostic-launch.json', dict(record, pid=child.pid))
    print(json.dumps(dict(status='launched', pid=child.pid, pod=POD, gpu_uuid=GPU)), flush=True)
    print('RECURRENT_LAUNCH_DONE', flush=True)
