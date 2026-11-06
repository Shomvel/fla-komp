import json,platform
from pathlib import Path
import torch,triton,google.protobuf
from google.protobuf.internal import api_implementation
Path('../evidence/environment.json').write_text(json.dumps(dict(python=platform.python_version(),torch=str(torch.__version__),triton=triton.__version__,protobuf=google.protobuf.__version__,protobuf_implementation=api_implementation.Type(),gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda),indent=2))
