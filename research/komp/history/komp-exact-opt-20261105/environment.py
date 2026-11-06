import json,sys,torch,triton,google.protobuf
from google.protobuf.internal.api_implementation import Type
print(json.dumps(dict(python=sys.version,torch=torch.__version__,triton=triton.__version__,
    cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
    protobuf=google.protobuf.__version__,protobuf_backend=Type()),indent=2))
