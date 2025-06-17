from backend.custom_op.producestream import ProduceStream
from backend.custom_op.consumestream import ConsumeStream
from backend.custom_op.tensorduplicator import TensorDuplicator
from backend.custom_op.bandwidthadjust import BandwidthAdjust
from backend.custom_op.paramstream import ParamStream

custom_op = {
    "ProduceStream": ProduceStream,
    "ConsumeStream": ConsumeStream,
    "TensorDuplicator": TensorDuplicator,
    "BandwidthAdjust": BandwidthAdjust,
    "ParamStream": ParamStream,
}