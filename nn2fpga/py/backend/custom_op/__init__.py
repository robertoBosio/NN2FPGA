from backend.custom_op.producestream import ProduceStream
from backend.custom_op.consumestream import ConsumeStream
from backend.custom_op.tensorduplicator import TensorDuplicator
from backend.custom_op.bandwidthadjust import BandwidthAdjustIncreaseChannels, BandwidthAdjustDecreaseChannels
from backend.custom_op.bandwidthadjust import BandwidthAdjustIncreaseStreams, BandwidthAdjustDecreaseStreams
from backend.custom_op.paramstream import ParamStream
from backend.custom_op.streaminglinebuffer import StreamingLineBuffer
from backend.custom_op.streamingconv import StreamingConv
from backend.custom_op.streamingglobalaveragepool import StreamingGlobalAveragePool
from backend.custom_op.streamingadd import StreamingAdd
from backend.custom_op.streamingrelu import StreamingRelu

custom_op = {
    "ProduceStream": ProduceStream,
    "ConsumeStream": ConsumeStream,
    "TensorDuplicator": TensorDuplicator,
    "ParamStream": ParamStream,
    "StreamingLineBuffer": StreamingLineBuffer,
    "StreamingConv": StreamingConv,
    "StreamingGlobalAveragePool": StreamingGlobalAveragePool,
    "StreamingAdd": StreamingAdd,
    "StreamingRelu": StreamingRelu,
    "BandwidthAdjustIncreaseChannels": BandwidthAdjustIncreaseChannels,
    "BandwidthAdjustDecreaseChannels": BandwidthAdjustDecreaseChannels,
    "BandwidthAdjustIncreaseStreams": BandwidthAdjustIncreaseStreams,
    "BandwidthAdjustDecreaseStreams": BandwidthAdjustDecreaseStreams,
}