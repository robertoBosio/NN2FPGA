from backend.custom_op.producestream import ProduceStream
from backend.custom_op.consumestream import ConsumeStream
from backend.custom_op.tensorduplicator import TensorDuplicator
from backend.custom_op.bandwidthadjust import (
    BandwidthAdjustIncreaseChannels, BandwidthAdjustDecreaseChannels,
    BandwidthAdjustIncreaseStreams, BandwidthAdjustDecreaseStreams,
)
from backend.custom_op.nn2fpgapartition import nn2fpgaPartition
from backend.custom_op.paramstream import ParamStream
from backend.custom_op.streaminglinebuffer import StreamingLineBuffer
from backend.custom_op.streamingconv import StreamingConv
from backend.custom_op.streamingglobalaveragepool import StreamingGlobalAveragePool
from backend.custom_op.streamingadd import StreamingAdd
from backend.custom_op.streamingrelu import StreamingRelu

custom_op = {
    "BandwidthAdjustDecreaseChannels": BandwidthAdjustDecreaseChannels,
    "BandwidthAdjustDecreaseStreams": BandwidthAdjustDecreaseStreams,
    "BandwidthAdjustIncreaseChannels": BandwidthAdjustIncreaseChannels,
    "BandwidthAdjustIncreaseStreams": BandwidthAdjustIncreaseStreams,
    "ConsumeStream": ConsumeStream,
    "ParamStream": ParamStream,
    "ProduceStream": ProduceStream,
    "nn2fpgaPartition": nn2fpgaPartition,
    "StreamingAdd": StreamingAdd,
    "StreamingConv": StreamingConv,
    "StreamingGlobalAveragePool": StreamingGlobalAveragePool,
    "StreamingLineBuffer": StreamingLineBuffer,
    "StreamingRelu": StreamingRelu,
    "TensorDuplicator": TensorDuplicator,
}