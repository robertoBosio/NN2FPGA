from backend.custom_op.producestream import ProduceStream
from backend.custom_op.consumestream import ConsumeStream
from backend.custom_op.tensorduplicator import TensorDuplicator

custom_op = {
    "ProduceStream": ProduceStream,
    "ConsumeStream": ConsumeStream,
    "TensorDuplicator": TensorDuplicator,
}