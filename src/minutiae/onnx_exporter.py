"""
Zero-Dependency ONNX Protobuf Exporter.

Serializes pure NumPy CNN weights and computation graphs into standard ONNX IR v8
(Opset 13) binary `.onnx` ModelProto files using `google.protobuf`, validated directly
against `onnxruntime.InferenceSession`. Requires no `torch` or `onnx` package.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


_ONNX_CLASSES = None


def _get_onnx_proto_classes():
    """Lazily builds and caches minimal ONNX IR v8 protobuf message classes."""
    global _ONNX_CLASSES
    if _ONNX_CLASSES is not None:
        return _ONNX_CLASSES

    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "cfp_onnx_ir_v8.proto"
    fdp.package = "cfp_onnx"
    fdp.syntax = "proto2"

    F = descriptor_pb2.FieldDescriptorProto

    def add_msg(name, fields, enums=None):
        m = fdp.message_type.add()
        m.name = name
        for fname, num, label, ftype, type_name in fields:
            f = m.field.add()
            f.name = fname
            f.number = num
            f.label = label
            f.type = ftype
            if type_name:
                f.type_name = type_name
        if enums:
            for ename, evals in enums:
                e = m.enum_type.add()
                e.name = ename
                for vname, vnum in evals:
                    ev = e.value.add()
                    ev.name = vname
                    ev.number = vnum
        return m

    add_msg(
        "OperatorSetIdProto",
        [
            ("domain", 1, F.LABEL_OPTIONAL, F.TYPE_STRING, None),
            ("version", 2, F.LABEL_OPTIONAL, F.TYPE_INT64, None),
        ],
    )

    ts = add_msg("TensorShapeProto", [])
    dim = ts.nested_type.add()
    dim.name = "Dimension"
    for fname, num, ftype in [("dim_value", 1, F.TYPE_INT64), ("dim_param", 2, F.TYPE_STRING)]:
        f = dim.field.add()
        f.name = fname
        f.number = num
        f.label = F.LABEL_OPTIONAL
        f.type = ftype
    f_dim = ts.field.add()
    f_dim.name = "dim"
    f_dim.number = 1
    f_dim.label = F.LABEL_REPEATED
    f_dim.type = F.TYPE_MESSAGE
    f_dim.type_name = ".cfp_onnx.TensorShapeProto.Dimension"

    tp = add_msg("TypeProto", [])
    tp_t = tp.nested_type.add()
    tp_t.name = "Tensor"
    for fname, num, ftype, tname in [
        ("elem_type", 1, F.TYPE_INT32, None),
        ("shape", 2, F.TYPE_MESSAGE, ".cfp_onnx.TensorShapeProto"),
    ]:
        f = tp_t.field.add()
        f.name = fname
        f.number = num
        f.label = F.LABEL_OPTIONAL
        f.type = ftype
        if tname:
            f.type_name = tname
    f_tt = tp.field.add()
    f_tt.name = "tensor_type"
    f_tt.number = 1
    f_tt.label = F.LABEL_OPTIONAL
    f_tt.type = F.TYPE_MESSAGE
    f_tt.type_name = ".cfp_onnx.TypeProto.Tensor"

    add_msg(
        "ValueInfoProto",
        [
            ("name", 1, F.LABEL_OPTIONAL, F.TYPE_STRING, None),
            ("type", 2, F.LABEL_OPTIONAL, F.TYPE_MESSAGE, ".cfp_onnx.TypeProto"),
        ],
    )
    add_msg(
        "TensorProto",
        [
            ("dims", 1, F.LABEL_REPEATED, F.TYPE_INT64, None),
            ("data_type", 2, F.LABEL_OPTIONAL, F.TYPE_INT32, None),
            ("float_data", 4, F.LABEL_REPEATED, F.TYPE_FLOAT, None),
            ("int64_data", 7, F.LABEL_REPEATED, F.TYPE_INT64, None),
            ("name", 8, F.LABEL_OPTIONAL, F.TYPE_STRING, None),
            ("raw_data", 9, F.LABEL_OPTIONAL, F.TYPE_BYTES, None),
        ],
    )
    add_msg(
        "AttributeProto",
        [
            ("name", 1, F.LABEL_OPTIONAL, F.TYPE_STRING, None),
            ("f", 2, F.LABEL_OPTIONAL, F.TYPE_FLOAT, None),
            ("i", 3, F.LABEL_OPTIONAL, F.TYPE_INT64, None),
            ("s", 4, F.LABEL_OPTIONAL, F.TYPE_BYTES, None),
            ("floats", 7, F.LABEL_REPEATED, F.TYPE_FLOAT, None),
            ("ints", 8, F.LABEL_REPEATED, F.TYPE_INT64, None),
            ("type", 20, F.LABEL_OPTIONAL, F.TYPE_ENUM, ".cfp_onnx.AttributeProto.AttributeType"),
        ],
        enums=[
            (
                "AttributeType",
                [
                    ("UNDEFINED", 0),
                    ("FLOAT", 1),
                    ("INT", 2),
                    ("STRING", 3),
                    ("FLOATS", 6),
                    ("INTS", 7),
                ],
            )
        ],
    )
    add_msg(
        "NodeProto",
        [
            ("input", 1, F.LABEL_REPEATED, F.TYPE_STRING, None),
            ("output", 2, F.LABEL_REPEATED, F.TYPE_STRING, None),
            ("name", 3, F.LABEL_OPTIONAL, F.TYPE_STRING, None),
            ("op_type", 4, F.LABEL_OPTIONAL, F.TYPE_STRING, None),
            ("attribute", 5, F.LABEL_REPEATED, F.TYPE_MESSAGE, ".cfp_onnx.AttributeProto"),
        ],
    )
    add_msg(
        "GraphProto",
        [
            ("node", 1, F.LABEL_REPEATED, F.TYPE_MESSAGE, ".cfp_onnx.NodeProto"),
            ("name", 2, F.LABEL_OPTIONAL, F.TYPE_STRING, None),
            ("initializer", 5, F.LABEL_REPEATED, F.TYPE_MESSAGE, ".cfp_onnx.TensorProto"),
            ("input", 11, F.LABEL_REPEATED, F.TYPE_MESSAGE, ".cfp_onnx.ValueInfoProto"),
            ("output", 12, F.LABEL_REPEATED, F.TYPE_MESSAGE, ".cfp_onnx.ValueInfoProto"),
        ],
    )
    add_msg(
        "ModelProto",
        [
            ("ir_version", 1, F.LABEL_OPTIONAL, F.TYPE_INT64, None),
            ("producer_name", 2, F.LABEL_OPTIONAL, F.TYPE_STRING, None),
            ("opset_import", 8, F.LABEL_REPEATED, F.TYPE_MESSAGE, ".cfp_onnx.OperatorSetIdProto"),
            ("graph", 7, F.LABEL_OPTIONAL, F.TYPE_MESSAGE, ".cfp_onnx.GraphProto"),
        ],
    )

    pool = descriptor_pool.Default()
    pool.AddSerializedFile(fdp.SerializeToString())
    _ONNX_CLASSES = {
        name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"cfp_onnx.{name}"))
        for name in [
            "ModelProto",
            "GraphProto",
            "NodeProto",
            "TensorProto",
            "ValueInfoProto",
            "AttributeProto",
            "OperatorSetIdProto",
        ]
    }
    return _ONNX_CLASSES


class ONNXGraphBuilder:
    """Imperative ONNX Opset-13 graph construction helper."""

    def __init__(self, graph_name: str = "MinutiaeNetV1", opset: int = 13):
        self.cls = _get_onnx_proto_classes()
        self.model = self.cls["ModelProto"]()
        self.model.ir_version = 8
        self.model.producer_name = "CancelableFingerprintHybrid"
        op = self.model.opset_import.add()
        op.domain = ""
        op.version = opset
        self.graph = self.model.graph
        self.graph.name = graph_name
        self._node_idx = 0

    def add_input(self, name: str, shape: Sequence[Optional[int]], elem_type: int = 1):
        vi = self.graph.input.add()
        vi.name = name
        vi.type.tensor_type.elem_type = elem_type
        for i, d in enumerate(shape):
            dim = vi.type.tensor_type.shape.dim.add()
            if d is None or d < 0:
                dim.dim_param = f"dim_{i}"
            else:
                dim.dim_value = int(d)

    def add_output(self, name: str, shape: Sequence[Optional[int]], elem_type: int = 1):
        vi = self.graph.output.add()
        vi.name = name
        vi.type.tensor_type.elem_type = elem_type
        for i, d in enumerate(shape):
            dim = vi.type.tensor_type.shape.dim.add()
            if d is None or d < 0:
                dim.dim_param = f"dim_{i}"
            else:
                dim.dim_value = int(d)

    def add_initializer(self, name: str, array: np.ndarray):
        t = self.graph.initializer.add()
        t.name = name
        if array.dtype == np.int64:
            t.data_type = 7
            arr = np.ascontiguousarray(array, dtype=np.int64)
        else:
            t.data_type = 1
            arr = np.ascontiguousarray(array, dtype=np.float32)
        t.dims.extend(list(arr.shape))
        t.raw_data = arr.tobytes()

    def add_node(
        self,
        op_type: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        name: Optional[str] = None,
        attrs: Optional[Dict[str, object]] = None,
    ):
        node = self.graph.node.add()
        node.op_type = op_type
        node.name = name or f"{op_type}_{self._node_idx}"
        self._node_idx += 1
        node.input.extend(inputs)
        node.output.extend(outputs)
        if attrs:
            for k, v in attrs.items():
                attr = node.attribute.add()
                attr.name = k
                if isinstance(v, (list, tuple)):
                    if all(isinstance(x, (int, np.integer)) for x in v):
                        attr.type = 7  # INTS
                        attr.ints.extend([int(x) for x in v])
                    else:
                        attr.type = 6  # FLOATS
                        attr.floats.extend([float(x) for x in v])
                elif isinstance(v, (int, np.integer)):
                    attr.type = 2  # INT
                    attr.i = int(v)
                elif isinstance(v, (float, np.floating)):
                    attr.type = 1  # FLOAT
                    attr.f = float(v)
                elif isinstance(v, (str, bytes)):
                    attr.type = 3  # STRING
                    attr.s = v.encode("utf-8") if isinstance(v, str) else v

    def serialize(self) -> bytes:
        return self.model.SerializeToString()

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.serialize())
