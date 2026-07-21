"""Deterministic six-camera MCAP bytes for integration tests."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Final

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, timestamp_pb2
from google.protobuf.message import Message
from mcap.writer import CompressionType, IndexType, Writer

SIX_CAMERA_TOPICS: Final = tuple(f"/fixture/camera{index}/compressed" for index in range(6))
SIX_CAMERA_MCAP_BYTES: Final = 7_966
SIX_CAMERA_MCAP_SHA256: Final = "d84c3bf77f001c163463a3fd681b161c9611266225e1cd1aa7dc536283433a14"
COMPRESSED_IMAGE_DESCRIPTOR_BYTES: Final = 592
COMPRESSED_IMAGE_DESCRIPTOR_SHA256: Final = (
    "7832ae85852b2fbbaa6908d5d5c15c35de0f9a1ba244daecd535f3c144491ac0"
)
H264_PACKET_SHA256: Final = (
    "db20250708f31853144e2c3ae3259761237d739ee1b6ad878bb7a42b59af1b9b",
    "24316eb28d8a9f86897e03b1142fd58b2fe27f5e41de285d5779da3f57bfb4bd",
)

_H264_PACKETS: Final = tuple(
    base64.b64decode(value, validate=True)
    for value in (
        "AAAAAWdCwArcR6EAAAMAAQAAAwA8jxIngAAAAAFozg/IAAABBgX//z/cRem95tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTY1IC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAyNSAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTAgcmVmPTEgZGVibG9jaz0wOjA6MCBhbmFseXNlPTA6MCBtZT1kaWEgc3VibWU9MCBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0wIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MCA4eDhkY3Q9MCBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0wIHRocmVhZHM9MSBsb29rYWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFjZWQ9MCBibHVyYXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTAgd2VpZ2h0cD0wIGtleWludD0xIGtleWludF9taW49MSBzY2VuZWN1dD0wIGludHJhX3JlZnJlc2g9MCByYz1jcmYgbWJ0cmVlPTAgY3JmPTIzLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MACAAAABZYiEOiYoAAi0ycnJ111111114A==",
        "AAAAAWdCwArcR6EAAAMAAQAAAwA8jxIngAAAAAFozg/IAAABZYiCA2icnJydddddddde",
    )
)

assert tuple(map(len, _H264_PACKETS)) == (637, 51)
assert tuple(hashlib.sha256(packet).hexdigest() for packet in _H264_PACKETS) == (H264_PACKET_SHA256)


def build_six_camera_mcap() -> bytes:
    """Build one byte-stable six-channel protobuf/H.264 MCAP."""

    compressed_image, schema_bytes = _compressed_image_contract()
    output = BytesIO()
    writer = Writer(
        output,
        chunk_size=1_048_576,
        compression=CompressionType.NONE,
        index_types=IndexType.ALL,
        repeat_channels=True,
        repeat_schemas=True,
        use_chunking=True,
        use_statistics=True,
        use_summary_offsets=True,
        enable_crcs=True,
        enable_data_crcs=False,
    )
    writer.start(
        profile="robata-development-fixture",
        library="robata-test-fixture-v1",
    )
    schema_id = writer.register_schema(
        name="foxglove.CompressedImage",
        encoding="protobuf",
        data=schema_bytes,
    )
    channels = tuple(
        writer.register_channel(
            topic=topic,
            message_encoding="protobuf",
            schema_id=schema_id,
        )
        for topic in SIX_CAMERA_TOPICS
    )

    for camera_index, channel_id in enumerate(channels):
        for frame_index, packet in enumerate(_H264_PACKETS):
            timestamp_ns = (
                1_781_051_907_271_610_000 + frame_index * 100_000_000 + camera_index * 1_000
            )
            message = compressed_image(
                data=packet,
                format="h264",
                frame_id=f"fixture_cam_{camera_index + 1}",
            )
            message.timestamp.seconds = timestamp_ns // 1_000_000_000
            message.timestamp.nanos = timestamp_ns % 1_000_000_000
            message.header.timestamp = timestamp_ns
            writer.add_message(
                channel_id=channel_id,
                log_time=timestamp_ns,
                publish_time=timestamp_ns,
                sequence=frame_index,
                data=message.SerializeToString(),
            )

    writer.finish()
    payload = output.getvalue()
    assert len(payload) == SIX_CAMERA_MCAP_BYTES
    assert hashlib.sha256(payload).hexdigest() == SIX_CAMERA_MCAP_SHA256
    return payload


def write_six_camera_mcap(destination: Path) -> Path:
    """Write the deterministic fixture to an integration-test temporary path."""

    destination = Path(destination)
    destination.write_bytes(build_six_camera_mcap())
    return destination


def _compressed_image_contract() -> tuple[type[Message], bytes]:
    header_proto = descriptor_pb2.FileDescriptorProto()
    header_proto.name = "header.proto"
    header_proto.package = "arnold.common.proto"
    header_proto.syntax = "proto3"
    header_message = header_proto.message_type.add()
    header_message.name = "Header"
    header_timestamp = header_message.field.add()
    header_timestamp.name = "timestamp"
    header_timestamp.number = 3
    header_timestamp.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    header_timestamp.type = descriptor_pb2.FieldDescriptorProto.TYPE_UINT64

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "foxglove/CompressedImage.proto"
    file_proto.package = "foxglove"
    file_proto.syntax = "proto3"
    file_proto.dependency.append("google/protobuf/timestamp.proto")
    file_proto.dependency.append("header.proto")

    message_proto = file_proto.message_type.add()
    message_proto.name = "CompressedImage"
    for name, number, field_type, type_name in (
        (
            "timestamp",
            1,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            ".google.protobuf.Timestamp",
        ),
        ("data", 2, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES, ""),
        ("format", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING, ""),
        ("frame_id", 4, descriptor_pb2.FieldDescriptorProto.TYPE_STRING, ""),
        (
            "header",
            8,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            ".arnold.common.proto.Header",
        ),
    ):
        field = message_proto.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = field_type
        if type_name:
            field.type_name = type_name

    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)
    pool.Add(header_proto)
    pool.Add(file_proto)
    message_type = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("foxglove.CompressedImage")
    )

    timestamp_file = descriptor_pb2.FileDescriptorProto()
    timestamp_pb2.DESCRIPTOR.CopyToProto(timestamp_file)
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    descriptor_set.file.extend((timestamp_file, header_proto, file_proto))
    schema_bytes = descriptor_set.SerializeToString()
    assert len(schema_bytes) == COMPRESSED_IMAGE_DESCRIPTOR_BYTES
    assert hashlib.sha256(schema_bytes).hexdigest() == COMPRESSED_IMAGE_DESCRIPTOR_SHA256
    return message_type, schema_bytes


__all__ = [
    "H264_PACKET_SHA256",
    "SIX_CAMERA_MCAP_BYTES",
    "SIX_CAMERA_MCAP_SHA256",
    "SIX_CAMERA_TOPICS",
    "build_six_camera_mcap",
    "write_six_camera_mcap",
]
