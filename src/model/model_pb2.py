# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: model.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0bmodel.proto\x12\x0b\x62lokusmodel\"5\n\x13StateRepresentation\x12\x0e\n\x06\x62oards\x18\x01 \x03(\x08\x12\x0e\n\x06player\x18\x03 \x01(\x05\"+\n\nPrediction\x12\x0e\n\x06policy\x18\x01 \x03(\x02\x12\r\n\x05value\x18\x02 \x03(\x02\"Z\n\x04\x44\x61ta\x12\x30\n\x06states\x18\x01 \x03(\x0b\x32 .blokusmodel.StateRepresentation\x12\x10\n\x08policies\x18\x02 \x03(\x02\x12\x0e\n\x06values\x18\x03 \x03(\x02\"\x16\n\x06Status\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x32\x88\x01\n\x0b\x42lokusModel\x12\x46\n\x07Predict\x12 .blokusmodel.StateRepresentation\x1a\x17.blokusmodel.Prediction\"\x00\x12\x31\n\x05Train\x12\x11.blokusmodel.Data\x1a\x13.blokusmodel.Status\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'model_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_STATEREPRESENTATION']._serialized_start=28
  _globals['_STATEREPRESENTATION']._serialized_end=81
  _globals['_PREDICTION']._serialized_start=83
  _globals['_PREDICTION']._serialized_end=126
  _globals['_DATA']._serialized_start=128
  _globals['_DATA']._serialized_end=218
  _globals['_STATUS']._serialized_start=220
  _globals['_STATUS']._serialized_end=242
  _globals['_BLOKUSMODEL']._serialized_start=245
  _globals['_BLOKUSMODEL']._serialized_end=381
# @@protoc_insertion_point(module_scope)
