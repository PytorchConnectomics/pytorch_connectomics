"""Structured schemas retain annotations understood by the OmegaConf dependency floor."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import pkgutil

import pytest
from omegaconf import OmegaConf

from connectomics.config import Config, schema
from connectomics.config.schema.root import MergeContext


def _schema_dataclasses() -> list[type]:
    classes: set[type] = set()
    for module_info in pkgutil.walk_packages(schema.__path__, schema.__name__ + "."):
        module = importlib.import_module(module_info.name)
        classes.update(
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and dataclasses.is_dataclass(value)
            and value.__module__ == module.__name__
        )
    return sorted(classes, key=lambda cls: (cls.__module__, cls.__name__))


@pytest.mark.parametrize("cls", _schema_dataclasses(), ids=lambda cls: cls.__name__)
def test_schema_dataclass_loads_through_omegaconf(cls):
    if cls is MergeContext:
        # This internal set-valued bookkeeping object is deliberately excluded
        # from structured Config serialization; it requires object support alone.
        config = OmegaConf.structured(cls, flags={"allow_objects": True})
        assert "_merge_context" not in OmegaConf.structured(Config)
    else:
        config = OmegaConf.structured(cls)
    assert OmegaConf.get_type(config) is cls


@pytest.mark.parametrize("cls", _schema_dataclasses(), ids=lambda cls: cls.__name__)
def test_schema_dataclass_fields_use_no_builtin_generics(cls):
    field_names = {field.name for field in dataclasses.fields(cls)}
    builtin_generics = {"list", "dict", "tuple", "set", "frozenset", "type"}
    definition = ast.parse(inspect.getsource(cls)).body[0]
    assert isinstance(definition, ast.ClassDef)
    for statement in definition.body:
        if not (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id in field_names
        ):
            continue
        annotation = statement.annotation
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            annotation = ast.parse(annotation.value, mode="eval").body
        for node in ast.walk(annotation):
            if not isinstance(node, ast.Subscript):
                continue
            generic = node.value
            if isinstance(generic, ast.Attribute):
                assert not (
                    isinstance(generic.value, ast.Name)
                    and generic.value.id == "builtins"
                    and generic.attr in builtin_generics
                ), f"{cls.__name__}.{statement.target.id} must use typing generics"
            else:
                assert not (isinstance(generic, ast.Name) and generic.id in builtin_generics), (
                    f"{cls.__name__}.{statement.target.id} must use typing generics"
                )
