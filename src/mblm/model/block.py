__copyright__ = """MIT License

Copyright (c) 2024 - IBM Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Literal, TypeVar

import torch
from pydantic import BaseModel, Field


class StageBlock(ABC, BaseModel):
    """
    Configuration for a single model block at an MBLM stage.
    """

    block_type: str = Field(
        description="A name for the block for easy identification, which can be any string name"
    )

    parse_keys: ClassVar[tuple[str, ...]] = ()
    """Extra keys that route a YAML block to this class, on top of the default of `block_type`"""

    pos_emb_type: Literal["fixed", "rope"] | None = Field(
        default=None,
        description="The type of positional embedding to add to tokens of the stage block",
    )

    @abstractmethod
    def to_model(
        self,
        model_dim: int,
        num_layers: int,
    ) -> torch.nn.Module: ...

    """
    An abstract method that creates a `torch.nn.Module` from the stage block
    configuration.
    """


TStageBlock = TypeVar("TStageBlock", bound=StageBlock)


class StageBlockRegistry(set[type[StageBlock]]):
    """
    Stage block registry that allows (custom) stage blocks to be registered for
    usage with MBLM. Blocks only need to be registered when they require parsing
    from a YAML config file. When a MBLM configuration is read from a YAML
    config file, its `block_type` key selects the stage block implementation
    directly; a YAML block without a key, or with an unknown one, is rejected
    rather than matched by trial parsing.
    """

    def __init__(self) -> None:
        super().__init__()
        self._parsers: dict[str, type[StageBlock]] = {}

    def register(self) -> Callable[[type[TStageBlock]], type[TStageBlock]]:
        """
        Decorator to register a stage block class so it can be validated from a YAML
        configuration file.

        Usage:
            @block_registry.register()
            class MyBlock(StageBlock):
                pass

        The keys of a registered block are its declared `block_type` default plus
        any extra `parse_keys` it declares. Reusing a key across classes fails
        registration.
        """

        def decorator(stage_block_klass: type[TStageBlock]) -> type[TStageBlock]:
            for key in self._parse_keys_of(stage_block_klass):
                registered = self._parsers.get(key)
                if registered is not None and registered is not stage_block_klass:
                    raise ValueError(
                        f"Cannot register {stage_block_klass.__name__} for '{key}': "
                        f"already used by {registered.__name__}"
                    )
                self._parsers[key] = stage_block_klass
            self.add(stage_block_klass)
            return stage_block_klass

        return decorator

    @staticmethod
    def _parse_keys_of(stage_block_klass: type[StageBlock]) -> tuple[str, ...]:
        default = stage_block_klass.model_fields["block_type"].default
        if not isinstance(default, str) or not default:
            raise ValueError(
                f"{stage_block_klass.__name__} needs a string default for `block_type` "
                "to be reachable from a YAML config"
            )
        return (default, *stage_block_klass.parse_keys)

    def supported_keys(self) -> list[str]:
        """The keys a YAML block may declare, in stable order."""
        return sorted(self._parsers)

    def try_parse(
        self,
        data: Any,
        parse_func: Callable[[type[StageBlock], Any], StageBlock],
    ):
        """
        Parse configuration data with the stage block class its `block_type` selects.
        """
        key = data.get("block_type") if isinstance(data, dict) else None
        if not isinstance(key, str):
            raise ValueError(
                f"Stage block config {data!r} does not declare a 'block_type'; "
                f"supported keys: {self.supported_keys()}"
            )

        stage_block_klass = self._parsers.get(key)
        if stage_block_klass is None:
            raise ValueError(f"Unknown block_type '{key}'; supported keys: {self.supported_keys()}")

        return parse_func(stage_block_klass, data)
