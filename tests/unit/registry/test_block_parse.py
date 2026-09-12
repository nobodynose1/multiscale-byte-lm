import pytest
from pydantic import ValidationError

from mblm.model import mamba_shim
from mblm.model.block import StageBlock, StageBlockRegistry
from mblm.model.config import block_registry
from mblm.model.mamba import MambaBlock
from mblm.model.transformer import TransformerBlock, TransformerEncoderBlock

TRANSFORMER_DATA = {
    "block_type": "transformer",
    "attn_head_dims": 64,
    "attn_num_heads": 16,
    "attn_use_rot_embs": True,
    "use_flash_attn": True,
    "pos_emb_type": "fixed",
}

ENCODER_DATA = {**TRANSFORMER_DATA, "block_type": "transformerEncoder"}

MAMBA_DATA = {
    "block_type": "mamba2",
    "d_state": 128,
    "d_conv": 4,
    "expand": 2,
    "headdim": 64,
    "mamba_backend": "mamba2",
    "pos_emb_type": None,
}


def parse(data, registry: StageBlockRegistry = block_registry) -> StageBlock:
    return registry.try_parse(data, lambda klass, block_data: klass.model_validate(block_data))


class LstmBlock(StageBlock):
    block_type: str = "lstm"

    dropout: float
    my_property: int

    def to_model(self, model_dim: int, num_layers: int):
        raise NotImplementedError


def registry_with(*blocks: type[StageBlock]) -> StageBlockRegistry:
    registry = StageBlockRegistry()
    for block in blocks:
        registry.register()(block)
    return registry


class TestExplicitKeyRouting:
    @pytest.mark.parametrize(
        "data,expected",
        [
            (TRANSFORMER_DATA, TransformerBlock),
            (ENCODER_DATA, TransformerEncoderBlock),
            (MAMBA_DATA, MambaBlock),
            ({**MAMBA_DATA, "block_type": "mamba1"}, MambaBlock),
            ({**MAMBA_DATA, "block_type": "unresolved"}, MambaBlock),
        ],
    )
    def test_a_key_routes_to_the_class_that_declares_it(self, data, expected):
        assert isinstance(parse(data), expected)

    def test_supported_keys_cover_every_registered_class(self):
        # a fresh registry: other tests register their own blocks globally
        registry = registry_with(TransformerBlock, MambaBlock, TransformerEncoderBlock)

        assert registry.supported_keys() == [
            mamba_shim.MAMBA1,
            mamba_shim.MAMBA2,
            "transformer",
            "transformerEncoder",
            mamba_shim.UNRESOLVED,
        ]

    def test_a_routing_key_does_not_declare_the_engine(self):
        parsed = parse({**MAMBA_DATA, "block_type": "mamba1"})

        assert isinstance(parsed, MambaBlock)
        assert parsed.mamba_backend == "mamba2"

    def test_a_custom_block_key_comes_from_its_own_default(self):
        registry = registry_with(LstmBlock)

        parsed = registry.try_parse(
            {"block_type": "lstm", "dropout": 0.1, "my_property": 1},
            lambda klass, data: klass.model_validate(data),
        )

        assert isinstance(parsed, LstmBlock)
        assert registry.supported_keys() == ["lstm"]


class TestRejection:
    def test_an_unknown_key_is_rejected_with_the_supported_keys(self):
        with pytest.raises(ValueError, match="Unknown block_type 'rnn'") as error:
            parse({**TRANSFORMER_DATA, "block_type": "rnn"})

        assert "supported keys" in str(error.value)
        assert mamba_shim.MAMBA2 in str(error.value)

    def test_a_missing_key_is_rejected(self):
        data = {key: value for key, value in TRANSFORMER_DATA.items() if key != "block_type"}

        with pytest.raises(ValueError, match="does not declare a 'block_type'") as error:
            parse(data)

        assert "supported keys" in str(error.value)

    @pytest.mark.parametrize("data", ["transformer", 3, None, ["transformer"]])
    def test_a_non_mapping_block_config_is_rejected(self, data):
        with pytest.raises(ValueError, match="does not declare a 'block_type'"):
            parse(data)

    def test_the_routed_class_that_cannot_parse_is_not_replaced_by_another(self):
        mamba_fields_with_transformer_key = {**MAMBA_DATA, "block_type": "transformer"}

        with pytest.raises(ValidationError, match="attn_head_dims"):
            parse(mamba_fields_with_transformer_key)

    def test_a_reused_key_fails_registration(self):
        class OtherBlock(StageBlock):
            block_type: str = "lstm"

            def to_model(self, model_dim: int, num_layers: int):
                raise NotImplementedError

        registry = registry_with(LstmBlock)

        with pytest.raises(ValueError, match="already used by LstmBlock"):
            registry.register()(OtherBlock)

    def test_a_class_without_a_key_default_fails_registration(self):
        class NoKeyBlock(StageBlock):
            block_type: str

            def to_model(self, model_dim: int, num_layers: int):
                raise NotImplementedError

        registry = StageBlockRegistry()

        with pytest.raises(ValueError, match="needs a string default for `block_type`"):
            registry.register()(NoKeyBlock)


class TestRegistrationOrderIndependence:
    def test_routing_is_stable_across_registration_orders(self):
        forwards = registry_with(TransformerBlock, MambaBlock, TransformerEncoderBlock)
        backwards = registry_with(TransformerEncoderBlock, MambaBlock, TransformerBlock)

        for _ in range(5):
            for registry in (forwards, backwards):
                assert type(parse(TRANSFORMER_DATA, registry)) is TransformerBlock
                assert type(parse(ENCODER_DATA, registry)) is TransformerEncoderBlock
                assert type(parse(MAMBA_DATA, registry)) is MambaBlock

    def test_parsing_needs_no_mamba_admission(self):
        parsed = parse(MAMBA_DATA)

        assert isinstance(parsed, MambaBlock)
        assert parsed.block_type == "mamba2"
