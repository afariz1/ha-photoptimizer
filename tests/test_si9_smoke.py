from __future__ import annotations

import importlib.util
import ast
from pathlib import Path
import unittest
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


const = _load_module(
    "photoptimizer_const_test", "custom_components/photoptimizer/const.py"
)
models = _load_module(
    "photoptimizer_models_test", "custom_components/photoptimizer/models.py"
)


def _load_config_flow_validation_function():
    config_flow_path = (
        REPO_ROOT / "custom_components" / "photoptimizer" / "config_flow.py"
    )
    source = config_flow_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(config_flow_path))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_inverter_step_input"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    compiled = compile(module, filename=str(config_flow_path), mode="exec")
    namespace = {
        "CONF_INVERTER_COMMAND_ONLY": const.CONF_INVERTER_COMMAND_ONLY,
        "CONF_INVERTER_MODE_ENTITY": const.CONF_INVERTER_MODE_ENTITY,
        "CONF_INVERTER_DISCHARGE_POWER_ENTITY": const.CONF_INVERTER_DISCHARGE_POWER_ENTITY,
        "CONF_INVERTER_CHARGE_POWER_ENTITY": const.CONF_INVERTER_CHARGE_POWER_ENTITY,
        "Any": object,
    }
    exec(compiled, namespace)
    return namespace["validate_inverter_step_input"]


validate_inverter_step_input = _load_config_flow_validation_function()


class PhotoptimizerSmokeTests(unittest.TestCase):
    def test_core_modules_import(self) -> None:
        self.assertEqual(const.DOMAIN, "photoptimizer")
        self.assertTrue(hasattr(models, "ExecutionPlan"))
        self.assertTrue(callable(validate_inverter_step_input))

    def test_inverter_validation_requires_charge_entity_when_control_enabled(
        self,
    ) -> None:
        errors = validate_inverter_step_input(
            {
                const.CONF_INVERTER_COMMAND_ONLY: False,
                const.CONF_INVERTER_MODE_ENTITY: "select.inverter_mode",
                const.CONF_INVERTER_DISCHARGE_POWER_ENTITY: "number.inverter_discharge",
            }
        )

        self.assertEqual(
            errors,
            {const.CONF_INVERTER_CHARGE_POWER_ENTITY: "required"},
        )

    def test_quality_scale_reflects_current_test_coverage(self) -> None:
        component_quality_scale = (
            REPO_ROOT / "custom_components" / "photoptimizer" / "quality_scale.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("config-flow-test-coverage: todo", component_quality_scale)
        self.assertIn("test-coverage: todo", component_quality_scale)


if __name__ == "__main__":
    unittest.main()
