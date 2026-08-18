from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path

import torch.nn as nn

COMFY_ROOT = Path(r"E:\Stable Diffusion\ComfyUI-aki-v3.2\ComfyUI")
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

NODE_IMPORT_ERROR = None
try:
    from h3_memcontrol import nodes
    from h3_memcontrol.manager import MemControlManager
except Exception as exc:
    nodes = None
    NODE_IMPORT_ERROR = exc


def h3_prefetch_like(self):
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
    for i, block in enumerate(self.blocks):
        block()
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)


def qwen_prefetch_like(self):
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, {"prefetch_dynamic_vbars": getattr(self, "prefetch_dynamic_vbars", False)})
    for i, layer in enumerate(self.layers):
        layer()
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, None)


class FakeQwenLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 1)])

    forward = qwen_prefetch_like


class FakeQwenRoot(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.model = FakeQwenLlama()


class FakePatcher:
    def __init__(self, model):
        self.model = model
        self.object_patches = {}

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        obj = self.model
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj


@unittest.skipUnless(nodes is not None, f"ComfyUI comfy_api unavailable: {NODE_IMPORT_ERROR}")
class NodeSchemaTests(unittest.TestCase):
    def test_extension_registers_nodes(self):
        extension = asyncio.run(nodes.comfy_entrypoint())
        classes = asyncio.run(extension.get_node_list())
        names = {cls.GET_SCHEMA().node_id for cls in classes}
        self.assertEqual(
            names,
            {
                "H3MemControlSetup",
                "H3MemControlCleanup",
                "H3MemControlVAECache",
                "H3MemControlDebug",
            },
        )

    def test_cleanup_passthrough(self):
        result = nodes.H3MemControlCleanup.execute("hello", stage="after_sampling")
        self.assertEqual(result[0], "hello")

    def test_debug_returns_string(self):
        result = nodes.H3MemControlDebug.execute()
        self.assertIsInstance(result[0], str)

    def test_h3_prefetch_forward_patch(self):
        new_forward = nodes._make_prefetch_free_forward(h3_prefetch_like)
        self.assertIsNotNone(new_forward)
        names = new_forward.__code__.co_names
        self.assertNotIn("enumerate", names)
        self.assertIn("range", names)

    def test_qwen_prefetch_forward_patch(self):
        new_forward = nodes._make_prefetch_free_forward(
            qwen_prefetch_like,
            container="layers",
            item="layer",
            prefetch_source=nodes.QWEN_PREFETCH_SOURCE,
        )
        self.assertIsNotNone(new_forward)
        names = new_forward.__code__.co_names
        self.assertNotIn("enumerate", names)
        self.assertIn("range", names)

    def test_setup_has_no_manage_qwen_switch(self):
        schema = nodes.H3MemControlSetup.GET_SCHEMA()
        self.assertEqual(len(schema.inputs), 4)

    def test_qwen_patch_applied_to_managed_layers_parent(self):
        root = FakeQwenRoot()
        manager = MemControlManager()
        manager.install_on_root(root, "clip")
        clip = types.SimpleNamespace(patcher=FakePatcher(root))

        nodes._patch_qwen_prefetch(manager, clip)

        self.assertIn("transformer.model.forward", clip.patcher.object_patches)
        bound = clip.patcher.object_patches["transformer.model.forward"]
        self.assertNotIn("enumerate", bound.__func__.__code__.co_names)
        self.assertIn("range", bound.__func__.__code__.co_names)


if __name__ == "__main__":
    unittest.main()
