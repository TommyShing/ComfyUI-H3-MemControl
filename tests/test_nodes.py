from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

COMFY_ROOT = Path(r"E:\Stable Diffusion\ComfyUI-aki-v3.2\ComfyUI")
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

NODE_IMPORT_ERROR = None
try:
    from h3_memcontrol import nodes
except Exception as exc:
    nodes = None
    NODE_IMPORT_ERROR = exc


def h3_prefetch_like(self):
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
    for i, block in enumerate(self.blocks):
        block()
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)


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


if __name__ == "__main__":
    unittest.main()
