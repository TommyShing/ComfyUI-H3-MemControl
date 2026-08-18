from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from h3_memcontrol.manager import MemControlManager, registry


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(3)])
        self.layers = nn.ModuleList([Block() for _ in range(2)])


class ManagerTests(unittest.TestCase):
    def test_find_and_restore_containers(self):
        model = DummyModel()
        manager = MemControlManager()
        manager.install_on_root(model, "model")
        self.assertTrue(hasattr(model, "blocks"))
        self.assertTrue(hasattr(model, "layers"))
        self.assertEqual(len(manager.installations), 2)
        manager.cleanup()
        self.assertEqual(len(manager.installations), 0)
        self.assertFalse(
            any(
                type(getattr(model, name, None)).__name__ == "MemControlModuleList"
                for name in ("blocks", "layers")
            )
        )

    def test_vae_cache_uses_path(self):
        class Patcher:
            cached_patcher_init = (object, ("models/vae/test.safetensors", None, None))

        class DummyVAE:
            patcher = Patcher()

        vae = DummyVAE()
        cached = registry.cache_vae(vae)
        self.assertIs(cached, vae)
        self.assertIs(registry.get_vae(vae), vae)


if __name__ == "__main__":
    unittest.main()
