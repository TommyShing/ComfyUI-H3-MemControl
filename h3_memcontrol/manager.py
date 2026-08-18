"""MemControl manager and monitoring helpers.

This is a test/monitoring implementation. It does not modify ComfyUI official
files, and it intentionally avoids aggressive model unloading until the block
access/memory logs show the real runtime behavior.
"""

from __future__ import annotations

import gc
import logging
import time
import uuid
import weakref
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger("H3MemControl")


def format_bytes(value: float | int) -> str:
    value = float(value)
    if value >= 1024 ** 3:
        return f"{value / 1024 ** 3:.2f}GB"
    if value >= 1024 ** 2:
        return f"{value / 1024 ** 2:.1f}MB"
    if value >= 1024:
        return f"{value / 1024:.0f}KB"
    return f"{value:.0f}B"


def log_memory(label: str) -> None:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        ram_used = vm.used
        ram_available = vm.available
    except Exception:
        ram_used = ram_available = -1

    vram_used = vram_free = -1
    try:
        if torch.cuda.is_available():
            vram_free, vram_total = torch.cuda.mem_get_info()
            vram_used = vram_total - vram_free
    except Exception:
        pass

    logger.info(
        "[MemControl] memory %s | VRAM used=%s free=%s | RAM used=%s available=%s",
        label,
        format_bytes(vram_used),
        format_bytes(vram_free),
        format_bytes(ram_used),
        format_bytes(ram_available),
    )


def module_bytes(module: nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def _has_meta_params(module: nn.Module) -> bool:
    for p in module.parameters(recurse=True):
        if p.device.type == "meta":
            return True
    return False


def _clear_comfy_module_state(module: nn.Module) -> None:
    try:
        import comfy.model_prefetch

        comfy_modules = [m for m in module.modules() if hasattr(m, "_prefetch")]
        if comfy_modules:
            comfy.model_prefetch.cleanup_prefetched_modules(module, comfy_modules)
    except Exception:
        pass

    for child in module.modules():
        for attr in ("_prefetch", "_v_weight", "_v_bias", "_v_signature", "_v_block_faulted"):
            if hasattr(child, attr):
                try:
                    delattr(child, attr)
                except Exception:
                    pass


def find_block_containers(
    root: nn.Module,
    depth: int = 0,
    seen: set[int] | None = None,
    path: str = "",
):
    if seen is None:
        seen = set()
    if depth > 12:
        return
    for name, child in root.named_children():
        child_path = f"{path}.{name}" if path else name
        if isinstance(child, (nn.ModuleList, list)) and len(child) > 0 and hasattr(child[0], "forward"):
            if id(child) not in seen:
                seen.add(id(child))
                yield name, child, root, child_path
        elif isinstance(child, nn.Module):
            yield from find_block_containers(child, depth + 1, seen, child_path)


def _vae_path(vae: Any) -> str | None:
    try:
        init = getattr(vae.patcher, "cached_patcher_init", None)
        if init and len(init) >= 2 and init[1] and isinstance(init[1][0], str):
            return init[1][0]
    except Exception:
        pass
    return None


class MemControlModuleList(nn.ModuleList):
    """ModuleList replacement that logs block access without moving weights yet."""

    def __init__(self, modules, manager, container_name: str, container_path: str, root_name: str, root_device=None):
        super().__init__(modules)
        self.memcontrol_manager_ref = weakref.ref(manager)
        self.container_name = container_name
        self.container_path = container_path
        self.root_name = root_name
        self.root_device = root_device
        self._block_sizes = [module_bytes(module) for module in modules]
        self._access_counts: dict[int, int] = {}
        self._last_resident: dict[int, bool] = {}

    def _note_access(self, idx: int) -> None:
        manager = self.memcontrol_manager_ref()
        if manager is None or idx < 0 or idx >= len(self):
            return
        count = self._access_counts.get(idx, 0) + 1
        self._access_counts[idx] = count
        block = self._modules[str(idx)]
        size = self._block_sizes[idx] if idx < len(self._block_sizes) else 0
        resident = False
        try:
            resident = manager.ensure_block(self.container_path, idx, block, size, self.root_device)
        except Exception as exc:
            logger.warning("[MemControl] ensure_block failed %s[%d]: %s", self.container_path, idx, exc)

        prev = self._last_resident.get(idx)
        should_log = count <= 3 or count % 50 == 0 or prev != resident
        if should_log:
            logger.info(
                "[MemControl] access %s[%d] count=%d size=%s resident=%s",
                self.container_path,
                idx,
                count,
                format_bytes(size),
                resident,
            )
        self._last_resident[idx] = resident

    def __getitem__(self, idx):
        if isinstance(idx, int):
            self._note_access(idx)
        return super().__getitem__(idx)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class MemControlManager:
    def __init__(self, headroom_bytes: int = 768 * 1024 ** 2):
        self.manager_id = uuid.uuid4().hex[:12]
        self.headroom_bytes = headroom_bytes
        self.installations: list[tuple[weakref.ReferenceType, str, nn.ModuleList, MemControlModuleList, str]] = []
        self.patcher_filters: list[tuple[Any, Any, str]] = []
        self.resident: dict[tuple[str, int], tuple[MemControlModuleList, int, str, int]] = {}
        self.last_used: dict[tuple[str, int], float] = {}
        self.managed_module_ids: set[int] = set()
        self.container_limits: dict[str, int] = {}
        self.path_to_swl: dict[str, MemControlModuleList] = {}
        self.model_accessed = False
        self.clip_accessed = False
        self.created_at = time.time()

    def install_on_root(
        self,
        root: nn.Module,
        root_name: str,
        seen: set[int] | None = None,
        root_device=None,
    ) -> None:
        for name, orig, parent, child_path in find_block_containers(root, seen=seen):
            if isinstance(orig, MemControlModuleList):
                continue
            swl = MemControlModuleList(orig, self, name, child_path, root_name, root_device)
            setattr(parent, name, swl)
            self.path_to_swl[child_path] = swl
            self.installations.append((weakref.ref(parent), name, orig, swl, root_name))
            for block in orig:
                for module in block.modules():
                    self.managed_module_ids.add(id(module))
            logger.info(
                "[MemControl] install root=%s container=%s blocks=%d device=%s",
                root_name,
                child_path,
                len(swl),
                root_device,
            )

    def patch_load_list(self, patcher: Any, root_name: str) -> None:
        if patcher is None or not hasattr(patcher, "_load_list"):
            return
        original = getattr(patcher, "_load_list")
        if getattr(original, "_memcontrol_patched", False):
            return

        def filtered_load_list(*args, **kwargs):
            entries = original(*args, **kwargs)
            result = []
            skipped = 0
            for entry in entries:
                module = entry[-2] if entry else None
                if module is not None and id(module) in self.managed_module_ids:
                    skipped += 1
                    continue
                result.append(entry)
            if skipped:
                logger.info(
                    "[MemControl] _load_list root=%s total=%d managed_skipped=%d kept=%d",
                    root_name,
                    len(entries),
                    skipped,
                    len(result),
                )
            return result

        filtered_load_list._memcontrol_patched = True
        filtered_load_list._memcontrol_original = original
        patcher._load_list = filtered_load_list
        self.patcher_filters.append((patcher, original, root_name))
        logger.info("[MemControl] patched _load_list root=%s", root_name)

    def set_container_limit(self, container_path: str, limit_bytes: int) -> None:
        self.container_limits[container_path] = int(limit_bytes)
        logger.info(
            "[MemControl] container limit %s=%s",
            container_path,
            format_bytes(limit_bytes),
        )

    def _resident_bytes(self) -> int:
        return sum(entry[3] for entry in self.resident.values())

    def _container_resident_bytes(self, container_path: str) -> int:
        return sum(entry[3] for key, entry in self.resident.items() if key[0] == container_path)

    def _evict(self, key: tuple[str, int]) -> None:
        resident_entry = self.resident.pop(key, None)
        self.last_used.pop(key, None)
        if resident_entry is None:
            return
        swl, idx, root_name, _ = resident_entry
        try:
            block = swl._modules[str(idx)]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _clear_comfy_module_state(block)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                "[MemControl] evict root=%s %s[%d]",
                root_name,
                swl.container_path,
                idx,
            )
            log_memory(f"evict_after_{swl.container_path}_{idx}")
        except Exception as exc:
            logger.warning("[MemControl] evict failed %s[%d]: %s", swl.container_path, idx, exc)

    def ensure_block(
        self,
        container_path: str,
        idx: int,
        block: nn.Module,
        size: int,
        compute_device,
    ) -> bool:
        key = (container_path, idx)
        root_name = self._root_for_path(container_path)
        if root_name == "model":
            self.model_accessed = True
        elif root_name == "clip":
            self.clip_accessed = True
        if key in self.resident:
            self.last_used[key] = time.time()
            return True
        if compute_device is None or getattr(compute_device, "type", "cpu") == "cpu":
            return False
        if _has_meta_params(block):
            return False
        if not torch.cuda.is_available():
            return False

        limit = self.container_limits.get(container_path)
        if limit is not None:
            while self._container_resident_bytes(container_path) + size > limit:
                victim_key = next(
                    (
                        key
                        for key in sorted(self.last_used, key=self.last_used.get)
                        if key[0] == container_path and key in self.resident
                    ),
                    None,
                )
                if victim_key is None:
                    break
                self._evict(victim_key)

        try:
            free_bytes, _ = torch.cuda.mem_get_info(compute_device)
            budget = max(0, free_bytes - self.headroom_bytes)
            needed = size - (budget - self._resident_bytes())
            if needed > 0:
                for victim_key in sorted(self.last_used, key=self.last_used.get):
                    if self.resident.get(victim_key) is None:
                        continue
                    self._evict(victim_key)
                    if self._resident_bytes() + size <= budget:
                        break
        except Exception as exc:
            logger.warning("[MemControl] budget check failed: %s", exc)

        self.resident[key] = (self._get_swl_by_path(container_path), idx, root_name, size)
        self.last_used[key] = time.time()
        logger.info(
            "[MemControl] activate %s[%d] size=%s active=%d",
            container_path,
            idx,
            format_bytes(size),
            len(self.resident),
        )
        return True

    def _get_swl_by_path(self, container_path: str) -> MemControlModuleList | None:
        return self.path_to_swl.get(container_path)

    def _root_for_path(self, container_path: str) -> str:
        swl = self.path_to_swl.get(container_path)
        return swl.root_name if swl is not None else "unknown"

    def container_stats(self) -> list[dict[str, Any]]:
        stats = []
        for parent_ref, name, orig, swl, root_name in self.installations:
            total = len(swl)
            sizes = list(swl._block_sizes)
            stats.append(
                {
                    "root": root_name,
                    "container": swl.container_path,
                    "blocks": total,
                    "size_bytes": sum(sizes),
                    "max_block_bytes": max(sizes, default=0),
                    "access_counts": dict(swl._access_counts),
                }
            )
        return stats

    def _release_root_resident(self, root_name: str) -> None:
        for key in [k for k, (_, _, r, _) in self.resident.items() if r == root_name]:
            self._evict(key)

    def cleanup_root(self, root_name: str) -> None:
        remaining = []
        for parent_ref, name, orig, swl, install_root in self.installations:
            if install_root == root_name:
                self._release_root_resident(root_name)
                parent = parent_ref()
                if parent is not None and getattr(parent, name, None) is swl:
                    setattr(parent, name, orig)
                logger.info(
                    "[MemControl] cleanup root=%s restore container=%s",
                    root_name,
                    swl.container_path,
                )
            else:
                remaining.append((parent_ref, name, orig, swl, install_root))
        self.installations = remaining

        kept_filters = []
        for patcher, original, filter_root in self.patcher_filters:
            if filter_root == root_name and getattr(patcher, "_load_list", None) is not None:
                patcher._load_list = original
                logger.info("[MemControl] restore _load_list root=%s", root_name)
            else:
                kept_filters.append((patcher, original, filter_root))
        self.patcher_filters = kept_filters
        log_memory(f"cleanup_{root_name}")

    def cleanup_auto(self) -> None:
        log_memory("cleanup_auto")
        if self.model_accessed:
            logger.info("[MemControl] cleanup_auto model_accessed=True -> cleanup model")
            self.cleanup_root("model")
        if self.clip_accessed:
            logger.info("[MemControl] cleanup_auto clip_accessed=True -> cleanup clip")
            self.cleanup_root("clip")
        if not self.model_accessed and not self.clip_accessed:
            logger.info("[MemControl] cleanup_auto no block accessed yet -> skip")

    def cleanup(self) -> None:
        for key in list(self.resident):
            self._evict(key)
        self.resident.clear()
        self.last_used.clear()
        for parent_ref, name, orig, swl, root_name in self.installations:
            parent = parent_ref()
            if parent is not None and getattr(parent, name, None) is swl:
                setattr(parent, name, orig)
            logger.info(
                "[MemControl] cleanup restore root=%s container=%s",
                root_name,
                swl.container_path,
            )
        self.installations.clear()
        for patcher, original, root_name in self.patcher_filters:
            try:
                patcher._load_list = original
            except Exception:
                pass
        self.patcher_filters.clear()
        log_memory("cleanup")


class _Registry:
    def __init__(self):
        self.managers: dict[str, MemControlManager] = {}
        self.vae_cache: dict[str, Any] = {}
        self.vae_order: list[str] = []

    def create_manager(self) -> MemControlManager:
        manager = MemControlManager()
        self.managers[manager.manager_id] = manager
        logger.info("[MemControl] create manager id=%s", manager.manager_id)
        return manager

    def get_manager(self, manager_id: str) -> MemControlManager | None:
        return self.managers.get(manager_id)

    def cleanup_root(self, root_name: str) -> None:
        for manager in list(self.managers.values()):
            manager.cleanup_root(root_name)

    def cleanup_auto(self) -> None:
        for manager in list(self.managers.values()):
            manager.cleanup_auto()

    def cleanup_all(self) -> None:
        for manager in list(self.managers.values()):
            manager.cleanup()
        self.managers.clear()
        log_memory("cleanup_all")

    def vae_cache_key(self, vae: Any) -> str:
        path = _vae_path(vae)
        if path:
            return f"path:{path}"
        return f"id:{id(vae)}"

    def cache_vae(self, vae: Any) -> Any:
        key = self.vae_cache_key(vae)
        existing = self.vae_cache.get(key)
        if existing is not None:
            logger.info("[MemControl] VAE cache hit key=%s", key)
            return existing
        self.vae_cache[key] = vae
        self.vae_order.append(key)
        logger.info("[MemControl] VAE cache store key=%s entries=%d", key, len(self.vae_cache))
        return vae

    def get_vae(self, vae: Any) -> Any | None:
        return self.vae_cache.get(self.vae_cache_key(vae))

    def status_text(self) -> str:
        vram_used = vram_free = ram_used = ram_available = -1
        try:
            import psutil  # type: ignore

            vm = psutil.virtual_memory()
            ram_used = vm.used
            ram_available = vm.available
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                vram_free, vram_total = torch.cuda.mem_get_info()
                vram_used = vram_total - vram_free
        except Exception:
            pass

        lines = [
            f"managers={len(self.managers)}",
            f"vae_cache_entries={len(self.vae_cache)}",
            f"vae_cache_order={self.vae_order[-8:]}",
            f"VRAM used={format_bytes(vram_used)} free={format_bytes(vram_free)}",
            f"RAM used={format_bytes(ram_used)} available={format_bytes(ram_available)}",
        ]
        for manager in self.managers.values():
            resident_entries = len(manager.resident)
            lines.append(
                f"manager={manager.manager_id} resident_blocks={resident_entries} "
                f"resident_bytes={format_bytes(manager._resident_bytes())} headroom={format_bytes(manager.headroom_bytes)}"
            )
            for stat in manager.container_stats():
                lines.append(
                    f"container={stat['container']} blocks={stat['blocks']} "
                    f"size={format_bytes(stat['size_bytes'])} max_block={format_bytes(stat['max_block_bytes'])} "
                    f"accesses={sum(stat['access_counts'].values())}"
                )
        return "\n".join(lines)


registry = _Registry()
