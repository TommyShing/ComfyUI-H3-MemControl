"""ComfyUI nodes for H3 MemControl test version."""

from __future__ import annotations

import inspect
import logging
import re
import textwrap
import types
from typing import Any

from comfy_api.latest import ComfyExtension, io

from .manager import format_bytes, log_memory, registry

logger = logging.getLogger("H3MemControl")


def _wrap_vae_lifecycle(vae: Any) -> Any:
    if vae is None or getattr(vae, "_memcontrol_lifecycle_wrapped", False):
        return vae

    def managed_call(method_name: str):
        original = getattr(vae, method_name)

        def wrapper(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            finally:
                try:
                    import comfy.model_management as model_management

                    patcher = getattr(vae, "patcher", None)
                    if patcher is not None:
                        model_management.unload_model_and_clones(patcher)
                        logger.info("[MemControl] VAE %s released from VRAM", method_name)
                except Exception as exc:
                    logger.warning("[MemControl] VAE %s release failed: %s", method_name, exc)

        return wrapper

    vae.decode = managed_call("decode")
    if hasattr(vae, "encode"):
        vae.encode = managed_call("encode")
    vae._memcontrol_lifecycle_wrapped = True
    return vae


def _cache_vae(value: Any | None) -> Any | None:
    if value is None:
        return None
    return _wrap_vae_lifecycle(registry.cache_vae(value))


def _make_prefetch_free_forward(
    original,
    container="blocks",
    item="block",
    prefetch_source=None,
):
    try:
        source = textwrap.dedent(inspect.getsource(original))
    except Exception as exc:
        logger.warning("[MemControl] prefetch patch source read failed: %s", exc)
        return None

    if prefetch_source is None:
        prefetch_source = (
            "prefetch_queue = comfy.model_prefetch.make_prefetch_queue("
            f"list(self.{container}), device, transformer_options)"
        )
    if prefetch_source not in source:
        logger.warning(
            "[MemControl] prefetch patch skipped: expected prefetch line not found for %s",
            container,
        )
        return None

    source = source.replace(prefetch_source, "prefetch_queue = None")
    loop_pattern = re.compile(
        rf"^(\s*)for i, {item} in enumerate\(self\.{container}\):\s*$",
        re.MULTILINE,
    )
    loop_match = loop_pattern.search(source)
    if loop_match is None:
        logger.warning(
            "[MemControl] prefetch patch skipped: expected %s loop not found",
            container,
        )
        return None

    indent = loop_match.group(1)
    loop_replacement = (
        f"{indent}for i in range(len(self.{container})):\n"
        f"{indent}    {item} = self.{container}[i]\n"
    )
    source = source[:loop_match.start()] + loop_replacement + source[loop_match.end():]

    namespace = dict(original.__globals__)
    try:
        exec(
            compile(source, inspect.getsourcefile(original) or "<H3MemControl>", "exec"),
            namespace,
        )
    except Exception as exc:
        logger.warning("[MemControl] prefetch patch compile failed: %s", exc)
        return None
    return namespace.get(original.__name__)


QWEN_PREFETCH_SOURCE = (
    "prefetch_queue = comfy.model_prefetch.make_prefetch_queue("
    'list(self.layers), x.device, {"prefetch_dynamic_vbars": '
    'getattr(self, "prefetch_dynamic_vbars", False)})'
)


def _patch_h3_prefetch(model) -> None:
    if model is None:
        return
    patcher = model
    existing = getattr(patcher, "_memcontrol_prefetch_forward", None)
    if existing is not None:
        try:
            if patcher.get_model_object("diffusion_model._forward") == existing:
                return
        except Exception:
            pass

    root = getattr(patcher, "model", None)
    diffusion = getattr(root, "diffusion_model", None)
    original = getattr(type(diffusion), "_forward", None) if diffusion is not None else None
    if original is None:
        return

    new_forward = _make_prefetch_free_forward(original)
    if new_forward is None:
        logger.warning("[MemControl] H3 prefetch patch skipped")
        return

    bound = types.MethodType(new_forward, diffusion)
    patcher._memcontrol_prefetch_forward = bound
    patcher.add_object_patch("diffusion_model._forward", bound)
    logger.info("[MemControl] patched H3 _forward: prefetch queue disabled, block loop uses MemControl indexing")


def _patch_qwen_prefetch(manager, clip) -> None:
    if clip is None:
        return
    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        return

    existing = getattr(patcher, "_memcontrol_qwen_prefetch_forward", None)
    if existing is not None:
        try:
            if patcher.get_model_object(existing["path"]) == existing["bound"]:
                return
        except Exception:
            pass

    for parent_ref, name, orig, swl, install_root in manager.installations:
        if install_root != "clip" or name != "layers":
            continue
        parent_path = swl.container_path.rsplit(".", 1)[0]
        if not parent_path:
            continue
        parent = parent_ref()
        original = getattr(type(parent), "forward", None) if parent is not None else None
        if original is None:
            continue

        new_forward = _make_prefetch_free_forward(
            original,
            container="layers",
            item="layer",
            prefetch_source=QWEN_PREFETCH_SOURCE,
        )
        if new_forward is None:
            continue

        bound = types.MethodType(new_forward, parent)
        patch_path = f"{parent_path}.forward"
        patcher.add_object_patch(patch_path, bound)
        patcher._memcontrol_qwen_prefetch_forward = {
            "path": patch_path,
            "bound": bound,
        }
        logger.info("[MemControl] patched qwen forward: prefetch queue disabled, layer loop uses MemControl indexing")
        return
    logger.warning("[MemControl] qwen prefetch patch skipped: managed layers container not found")


def _install_roots(manager, model, clip):
    seen: set[int] = set()
    if model is not None:
        try:
            model_device = getattr(model, "load_device", None)
            manager.install_on_root(model.model, "model", seen=seen, root_device=model_device)
            manager.patch_load_list(model, "model")
            _patch_h3_prefetch(model)
        except Exception as exc:
            logger.warning("[MemControl] model container scan failed: %s", exc)
    if clip is not None:
        patcher = getattr(clip, "patcher", None)
        clip_device = getattr(patcher, "load_device", None)
        roots = []
        root = getattr(clip, "cond_stage_model", None)
        if root is not None:
            roots.append(root)
        root = getattr(patcher, "model", None)
        if root is not None:
            roots.append(root)
        for root in roots:
            try:
                manager.install_on_root(root, "clip", seen=seen, root_device=clip_device)
            except Exception as exc:
                logger.warning("[MemControl] clip container scan failed: %s", exc)
        try:
            manager.patch_load_list(patcher, "clip")
            _patch_qwen_prefetch(manager, clip)
            logger.info("[MemControl] qwen managed by MemControl")
        except Exception as exc:
            logger.warning("[MemControl] clip patcher filter failed: %s", exc)


class H3MemControlSetup(io.ComfyNode):
    """Register H3/Qwen block containers and VAE cache, then pass through all objects."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MemControlSetup",
            display_name="H3 MemControl Setup",
            category="h3/memcontrol",
            description=(
                "Test setup for H3 MemControl. Scans H3/Qwen block containers, "
                "logs memory state and block access, and registers VAE instances."
            ),
            search_aliases=["h3 memcontrol setup", "h3 mem control setup"],
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", optional=True),
                io.Vae.Input("vae", optional=True),
                io.Vae.Input("audio_vae", optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.Vae.Output(display_name="vae"),
                io.Vae.Output(display_name="audio_vae"),
            ],
        )

    @classmethod
    def execute(cls, model, clip=None, vae=None, audio_vae=None) -> io.NodeOutput:
        manager = registry.create_manager()
        log_memory("setup_start")
        _install_roots(manager, model, clip)
        vae = _cache_vae(vae)
        audio_vae = _cache_vae(audio_vae)
        logger.info(
            "[MemControl] setup complete manager_id=%s vae=%s audio_vae=%s",
            manager.manager_id,
            vae is not None,
            audio_vae is not None,
        )
        for stat in manager.container_stats():
            logger.info(
                "[MemControl] container root=%s path=%s blocks=%d size=%s max_block=%s",
                stat["root"],
                stat["container"],
                stat["blocks"],
                format_bytes(stat["size_bytes"]),
                format_bytes(stat["max_block_bytes"]),
            )
            if stat["root"] == "clip" and stat["container"].endswith(".layers"):
                manager.set_container_limit(stat["container"], stat["max_block_bytes"])
            if stat["root"] == "model" and stat["container"].endswith(".blocks"):
                manager.set_container_limit(stat["container"], stat["max_block_bytes"])
        log_memory("setup_done")
        return io.NodeOutput(model, clip, vae, audio_vae)


class H3MemControlCleanup(io.ComfyNode):
    """Release managed block state while passing data through."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MemControlCleanup",
            display_name="H3 MemControl Cleanup",
            category="h3/memcontrol",
            description=(
                "Restores MemControl-wrapped block containers and releases managed block state. "
                "Auto cleanup releases qwen after text encoding and H3 after sampling when those "
                "managed roots were used. It does not clear user IO such as prompt, seed, images, or video previews."
            ),
            search_aliases=["h3 memcontrol cleanup", "h3 mem control cleanup"],
            inputs=[
                io.AnyType.Input("passthrough"),
                io.Combo.Input(
                    "stage",
                    options=["auto", "after_te", "after_sampling", "full"],
                    default="auto",
                    tooltip="after_te cleans qwen containers; after_sampling cleans H3 containers.",
                ),
            ],
            outputs=[io.AnyType.Output(display_name="passthrough")],
        )

    @classmethod
    def execute(cls, passthrough, stage="auto") -> io.NodeOutput:
        log_memory(f"cleanup_start_{stage}")
        if stage == "auto":
            registry.cleanup_auto()
        elif stage == "after_sampling":
            registry.cleanup_root("model")
        elif stage == "after_te":
            registry.cleanup_root("clip")
        elif stage == "full":
            registry.cleanup_all()
        logger.info("[MemControl] cleanup stage=%s complete", stage)
        log_memory(f"cleanup_done_{stage}")
        return io.NodeOutput(passthrough)


class H3MemControlVAECache(io.ComfyNode):
    """Return one process-lifetime VAE instance per file path."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MemControlVAECache",
            display_name="H3 MemControl VAE Cache",
            category="h3/memcontrol",
            description=(
                "Keeps one VAE instance per file path for the Comfy process lifetime and releases "
                "it from VRAM after each encode/decode use. The cached VAE is outside Comfy output "
                "cache and is not evicted by clean_unused()."
            ),
            search_aliases=["h3 vae cache", "h3 mem control vae cache"],
            inputs=[
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae", optional=True),
            ],
            outputs=[
                io.Vae.Output(display_name="vae"),
                io.Vae.Output(display_name="audio_vae"),
            ],
        )

    @classmethod
    def execute(cls, vae, audio_vae=None) -> io.NodeOutput:
        vae = _cache_vae(vae)
        audio_vae = _cache_vae(audio_vae)
        logger.info(
            "[MemControl] VAE cache node vae_cached=%s audio_vae_cached=%s entries=%d",
            vae is not None,
            audio_vae is not None,
            len(registry.vae_cache),
        )
        return io.NodeOutput(vae, audio_vae)


class H3MemControlDebug(io.ComfyNode):
    """Return MemControl status as a string for console/workflow debugging."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MemControlDebug",
            display_name="H3 MemControl Debug",
            category="h3/memcontrol",
            description="Returns MemControl manager, container, and VAE cache status.",
            search_aliases=["h3 memcontrol debug", "h3 mem control status"],
            inputs=[io.Model.Input("model", optional=True)],
            outputs=[io.String.Output(display_name="status")],
        )

    @classmethod
    def execute(cls, model=None) -> io.NodeOutput:
        text = registry.status_text()
        logger.info("[MemControl] debug status:\n%s", text)
        return io.NodeOutput(text)


class H3MemControlExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            H3MemControlSetup,
            H3MemControlCleanup,
            H3MemControlVAECache,
            H3MemControlDebug,
        ]


async def comfy_entrypoint() -> H3MemControlExtension:
    return H3MemControlExtension()
