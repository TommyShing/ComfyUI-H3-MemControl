"""ComfyUI nodes for H3 MemControl test version."""

from __future__ import annotations

import logging
from typing import Any

from comfy_api.latest import ComfyExtension, io

from .manager import format_bytes, log_memory, registry

logger = logging.getLogger("H3MemControl")


def _cache_vae(value: Any | None) -> Any | None:
    if value is None:
        return None
    return registry.cache_vae(value)


def _install_roots(manager, model, clip):
    seen: set[int] = set()
    if model is not None:
        try:
            model_device = getattr(model, "load_device", None)
            manager.install_on_root(model.model, "model", seen=seen, root_device=model_device)
            manager.patch_load_list(model, "model")
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
        log_memory("setup_done")
        return io.NodeOutput(model, clip, vae, audio_vae)


class H3MemControlCleanup(io.ComfyNode):
    """Release managed block state, buffers, and LoRA references while passing data through."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MemControlCleanup",
            display_name="H3 MemControl Cleanup",
            category="h3/memcontrol",
            description=(
                "Restores MemControl-wrapped block containers and releases buffer/LoRA state. "
                "It does not clear user IO such as prompt, seed, images, or video previews."
            ),
            search_aliases=["h3 memcontrol cleanup", "h3 mem control cleanup"],
            inputs=[
                io.AnyType.Input("passthrough"),
                io.Combo.Input(
                    "stage",
                    options=["auto", "after_te", "after_sampling", "full"],
                    default="auto",
                    tooltip="after_te cleans qwen containers; after_sampling cleans H3 containers and releases buffer.",
                ),
            ],
            outputs=[io.AnyType.Output(display_name="passthrough")],
        )

    @classmethod
    def execute(cls, passthrough, stage="auto") -> io.NodeOutput:
        log_memory(f"cleanup_start_{stage}")
        if stage in ("auto", "full", "after_sampling"):
            registry.cleanup_root("model", release_buffer=True)
        if stage in ("auto", "full", "after_te"):
            registry.cleanup_root("clip", release_buffer=False)
        if stage == "full":
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
                "Keeps one VAE instance per file path for the Comfy process lifetime. "
                "The cached VAE is outside Comfy output cache and is not evicted by clean_unused()."
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
