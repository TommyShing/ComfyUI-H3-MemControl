"""ComfyUI entrypoint for H3 MemControl custom node package."""


async def comfy_entrypoint():
    from .h3_memcontrol.nodes import H3MemControlExtension

    return H3MemControlExtension()
