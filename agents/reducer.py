from __future__ import annotations

import os
import json
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from graph.schema import State, Plan, ImageSpec
from graph.llm import llm
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk import client as stability_client
from langchain_core.messages import SystemMessage, HumanMessage


def merge_content(state: State) -> dict:
    plan = state["plan"]
    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered).strip()
    return {"merged_md": f"# {plan.blog_title}\n\n{body}\n"}


def decide_image(state: State) -> dict:
    merged = state.get("merged_md", "")

    headings = "\n".join(
        line for line in merged.split("\n")
        if line.startswith("#")
    )

    response = llm.invoke([
        SystemMessage(content='You are a technical blog editor. Return ONLY valid JSON, no markdown, no backticks. Format: {"md_with_placeholders": "<markdown with [[IMAGE_N]] inserted>", "images": [{"placeholder": "[[IMAGE_1]]", "filename": "img1.png", "alt": "alt text", "caption": "caption", "prompt": "detailed SD prompt for technical diagram", "size": "1536x1024", "quality": "high"}]}. Max 3 images. Only where genuinely useful. If no images needed return images=[].'),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Headings:\n{headings}\n\n"
            f"Markdown:\n{merged[:1500]}"
        )),
    ])

    raw = response.content.strip()

    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                data = json.loads(part)
                break
            except Exception:
                continue
        else:
            return {"md_with_placeholders": merged, "image_specs": []}
    else:
        try:
            data = json.loads(raw)
        except Exception:
            return {"md_with_placeholders": merged, "image_specs": []}

    images = []
    for img in data.get("images", []):
        try:
            images.append(ImageSpec(**img).model_dump())
        except Exception:
            continue

    return {
        "md_with_placeholders": data.get("md_with_placeholders", merged),
        "image_specs": images,
    }


def stability_image_generation(prompt: str, size: str = "1024x1024") -> bytes:
    api_key = os.environ.get("STABILITY_API_KEY")
    if not api_key:
        raise RuntimeError("STABILITY_API_KEY is not set.")

    width, height = (int(x) for x in size.split("x"))

    stability = stability_client.StabilityInference(
        key=api_key,
        verbose=False,
        engine="stable-diffusion-xl-1024-v1-0",
    )

    answers = stability.generate(
        prompt=prompt,
        width=width,
        height=height,
        steps=30,
        cfg_scale=7.0,
        sampler=generation.SAMPLER_K_DPMPP_2M,
    )

    for answer in answers:
        for artifact in answer.artifacts:
            if artifact.finish_reason == generation.FILTER:
                raise RuntimeError("Blocked by Stability AI safety filter.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                return artifact.binary

    raise RuntimeError("No image returned from Stability AI.")


def generate_and_place_images(state: State) -> dict:
    plan: Plan = state["plan"]
    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs") or []

    if not image_specs:
        Path(f"{plan.blog_title}.md").write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        out_path = images_dir / spec["filename"]

        if not out_path.exists():
            try:
                img_bytes = stability_image_generation(
                    prompt=spec["prompt"],
                    size=spec.get("size", "1024x1024"),
                )
                out_path.write_bytes(img_bytes)
            except Exception as e:
                fallback = (
                    f"> **[IMAGE FAILED]** {spec.get('caption', '')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, fallback)
                continue

        img_md = f"![{spec['alt']}](images/{spec['filename']})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    Path(f"{plan.blog_title}.md").write_text(md, encoding="utf-8")
    return {"final": md}


reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_image)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)

reducer_subgraph = reducer_graph.compile()