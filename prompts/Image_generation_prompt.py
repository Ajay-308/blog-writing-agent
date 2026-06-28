from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts import PromptTemplate


image_prompt = PromptTemplate(
    input_variables=["topic", "mode", "research_results"],
    template="""
You are an expert technical editor.

Decide whether images or diagrams are needed for this blog.

Rules:
- Maximum 3 images.
- Each image must materially improve understanding.
- Prefer architecture diagrams, workflows, comparisons, or technical illustrations.
- Avoid decorative or stock images.
- Insert placeholders exactly:
  [[IMAGE_1]]
  [[IMAGE_2]]
  [[IMAGE_3]]
- If no images are needed:
  - md_with_placeholders must equal the original input.
  - images must be [].
- Use short labels inside diagrams.

Return strictly a GlobalImagePlan object.

Topic:
{topic}

Blog kind:
{mode}

Research evidence:
{research_results}

Plan:
"""
)


def get_image_prompt() -> BasePromptTemplate:
    return image_prompt