import re

def remove_think_tags(text: str) -> str:
  """
  Removes <think>...</think> tags and their content from a string.

  Args:
    text: The input string.

  Returns:
    The string with <think> tags and their content removed.
  """
  return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
