import pytest
from src.utils.text_utils import remove_think_tags

def test_single_think_tag():
    text = "This is a sentence. <think>This part should be removed.</think> This part should remain."
    expected = "This is a sentence.  This part should remain."
    assert remove_think_tags(text) == expected

def test_multiple_think_tags():
    text = "<think>Remove this.</think>Keep this.<think>And this.</think>"
    expected = "Keep this."
    assert remove_think_tags(text) == expected

def test_no_think_tags():
    text = "This string has no think tags."
    expected = "This string has no think tags."
    assert remove_think_tags(text) == expected

def test_think_tags_at_beginning():
    text = "<think>Remove this.</think>Keep this."
    expected = "Keep this."
    assert remove_think_tags(text) == expected

def test_think_tags_at_end():
    text = "Keep this.<think>Remove this.</think>"
    expected = "Keep this."
    assert remove_think_tags(text) == expected

def test_think_tags_in_middle():
    text = "Keep <think>Remove this.</think> this."
    expected = "Keep  this."
    assert remove_think_tags(text) == expected

def test_empty_string():
    text = ""
    expected = ""
    assert remove_think_tags(text) == expected

def test_only_think_tag():
    text = "<think>This is only a think tag.</think>"
    expected = ""
    assert remove_think_tags(text) == expected

def test_multiline_think_tags():
    text = "Before <think>This is a\nmultiline think tag\ncontent.</think> After"
    expected = "Before  After"
    assert remove_think_tags(text) == expected

def test_nested_think_tags():
    # The current regex removes the outermost tags.
    text = "<think>outer <think>inner</think> outer</think>"
    expected = "" 
    assert remove_think_tags(text) == expected

def test_nested_think_tags_with_content_outside():
    text = "Content before <think>outer <think>inner</think> outer</think> content after"
    expected = "Content before  content after"
    assert remove_think_tags(text) == expected

def test_tags_with_leading_trailing_whitespace_content():
    text = "Text <think>  content  </think> more text"
    expected = "Text  more text"
    assert remove_think_tags(text) == expected

def test_incomplete_tag_no_end():
    text = "<think>no end tag"
    expected = "<think>no end tag"
    assert remove_think_tags(text) == expected

def test_incomplete_tag_no_start():
    text = "no start tag</think>"
    expected = "no start tag</think>"
    assert remove_think_tags(text) == expected

def test_mismatched_tags():
    text = "<think>mismatched</thinkk>"
    expected = "<think>mismatched</thinkk>"
    assert remove_think_tags(text) == expected

def test_think_tag_with_special_chars_inside():
    text = "Start <think>content with !@#$%^&*()_+=-`~[]{}|;':\",./<>?</think> End"
    expected = "Start  End"
    assert remove_think_tags(text) == expected

def test_multiple_tags_adjacent():
    text = "<think>one</think><think>two</think>three"
    expected = "three"
    assert remove_think_tags(text) == expected

def test_multiple_tags_with_spaces_between():
    text = "<think>one</think> <think>two</think> three"
    expected = "  three" # Note: spaces where tags were remain
    assert remove_think_tags(text) == expected
