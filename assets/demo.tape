# vhs tape — records the end-to-end selftest.
#
# Regenerate after `pip install -e ".[dev]"` (so fastembed + sqlite-vec
# are importable by `python`), from the repo root:
#
#     vhs assets/demo.tape
#
Output assets/demo.gif

Set Shell "bash"
Set FontSize 17
Set Width 1180
Set Height 680
Set Padding 22
Set Theme "Catppuccin Mocha"

Hide
Type "clear"
Enter
Show

Sleep 700ms
Type "python mnemosyne/unified_index.py selftest"
Sleep 600ms
Enter
Sleep 14s
Sleep 2500ms
