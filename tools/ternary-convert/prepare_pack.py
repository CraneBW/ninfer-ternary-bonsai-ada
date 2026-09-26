"""把 pack.py 改造成 zatfung 版：路径参数化，不动原仓库。"""
from pathlib import Path

SRC = Path(r"G:/Agents/ninfer-works/ninfer-ternary-qwen3.8-27b-3090--may-has-bug/pack/pack.py")
DST = Path(r"C:/WorkModels/Qwen3.8-27B/_pack/pack_zatfung.py")
src = SRC.read_text(encoding="utf-8")

old_block = '''NINFER_ROOT = r"H:/ninfer-ternary/src/ninfer"
if NINFER_ROOT not in sys.path:
    sys.path.insert(0, NINFER_ROOT)'''
new_block = '''# --- zatfung 改造：路径全部环境变量化 -------------------------------------
# 原版硬编码 H:/... 路径。改为从环境变量取，便于同一脚本处理两个模型、
# 且不把 zatfung 的源码树和模型耦合进脚本。
import os
NINFER_ROOT = os.environ.get("ZATFUNG_ROOT", r"G:/Agents/ninfer-works/zatfung")
if NINFER_ROOT not in sys.path:
    sys.path.insert(0, NINFER_ROOT)'''
assert old_block in src, "NINFER_ROOT block not found"
src = src.replace(old_block, new_block, 1)

old_tpl = 'TEMPLATE = r"H:/ninfer-3090/models/qwen3_8_27b_huihui_abliterated.ninfer"'
new_tpl = ('TEMPLATE = os.environ.get("ZATFUNG_TEMPLATE",\n'
           '            r"C://WorkModels//Qwen3.8-27B//_pack_template//qwen3_8_27b_v2.ninfer")')
assert old_tpl in src, "TEMPLATE not found"
src = src.replace(old_tpl, new_tpl, 1)

old_gguf = 'GGUF = r"H:/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf"'
new_gguf = ('GGUF = os.environ.get("ZATFUNG_GGUF",\n'
            '            r"C://WorkModels//Qwen3.8-27B//Ternary-Bonsai-2-27B-PQ2_0.gguf")')
assert old_gguf in src, "GGUF not found"
src = src.replace(old_gguf, new_gguf, 1)

DST.write_text(src, encoding="utf-8")
print("wrote", DST)
print("  NINFER_ROOT -> env ZATFUNG_ROOT (default zatfung)")
print("  TEMPLATE    -> env ZATFUNG_TEMPLATE")
print("  GGUF        -> env ZATFUNG_GGUF")
