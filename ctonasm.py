
"""
C to NASM Struct Converter
Author  : Alon Alush / alonalush5@gmail.com
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# 1.  TYPE REGISTRY
PTR64_DIR = "resq"
PTR32_DIR = "resd"

def _byte(n: int = 1) -> str: return f"resb {n}"
def _word(n: int = 1) -> str: return f"resw {n}"
def _dword(n: int = 1) -> str: return f"resd {n}"
def _qword(n: int = 1) -> str: return f"resq {n}"

BASE_TYPE_MAP: Dict[str, str | None] = {
    # C / stdint
    "void": _byte(1),  # only when 'void *'
    "char": _byte(), "signed char": _byte(), "unsigned char": _byte(),
    "short": _word(), "unsigned short": _word(),
    "int": _dword(), "unsigned int": _dword(),
    "long": _dword(), "unsigned long": _dword(),  # LLP64 handled later
    "long long": _qword(), "unsigned long long": _qword(),
    "float": _dword(), "double": _qword(), "long double": _qword(),

    "int8_t": _byte(),  "uint8_t": _byte(),
    "int16_t": _word(), "uint16_t": _word(),
    "int32_t": _dword(), "uint32_t": _dword(),
    "int64_t": _qword(), "uint64_t": _qword(),

    # GCC / Clang vectors
    "__int128": _qword(2), "__uint128_t": _qword(2), "__float128": _qword(2),
    "__m64": _byte(8),
    "__m128": _byte(16), "__m128i": _byte(16), "__m128d": _byte(16),
    "__m256": _byte(32), "__m256i": _byte(32), "__m256d": _byte(32),
    "__m512": _byte(64), "__m512i": _byte(64), "__m512d": _byte(64),

    # WinAPI – scalar (non-pointer) types
    "BOOL": _dword(), "BOOLEAN": _byte(),
    "BYTE": _byte(), "WORD": _word(), "DWORD": _dword(),
    "INT": _dword(), "UINT": _dword(), "LONG": _dword(), "ULONG": _dword(),
    "LONGLONG": _qword(), "ULONGLONG": _qword(),
    "SHORT": _word(), "USHORT": _word(),
    "CHAR": _byte(), "WCHAR": _word(), "TCHAR": _word(),
    "HRESULT": _dword(), "NTSTATUS": _dword(),  # new
    "GUID": _byte(16),                          # struct, but length is fixed

    # Pointer-sized scalars – will be patched to resd/resq later
    "SIZE_T": None, "SSIZE_T": None,
    "INT_PTR": None, "UINT_PTR": None,
    "LONG_PTR": None, "ULONG_PTR": None,
    "DWORD_PTR": None, "HANDLE": None,
    # Obvious pointers
    "PVOID": None, "LPVOID": None, "LPCVOID": None,
    # A few more ubiquitous Win handles
    "HMODULE": None, "HINSTANCE": None, "HWND": None, "HDC": None,
}

# Anything ALL-CAPS **or** matching *_PTR / PFOO / LPBAR is likely a pointer
_CAPS_PTR_RX = re.compile(r"^[A-Z0-9_]+$")
_POINTER_RX  = re.compile(r"^(?:P|LP)?[A-Z0-9_]*_?PTR$|^P\w+$")

class TypeRegistry:
    """Resolve ‘C type’ → NASM reservation directive."""
    _qual_rx  = re.compile(r"\b(?:const|volatile|static|extern|struct|union|enum)\b")
    _spaces   = re.compile(r"\s+")

    def __init__(self, ptr_size: int = 8) -> None:
        self.ptr_size = ptr_size
        self.dir_ptr  = PTR64_DIR if ptr_size == 8 else PTR32_DIR
        self.map: Dict[str, str] = BASE_TYPE_MAP.copy()
        # patch pointer-sized entries
        for k, v in list(self.map.items()):
            if v is None:
                self.map[k] = self.dir_ptr

    # ── helpers ───────────────────────────────────────────────────────────
    @classmethod
    def _norm(cls, t: str) -> str:
        t = cls._qual_rx.sub("", t)
        t = cls._spaces.sub(" ", t).strip()
        t = t.replace(" *", "*").replace("* ", "*")
        return re.sub(r"\*{2,}", "*", t)

    # ── public API ────────────────────────────────────────────────────────
    def add_typedef(self, alias: str, target: str) -> None:
        alias, target = self._norm(alias), self._norm(target)
        if alias in self.map:
            return
        if target in self.map:
            self.map[alias] = self.map[target]
        elif target.endswith('*'):
            self.map[alias] = self.dir_ptr

    def resolve(self, c_type: str) -> str:
        c_type = self._norm(c_type)
        if c_type.endswith(('*', '&')):
            return self.dir_ptr
        if c_type in self.map:
            return self.map[c_type]
        if _POINTER_RX.match(c_type) or (_CAPS_PTR_RX.match(c_type)):
            return self.dir_ptr
        return _dword()  # fallback – assume 32-bit scalar

# 2.  LEXING / PARSING HELPERS
COMMENT_RX = re.compile(
    r"//.*?$"          # C++ 1-line
    r"|/\*.*?\*/",     # C   multi-line
    re.S | re.M,
)

def strip_comments(code: str) -> str:
    return COMMENT_RX.sub("", code)

# Field patterns (<<< the bug was here)
FIELD_SCALAR_RX = re.compile(
    r"^\s*(?P<type>[^$$\]:;]+?)\s+(?P<name>\w+)\s*;\s*$"
)
FIELD_ARRAY_RX = re.compile(
    r"^\s*(?P<type>[^\[$$:;]+?)\s+(?P<name>\w+)\s*$$\s*(?P<size>[^$$]+?)\s*]\s*;\s*$"
)
FIELD_BITFIELD_RX = re.compile(
    r"^\s*(?P<type>.+?)\s+(?P<name>\w+)\s*:\s*(?P<bits>\d+)\s*;\s*$"
)
TYPEDEF_RX = re.compile(
    r"^\s*typedef\s+(?P<body>.+?)\s+(?P<alias>\w+)\s*;\s*$"
)

def canonicalise_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# 3.  STRUCT PARSER
def _find_matching_brace(src: str, pos: int) -> int:
    depth = 0
    for i, c in enumerate(src[pos:], start=pos):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1

def parse_typedefs(code: str, reg: TypeRegistry) -> None:
    for m in TYPEDEF_RX.finditer(code):
        reg.add_typedef(m.group("alias"), canonicalise_whitespace(m.group("body")))

def extract_structs(code: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(code):
        if code.startswith("struct", i) or code.startswith("typedef struct", i):
            brace = code.find("{", i)
            if brace == -1:
                break
            end = _find_matching_brace(code, brace)
            if end == -1:
                break
            body = code[brace + 1 : end]
            tail = code[end + 1 :].lstrip()
            if (m := re.match(r"(?P<name>\w+)", tail)):
                out.append((m.group("name"), body))
            i = end + 1
        else:
            i += 1
    return out

def parse_struct_body(body: str) -> List[dict]:
    fields: List[dict] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if (m := FIELD_ARRAY_RX.match(line)):
            fields.append(
                dict(type=m.group("type"), name=m.group("name"), array_size=m.group("size"))
            )
        elif (m := FIELD_BITFIELD_RX.match(line)):
            fields.append(
                dict(
                    type=m.group("type"),
                    name=m.group("name"),
                    is_bitfield=True,
                    bits=int(m.group("bits")),
                )
            )
        elif (m := FIELD_SCALAR_RX.match(line)):
            fields.append(dict(type=m.group("type"), name=m.group("name")))
    return fields

# 4.  NASM GENERATOR (minor cosmetics)
def _render_field(reg: TypeRegistry, fld: dict) -> str:
    name, ctype = fld["name"], fld["type"]
    if fld.get("is_bitfield"):
        return f"    .{name:<24} resd 1    ; bit-field {fld['bits']} bits"
    nasm_dir = reg.resolve(ctype)
    comment = "" if ctype in reg.map or ctype.endswith("*") else f" ; {ctype}"
    if "array_size" in fld:
        return f"    .{name:<24} {nasm_dir} {fld['array_size']}{comment}"
    space = "" if nasm_dir.startswith("res") and nasm_dir[-1].isdigit() else " 1"
    return f"    .{name:<24} {nasm_dir}{space}{comment}"

def generate_nasm(name: str, fields: Iterable[dict], reg: TypeRegistry) -> str:
    lines = [f"struc {name}"]
    seen: set[str] = set()
    for f in fields:
        n = f["name"]
        while n in seen:  # avoid dups
            n += "_"
        seen.add(n)
        lines.append(_render_field(reg, {**f, "name": n}))
    lines.append("endstruc\n")
    return "\n".join(lines)

# 5.  CLI / MAIN
def banner() -> str:
    return "=" * 55 + "\n  C → NASM struct converter (v3.1)\n" + "=" * 55

def _selftest() -> None:
    header = """
        typedef struct _FOO {
            int     a;
            BYTE    b[4];
            HRESULT c;
            HWND    hWnd;
        } FOO, *PFOO;
    """
    reg = TypeRegistry(8)
    parse_typedefs(header, reg)
    body = extract_structs(header)[0][1]
    print(generate_nasm("FOO", parse_struct_body(body), reg))

def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Convert C structs in <header.h> to NASM 'struc' blocks.",
    )
    ap.add_argument("-i", "--input", help="C header file")
    ap.add_argument("-o", "--output", help=".inc to write (default: <input>.inc)")
    ap.add_argument("-m", "--mode", choices=("32", "64"), help="Force 32/64-bit")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--list-types", action="store_true", help="Print builtin type table")
    ap.add_argument("--show-regex", action="store_true", help="Dump the major regexes")
    ap.add_argument("--selftest", action="store_true", help="Run a quick parser test")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return
    if args.show_regex:
        for n, rx in [("SCALAR", FIELD_SCALAR_RX), ("ARRAY", FIELD_ARRAY_RX), ("BITFIELD", FIELD_BITFIELD_RX)]:
            print(f"{n}:\n  {rx.pattern}\n")
        return
    if args.list_types:
        print("\n".join(sorted(BASE_TYPE_MAP)))
        return
    if not args.input:
        ap.error("-i/--input is required")

    ptr_size = 8 if (args.mode or ("64" if sys.maxsize > 2 ** 32 else "32")) == "64" else 4
    reg = TypeRegistry(ptr_size)

    src_path = Path(args.input)
    if not src_path.exists():
        sys.exit(f"error: {src_path} not found")

    code = strip_comments(src_path.read_text(encoding="utf-8", errors="ignore"))
    parse_typedefs(code, reg)

    structs = extract_structs(code)
    if not structs:
        sys.exit("No structs found.")

    out_lines: List[str] = [
        "; ------------------------------------------------------------------",
        ";  generated by  ctonasm / alonalush5@gmail.com",
        f";  source : {src_path.name}",
        "; ------------------------------------------------------------------\n",
    ]

    for name, body in structs:
        if not args.quiet:
            print(f"  > {name}")
        out_lines.append(generate_nasm(name, parse_struct_body(body), reg))

    out_path = Path(args.output) if args.output else src_path.with_suffix(".inc")
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    if not args.quiet:
        print(f"Wrote to {out_path}")

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
