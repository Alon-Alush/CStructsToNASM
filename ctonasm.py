
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
from typing import Dict, Iterable, List, Tuple, Set

# 1.  TYPE REGISTRY
PTR64_DIR = "resq"
PTR32_DIR = "resd"

def _byte(n: int = 1) -> str: return f"resb {n}"
def _word(n: int = 1) -> str: return f"resw {n}"
def _dword(n: int = 1) -> str: return f"resd {n}"
def _qword(n: int = 1) -> str: return f"resq {n}"

BASE_TYPE_MAP: Dict[str, str | None] = {
    # C / stdint basic types
    "void": _byte(1),  # only when 'void *'
    "char": _byte(), "signed char": _byte(), "unsigned char": _byte(),
    "short": _word(), "unsigned short": _word(),
    "int": _dword(), "unsigned int": _dword(),
    "long": _dword(), "unsigned long": _dword(),  # LLP64 handled later
    "long long": _qword(), "unsigned long long": _qword(),
    "float": _dword(), "double": _qword(), "long double": _qword(),

    # stdint.h types
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

    # Windows API - Basic scalar types
    "BOOL": _dword(), "BOOLEAN": _byte(),
    "BYTE": _byte(), "WORD": _word(), "DWORD": _dword(),
    "INT": _dword(), "UINT": _dword(), 
    "LONG": _dword(), "ULONG": _dword(),
    "LONGLONG": _qword(), "ULONGLONG": _qword(),
    "SHORT": _word(), "USHORT": _word(),
    "CHAR": _byte(), "UCHAR": _byte(),
    "WCHAR": _word(), "TCHAR": _word(),
    "CCHAR": _byte(), "TBYTE": _byte(),
    
    # Windows API - Return codes and special types
    "HRESULT": _dword(), "NTSTATUS": _dword(),
    "LRESULT": None,  # LONG_PTR
    "WPARAM": None, "LPARAM": None,  # pointer-sized
    
    # Windows API - Sized integers
    "INT8": _byte(), "UINT8": _byte(),
    "INT16": _word(), "UINT16": _word(),
    "INT32": _dword(), "UINT32": _dword(),
    "INT64": _qword(), "UINT64": _qword(),
    "LONG32": _dword(), "ULONG32": _dword(),
    "LONG64": _qword(), "ULONG64": _qword(),
    "DWORD32": _dword(), "DWORD64": _qword(),
    "DWORDLONG": _qword(),
    
    # Windows API - Large integers and special structs
    "LARGE_INTEGER": _qword(),
    "ULARGE_INTEGER": _qword(),
    "GUID": _byte(16),
    "COLORREF": _dword(),
    "LCID": _dword(), "LANGID": _word(),
    "LCTYPE": _dword(), "LGRPID": _dword(),
    "ATOM": _word(), "USN": _qword(),
    "FLOAT": _dword(), "QWORD": _qword(),
    
    # Unicode string structure (known size)
    "UNICODE_STRING": _byte(16),  # USHORT + USHORT + PWSTR = 2+2+8 on x64

    # Pointer-sized scalars – will be patched to resd/resq later
    "SIZE_T": None, "SSIZE_T": None,
    "INT_PTR": None, "UINT_PTR": None,
    "LONG_PTR": None, "ULONG_PTR": None,
    "DWORD_PTR": None, "HALF_PTR": None, "UHALF_PTR": None,
    
    # All Windows handles - these are all pointers
    "HANDLE": None, "PHANDLE": None, "LPHANDLE": None,
    "HACCEL": None, "HBITMAP": None, "HBRUSH": None, "HCOLORSPACE": None,
    "HCONV": None, "HCONVLIST": None, "HCURSOR": None, "HDC": None,
    "HDDEDATA": None, "HDESK": None, "HDROP": None, "HDWP": None,
    "HENHMETAFILE": None, "HFILE": None, "HFONT": None, "HGDIOBJ": None,
    "HGLOBAL": None, "HHOOK": None, "HICON": None, "HINSTANCE": None,
    "HKEY": None, "PHKEY": None, "HKL": None, "HLOCAL": None,
    "HMENU": None, "HMETAFILE": None, "HMODULE": None, "HMONITOR": None,
    "HPALETTE": None, "HPEN": None, "HRGN": None, "HRSRC": None,
    "HSZ": None, "HWINSTA": None, "HWND": None,
    "SC_HANDLE": None, "SC_LOCK": None, "SERVICE_STATUS_HANDLE": None,
    
    # Obvious pointer types
    "PVOID": None, "LPVOID": None, "LPCVOID": None,
    
    # String pointer types
    "PSTR": None, "LPSTR": None, "LPCSTR": None, "PCSTR": None,
    "PWSTR": None, "LPWSTR": None, "LPCWSTR": None, "PCWSTR": None,
    "PTSTR": None, "LPTSTR": None, "LPCTSTR": None, "PCTSTR": None,
    "PTCHAR": None, "PTBYTE": None, "PWCHAR": None, "PCHAR": None,
    
    # Pointer to basic types
    "PBOOL": None, "LPBOOL": None, "PBOOLEAN": None,
    "PBYTE": None, "LPBYTE": None, "PUCHAR": None,
    "PWORD": None, "LPWORD": None, "PUSHORT": None,
    "PDWORD": None, "LPDWORD": None, "PULONG": None, "PULONGLONG": None,
    "PINT": None, "LPINT": None, "PUINT": None,
    "PLONG": None, "LPLONG": None, "PLONGLONG": None,
    "PFLOAT": None, "LPCOLORREF": None, "PLCID": None,
    
    # Pointer to sized types
    "PINT8": None, "PUINT8": None, "PINT16": None, "PUINT16": None,
    "PINT32": None, "PUINT32": None, "PINT64": None, "PUINT64": None,
    "PLONG32": None, "PULONG32": None, "PLONG64": None, "PULONG64": None,
    "PDWORD32": None, "PDWORD64": None, "PDWORDLONG": None,
    
    # Pointer to pointer-sized types
    "PSIZE_T": None, "PSSIZE_T": None,
    "PINT_PTR": None, "PUINT_PTR": None,
    "PLONG_PTR": None, "PULONG_PTR": None,
    "PDWORD_PTR": None, "PHALF_PTR": None, "PUHALF_PTR": None,
    "PSHORT": None,
}

# Anything ALL-CAPS **or** matching *_PTR / PFOO / LPBAR is likely a pointer
_CAPS_PTR_RX = re.compile(r"^[A-Z0-9_]+$")
_POINTER_RX  = re.compile(r"^(?:P|LP)?[A-Z0-9_]*_?PTR$|^P\w+$|^LP\w+$")

class TypeRegistry:
    """Resolve 'C type' → NASM reservation directive."""
    _qual_rx  = re.compile(r"\b(?:const|volatile|static|extern|struct|union|enum)\b")
    _spaces   = re.compile(r"\s+")

    def __init__(self, ptr_size: int = 8) -> None:
        self.ptr_size = ptr_size
        self.dir_ptr  = PTR64_DIR if ptr_size == 8 else PTR32_DIR
        self.map: Dict[str, str] = BASE_TYPE_MAP.copy()
        self.known_structs: Set[str] = set()  # Track defined structs
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
    def add_struct(self, struct_name: str) -> None:
        """Register a struct as known for size references."""
        self.known_structs.add(struct_name)

    def add_typedef(self, alias: str, target: str) -> None:
        alias, target = self._norm(alias), self._norm(target)
        if alias in self.map:
            return
        if target in self.map:
            self.map[alias] = self.map[target]
        elif target.endswith('*'):
            self.map[alias] = self.dir_ptr

    def resolve(self, c_type: str, is_nested_struct: bool = False) -> Tuple[str, str]:
        """
        Returns (nasm_directive, comment)
        Always includes original type in comment for clarity.
        """
        original_type = c_type
        c_type = self._norm(c_type)
        
        # Handle pointers - always comment with original type
        if c_type.endswith(('*', '&')):
            base_type = c_type.rstrip('*&').strip()
            comment = f" ; {original_type}"
            if base_type in self.known_structs:
                return self.dir_ptr, comment
            return self.dir_ptr, comment
        
        # Check if it's a known struct (for nested structs)
        if c_type in self.known_structs and is_nested_struct:
            return f"resb {c_type}_size", f" ; {original_type}"
        
        # Standard type mapping - always comment with original type unless it's a basic C type
        if c_type in self.map:
            # Only comment non-basic types for clarity
            basic_c_types = {"char", "short", "int", "long", "float", "double", 
                           "signed char", "unsigned char", "unsigned short", 
                           "unsigned int", "unsigned long", "long long", 
                           "unsigned long long", "void"}
            if c_type in basic_c_types and original_type == c_type:
                return self.map[c_type], ""
            else:
                return self.map[c_type], f" ; {original_type}"
        
        # Heuristic detection for pointers
        if _POINTER_RX.match(c_type) or (_CAPS_PTR_RX.match(c_type)):
            return self.dir_ptr, f" ; {original_type}"
        
        # Unknown type - comment with original type
        return _dword(), f" ; {original_type}"

# 2.  LEXING / PARSING HELPERS
COMMENT_RX = re.compile(
    r"//.*?$"          # C++ 1-line
    r"|/\*.*?\*/",     # C   multi-line
    re.S | re.M,
)

def strip_comments(code: str) -> str:
    return COMMENT_RX.sub("", code)

# Fixed Field patterns
FIELD_SCALAR_RX = re.compile(
    r"^\s*(?P<type>[^\[\]:;]+?)\s+(?P<name>\w+)\s*;\s*$"
)
FIELD_ARRAY_RX = re.compile(
    r"^\s*(?P<type>[^\[\]:;]+?)\s+(?P<name>\w+)\s*\[\s*(?P<size>[^\]]+?)\s*\]\s*;\s*$"
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
    """Parse struct body, handling unions better."""
    fields: List[dict] = []
    lines = body.splitlines()
    i = 0
    in_union = False
    union_depth = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Handle union start
        if line.startswith("union"):
            in_union = True
            union_depth = 0
            i += 1
            continue
            
        # Handle braces
        if "{" in line:
            union_depth += line.count("{")
            i += 1
            continue
            
        if "}" in line:
            union_depth -= line.count("}")
            if union_depth <= 0 and in_union:
                in_union = False
                # Skip the union variable name if present
                if i + 1 < len(lines) and lines[i + 1].strip().endswith(";"):
                    i += 1
            i += 1
            continue
            
        # Skip struct definitions inside unions (like s1, s2, etc.)
        if line.startswith("struct") and in_union:
            # Skip until we find the closing brace
            struct_depth = 0
            while i < len(lines):
                if "{" in lines[i]:
                    struct_depth += lines[i].count("{")
                if "}" in lines[i]:
                    struct_depth -= lines[i].count("}")
                    if struct_depth <= 0:
                        break
                i += 1
            i += 1
            continue
        
        # Parse actual field lines
        if (m := FIELD_ARRAY_RX.match(line)):
            fields.append(
                dict(type=m.group("type"), name=m.group("name"), array_size=m.group("size"))
            )
        elif (m := FIELD_BITFIELD_RX.match(line)):
            # Skip bitfields in unions - they're usually overlapping
            if not in_union:
                fields.append(
                    dict(
                        type=m.group("type"),
                        name=m.group("name"),
                        is_bitfield=True,
                        bits=int(m.group("bits")),
                    )
                )
        elif (m := FIELD_SCALAR_RX.match(line)):
            # Only add the first field of a union to avoid duplication
            if not in_union or not fields or not getattr(fields[-1], 'in_union', False):
                field_dict = dict(type=m.group("type"), name=m.group("name"))
                if in_union:
                    field_dict['in_union'] = True
                fields.append(field_dict)
        
        i += 1
    
    return fields

# 4.  NASM GENERATOR
def _render_field(reg: TypeRegistry, fld: dict) -> str:
    name, ctype = fld["name"], fld["type"]
    
    if fld.get("is_bitfield"):
        # For bitfields, we usually want to reserve space for the underlying type
        nasm_dir, comment = reg.resolve(ctype)
        if not comment:
            comment = f" ; {ctype} bit-field {fld['bits']} bits"
        else:
            comment = f"{comment} bit-field {fld['bits']} bits"
        return f"    .{name:<24} {nasm_dir}{comment}"
    
    # Check if this might be a nested struct
    normalized_type = reg._norm(ctype)
    is_nested = (normalized_type in reg.known_structs and 
                not normalized_type.endswith('*') and 
                normalized_type not in reg.map and
                not _POINTER_RX.match(normalized_type) and 
                not _CAPS_PTR_RX.match(normalized_type))
    
    nasm_dir, comment = reg.resolve(ctype, is_nested_struct=is_nested)
    
    if "array_size" in fld:
        return f"    .{name:<24} {nasm_dir} {fld['array_size']}{comment}"
    
    # Handle spacing for directives
    if nasm_dir.startswith("res") and (nasm_dir[-1].isdigit() or "size" in nasm_dir):
        space = ""
    else:
        space = " 1"
    
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
    return "=" * 55 + "\n  C → NASM struct converter\n" + "=" * 55

def _selftest() -> None:
    header = """
        typedef struct _FOO {
            int     a;
            BYTE    b[4];
            HRESULT c;
            HWND    hWnd;
            UNICODE_STRING str;
            union {
                ULONG flags;
                struct {
                    ULONG flag1 : 1;
                    ULONG flag2 : 1;
                } s;
            } u;
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

    # First pass: register all struct names
    for name, _ in structs:
        reg.add_struct(name)

    out_lines: List[str] = [
        "; ------------------------------------------------------------------",
        ";  generated by  ctonasm (enhanced) / alonalush5@gmail.com",
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
