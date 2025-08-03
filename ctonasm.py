"""
C to NASM Struct Converter
Author  : Alon Alush / alonalush5@gmail.com
"""

import re
import sys
import os
import argparse
from collections import defaultdict


#  TYPE MAP
type_map = {
    # Basic C types  ------------------------------------------------------------
    "char": "resb",
    "signed char": "resb",
    "unsigned char": "resb",
    "short": "resw",
    "short int": "resw",
    "signed short": "resw",
    "signed short int": "resw",
    "unsigned short": "resw",
    "unsigned short int": "resw",
    "int": "resd",
    "signed int": "resd",
    "unsigned int": "resd",
    "unsigned": "resd",
    "long": "resd",          # Windows long = 32-bit
    "long int": "resd",
    "signed long": "resd",
    "signed long int": "resd",
    "unsigned long": "resd",
    "unsigned long int": "resd",
    "long long": "resq",
    "long long int": "resq",
    "signed long long": "resq",
    "signed long long int": "resq",
    "unsigned long long": "resq",
    "unsigned long long int": "resq",
    "float": "resd",
    "double": "resq",
    "long double": "resq",   # assume 8 bytes on win64

    # stdint.h ---------------------------------------------------------------
    "int8_t": "resb",
    "uint8_t": "resb",
    "int16_t": "resw",
    "uint16_t": "resw",
    "int32_t": "resd",
    "uint32_t": "resd",
    "int64_t": "resq",
    "uint64_t": "resq",
    "intptr_t": "resq",
    "uintptr_t": "resq",
    "size_t": "resq",
    "ssize_t": "resq",
    "ptrdiff_t": "resq",

    # Windows / MS types ------------------------------------------------------
    "BOOL": "resd",
    "BOOLEAN": "resb",
    "BYTE": "resb",
    "WORD": "resw",
    "DWORD": "resd",
    "ULONG": "resd",
    "HANDLE": "resq",
    "PVOID": "resq",
    "LPVOID": "resq",
}


#  SMALL HELPERS
def print_banner():
    print("=" * 60)
    print("        C to NASM Struct Converter  –  v2.0")
    print("=" * 60)
    print()


def normalize_type(c_type: str) -> str:
    """Strip qualifiers, squeeze spaces, canonicalise pointers."""
    c_type = re.sub(r'\b(const|volatile|static|extern|struct|union|enum)\b', '',
                    c_type).strip()
    c_type = re.sub(r'\s+', ' ', c_type)
    c_type = c_type.replace(' *', '*').replace('* ', '*')
    c_type = re.sub(r'\*+', '*', c_type)
    return c_type.strip()


def get_nasm_type(c_type: str):
    """Return the nasm reservation directive (or dict for compound)."""
    original = c_type
    c_type = normalize_type(c_type)

    # Exact hit in table
    if c_type in type_map:
        return type_map[c_type]

    # Pointers / references
    if c_type.endswith(('*', '&')):
        return "resq"

    # Windows HANDLE-like uppercase names
    if c_type.isupper():
        return "resq"

    # Fallback
    print(f"Warning: unknown type '{original}', assuming 32-bit resd")
    return "resd"


def convert_field_to_nasm(c_type, name, array_size=None,
                          is_bitfield=False, bitfield_size=None, comment=""):
    """One C member  → one or more NASM lines."""
    if is_bitfield:
        return [f"    .{name}    resd 1 ; bitfield {bitfield_size} bits{comment}"]

    nasm_type = get_nasm_type(c_type)

    # compound struct in lookup table
    if isinstance(nasm_type, dict):
        out = []
        for sub_name, dir_, cnt in nasm_type["fields"]:
            out.append(
                f"    .{name}_{sub_name}    {dir_} {cnt}"
            )
        return out

    # arrays
    if array_size:
        return [f"    .{name}    {nasm_type} {array_size}{comment}"]

    # simple scalar
    if ' ' in nasm_type:          # “resb 16” etc.
        return [f"    .{name}    {nasm_type}{comment}"]
    return [f"    .{name}    {nasm_type} 1{comment}"]


def parse_field_line(line):
    """
    Extract information for one member inside a struct:
        • 'int value;'                    → scalar
        • 'char name[32];'                → array
        • 'uint32_t Flags : 3;'           → bit-field
    Returns a dict with keys:
        type, name, and optionally array_size / is_bitfield / bitfield_size
    """
    # Remove inline comments and surrounding whitespace/semicolon
    line = re.sub(r'/\*.*?\*/', '', line)        # strip /* ... */ on the same line
    line = re.sub(r'//.*$', '', line).strip()    # strip // ...
    line = line.rstrip(';').strip()

    # ignore blank lines
    if not line:
        return None

    # bit-field: type name : bits;
    m = re.match(r'^(?P<type>.+?)\s+(?P<name>\w+)\s*:\s*(?P<bits>\d+)\s*$', line)
    if m:
        return {
            "type": m.group('type'),
            "name": m.group('name'),
            "is_bitfield": True,
            "bitfield_size": int(m.group('bits')),
        }

    # array: type name[expr];
    m = re.match(r'^(?P<type>.+?)\s+(?P<name>\w+)\s*$$\s*(?P<size>[^$$]+)\s*\]\s*$', line)
    if m:
        return {
            "type": m.group('type'),
            "name": m.group('name'),
            "array_size": m.group('size').strip(),
        }

    # scalar: type name;
    m = re.match(r'^(?P<type>.+?)\s+(?P<name>\w+)\s*$', line)
    if m:
        return {
            "type": m.group('type'),
            "name": m.group('name'),
        }

    # could not parse this line
    return None


def find_matching_brace(lines, start_idx):
    depth = 0
    for i in range(start_idx, len(lines)):
        depth += lines[i].count('{')
        depth -= lines[i].count('}')
        if depth == 0:
            return i
    return -1


#  MAIN PARSER
def parse_struct_body(lines):
    """Return a list[dict] describing each member inside a struct."""
    fields = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # skip comments / blank
        if not line or line.startswith(('#', '//', '/*')):
            i += 1
            continue

        # unions – we take the *largest* member as placeholder
        if line.startswith('union'):
            start = i
            if '{' not in line:
                i += 1
                while i < len(lines) and '{' not in lines[i]:
                    i += 1
            end = find_matching_brace(lines, i)
            if end == -1:
                break
            # naïvely pick first member for now
            union_member = parse_field_line(lines[i+1])
            if union_member:
                union_member['name'] = re.sub(r'.*}\s*', '', lines[end]).strip() or \
                                       union_member['name']
                fields.append(union_member)
            i = end + 1
            continue

        # nested struct – flatten with prefix
        if line.startswith('struct'):
            start = i
            if '{' not in line:
                i += 1
                while i < len(lines) and '{' not in lines[i]:
                    i += 1
            end = find_matching_brace(lines, i)
            if end == -1:
                break
            nested_name = re.sub(r'.*}\s*', '', lines[end]).strip()
            nested_body = lines[i+1:end]
            for nf in parse_struct_body(nested_body):
                nf['name'] = f"{nested_name}_{nf['name']}"
                fields.append(nf)
            i = end + 1
            continue

        # plain member
        info = parse_field_line(line)
        if info:
            fields.append(info)
        i += 1
    return fields


#  NASM GENERATOR
def generate_nasm_struct(name, fields):
    out = [f"struc {name}"]
    used = set()
    for fld in fields:
        fname = fld['name']
        # avoid duplicate names
        n = 1
        while fname in used:
            fname = f"{fld['name']}_{n}"
            n += 1
        used.add(fname)

        original_type = normalize_type(fld.get('type', ''))
        comment = ''
        if original_type and original_type not in type_map and not original_type.endswith('*'):
            comment = f" ; {original_type}"

        out.extend(
            convert_field_to_nasm(fld.get('type', 'unsigned long'),
                                  fname,
                                  fld.get('array_size'),
                                  fld.get('is_bitfield', False),
                                  fld.get('bitfield_size'),
                                  comment)
        )
    out.append("endstruc")
    return out


#  FILE-LEVEL UTILITIES
def extract_structs(code: str):
    structs = {}
    lines = code.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('typedef struct') or \
           (line.startswith('struct') and '{' in line):

            # locate opening brace
            while '{' not in lines[i]:
                i += 1
            start = i
            end = find_matching_brace(lines, start)
            if end == -1:
                break

            body = lines[start + 1:end]
            # struct name after closing brace
            name_match = re.search(r'}\s*(\w+)', lines[end])
            if name_match:
                name = name_match.group(1)
                structs[name] = body
            i = end + 1
            continue

        i += 1
    return structs


def convert_file(in_path, out_path, verbose=True):
    try:
        code = open(in_path, encoding='utf-8').read()
    except OSError as e:
        print(f"Cannot read {in_path}: {e}")
        return False

    structs = extract_structs(code)
    if not structs:
        print("No structs found.")
        return False

    output = [
        "; Generated by C-to-NASM Struct Converter v2.0",
        f"; Source: {in_path}",
        ""
    ]
    for name, body in structs.items():
        if verbose:
            print(f"  > {name}")
        output.extend(generate_nasm_struct(name, parse_struct_body(body)))
        output.append("")

    try:
        with open(out_path, 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(output))
    except OSError as e:
        print(f"Cannot write {out_path}: {e}")
        return False

    if verbose:
        print(f"Written to {out_path}")
    return True


#  MAIN
def main():
    ap = argparse.ArgumentParser(
        description="C-struct → NASM struc converter",
        add_help=False
    )
    ap.add_argument('-i', '--input', help='C header file')
    ap.add_argument('-o', '--output', help='Output .asm file')
    ap.add_argument('-q', '--quiet', action='store_true')
    ap.add_argument('-h', '--help', action='store_true')

    args = ap.parse_args()

    if args.help or not args.input:
        print_banner()
        if not args.input:
            print("Error: Missing required input file (-i)")
        print()
        print("Correct usage:")
        print("    ctonasm.py -i <input.h> [-o <output.inc>]")
        print()
        print("Example:")
        print("    ctonasm.py -i my_structs.h -o my_structs.inc")
        return

    out_file = args.output or (
        os.path.splitext(os.path.basename(args.input))[0] + "_structs.asm"
    )

    if not args.quiet:
        print_banner()
        print(f"Converting '{args.input}' → '{out_file}'")
        print()

    convert_file(args.input, out_file, verbose=not args.quiet)
if __name__ == '__main__':

    main()
