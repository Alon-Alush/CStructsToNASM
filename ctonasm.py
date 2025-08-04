"""
C to NASM Struct Converter
Author  : Alon Alush / alonalush5@gmail.com
"""

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

from pycparser import c_ast, c_parser


# ----------------------------------------------------------------------------
# Type mapping
#
# The mapping between C primitive types and NASM reservation directives.
# ``resb`` reserves a byte, ``resw`` reserves a word (2 bytes), ``resd``
# reserves a double word (4 bytes) and ``resq`` reserves a quad word (8
# bytes).  Note that long on Windows is considered a 32‑bit type, hence
# mapped to ``resd``.  Unrecognised types fall back to ``resd`` with a
# warning.
type_map: Dict[str, str] = {
    # Basic C types -----------------------------------------------------------
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
    "long": "resd",  # Windows long = 32‑bit
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
    "long double": "resq",  # assume 8 bytes on win64

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


def print_banner():
    print("=" * 60)
    print("        C to NASM Struct Converter  –  pycparser edition")
    print("=" * 60)
    print()


def normalize_type(c_type: str) -> str:
    """Strip qualifiers, squeeze spaces, canonicalise pointers.

    ``pycparser`` returns type strings as lists (e.g. ['unsigned', 'int']) for
    primitive types.  We join them into a canonical form and perform a
    normalisation similar to the original script.  Qualifiers such as
    'const', 'volatile' etc. are removed, whitespace is collapsed and
    multiple consecutive ``*`` characters are reduced to a single ``*``.
    """
    # remove qualifiers and storage specifiers
    c_type = re.sub(r'\b(const|volatile|static|extern|struct|union|enum)\b', '', c_type).strip()
    c_type = re.sub(r'\s+', ' ', c_type)
    # normalise pointer spacing
    c_type = c_type.replace(' *', '*').replace('* ', '*')
    c_type = re.sub(r'\*+', '*', c_type)
    return c_type.strip()


def get_nasm_type(c_type: str) -> str:
    """Return the NASM reservation directive for a given C type.

    If the type is present in ``type_map`` we return the associated value.
    Pointers and uppercase names are assumed to be 64‑bit and mapped to
    ``resq``.  Unknown types trigger a warning and default to ``resd``.
    """
    original = c_type
    c_type = normalize_type(c_type)

    # exact hit in table
    if c_type in type_map:
        return type_map[c_type]

    # pointer or reference
    if c_type.endswith(('*', '&')):
        return 'resq'

    # uppercase (e.g. HANDLE)
    if c_type.isupper():
        return 'resq'

    # fallback with warning
    print(f"Warning: unknown type '{original}', assuming 32‑bit resd")
    return 'resd'


def convert_field_to_nasm(c_type: str, name: str, array_size: Optional[str] = None,
                          is_bitfield: bool = False, bitfield_size: Optional[int] = None,
                          comment: str = "") -> List[str]:
    """Convert a C member into one or more NASM lines.

    The function mirrors the behaviour of the original implementation.  A
    bit‑field always reserves a single 32‑bit word regardless of its
    width and appends the number of bits as a comment.  Arrays multiply
    the base directive by the given size.  Scalar fields without an
    explicit count default to ``1`` when the NASM directive does not
    already include a count (e.g. 'resb 16' is preserved as given).
    """
    if is_bitfield:
        # bitfields reserve one 32‑bit unit and annotate the width
        return [f"    .{name}    resd 1 ; bitfield {bitfield_size} bits{comment}"]

    nasm_type = get_nasm_type(c_type)

    # arrays
    if array_size:
        return [f"    .{name}    {nasm_type} {array_size}{comment}"]

    # simple scalar
    if ' ' in nasm_type:  # e.g. 'resb 16'
        return [f"    .{name}    {nasm_type}{comment}"]
    return [f"    .{name}    {nasm_type} 1{comment}"]


class FieldInfo:
    """Structure for representing a flattened struct field.

    Attributes
    ----------
    type : str
        Canonical C type string (possibly with pointer or array notation).
    name : str
        Name of the field, including any prefix for nested structs.
    array_size : Optional[str]
        String representation of the array size if the field is an array.
    is_bitfield : bool
        Whether the field represents a bit‑field.
    bitfield_size : Optional[int]
        Number of bits in a bit‑field.
    comment : str
        Optional comment appended to the NASM line describing the original
        type when it is not found in the type map.
    """

    def __init__(self, type: str, name: str,
                 array_size: Optional[str] = None,
                 is_bitfield: bool = False,
                 bitfield_size: Optional[int] = None,
                 comment: str = ""):
        self.type = type
        self.name = name
        self.array_size = array_size
        self.is_bitfield = is_bitfield
        self.bitfield_size = bitfield_size
        self.comment = comment


def _typename_from_type(type_node: c_ast.Node) -> str:
    """Return a human readable type name from a pycparser type node.

    This helper walks through nested type declarations (e.g. pointers,
    arrays, function pointers) until it reaches a ``TypeDecl``.  It then
    extracts and joins the tokens from the contained ``IdentifierType``.
    Pointer notation is appended using ``*``.  Array dimensions are
    ignored here and handled separately by the caller.
    """
    # handle pointer: add '*' and continue
    if isinstance(type_node, c_ast.PtrDecl):
        base = _typename_from_type(type_node.type)
        return f"{base}*"

    # handle array: treat as base type; actual dimension will be handled
    if isinstance(type_node, c_ast.ArrayDecl):
        base = _typename_from_type(type_node.type)
        return base

    # handle TypeDecl -> IdentifierType or another nested type
    if isinstance(type_node, c_ast.TypeDecl):
        decl = type_node.type
        if isinstance(decl, c_ast.IdentifierType):
            return ' '.join(decl.names)
        if isinstance(decl, c_ast.Struct):
            # struct as a type; return its tag name if available
            return f"struct {decl.name}" if decl.name else "struct"
        if isinstance(decl, c_ast.Enum):
            return f"enum {decl.name}" if decl.name else "enum"
        # unexpected
        return str(decl)

    # fallback for unhandled nodes
    return str(type_node)


def _flatten_decl(decl: c_ast.Decl, prefix: str = "") -> List[FieldInfo]:
    """Flatten a pycparser ``Decl`` into a list of ``FieldInfo`` records.

    Nested structs are flattened by concatenating the prefix with
    an underscore separator.  Unions pick the first declared member as
    representative, mirroring the original script.  Bit‑fields are
    recognised via ``Decl.bitsize`` and recorded with their width.
    ``PtrDecl`` and ``ArrayDecl`` nodes are translated into pointer and
    array representations; array sizes are preserved as strings to
    accommodate expression lengths.
    """
    fields: List[FieldInfo] = []

    # compute current field name with prefix
    field_name = prefix + decl.name if decl.name else prefix.rstrip('_')

    # bitfield size (if any) is stored in Decl.bitsize
    bits = decl.bitsize
    is_bitfield = bits is not None
    bitfield_size = None
    if is_bitfield:
        # bits may be a Constant or an ID; we take the value if constant
        if isinstance(bits, c_ast.Constant):
            bitfield_size = int(bits.value)
        else:
            # non‑constant bitfield widths are preserved as None
            bitfield_size = None

    # underlying type node
    type_node = decl.type

    # handle array declarations
    if isinstance(type_node, c_ast.ArrayDecl):
        # dimension may be a Constant, ID, or expression; we preserve the
        # string representation as given in the original C code
        dim = type_node.dim
        array_size = None
        if dim is not None:
            # pycparser can produce an ID or a Constant or a BinaryOp
            array_size = _expr_to_str(dim)
        base_type_node = type_node.type
        c_type = _typename_from_type(base_type_node)
        comment = ''
        norm_type = normalize_type(c_type)
        # annotate if the type is not in the mapping and not a pointer
        if norm_type and norm_type not in type_map and not norm_type.endswith('*'):
            comment = f" ; {norm_type}"
        fields.append(FieldInfo(c_type, field_name, array_size=array_size,
                                is_bitfield=is_bitfield,
                                bitfield_size=bitfield_size,
                                comment=comment))
        return fields

    # handle pointer declarations
    if isinstance(type_node, c_ast.PtrDecl):
        c_type = _typename_from_type(type_node)
        comment = ''
        norm_type = normalize_type(c_type)
        if norm_type and norm_type not in type_map and not norm_type.endswith('*'):
            comment = f" ; {norm_type}"
        fields.append(FieldInfo(c_type, field_name,
                                is_bitfield=is_bitfield,
                                bitfield_size=bitfield_size,
                                comment=comment))
        return fields

    # handle nested struct declarations
    if isinstance(type_node, c_ast.TypeDecl) and isinstance(type_node.type, c_ast.Struct):
        struct_type = type_node.type
        # nested struct may not have decls (forward declaration)
        if not struct_type.decls:
            # treat as opaque: we don't know the fields; reserve based on type
            c_type = _typename_from_type(type_node)
            comment = ''
            norm_type = normalize_type(c_type)
            if norm_type and norm_type not in type_map and not norm_type.endswith('*'):
                comment = f" ; {norm_type}"
            fields.append(FieldInfo(c_type, field_name,
                                    is_bitfield=is_bitfield,
                                    bitfield_size=bitfield_size,
                                    comment=comment))
            return fields
        # flatten nested struct: prefix each field
        for sub_decl in struct_type.decls:
            sub_fields = _flatten_decl(sub_decl, prefix=f"{field_name}_")
            fields.extend(sub_fields)
        return fields

    # handle union declarations
    if isinstance(type_node, c_ast.TypeDecl) and isinstance(type_node.type, c_ast.Union):
        union_type = type_node.type
        # choose the first member of the union as representative
        if union_type.decls:
            first_decl = union_type.decls[0]
            # union name is preserved; type and array information come from the first member
            # compute type name and array size from the first member
            sub_type_node = first_decl.type
            # array decl: propagate array dimension
            if isinstance(sub_type_node, c_ast.ArrayDecl):
                dim = sub_type_node.dim
                array_size = None
                if dim is not None:
                    array_size = _expr_to_str(dim)
                base_type_node = sub_type_node.type
                c_type = _typename_from_type(base_type_node)
                comment = ''
                norm = normalize_type(c_type)
                if norm and norm not in type_map and not norm.endswith('*'):
                    comment = f" ; {norm}"
                fields.append(FieldInfo(c_type, field_name, array_size=array_size,
                                        is_bitfield=is_bitfield,
                                        bitfield_size=bitfield_size,
                                        comment=comment))
                return fields
            # pointer decl: treat as pointer
            if isinstance(sub_type_node, c_ast.PtrDecl):
                c_type = _typename_from_type(sub_type_node)
                comment = ''
                norm = normalize_type(c_type)
                if norm and norm not in type_map and not norm.endswith('*'):
                    comment = f" ; {norm}"
                fields.append(FieldInfo(c_type, field_name,
                                        is_bitfield=is_bitfield,
                                        bitfield_size=bitfield_size,
                                        comment=comment))
                return fields
            # plain TypeDecl or nested struct/enum: take its type name
            c_type = _typename_from_type(sub_type_node)
            comment = ''
            norm = normalize_type(c_type)
            if norm and norm not in type_map and not norm.endswith('*'):
                comment = f" ; {norm}"
            fields.append(FieldInfo(c_type, field_name,
                                    is_bitfield=is_bitfield,
                                    bitfield_size=bitfield_size,
                                    comment=comment))
            return fields
        # empty union – reserve as 32 bits
        c_type = _typename_from_type(type_node)
        comment = ''
        norm_type = normalize_type(c_type)
        if norm_type and norm_type not in type_map and not norm_type.endswith('*'):
            comment = f" ; {norm_type}"
        fields.append(FieldInfo(c_type, field_name,
                                is_bitfield=is_bitfield,
                                bitfield_size=bitfield_size,
                                comment=comment))
        return fields

    # handle plain TypeDecl with IdentifierType or Enum
    if isinstance(type_node, c_ast.TypeDecl):
        c_type = _typename_from_type(type_node)
        comment = ''
        norm_type = normalize_type(c_type)
        if norm_type and norm_type not in type_map and not norm_type.endswith('*'):
            comment = f" ; {norm_type}"
        fields.append(FieldInfo(c_type, field_name,
                                is_bitfield=is_bitfield,
                                bitfield_size=bitfield_size,
                                comment=comment))
        return fields

    # catch all: treat as opaque scalar
    c_type = _typename_from_type(type_node)
    comment = ''
    norm_type = normalize_type(c_type)
    if norm_type and norm_type not in type_map and not norm_type.endswith('*'):
        comment = f" ; {norm_type}"
    fields.append(FieldInfo(c_type, field_name,
                            is_bitfield=is_bitfield,
                            bitfield_size=bitfield_size,
                            comment=comment))
    return fields


def _expr_to_str(expr: c_ast.Node) -> str:
    """Convert an expression node to source string.

    For array sizes and bit‑field widths, we need to preserve the literal
    text.  ``pycparser`` does not provide the original text, so we fall
    back to ``.value`` for constants or recursively reconstruct
    identifiers and binary expressions.
    """
    if isinstance(expr, c_ast.Constant):
        return expr.value
    if isinstance(expr, c_ast.ID):
        return expr.name
    if isinstance(expr, c_ast.BinaryOp):
        left = _expr_to_str(expr.left)
        right = _expr_to_str(expr.right)
        return f"({left} {expr.op} {right})"
    # fallback
    return str(expr)


def parse_structs(code: str) -> Dict[str, List[FieldInfo]]:
    """Parse C code and return a mapping of struct names to flattened fields.

    Only ``typedef`` statements that introduce a new struct alias are
    considered.  Anonymous structs without a typedef name are ignored,
    mirroring common usage patterns.  Each struct's field list is
    produced by flattening nested structs and processing arrays,
    pointers, bit‑fields and unions accordingly.
    """
    # Build a preamble of dummy typedefs for non‑builtin types in our map.
    # pycparser requires identifiers to be declared as types before use.
    dummy_typedefs = []
    # known C fundamental types to avoid redefining
    builtin_types = {
        'char', 'short', 'int', 'long', 'float', 'double', 'void', '_Bool',
        'signed', 'unsigned'
    }
    for t in type_map:
        # skip names containing whitespace (composite builtins) or pointer stars
        if ' ' in t or '*' in t:
            continue
        # skip built‑in fundamental types that would result in invalid typedefs
        if t in builtin_types:
            continue
        # create a dummy typedef to appease pycparser
        dummy_typedefs.append(f"typedef int {t};")
    preamble = '\n'.join(dummy_typedefs)
    # Strip preprocessor directives (lines starting with '#') to help
    # pycparser handle header files without running a real C preprocessor.
    filtered_code = '\n'.join(
        line for line in code.splitlines()
        if not line.strip().startswith('#')
    )
    # Concatenate preamble and filtered user code.  The preamble adds
    # typedefs for types such as uint32_t so that pycparser accepts
    # them even if the corresponding <stdint.h> is not included.
    filtered_code = preamble + '\n' + filtered_code
    parser = c_parser.CParser()
    try:
        ast = parser.parse(filtered_code)
    except Exception as e:
        raise RuntimeError(f"C parsing failed: {e}")

    structs: Dict[str, List[FieldInfo]] = {}
    for ext in ast.ext:
        # handle typedefs that alias a struct
        if isinstance(ext, c_ast.Typedef):
            typedef_name = ext.name
            type_decl = ext.type
            if isinstance(type_decl, c_ast.TypeDecl):
                underlying = type_decl.type
                if isinstance(underlying, c_ast.Struct):
                    struct_type = underlying
                    if not struct_type.decls:
                        continue
                    fields: List[FieldInfo] = []
                    for decl in struct_type.decls:
                        fields.extend(_flatten_decl(decl))
                    structs[typedef_name] = fields
            continue
        # handle plain struct declarations (not typedef) of the form
        #   struct Foo { ... };
        # They appear as Decl nodes with type=TypeDecl(type=Struct)
        if isinstance(ext, c_ast.Decl):
            # Case 1: ext.type is a TypeDecl whose underlying type is a Struct
            type_decl = ext.type
            if isinstance(type_decl, c_ast.TypeDecl):
                underlying = type_decl.type
                if isinstance(underlying, c_ast.Struct):
                    struct_type = underlying
                    # ensure this is a definition (has decls) and has a tag name
                    if not struct_type.decls or not struct_type.name:
                        continue
                    struct_name = struct_type.name
                    if struct_name in structs:
                        continue
                    fields: List[FieldInfo] = []
                    for decl in struct_type.decls:
                        fields.extend(_flatten_decl(decl))
                    structs[struct_name] = fields
            # Case 2: ext.type itself is a Struct (not wrapped in TypeDecl)
            elif isinstance(type_decl, c_ast.Struct):
                struct_type = type_decl
                # ensure has a tag name and decls
                if not struct_type.decls or not struct_type.name:
                    continue
                struct_name = struct_type.name
                if struct_name in structs:
                    continue
                fields: List[FieldInfo] = []
                for decl in struct_type.decls:
                    fields.extend(_flatten_decl(decl))
                structs[struct_name] = fields
    return structs


def generate_nasm_struct(name: str, fields: List[FieldInfo], known_structs: Optional[set] = None) -> List[str]:
    """Generate NASM ``struc`` block for a struct.

    Names are deduplicated if necessary by appending an underscore and a
    counter.  Fields whose type matches a previously defined struct name
    are emitted using the NASM ``_size`` convention: for example,
    ``struct Address address;`` becomes ``resb Address_size``.  Arrays of
    such structs multiply the size accordingly.  All other fields are
    converted via ``convert_field_to_nasm``.  The function returns a list
    of strings ready for writing to the output file.
    """
    known_structs = known_structs or set()
    out: List[str] = [f"struc {name}"]
    used = set()
    for fld in fields:
        fname = fld.name
        # deduplicate field names
        suffix = 1
        orig_name = fname
        while fname in used:
            fname = f"{orig_name}_{suffix}"
            suffix += 1
        used.add(fname)
        norm_type = normalize_type(fld.type)
        # Check for struct reference: either 'struct Name' or alias 'Name'
        struct_ref = None
        if norm_type.startswith('struct '):
            tag = norm_type.split(' ', 1)[1]
            if tag in known_structs:
                struct_ref = tag
        elif norm_type in known_structs:
            struct_ref = norm_type
        # If it's a reference to a known struct and not a bitfield or pointer
        if struct_ref and not fld.is_bitfield:
            # If array, multiply size
            if fld.array_size:
                # produce expression <struct_ref>_size * <array_size>
                out.append(f"    .{fname}    resb {struct_ref}_size * {fld.array_size}")
            else:
                out.append(f"    .{fname}    resb {struct_ref}_size")
            continue
        # Otherwise fall back to normal conversion
        comment = fld.comment
        lines = convert_field_to_nasm(fld.type, fname, fld.array_size,
                                      fld.is_bitfield, fld.bitfield_size, comment)
        out.extend(lines)
    out.append("endstruc")
    return out


def convert_file(in_path: str, out_path: str, verbose: bool = True) -> bool:
    """Convert a C header file into NASM structures using pycparser.

    This function reads the source file, extracts all typedef'd structs
    and writes corresponding NASM ``struc`` definitions into the output
    file.  Verbose mode prints progress to stdout.  Returns ``True`` on
    success and ``False`` on error.
    """
    try:
        with open(in_path, encoding='utf-8') as fp:
            code = fp.read()
    except OSError as e:
        print(f"Cannot read {in_path}: {e}")
        return False

    try:
        structs = parse_structs(code)
    except RuntimeError as e:
        print(e)
        return False

    if not structs:
        print("No structs found.")
        return False

    output: List[str] = [
        "; Generated by C‑to‑NASM Struct Converter (pycparser edition)",
        f"; Source: {in_path}",
        "",
    ]

    # set of struct names for reference lookup
    known_structs = set(structs.keys())
    for name, fields in structs.items():
        if verbose:
            print(f"  > {name}")
        output.extend(generate_nasm_struct(name, fields, known_structs))
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


def main() -> None:
    """Entry point for command line execution."""
    ap = argparse.ArgumentParser(
        description="C‑struct → NASM struc converter (pycparser edition)",
        add_help=False,
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
        print("    ctonasm_pycparser.py -i <input.h> [-o <output.asm>]")
        print()
        print("Example:")
        print("    ctonasm_pycparser.py -i my_structs.h -o my_structs.asm")
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
