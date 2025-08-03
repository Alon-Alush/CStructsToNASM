#!/usr/bin/env python3
"""
C to NASM Struct Converter (Final Perfect Version)
Author: Alon Alush / alonalush5@gmail.com
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import datetime

# Comprehensive type definitions
TYPES = {
    # Basic C types
    "void": (1, "resb"), "char": (1, "resb"), "signed char": (1, "resb"), 
    "unsigned char": (1, "resb"), "short": (2, "resw"), "unsigned short": (2, "resw"),
    "int": (4, "resd"), "unsigned int": (4, "resd"), "long": (4, "resd"), 
    "unsigned long": (4, "resd"), "long long": (8, "resq"), "unsigned long long": (8, "resq"),
    "float": (4, "resd"), "double": (8, "resq"), "long double": (8, "resq"),
    
    # stdint types
    "int8_t": (1, "resb"), "uint8_t": (1, "resb"), "int16_t": (2, "resw"), 
    "uint16_t": (2, "resw"), "int32_t": (4, "resd"), "uint32_t": (4, "resd"),
    "int64_t": (8, "resq"), "uint64_t": (8, "resq"),
    
    # Windows types
    "BOOLEAN": (1, "resb"), "BYTE": (1, "resb"), "WORD": (2, "resw"), "DWORD": (4, "resd"),
    "QWORD": (8, "resq"), "BOOL": (4, "resd"), "INT": (4, "resd"), "UINT": (4, "resd"),
    "LONG": (4, "resd"), "ULONG": (4, "resd"), "LONGLONG": (8, "resq"), "ULONGLONG": (8, "resq"),
    "SHORT": (2, "resw"), "USHORT": (2, "resw"), "CHAR": (1, "resb"), "UCHAR": (1, "resb"),
    "WCHAR": (2, "resw"), "TCHAR": (2, "resw"), "INT8": (1, "resb"), "UINT8": (1, "resb"),
    "INT16": (2, "resw"), "UINT16": (2, "resw"), "INT32": (4, "resd"), "UINT32": (4, "resd"),
    "INT64": (8, "resq"), "UINT64": (8, "resq"), "LONG32": (4, "resd"), "ULONG32": (4, "resd"),
    "LONG64": (8, "resq"), "ULONG64": (8, "resq"), "DWORD32": (4, "resd"), "DWORD64": (8, "resq"),
    "DWORDLONG": (8, "resq"), "HRESULT": (4, "resd"), "NTSTATUS": (4, "resd"),
    "COLORREF": (4, "resd"), "LCID": (4, "resd"), "LANGID": (2, "resw"), "ATOM": (2, "resw"),
    "USN": (8, "resq"), "FLOAT": (4, "resd"),
    
    # Special Windows structures (known sizes)
    "LARGE_INTEGER": (8, "resq"), "ULARGE_INTEGER": (8, "resq"), "GUID": (16, "resb"),
    "UNICODE_STRING": (16, "resb"), "LIST_ENTRY": (16, "resb"), "CLIENT_ID": (16, "resb"),
    "NT_TIB": (56, "resb"), "ACTIVATION_CONTEXT_STACK": (40, "resb"), "GDI_TEB_BATCH": (1256, "resb"),
    "PROCESSOR_NUMBER": (4, "resd"), "TEB_ACTIVE_FRAME": (24, "resb"), "TEB_ACTIVE_FRAME_CONTEXT": (16, "resb")
}

@dataclass
class Field:
    name: str
    type_name: str
    array_size: Optional[str] = None
    bitfield_bits: Optional[int] = None
    in_union: bool = False
    comment: str = ""

class TypeRegistry:
    def __init__(self, ptr_size: int = 8):
        self.ptr_size = ptr_size
        self.ptr_directive = "resq" if ptr_size == 8 else "resd"
        self.types = TYPES.copy()
        self.defines = {}
        self.structs = {}
        
        # Add all pointer types
        pointer_types = [
            "SIZE_T", "SSIZE_T", "INT_PTR", "UINT_PTR", "LONG_PTR", "ULONG_PTR", "DWORD_PTR",
            "LRESULT", "WPARAM", "LPARAM", "HANDLE", "PVOID", "LPVOID", "LPCVOID",
            "PSTR", "LPSTR", "LPCSTR", "PWSTR", "LPWSTR", "LPCWSTR", "PTSTR", "LPTSTR", "LPCTSTR",
            "PBYTE", "PWORD", "PDWORD", "PULONG", "PLONG", "PULONG_PTR", "PLONG_PTR",
            "PPEB", "PPEB_LDR_DATA", "PRTL_USER_PROCESS_PARAMETERS", "PRTL_CRITICAL_SECTION",
            "PTEB_ACTIVE_FRAME", "PACTIVATION_CONTEXT_STACK"
        ]
        
        for ptype in pointer_types:
            self.types[ptype] = (ptr_size, self.ptr_directive)
        
        # Add P-prefixed versions
        for base_type in list(TYPES.keys()):
            if base_type.isupper() and not base_type.startswith('P'):
                self.types[f"P{base_type}"] = (ptr_size, self.ptr_directive)
    
    def add_define(self, name: str, value: str):
        self.defines[name] = value.strip()
    
    def add_struct(self, name: str, size: int):
        self.structs[name] = size
        self.types[name] = (size, "resb")
    
    def is_pointer(self, type_str: str) -> bool:
        clean_type = re.sub(r'\b(?:const|volatile|struct|union)\b', '', type_str).strip()
        return ('*' in clean_type or clean_type.endswith('_PTR') or 
                (clean_type.startswith('P') and clean_type not in ['PROCESSOR_NUMBER']))
    
    def get_type_info(self, type_str: str) -> Tuple[int, str]:
        clean_type = re.sub(r'\b(?:const|volatile|struct|union)\b', '', type_str).strip()
        
        if self.is_pointer(type_str):
            return (self.ptr_size, self.ptr_directive)
        
        base_type = clean_type.rstrip('*& ')
        if base_type in self.types:
            return self.types[base_type]
        
        return (4, "resd")  # fallback
    
    def evaluate_expr(self, expr: str) -> int:
        # Replace defines first
        for name, value in self.defines.items():
            expr = re.sub(r'\b' + re.escape(name) + r'\b', value, expr)
        
        # Handle sizeof expressions
        def replace_sizeof(match):
            type_name = match.group(1).strip()
            size, _ = self.get_type_info(type_name)
            return str(size)
        
        expr = re.sub(r'sizeof\s*\(\s*([^)]+)\s*\)', replace_sizeof, expr)
        
        # Try to evaluate arithmetic
        try:
            if all(c in '0123456789+-*/() ' for c in expr):
                return int(eval(expr))
        except:
            pass
        
        try:
            return int(expr)
        except:
            return 1

def find_matching_brace(text: str, start: int) -> int:
    """Find matching closing brace"""
    if text[start] != '{':
        return -1
    
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1

def strip_comments(code: str) -> str:
    """Remove C/C++ comments"""
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code

def handle_conditionals(code: str, is_64bit: bool) -> str:
    """Handle preprocessor conditionals"""
    lines = code.split('\n')
    result = []
    skip_stack = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#ifdef _WIN64'):
            skip_stack.append(not is_64bit)
        elif stripped.startswith('#ifndef _WIN64'):
            skip_stack.append(is_64bit)
        elif stripped.startswith('#else') and skip_stack:
            skip_stack[-1] = not skip_stack[-1]
        elif stripped.startswith('#endif') and skip_stack:
            skip_stack.pop()
        elif stripped.startswith('#if'):
            skip_stack.append(False)
        elif not any(skip_stack):
            result.append(line)
    
    return '\n'.join(result)

def extract_structs(code: str) -> List[Tuple[str, str]]:
    """Extract typedef struct definitions"""
    structs = []
    i = 0
    
    while i < len(code):
        # Look for typedef struct
        typedef_match = re.search(r'typedef\s+struct\s+(?:_\w+\s*)?', code[i:])
        if not typedef_match:
            break
        
        typedef_start = i + typedef_match.start()
        struct_def_end = i + typedef_match.end()
        
        # Find opening brace
        brace_match = re.search(r'\{', code[struct_def_end:])
        if not brace_match:
            i = struct_def_end
            continue
        
        brace_start = struct_def_end + brace_match.start()
        brace_end = find_matching_brace(code, brace_start)
        if brace_end == -1:
            i = brace_start + 1
            continue
        
        # Extract body
        body = code[brace_start + 1:brace_end]
        
        # Find typedef name
        remaining = code[brace_end + 1:brace_end + 200]
        name_match = re.match(r'\s*(\w+)(?:\s*,\s*\*\s*\w+)?\s*;', remaining)
        if name_match:
            name = name_match.group(1)
            if not name.startswith('P'):
                structs.append((name, body))
        
        i = brace_end + 1
    
    return structs

def parse_struct_fields(body: str) -> List[Field]:
    """Parse struct fields handling unions properly"""
    fields = []
    
    # Clean up the body
    lines = []
    for line in body.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('//'):
            lines.append(line)
    
    i = 0
    union_depth = 0
    struct_depth = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Track union/struct nesting
        if line.startswith('union'):
            union_depth += 1
            # Add union marker comment
            if union_depth == 1:  # Top-level union
                fields.append(Field(name=f"union_{len(fields)}", type_name="union", comment="union start"))
            i += 1
            continue
        elif line.startswith('struct') and not line.endswith(';'):
            struct_depth += 1
            i += 1
            continue
        elif line == '{':
            i += 1
            continue
        elif line.startswith('}'):
            if union_depth > 0:
                union_depth -= 1
            elif struct_depth > 0:
                struct_depth -= 1
            i += 1
            continue
        
        # Parse field declarations
        if line.endswith(';'):
            field = parse_field(line, in_union=(union_depth > 0))
            if field:
                fields.append(field)
        
        i += 1
    
    return fields

def parse_field(line: str, in_union: bool = False) -> Optional[Field]:
    """Parse field declaration"""
    line = line.rstrip(';').strip()
    
    # Bitfield
    bitfield_match = re.match(r'(.+?)\s+(\w+)\s*:\s*(\d+)', line)
    if bitfield_match:
        return Field(
            name=bitfield_match.group(2),
            type_name=bitfield_match.group(1).strip(),
            bitfield_bits=int(bitfield_match.group(3)),
            in_union=in_union
        )
    
    # Array
    array_match = re.match(r'(.+?)\s+(\w+)\s*\[\s*([^\]]+)\s*\]', line)
    if array_match:
        return Field(
            name=array_match.group(2),
            type_name=array_match.group(1).strip(),
            array_size=array_match.group(3).strip(),
            in_union=in_union
        )
    
    # Regular field
    field_match = re.match(r'(.+?)\s+(\w+)$', line)
    if field_match:
        return Field(
            name=field_match.group(2),
            type_name=field_match.group(1).strip(),
            in_union=in_union
        )
    
    return None

def calculate_struct_size(fields: List[Field], registry: TypeRegistry) -> int:
    """Calculate struct size with alignment"""
    offset = 0
    max_align = 1
    
    for field in fields:
        if field.type_name == "union":
            continue  # Skip union markers
            
        size, _ = registry.get_type_info(field.type_name)
        align = min(size, 8) if size <= 8 else 8
        
        if field.array_size:
            try:
                count = registry.evaluate_expr(field.array_size)
                size *= count
            except:
                pass
        
        # Apply alignment
        if align > 1:
            offset = (offset + align - 1) // align * align
        
        offset += size
        max_align = max(max_align, align)
    
    # Final padding
    if max_align > 1:
        offset = (offset + max_align - 1) // max_align * max_align
    
    return offset

def generate_nasm_field(field: Field, registry: TypeRegistry) -> str:
    """Generate NASM field"""
    if field.type_name == "union":
        return "    ; ── union ──"
    
    size, directive = registry.get_type_info(field.type_name)
    
    # Bitfield
    if field.bitfield_bits:
        comment = f" ; {field.type_name}:{field.bitfield_bits} bits"
        return f"    .{field.name:<24} {directive} 1{comment}"
    
    # Array
    if field.array_size:
        try:
            count = registry.evaluate_expr(field.array_size)
            total_size = size * count
            comment = f" ; {field.type_name}[{field.array_size}]"
            return f"    .{field.name:<24} resb {total_size}{comment}"
        except:
            comment = f" ; {field.type_name}[{field.array_size}]"
            return f"    .{field.name:<24} {directive} {field.array_size}{comment}"
    
    # Regular field
    comment = f" ; {field.type_name}" if field.type_name not in TYPES or field.type_name in registry.structs else ""
    return f"    .{field.name:<24} {directive} 1{comment}"

def generate_struct(name: str, fields: List[Field], registry: TypeRegistry) -> str:
    """Generate NASM struct"""
    lines = [f"struc {name}"]
    
    for field in fields:
        line = generate_nasm_field(field, registry)
        if line:
            lines.append(line)
    
    lines.append("endstruc")
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description="Convert C structs to NASM")
    parser.add_argument("-i", "--input", required=True, help="Input C header file")
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-m", "--mode", choices=["32", "64"], default="64", help="Pointer size")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    ptr_size = 8 if args.mode == "64" else 4
    registry = TypeRegistry(ptr_size)
    
    # Read input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        return 1
    
    try:
        code = input_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading {input_path}: {e}", file=sys.stderr)
        return 1
    
    # Process code
    code = strip_comments(code)
    code = handle_conditionals(code, ptr_size == 8)
    
    # Extract defines
    for match in re.finditer(r'#define\s+(\w+)\s+([^\r\n]*)', code):
        registry.add_define(match.group(1), match.group(2).strip())
    
    # Extract structs
    structs = extract_structs(code)
    if not structs:
        print("No structs found", file=sys.stderr)
        return 1
    
    # First pass: register struct sizes
    for name, body in structs:
        fields = parse_struct_fields(body)
        size = calculate_struct_size(fields, registry)
        registry.add_struct(name, size)
    
    # Generate output
    output_lines = [
        "; " + "=" * 70,
        f";  Generated by ctonasm_final.py",
        f";  Source: {input_path.name}",
        f";  Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f";  Mode: {ptr_size * 8}-bit",
        "; " + "=" * 70,
        ""
    ]
    
    for name, body in structs:
        if not args.quiet:
            size = registry.structs[name]
            print(f"  {name} ({size} bytes)")
        
        fields = parse_struct_fields(body)
        output_lines.append(generate_struct(name, fields, registry))
        output_lines.append("")
    
    # Write output
    output_path = Path(args.output) if args.output else input_path.with_suffix('.inc')
    try:
        output_path.write_text('\n'.join(output_lines), encoding='utf-8')
        if not args.quiet:
            print(f"Generated {output_path}")
    except Exception as e:
        print(f"Error writing {output_path}: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())