<img width="899" height="101" alt="generated_text" src="https://github.com/user-attachments/assets/42fe9ad7-d96d-4ab2-99f2-61ba44f059df" />

# Introduction

This tool converts **C struct definitions to NASM (Netwide Assembler) format**. It supports a lot of platforms specific types (GCC/Clang/WinAPI) etc, for example (`uint64_t`, `ULONGLONG`, `unsigned __int64`, `UNICODE_STRING`). 

# Example:

This is the C struct in `my_structs.h`:
```c
typedef struct _TEB_ACTIVE_FRAME_CONTEXT
{
    ULONG Flags;
    PSTR FrameName;
} TEB_ACTIVE_FRAME_CONTEXT, * PTEB_ACTIVE_FRAME_CONTEXT;
```

This is the output struct in NASM syntax:
```asm
struc TEB_ACTIVE_FRAME_CONTEXT
    .Flags    resd 1
    .FrameName    resq 1
endstruc
```

# Usage 

## Script 
To use the script, you need to have Python 3 installed on your machine.
```
ctonasm.py -i <structs.h> -o <structs.inc>
```

<img width="672" height="347" alt="image" src="https://github.com/user-attachments/assets/0762a8cf-7a28-45f5-b43c-b6e5eaf14ca6" />

We can see that the output nasm structs are nicely and correctly structred:
<img width="998" height="567" alt="image" src="https://github.com/user-attachments/assets/344cc098-ef90-42f9-b730-5b74cebe7fee" />
## Binaries

If you're too lazy to install Python, you can download the Windows `.exe` binary in the [Releases page](https://github.com/Alon-Alush/CStructsToNASM/releases/new).




# License

This tool is licened under MIT.
