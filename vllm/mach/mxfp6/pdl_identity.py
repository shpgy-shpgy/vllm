# SPDX-License-Identifier: Apache-2.0
"""Pure source-identity helpers from vllm-mach 05aeea3; no compile policy hooks."""

import ast
import copy
import hashlib
import json
import textwrap


def _json_hash(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _triton_type_code(value):
    """Invert real Triton scalar/pointer types without stringifying objects."""
    if type(value).__module__ != "triton.language.core":
        return None
    from triton.language import core as tl
    from triton.language import str_to_ty

    if type(value) is tl.dtype:
        code = value.mangle()
    elif type(value) is tl.pointer_type:
        # The generated signature syntax represents address-space one only.
        # mangle() omits const, so reconstruct the explicit *k spelling.
        if value.address_space != 1 or type(value.const) is not bool:
            raise TypeError("Unsupported Triton pointer address space/const qualifier")
        element = _triton_type_code(value.element_ty)
        if element is None:
            raise TypeError("Unsupported Triton pointer element type")
        code = "*" + ("k" if value.const else "") + element
    else:
        return None
    try:
        restored = str_to_ty(code, None)
    except (KeyError, ValueError, IndexError) as error:
        raise TypeError("Unsupported Triton scalar/pointer type") from error
    if type(restored) is not type(value) or restored != value:
        raise TypeError("Triton type signature roundtrip changed its type")
    return code


def _signature_encoded(signature):
    def argument(value):
        code = _triton_type_code(value)
        if code is not None:
            return code
        if type(value).__module__ == "triton.language.core":
            from triton.language import core as tl

            if type(value) is tl.constexpr_type and value.value is None:
                return "constexpr"
        return _encoded(value)

    return _encoded({name: argument(value) for name, value in signature.items()})


def _encoded(value):
    # Preserve tuple keys in Triton's divisibility specialization metadata.
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: _encoded(v) for k, v in value.items()}
        pairs = [[_encoded(k), _encoded(v)] for k, v in value.items()]
        pairs.sort(key=lambda item: json.dumps(item[0], sort_keys=True))
        return {"__mapping__": pairs}
    if isinstance(value, tuple):
        return {"__tuple__": [_encoded(item) for item in value]}
    if isinstance(value, list):
        return [_encoded(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    code = _triton_type_code(value)
    if code is not None:
        # Constants are typed values, unlike their signature declarations.
        return {"__triton_type__": code}
    raise TypeError(f"Unsupported Triton metadata value: {type(value).__name__}")


def _ast_value(node):
    if isinstance(node, ast.AST):
        fields = {}
        for name in node._fields:
            value = getattr(node, name, None)
            # Empty type_params is absent in Python 3.10/3.11. It has no effect
            # on these ordinary generated functions and must not vary the key.
            if name == "type_params" and not value:
                continue
            fields[name] = _ast_value(value)
        return {"node": type(node).__name__, **fields}
    if isinstance(node, list):
        return [_ast_value(item) for item in node]
    return node


def canonical_function_sha256(source: str, kernel_name: str | None = None) -> str:
    """Hash exact math/args AST, ignoring comments, name, decorators and paths."""
    tree = ast.parse(textwrap.dedent(source))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if kernel_name is not None:
        functions = [node for node in functions if node.name == kernel_name]
    if len(functions) != 1:
        raise ValueError("Expected one Triton function or an exact kernel_name")
    node = copy.deepcopy(functions[0])
    node.name = "kernel"
    node.decorator_list = []
    return _json_hash(_ast_value(node))


def descriptor(source: str, size_hints, triton_meta, *, kernel_name=None) -> dict:
    """Construct the public exact source/shape/compiler matching boundary."""
    device = triton_meta.get("device")
    if isinstance(device, dict):
        arch = {field: device.get(field) for field in ("type", "cc")}
    else:
        arch = {field: getattr(device, field, None) for field in ("type", "cc")}
    return {
        "function_sha256": canonical_function_sha256(source, kernel_name),
        "size_hints": _encoded(size_hints),
        "signature": _signature_encoded(triton_meta.get("signature", {})),
        "constants": _encoded(triton_meta.get("constants", {})),
        "specialization": _encoded(triton_meta.get("configs", [])),
        "compiler_flags": {
            name: triton_meta.get(name, default)
            for name, default in (
                ("enable_fp_fusion", True),
                ("disable_ftz", False),
                ("launch_pdl", False),
                ("native_matmul", False),
            )
        },
        "architecture": arch,
    }


def recipe_from_source(
    source: str, kernel_name: str, selected_fields: dict, binding_id: str
) -> dict:
    """Build an inspectable recipe from generated source without executing it.

    This pure-CPU helper is for maintainers updating the versioned policy. It
    reads no cache and is not used by an installed model's startup path.
    """
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name
    )
    decorator = next(
        node
        for node in function.decorator_list
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "triton_heuristics"
    )
    keywords = {item.arg: item.value for item in decorator.keywords}
    meta_node = keywords["triton_meta"]
    meta = {}
    for key, value in zip(meta_node.keys, meta_node.values):
        key = ast.literal_eval(key)
        if key == "device":
            meta[key] = {
                item.arg: ast.literal_eval(item.value) for item in value.keywords
            }
        else:
            meta[key] = ast.literal_eval(value)
    boundary = descriptor(
        source, ast.literal_eval(keywords["size_hints"]), meta, kernel_name=kernel_name
    )
    imeta_node = keywords["inductor_meta"]
    site = next(
        ast.literal_eval(value)
        for key, value in zip(imeta_node.keys, imeta_node.values)
        if ast.literal_eval(key) == "kernel_name"
    )
    return {
        "id": binding_id,
        "kernel_alias": kernel_name,
        "kernel_site": site,
        "heuristic": decorator.func.attr,
        "boundary": boundary,
        "semantic_key": _json_hash(boundary),
        "selected_fields": dict(selected_fields),
    }
