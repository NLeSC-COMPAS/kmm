def rst_comment():
    text = "This file has been auto-generated. DO NOT MODIFY ITS CONTENT"
    bars = "=" * len(text)
    return f"..\n  {bars}\n  {text}\n  {bars}\n\n"


def build_doxygen_page(name, items):
    print(f"name : {name}")
    print(f"items : {items}")
    content = rst_comment()
    content += f".. _{name}:\n\n"
    content += name + "\n" + "=" * len(name) + "\n"

    for item in items:
        directive = "doxygenclass" if item[0].isupper() else "doxygenfunction"
        content += f".. {directive}:: kmm::{item}\n"

    filename = f"api/{name}.rst"
    print(f"writing to {filename}")

    with open(filename, "w") as f:
        f.write(content)

    return filename


def build_index_page(groups):
    body = ""
    children = []

    for groupname, symbols in groups.items():
        body += f".. raw:: html\n\n   <h2>{groupname}</h2>\n\n"

        for symbol in symbols:
            if isinstance(symbol, str):
                name = symbol
                items = [symbol]
            else:
                name, items = symbol

            filename = build_doxygen_page(name, items)
            children.append(filename)

            filename = filename.replace(".rst", "")
            body += f"* :doc:`{name} <{filename}>`\n"

        body += "\n"

    title = "API Reference"
    content = rst_comment()
    content += title + "\n" + "=" * len(title) + "\n"
    content += ".. toctree::\n"
    content += "   :titlesonly:\n"
    content += "   :hidden:\n\n"

    for filename in sorted(children):
        content += f"   {filename}\n"

    content += "\n"
    content += body + "\n"

    filename = "api.rst"
    print(f"writing to {filename}")

    with open(filename, "w") as f:
        f.write(content)

    return filename


groups = {
    "Runtime": ["make_runtime", "RuntimeHandle", "Runtime", "RuntimeConfig"],
    "Data": ["Array", "Dim", "Range", "Bounds", "bounds", "Domain", "TileDomain", "write"],
    "Views": ["View", "GPUSubview", "GPUSubviewMut"],
    "Resources": ["ResourceId", "MemoryId", "DeviceId", "SystemInfo"],
    "Events and Execution": ["EventId", "GPUKernel", "Host"],
    "Reductions": ["reduce", "Reduction"],
}


build_index_page(groups)
