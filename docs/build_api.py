def rst_comment():
    text = "This file has been auto-generated. DO NOT MODIFY ITS CONTENT"
    bars = "=" * len(text)
    return f"..\n  {bars}\n  {text}\n  {bars}\n\n"


def build_doxygen_page(symbol):
    name = symbol[0]
    kind = symbol[1]
    content = rst_comment()
    content += f".. _{name}:\n\n"
    content += name + "\n" + "=" * len(name) + "\n"
    content += f".. doxygen{kind}:: kmm::{name}\n"

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
            filename = build_doxygen_page(symbol)
            children.append(filename)

            filename = filename.replace(".rst", "")
            body += f"* :doc:`{symbol[0]} <{filename}>`\n"

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
    "Runtime": [["make_runtime", "function"], ["RuntimeHandle", "class"], ["Runtime", "class"], ["RuntimeConfig", "struct"]],
    "Data": [["Array", "class"], ["Dim", "struct"], ["Range", "class"], ["Bounds", "class"], ["bounds", "function"], ["Domain", "struct"], ["TileDomain", "struct"], ["write", "function"]],
    #"Views": [["View", "struct"], ["GPUSubview", "struct"], ["GPUSubviewMut", "struct"]],
    "Resources": [["ResourceId", "class"], ["MemoryId", "struct"], ["DeviceId", "struct"], ["SystemInfo", "class"]],
    "Events and Execution": [["EventId", "struct"], ["GPUKernel", "struct"], ["Host", "struct"]],
    "Reductions": [["reduce", "function"]],
}


build_index_page(groups)
