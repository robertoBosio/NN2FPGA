
def parse(name, node):

    block = {} 
    block["declare"] = []
    block["pragma"] = []
    block["template"] = []
    block["args"] = []
    block["defines"] = {}
    block["func"] = "tensor_duplicator"

    producer = node["input"][0]

    for consumer in node["output"]:
        
        
        block["defines"][f"t_{consumer}"] = ["alias", f"t_{producer}"]
        block["defines"][f"t_{consumer}_struct"] = ["alias", f"t_{producer}_struct"]
        block["defines"][f"t_{consumer}_vector"] = ["alias", f"t_{producer}_vector"]
        # block["defines"][f"c_{consumer}_add_ops"] = ["const", f"c_{producer}_add_ops"]
        declare = {}
        declare["name"] = f"s_{consumer}"
        declare["type"] = f"t_{consumer}_struct"
        declare["is_array"] = True
        declare["dim"] = node["ow_ops"]
        block["declare"].append(declare)

        depth = int((node["C"] / node["ops"]) + 1)
        if (f"depth_{consumer}" in node.keys()):
            print(f"Found depth_{consumer} in node.keys() for {consumer}")
            depth = node[f"depth_{consumer}"]
        
        pragma = {}
        pragma["name"] = "stream"
        pragma_name = f"s_{consumer}"
        options = [
            ["variable", pragma_name],
            ["depth", depth],
            ["type", "fifo"],
        ]
        pragma["options"] = options
        block["pragma"].append(pragma)

    block["template"].append(f"t_{producer}_struct")
    block["template"].append({"name" : node["C"], "comment" : "Tensor channels"})
    block["template"].append({"name" : node["H"], "comment" : "Tensor height"})
    block["template"].append({"name" : node["W"], "comment" : "Tensor width"})
    block["template"].append({"name" : node["ops"], "comment" : "Number of channel packed per packet"})
    block["template"].append({"name" : node["ow_ops"], "comment" : "Number of streams in parallel"})

    block["args"].append(f"s_{producer}")
    for consumer in node["output"]:
        block["args"].append(f"s_{consumer}")

    return block