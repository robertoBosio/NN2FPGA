import json

def read_board_info(board):
    """ Read the board json file and returns a dictionary with the available resources"""
    file_path = f"/workspace/NN2FPGA/nn2fpga/boards/{board}.json"

    # Opening JSON file with board resources
    with open(file_path) as f:
        board_dict = json.load(f)

    # Right now consider the board as a monolithic block 
    board_res = {"uram" : 0, "bram" : 0, "dsp" : 0, "lut" : 0, "ff" : 0}

    # Check that the board has the required fields
    if "resource" not in board_dict:
        raise ValueError(f"Board {board} does not have a 'resource' field in its JSON file.")

    if "axi_bitwidth" not in board_dict or not isinstance(board_dict["axi_bitwidth"], int):
        raise ValueError(f"Board {board} does not have a valid 'axi_bitwidth' field in its JSON file.")

    if "PLL_frequency" not in board_dict or not isinstance(board_dict["PLL_frequency"], int):
        raise ValueError(f"Board {board} does not have a valid 'PLL_frequency' field in its JSON file.")

    if "board_part" not in board_dict or not isinstance(board_dict["board_part"], str):
        raise ValueError(f"Board {board} does not have a valid 'board_part' field in its JSON file.")

    if "part" not in board_dict or not isinstance(board_dict["part"], str):
        raise ValueError(f"Board {board} does not have a valid 'part' field in its JSON file.")

    if not isinstance(board_dict["resource"], list):
        raise ValueError(f"Board {board} 'resource' field must be a list of blocks (dictionaries).")

    for resource in board_dict["resource"]:
        if not isinstance(resource, dict):
            raise ValueError(f"Board {board} 'resource' field must contain dictionaries.")

        if "bram" not in resource or not isinstance(resource["bram"], int):
            raise ValueError(f"Board {board} does not have a valid 'bram' field in its 'resource' dictionary.")

        if "uram" not in resource or not isinstance(resource["uram"], int):
            raise ValueError(f"Board {board} does not have a valid 'uram' field in its 'resource' dictionary.")

        if "dsp" not in resource or not isinstance(resource["dsp"], int):
            raise ValueError(f"Board {board} does not have a valid 'dsp' field in its 'resource' dictionary.")

        if "lut" not in resource or not isinstance(resource["lut"], int):
            raise ValueError(f"Board {board} does not have a valid 'lut' field in its 'resource' dictionary.")

        if "ff" not in resource or not isinstance(resource["ff"], int):
            raise ValueError(f"Board {board} does not have a valid 'ff' field in its 'resource' dictionary.")

    for block in board_dict['resource']:
        for res in block.keys():
            if res in board_res:
                board_res[res] += block[res]
    board_res["axi_bitwidth"] = board_dict["axi_bitwidth"]
    board_res["PLL_frequency"] = board_dict["PLL_frequency"]
    board_res["board_part"] = board_dict["board_part"]
    board_res["part"] = board_dict["part"]
    
    return board_res

    
    