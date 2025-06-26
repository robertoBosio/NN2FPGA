import json

board_part_dict = {
    "PYNQ": "xc7z020clg400-1",
    "ZC706": "xc7z045ffg900-2",
    "ULTRA96v2": "xczu3eg-sbva484-1-i",
    "ZCU102": "xczu9eg-ffvb1156-2-e",
    "KRIA": "xck26-sfvc784-2LV-c",
    "U280": "xcu280-fsvh2892-2L-e",
    "U250": "xcu250-figd2104-2L-e",
    "U55C": "xcu55c-fsvh2892-2L-e"
}

def board_to_part(board: str) -> str:
    """
    Convert board name to FPGA part name and board part.
    
    Args:
        board (str): Name of the board.
        
    Returns:
        str: FPGA part name.
    """
    if board not in board_part_dict:
        raise ValueError(f"Unsupported board '{board}'. Allowed boards are: {', '.join(board_part_dict.keys())}")
    
    return board_part_dict[board]

def read_board_info(board, prj_root):
    """ Read the board json file and returns a dictionary with the available resources"""
    
    # Remove the part before NN2FPGA from the path
    file_path = prj_root.split("NN2FPGA")[0]
    file_path = f"{file_path}/NN2FPGA/nn2fpga/boards/{board}.json"

    # Opening JSON file with board resources
    with open(file_path) as f:
        board_dict = json.load(f)

    # Right now consider the board as a monolithic block 
    board_res = {"uram" : 0, "bram" : 0, "dsp" : 0, "lut" : 0, "ff" : 0}
    for block in board_dict['resource']:
        for res in block.keys():
            if res in board_res:
                board_res[res] += block[res]
    board_res["axi_bitwidth"] = board_dict["axi_bitwidth"]
    
    return board_res

    
    