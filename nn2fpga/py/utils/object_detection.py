from torch import nn
import torch

def quantize_tensor(x, bit_width=8, scale=-9.0):
    # Quantize the input tensor
    x = (x * 2**scale)
    return x


def detect_lut(nc, anchors, ch, input_shape, stride=32, bit_width=8, scale=-9.0):

    # Declare the Detect module
    detect = Detect(nc=nc, anchors=anchors, ch=ch)

    detect.eval()
    detect.stride = stride

    # Create a file to write the output
    # with open("../work/detect_lut.txt", "w") as f:
    #     f.write("")

    val_list = []
    # Iterate over signed integer range with number of bits equal to bit_width
    for i in range(0, 2**(bit_width-1)):
        # Declare the input tensor
        x = torch.ones(input_shape)*i

        # Quantize the input tensor
        x = quantize_tensor(x, bit_width=bit_width, scale=scale)

        y = detect([x])

        # Write the output to a file
        val_list.append(
            [y[0][0, 0, ...].tolist()[0],
            y[0][0, 0, ...].tolist()[2],
            y[0][0, 0, ...].tolist()[4]])

    print(len(val_list))

    for i in range(-2**(bit_width-1), 0):
        # Declare the input tensor
        x = torch.ones(input_shape)*i

        # Quantize the input tensor
        x = quantize_tensor(x, bit_width=bit_width, scale=scale)

        y = detect([x])

        # Write the output to a file
        val_list.append(
            [y[0][0, 0, ...].tolist()[0],
            y[0][0, 0, ...].tolist()[2],
            y[0][0, 0, ...].tolist()[4]])

    print(len(val_list))
    anchor_grid = []
    for data in detect.anchor_grid:
        # get possible values of grid from detect.grid element
        anchor_grid_values_x = data[..., :, 0].unique().tolist()
        anchor_grid_values_y = data[..., :, 1].unique().tolist()
        anchor_grid.append([anchor_grid_values_x, anchor_grid_values_y])
    
    # print("anchor_grid: ", anchor_grid)
    # exit()
    
    grid_h = []
    grid_w = []
    for data in detect.grid:
        # get possible values of grid from detect.grid element
        grid_values_w = data[..., :, 0].unique().tolist()
        grid_values_h = data[..., :, 1].unique().tolist()
        grid_h.append(grid_values_h)
        grid_w.append(grid_values_w)

    return val_list, grid_h, grid_w, anchor_grid

    # write the output to a file in ../work/cc/include/ directory
    with open("../work/cc/include/detect_lut.h", "w") as f:
        # write the header file for the lut
        f.write("#ifndef DETECT_LUT_H\n")
        f.write("#define DETECT_LUT_H\n")
        f.write("\n")
        f.write("#include <ap_int.h>\n")
        f.write("#include \"hls_vector.h\"\n")
        f.write("\n")
        f.write("const hls::vector<ap_fixed<32, 16>, 3> detect_lut[%0d] = {\n" % int(2**bit_width))
        # write the lut values
        for i in range(len(val_list)):
            # for each field of the val_list row
            f.write("\t{")
            for j in range(len(val_list[i])):
                # write the values in the form of hls::vector
                f.write("%f" % val_list[i][j])
                if j < (len(val_list[i])-1):
                    # write comma to file
                    f.write(", ")

            f.write("}")
            # write comma to file
            if i < (len(val_list)-1): 
                f.write(",")
            f.write("\n")

        f.write("};")

        f.write("\n")

        # print(detect.anchor_grid)
        f.write("const ap_fixed<8, 8> grid[%0d] = {\n" % int(x.shape[-1]))
        # write detect.grid to file
        for grid in detect.grid:
            # get possible values of grid from detect.grid element
            grid_values_w = grid[..., :, 0].unique().tolist()
            grid_values_h = grid[..., :, 1].unique().tolist()
            # for each value in grid_values
            for i in range(len(grid_values)):
                # write the values in the form of hls::vector
                f.write("\t%f" % grid_values[i])
                if i < (len(grid_values)-1):
                    # write comma to file
                    f.write(", ")
                f.write("\n")
        f.write("};\n")

        f.write("const hls::vector<ap_uint<16>, 2> anchor_grid[%0d] = {\n" % 3)
        # write detect.grid to file
        for anchor_grid in detect.anchor_grid:
            # get possible values of grid from detect.grid element
            anchor_grid_values_x = anchor_grid[..., :, 0].unique().tolist()
            anchor_grid_values_y = anchor_grid[..., :, 1].unique().tolist()
            # for each value in anchor_grid_values
            for i in range(len(anchor_grid_values_x)):
                # write the values in the form of hls::vector
                f.write("\t\t{%d, %d}" % (anchor_grid_values_x[i], anchor_grid_values_y[i]))
                if i < (len(anchor_grid_values_x)-1):
                    # write comma to file
                    f.write(", ")
                f.write("\n")
        f.write("};\n")

        f.write("const uint16_t stride[%0d] = {\n" % len(detect.stride))
        for i in range(len(detect.stride)):
            f.write("\t%d" % detect.stride[i])
            if i < (len(detect.stride)-1):
                # write comma to file
                f.write(", ")
            f.write("\n")
        f.write("};")

        # end the header file
        f.write("\n")
        f.write("#endif\n")

class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer(
            "anchors", torch.tensor(anchors).float().view(self.nl, -1, 2)
        )  # shape(nl,na,2)
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                # xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                # wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                xy = (xy * 2)  # xy
                wh = (wh * 2) ** 2  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return (
            x
            if self.training
            else (torch.cat(z, 1),)
            if self.export
            else (torch.cat(z, 1), x)
        )

    def _make_grid(
        self, nx=20, ny=20, i=0
    ):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = (
            torch.meshgrid(y, x)
        )  # torch>=0.7 compatibility
        grid = (
            torch.stack((xv, yv), 2).expand(shape) - 0.5
        )  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (
            (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        )
        return grid, anchor_grid

# declare main function
if __name__ == "__main__":
    detect_lut(
        nc = 7,
        anchors=[[2,2,5,4,11,8]],
        ch=[208],
        input_shape=[1, 36, 25, 25],
        stride=[32],
    )