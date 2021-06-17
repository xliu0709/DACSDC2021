
import numpy
infilename = [
    "bn_0_bias",
    "bn_1_bias",
    "bn_2_bias",
    "bn_3_bias",
    "bn_4_bias",
    "bn_5_bias",
    "bn_6_bias",
    "bn_7_bias",
    "bn_0_weight",
    "bn_1_weight",
    "bn_2_weight",
    "bn_3_weight",
    "bn_4_weight",
    "bn_5_weight",
    "bn_6_weight",
    "bn_7_weight",
    "conv_0",
    "conv_1",
    "conv_2",
    "conv_3",
    "conv_4",
    "conv_5",
    "conv_6",
    "conv_7",
    "conv_8",
]


outfilename = [
    "conv_0_bias",
    "conv_1_bias",
    "conv_2_bias",
    "conv_3_bias",
    "conv_4_bias",
    "conv_5_bias",
    "conv_6_bias",
    "conv_7_bias",
    "conv_0_inc",
    "conv_1_inc",
    "conv_2_inc",
    "conv_3_inc",
    "conv_4_inc",
    "conv_5_inc",
    "conv_6_inc",
    "conv_7_inc",
    "conv_0_w",
    "conv_1_w",
    "conv_2_w",
    "conv_3_w",
    "conv_4_w",
    "conv_5_w",
    "conv_6_w",
    "conv_7_w",
    "conv_8_w",
]

if __name__ == "__main__":
    for i, v in enumerate(infilename):
        path = "weightExtract/weights/4w4a/"+infilename[i]+".npy"
        arr = numpy.load(path)
        print(arr.dtype, arr.max(), arr.min())
        print(outfilename[i], 1+max(
            int(arr.max()).bit_length(), int(arr.min()).bit_length()))
        outpath = "weightExtract/weightbin/4w4a/"+outfilename[i]+".bin"

        arr.tofile(outpath)
