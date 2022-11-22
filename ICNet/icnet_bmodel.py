import os
import sys
import bmnett


def main():
    bmnett.compile(
        model="weights/ICNet.pb",
        input_names="Placeholder",
        shapes=[[1024, 2048, 3]],
        output_names="conv6_cls/BiasAdd",
        net_name="ICNet_Cityscapes",
        outdir="./ICNet_bmodel",
        target="BM1684",
        opt=2,
        dyn=False,
        descs='[0, fp32, 0, 1]',
        cmp=True
    )


if __name__ == '__main__':
    main()
